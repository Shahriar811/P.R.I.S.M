"""
ids_server.py — SDN-IoT multi-page SDN-IoT IDS (Latest One FL+RL)
==============================================================
Pages: Home, Predict, Federated FL, RL Agents, History, Classes, Settings.
APIs: /info, /live, /predict, /explain, /predict_csv
Day/Night theme is client-side (localStorage) and applies across all pages.

    python ids_server.py
    open http://127.0.0.1:7860
"""

import os
import sys
import logging
import time
import threading

import numpy as np
import pandas as pd
import joblib
import torch
from flask import Flask, request, jsonify, render_template

from ids_models import ModelRegistry
from feedback_store import FeedbackStore
from rl_learner import LearnableRLAgent
from rl_learn_routes import register_learn_routes
from rlfl_orchestrator import RLFLSyncManager

try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

HERE = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(HERE, 'Saved model')
ART_DIR = SAVE_DIR
TM_DIR = os.path.join(SAVE_DIR, 'TM')
FL_DIR = os.path.join(HERE, 'federated')


def _require_file(path, hint=''):
    if not os.path.exists(path):
        print(f'\n[ERROR] Missing file: {path}')
        if hint:
            print(f'        → {hint}')
        sys.exit(1)


print('Loading preprocessing artifacts from Saved model/ ...')
_require_file(
    os.path.join(ART_DIR, 'scaler.joblib'),
    'Run the notebook through Section 11 (save artifacts) first.',
)
_require_file(os.path.join(ART_DIR, 'label_encoder.joblib'))
_require_file(os.path.join(ART_DIR, 'feature_cols.joblib'))

scaler = joblib.load(os.path.join(ART_DIR, 'scaler.joblib'))
label_encoder = joblib.load(os.path.join(ART_DIR, 'label_encoder.joblib'))
feature_cols = joblib.load(os.path.join(ART_DIR, 'feature_cols.joblib'))
CLASS_NAMES = label_encoder.classes_.tolist()
N_FEATURES = len(feature_cols)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'  Classes    : {CLASS_NAMES}')
print(f'  N_FEATURES : {N_FEATURES}')
print(f'  Device     : {DEVICE}')

registry = ModelRegistry(SAVE_DIR, N_FEATURES, CLASS_NAMES, DEVICE, fl_dir=FL_DIR)
BEST = registry.BEST_MODEL_NAME

_explainer_cache = {}
_BG = None
_bg_path = os.path.join(FL_DIR, 'data', 'test.npz')
if _HAS_SHAP and os.path.exists(_bg_path):
    try:
        _bgd = np.load(_bg_path)['X'].astype(np.float32)
        idx = np.random.RandomState(0).choice(len(_bgd), min(50, len(_bgd)), replace=False)
        _BG = _bgd[idx]
    except Exception:
        pass


def _explain_row(Xs, k):
    model = registry._best
    if model is None:
        raise RuntimeError('Best model not loaded.')
    cls = type(model).__name__
    
    if _HAS_SHAP:
        try:
            if cls == 'CatBoostClassifier':
                from catboost import Pool
                raw = np.array(model.get_feature_importance(Pool(Xs), type='ShapValues'))
                return raw[0, k, :-1], float(raw[0, k, -1]), 'CatBoost SHAP'
            if cls in {
                'XGBClassifier', 'LGBMClassifier', 'RandomForestClassifier',
                'ExtraTreesClassifier', 'DecisionTreeClassifier',
            }:
                ex = _explainer_cache.get('best')
                if ex is None:
                    ex = shap.TreeExplainer(model)
                    _explainer_cache['best'] = ex
                sv = np.array(ex.shap_values(Xs))
                C = len(CLASS_NAMES)
                F = N_FEATURES
                if sv.ndim == 3 and sv.shape == (1, F, C):
                    vec = sv[0, :, k]
                elif sv.ndim == 3 and sv.shape[0] == C:
                    vec = sv[k, 0, :]
                else:
                    vec = sv.reshape(-1)[:F]
                ev = np.array(ex.expected_value).ravel()
                base = float(ev[k] if ev.size > 1 else ev[0])
                return vec, base, 'TreeExplainer'
        except Exception:
            pass

    # Fallback: model feature_importances_ scaled by feature magnitude
    if hasattr(model, 'feature_importances_'):
        fi = np.array(model.feature_importances_, dtype=np.float32)
        dev = (Xs[0] - 0.5)
        vec = fi * dev
        return vec, 0.0, 'Feature Importance XAI'

    # Pure feature deviation fallback
    dev = (Xs[0] - 0.5)
    return dev, 0.0, 'Feature Deviation'


def _prep(df_in: pd.DataFrame) -> np.ndarray:
    df_in = df_in.copy()
    for c in feature_cols:
        if c not in df_in.columns:
            df_in[c] = 0.0
    X = df_in[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
    return np.nan_to_num(
        scaler.transform(X), nan=0.0, posinf=1.0, neginf=0.0,
    ).astype(np.float32)


def _scale_row(vals):
    raw = pd.DataFrame([vals], columns=feature_cols)
    return np.nan_to_num(
        scaler.transform(raw.values.astype(np.float32)),
        nan=0.0, posinf=1.0, neginf=0.0,
    ).astype(np.float32)


logging.getLogger('werkzeug').setLevel(logging.ERROR)
app = Flask(
    __name__,
    template_folder=os.path.join(HERE, 'templates'),
    static_folder=os.path.join(HERE, 'static'),
)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024

_class_counts = None
_test_path = os.path.join(FL_DIR, 'data', 'test.npz')
if os.path.exists(_test_path):
    try:
        _ty = np.load(_test_path)['y'].astype(np.int64).ravel()
        _class_counts = np.bincount(_ty, minlength=len(CLASS_NAMES)).astype(np.float32)
    except Exception:
        pass

_learner = LearnableRLAgent(
    os.path.join(SAVE_DIR, 'RL'), N_FEATURES, CLASS_NAMES, DEVICE,
    class_counts=_class_counts)
# Broadcast feedback to all 3 FL clients so every client benefits from
# user corrections in the next FedAvg round (not just client 0).
_feedback_stores = [
    FeedbackStore(FL_DIR, N_FEATURES, CLASS_NAMES, client_id=cid)
    for cid in range(3)
]
_sync_settings = {
    'auto_save_new': True,
    'auto_federate': True,
    'novelty_threshold': 0.70,
    'flag_disagreement_as_novel': True,
}


def _reload_federated_global():
    registry.reload_fl()


_sync_manager = RLFLSyncManager(HERE, on_complete=_reload_federated_global)
registry.attach_live_rl(_learner)
registry.attach_feedback_status(lambda: _feedback_stores[0].status())
registry.attach_sync_status(_sync_manager.status)


# ── Background FL model file-watcher ─────────────────────────────────────────
# Polls all candidate FL .pt files every 8 seconds.
# When ANY file's mtime changes (written by the .bat OR the dashboard trigger),
# registry.reload_fl() fires instantly without needing a server restart.
# This means: run_rl_federated.bat finishes → sniffer output switches live.
_fl_watch_paths = [
    os.path.join(SAVE_DIR, 'FL', 'federated_rl_fedavg.pt'),
    os.path.join(SAVE_DIR, 'FL', 'federated_fedavg.pt'),
    os.path.join(FL_DIR,   'global_model.pt'),
    os.path.join(SAVE_DIR, 'DL', 'federated_fedavg.pt'),
]


def _fl_file_watcher():
    """Daemon thread: reload FL global model the moment any .pt file changes on disk."""
    # Seed the initial mtimes so we don't trigger a false reload at startup
    _mtimes = {}
    for p in _fl_watch_paths:
        try:
            _mtimes[p] = os.path.getmtime(p) if os.path.exists(p) else None
        except Exception:
            _mtimes[p] = None

    while True:
        time.sleep(8)   # poll interval — low enough to feel instant, light on CPU
        for p in _fl_watch_paths:
            try:
                mtime = os.path.getmtime(p) if os.path.exists(p) else None
            except Exception:
                mtime = None

            prev = _mtimes.get(p)
            if mtime is not None and mtime != prev:
                _mtimes[p] = mtime
                # Only reload if the registry doesn't already know this file
                # (avoids double-reload when dashboard trigger also fired)
                current_path = (registry._fl or {}).get('path')
                if current_path != p or registry._fl is None:
                    print(f'\n[FL-WATCHER] Detected new global model: {os.path.basename(p)}')
                    print( '[FL-WATCHER] Reloading FL model live — sniffer output will switch now.')
                    registry.reload_fl()
                    loaded = registry._fl
                    if loaded:
                        print(f'[FL-WATCHER] FL model loaded: arch={loaded.get("arch")} '
                              f'pipeline={loaded.get("pipeline")} '
                              f'round={loaded.get("round")}')
                        print('[FL-WATCHER] Sniffer will now show: | Routed to FL Client X\n')
                    else:
                        print('[FL-WATCHER] Reload attempted but model could not be parsed.\n')
                    break  # one reload per poll cycle is enough


_watcher_thread = threading.Thread(target=_fl_file_watcher, name='fl-file-watcher', daemon=True)
_watcher_thread.start()
print('[FL-WATCHER] Background FL file-watcher started (polling every 8s).')


# ── Cached FL-processes-alive check ─────────────────────────────────────────
# Returns True ONLY when the Flower RL-FL processes (rl_server.py + rl_client.py)
# are CURRENTLY running. This is distinct from "a FL model exists on disk".
# Reads the status.json files that Flower writes during training (cheap I/O, no psutil).
# Result is cached for 10 seconds so /predict stays fast.
_fl_alive_cache = {'value': False, 'ts': 0.0}
_FL_ALIVE_CACHE_TTL = 10.0  # seconds


def _fl_processes_alive() -> bool:
    """Return True if the FL Flower server or any client is currently running.

    Logic: reads status.json files written by rl_server.py / rl_client.py.
    A process is considered alive when BOTH conditions hold:
      1. state is NOT a terminal value ('finished', 'done', 'stopped', 'failed')
      2. The PID stored in the file is still a live OS process.
    This correctly handles stale files left over from previous runs.
    """
    now = time.time()
    if now - _fl_alive_cache['ts'] < _FL_ALIVE_CACHE_TTL:
        return _fl_alive_cache['value']

    # States that mean "the process has already exited"
    TERMINAL_STATES = {'finished', 'done', 'stopped', 'failed', 'error', 'complete', 'completed'}

    alive = False
    try:
        import json as _json

        # Check server status.json
        sp = os.path.join(FL_DIR, 'status.json')
        if os.path.exists(sp):
            try:
                with open(sp, encoding='utf-8') as fh:
                    s = _json.load(fh)
                state = str(s.get('state', 'unknown')).lower()
                pid = s.get('pid')
                if state not in TERMINAL_STATES and registry._pid_alive(pid):
                    alive = True
            except Exception:
                pass

        # Check any of the 3 client status files
        if not alive:
            for cid in range(3):
                cp = os.path.join(FL_DIR, f'client_{cid}_status.json')
                if os.path.exists(cp):
                    try:
                        with open(cp, encoding='utf-8') as fh:
                            c = _json.load(fh)
                        state = str(c.get('state', 'unknown')).lower()
                        pid = c.get('pid')
                        if state not in TERMINAL_STATES and registry._pid_alive(pid):
                            alive = True
                            break
                    except Exception:
                        pass
    except Exception:
        pass

    _fl_alive_cache['value'] = alive
    _fl_alive_cache['ts'] = now
    return alive


def _novelty_flags(prediction, confidence, rl_info, fl_info):
    reasons = []
    rl_primary = (rl_info or {}).get('primary') or {}
    fl_pred = (fl_info or {}).get('prediction')
    if confidence < float(_sync_settings['novelty_threshold']):
        reasons.append('best_model_low_confidence')
    if rl_primary and float(rl_primary.get('confidence', 1.0)) < float(_sync_settings['novelty_threshold']):
        reasons.append('local_rl_low_confidence')
    if fl_info and float(fl_info.get('confidence', 1.0)) < float(_sync_settings['novelty_threshold']):
        reasons.append('federated_rl_low_confidence')
    if (
        _sync_settings['flag_disagreement_as_novel']
        and rl_primary and fl_pred
        and (rl_primary.get('action') != fl_pred or rl_primary.get('action') != prediction)
    ):
        reasons.append('models_disagree')
    return {
        'candidate': bool(reasons),
        'reasons': reasons,
        'threshold': float(_sync_settings['novelty_threshold']),
        'message': (
            'New attack candidate from another dataset/domain. Save it to local RL and optionally trigger a FedAvg round.'
            if reasons else
            'Looks similar to known training distribution.'
        ),
    }


def _handle_feedback(Xs, true_index, meta):
    out = {}
    if _sync_settings['auto_save_new'] or meta.get('mark_new'):
        saved = None
        for store in _feedback_stores:
            saved = store.append(Xs, true_index, meta=meta)
        out['feedback_saved'] = True
        out['feedback'] = saved
        out['feedback_clients_updated'] = len(_feedback_stores)
        try:
            _learner.save(os.path.join(SAVE_DIR, 'RL', 'dqn_improved_learned.pt'))
            out['local_checkpoint_saved'] = True
        except Exception as e:
            out['local_checkpoint_saved'] = False
            out['local_checkpoint_error'] = str(e)
        if _sync_settings['auto_federate']:
            out['federated_sync'] = _sync_manager.trigger(reason='auto_feedback')
    return out


@app.route('/')
def home():
    return render_template('home.html', active='home')


@app.route('/predict-ui')
def predict_page():
    return render_template('predict.html', active='predict')


@app.route('/fl')
def fl_page():
    return render_template('fl.html', active='fl')


@app.route('/rl')
def rl_page():
    return render_template('rl.html', active='rl')


@app.route('/advisor')
def advisor_page():
    return render_template('advisor.html', active='advisor')


@app.route('/history')
def history_page():
    return render_template('history.html', active='history')


@app.route('/about')
def about_page():
    return render_template('about.html', active='about')


@app.route('/settings')
def settings_page():
    return render_template('settings.html', active='settings')


@app.route('/poster')
def poster_page():
    return render_template('poster.html', active='poster')


@app.route('/dashboard')
def dashboard_legacy():
    """Old single-page URL — keep working by sending users to Predict."""
    return render_template('predict.html', active='predict')


@app.route('/info')
def info():
    import groq_explain
    return jsonify({
        'n_features': N_FEATURES,
        'class_names': CLASS_NAMES,
        'feature_columns': feature_cols,
        'device': str(DEVICE),
        'best_model': BEST,
        'best_available': registry.best_available,
        'rl_agents': list(registry._rl.keys()),
        'rl_count': len(registry._rl),
        'fl_loaded': registry._fl is not None,
        'fl_is_rl_fl': bool(registry._fl and registry._fl.get('is_rl_fl')),
        'fl_pipeline': (registry._fl or {}).get('pipeline', 'RL -> FL'),
        'fl_arch': (registry._fl or {}).get('arch'),
        'sync_settings': _sync_settings,
        'feedback': _feedback_stores[0].status(),
        'sync': _sync_manager.status(),
        'llm_available': groq_explain.available(),
        'llm_model': groq_explain.GROQ_MODEL if groq_explain.available() else None,
        'pages': ['/', '/predict-ui', '/fl', '/rl', '/history', '/about', '/settings'],
    })



@app.route('/live')
def live():
    """Live FL + RL status — proves real multi-process FL and live RL inference."""
    return jsonify(registry.live_status())


@app.route('/rlfl_status')
def rlfl_status():
    return jsonify({
        'settings': dict(_sync_settings),
        'feedback': _feedback_stores[0].status(),
        'sync': _sync_manager.status(),
        'learner': _learner.status(),
    })


@app.route('/rlfl_settings', methods=['POST'])
def rlfl_settings():
    d = request.get_json(force=True) or {}
    if 'auto_save_new' in d:
        _sync_settings['auto_save_new'] = bool(d['auto_save_new'])
    if 'auto_federate' in d:
        _sync_settings['auto_federate'] = bool(d['auto_federate'])
    if 'flag_disagreement_as_novel' in d:
        _sync_settings['flag_disagreement_as_novel'] = bool(d['flag_disagreement_as_novel'])
    if 'novelty_threshold' in d:
        try:
            _sync_settings['novelty_threshold'] = float(d['novelty_threshold'])
        except Exception:
            pass
    return jsonify({'ok': True, 'settings': dict(_sync_settings)})


@app.route('/rlfl_trigger', methods=['POST'])
def rlfl_trigger():
    d = request.get_json(force=True) or {}
    reason = d.get('reason') or 'manual_ui'
    return jsonify(_sync_manager.trigger(reason=reason))


_SAMPLE_CACHE = None


def _load_sample_pool():
    """Load held-out test rows once (scaled) for diverse Sample-button draws."""
    global _SAMPLE_CACHE
    if _SAMPLE_CACHE is not None:
        return _SAMPLE_CACHE
    path = os.path.join(FL_DIR, 'data', 'test.npz')
    if not os.path.exists(path):
        _SAMPLE_CACHE = False
        return False
    data = np.load(path)
    X = data['X'].astype(np.float32)
    y = data['y'].astype(np.int64).ravel()
    by_class = {int(c): np.where(y == c)[0] for c in np.unique(y)}
    _SAMPLE_CACHE = {'X': X, 'y': y, 'by_class': by_class}
    return _SAMPLE_CACHE


@app.route('/sample')
def sample_flow():
    """Return a random real test-flow feature vector (class-balanced).

    Features are inverse-scaled to the same space the Predict textarea expects,
    so /predict can MinMax-scale them normally.
    """
    pool = _load_sample_pool()
    prefer = request.args.get('class')  # optional class name
    rng = np.random.default_rng()

    if pool:
        class_ids = sorted(pool['by_class'].keys())
        if prefer:
            # allow name or index
            if prefer.isdigit():
                cid = int(prefer)
            else:
                try:
                    cid = CLASS_NAMES.index(prefer)
                except ValueError:
                    return jsonify({'error': f'Unknown class {prefer!r}. Use one of {CLASS_NAMES}'}), 400
            if cid not in pool['by_class'] or len(pool['by_class'][cid]) == 0:
                return jsonify({'error': f'No samples for class {cid}'}), 404
        else:
            # uniform over classes so DDoS/Benign don't dominate
            nonempty = [c for c in class_ids if len(pool['by_class'][c]) > 0]
            cid = int(rng.choice(nonempty))
        idx = int(rng.choice(pool['by_class'][cid]))
        scaled = pool['X'][idx:idx + 1]
        try:
            raw = scaler.inverse_transform(scaled)[0]
        except Exception:
            raw = scaled[0]
        raw = np.nan_to_num(raw, nan=0.0, posinf=1e6, neginf=-1e6)
        # small jitter so repeated clicks are not identical even for same row
        jitter = rng.normal(0.0, 1e-6, size=raw.shape).astype(np.float64)
        vals = (raw.astype(np.float64) + jitter).tolist()
        return jsonify({
            'features': vals,
            'n_features': len(vals),
            'source': 'test.npz',
            'true_label': CLASS_NAMES[cid] if 0 <= cid < len(CLASS_NAMES) else str(cid),
            'true_index': cid,
            'row_index': idx,
        })

    # Fallback: diverse synthetic profiles if test.npz is missing
    profiles = ['sparse', 'burst', 'heavy', 'probe', 'mixed']
    profile = rng.choice(profiles)
    vals = []
    for i in range(N_FEATURES):
        if profile == 'sparse':
            v = abs(rng.normal(0.05, 0.04)) if rng.random() > 0.35 else abs(rng.normal(0.4, 0.2))
        elif profile == 'burst':
            v = abs(rng.lognormal(mean=-0.5, sigma=0.9))
        elif profile == 'heavy':
            v = abs(rng.normal(0.55, 0.25))
        elif profile == 'probe':
            v = abs(rng.uniform(0.0, 0.25) if i % 3 else rng.uniform(0.3, 0.9))
        else:
            v = abs(rng.normal(0.3, 0.35))
        vals.append(float(min(v, 5.0)))
    return jsonify({
        'features': vals,
        'n_features': len(vals),
        'source': f'synthetic:{profile}',
        'true_label': None,
        'true_index': None,
        'row_index': None,
    })


@app.route('/simulate_client_attack', methods=['POST'])
def simulate_client_attack():
    d = request.get_json(force=True)
    vals = d.get('features', [])
    if len(vals) != N_FEATURES:
        return jsonify({'error': f'Expected {N_FEATURES} features, got {len(vals)}.'}), 400
    
    Xs = _scale_row(vals)
    try:
        proba = registry.predict_best(Xs)[0]
    except Exception as e:
        return jsonify({'error': str(e)}), 500
        
    k = int(np.argmax(proba))
    prediction = CLASS_NAMES[k]
    
    # Assign to a random client (0, 1, or 2)
    rng = np.random.default_rng()
    client_id = int(rng.choice([0, 1, 2]))
    
    feedback_result = {}
    is_attack = prediction != 'Benign'
    
    if is_attack:
        # Simulate the client updating its local RL agent and sending feedback
        try:
            learn_res = _learner.learn_from_correction(Xs, k)
        except Exception:
            learn_res = {}
        meta = {
            'source': 'attack_simulation',
            'simulated_client_id': client_id,
            'mark_new': True
        }
        feedback_result = _handle_feedback(Xs, k, meta)
        feedback_result['learner_update'] = learn_res
        
    return jsonify({
        'client_id': client_id,
        'prediction': prediction,
        'confidence': float(proba[k]),
        'is_attack': is_attack,
        'feedback_result': feedback_result,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    })


@app.route('/predict', methods=['POST'])
def predict():
    d = request.get_json(force=True)
    vals = d.get('features', [])
    if registry._best is None:
        return jsonify({'error': f'Best model ({BEST}) not available. Train the notebook first.'}), 400
    if len(vals) != N_FEATURES:
        return jsonify({'error': f'Expected {N_FEATURES} features, got {len(vals)}.'}), 400
    Xs = _scale_row(vals)
    try:
        proba = registry.predict_best(Xs)[0]
        rl_info = registry.predict_rl_ensemble(Xs)
        fl_info = registry.predict_fl(Xs)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    k = int(np.argmax(proba))
    novelty = _novelty_flags(CLASS_NAMES[k], float(proba[k]), rl_info, fl_info)
    rng = np.random.default_rng()
    client_id = int(rng.choice([0, 1, 2]))
    return jsonify({
        'client_id': client_id,
        'prediction': CLASS_NAMES[k],
        'confidence': float(proba[k]),
        'classes': CLASS_NAMES,
        'proba': [float(x) for x in proba],
        'model': BEST,
        'rl': rl_info,
        'fl': fl_info,
        # True ONLY when Flower processes are currently running (not just "model exists")
        'fl_processes_alive': _fl_processes_alive(),
        'novelty': novelty,
        'feedback': _feedback_stores[0].status(),
        'sync': _sync_manager.status(),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'proof': {
            'supervised': f'real_{BEST}_inference',
            'rl': 'real_pytorch_forward' if rl_info else 'rl_models_not_trained_yet',
            'fl': (
                'real_fedavg_rl_weights'
                if fl_info and fl_info.get('is_rl_fl')
                else ('real_fedavg_global_weights' if fl_info else 'fl_global_model_not_ready')
            ),
            'pipeline': (fl_info or {}).get('pipeline', 'RL -> FL'),
        },
    })


@app.route('/explain', methods=['POST'])
def explain():
    d = request.get_json(force=True)
    vals = d.get('features', [])
    topk = int(d.get('top_k', 8))
    if len(vals) != N_FEATURES:
        return jsonify({'error': f'Expected {N_FEATURES} features, got {len(vals)}.'}), 400
    Xs = _scale_row(vals)
    try:
        proba = registry.predict_best(Xs)[0]
        k = int(np.argmax(proba))
        vec, base, method = _explain_row(Xs, k)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    order = np.argsort(np.abs(vec))[::-1][:topk]
    contribs = [{
        'feature': feature_cols[i],
        'impact': float(vec[i]),
        'input': float(vals[i]),
    } for i in order]
    return jsonify({
        'prediction': CLASS_NAMES[k],
        'predicted_index': k,
        'confidence': float(proba[k]),
        'base_value': base,
        'fx': float(base + float(np.sum(vec))),
        'method': method,
        'contributions': contribs,
        'model': BEST,
    })


@app.route('/explain_llm', methods=['POST'])
def explain_llm():
    """Multi-mode Groq explanations: predict|status|fl|rl|history|class|batch|learn."""
    import groq_explain
    if not groq_explain.available():
        return jsonify({
            'error': 'Groq API key not configured. Add GROQ_API_KEY to .env and restart.',
        }), 503
    d = request.get_json(force=True) or {}
    mode = (d.get('mode') or 'predict').strip().lower()

    try:
        if mode == 'status':
            live = registry.live_status()
            info = {
                'best_model': BEST,
                'device': str(DEVICE),
            }
            out = groq_explain.explain_status(live, info)
        elif mode == 'fl':
            out = groq_explain.explain_fl(registry.live_status())
        elif mode == 'rl':
            learn = d.get('learn')
            if learn is None:
                try:
                    learn = _learner.status()
                except Exception:
                    learn = {}
            out = groq_explain.explain_rl(registry.live_status(), learn)
        elif mode == 'history':
            out = groq_explain.explain_history_item(d.get('item') or {})
        elif mode == 'class':
            label = d.get('label') or d.get('class')
            if not label:
                return jsonify({'error': 'label required'}), 400
            out = groq_explain.explain_attack_class(label, d.get('desc'))
        elif mode == 'batch':
            out = groq_explain.explain_batch(
                d.get('counts') or [],
                int(d.get('total') or 0),
                d.get('model') or BEST,
            )
        elif mode == 'learn':
            out = groq_explain.explain_learn_event(
                true_label=d.get('true_label') or '',
                predicted=d.get('predicted'),
                updated=bool(d.get('updated')),
                learn_status=d.get('learn_status') or {},
            )
        else:
            # default: predict explanation (needs features)
            vals = d.get('features', [])
            if len(vals) != N_FEATURES:
                return jsonify({'error': f'Expected {N_FEATURES} features, got {len(vals)}.'}), 400
            Xs = _scale_row(vals)
            proba = registry.predict_best(Xs)[0]
            k = int(np.argmax(proba))
            prediction = CLASS_NAMES[k]
            confidence = float(proba[k])
            contribs = d.get('contributions')
            if not contribs and _HAS_SHAP:
                vec, base, method = _explain_row(Xs, k)
                order = np.argsort(np.abs(vec))[::-1][:8]
                contribs = [{
                    'feature': feature_cols[i],
                    'impact': float(vec[i]),
                    'input': float(vals[i]),
                } for i in order]
            rl_info = registry.predict_rl_ensemble(Xs)
            fl_info = registry.predict_fl(Xs)
            out = groq_explain.explain_prediction(
                prediction=prediction,
                confidence=confidence,
                model_name=BEST,
                contributions=contribs or [],
                rl_action=(rl_info or {}).get('primary', {}).get('action') if rl_info else None,
                fl_action=(fl_info or {}).get('prediction') if fl_info else None,
                class_proba={CLASS_NAMES[i]: float(proba[i]) for i in range(len(CLASS_NAMES))},
            )
            if out.get('ok'):
                out['prediction'] = prediction
                out['confidence'] = confidence
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    if out.get('error'):
        return jsonify(out), 502
    out['mode'] = mode
    return jsonify(out)


@app.route('/chat_advisor', methods=['POST'])
def chat_advisor_route():
    """Interactive incident response advisory chat with LLM."""
    import groq_explain
    d = request.get_json(force=True) or {}
    messages = d.get('messages') or []
    context = d.get('context') or {}
    
    if not context.get('prediction'):
        last_inf = registry.live_status().get('federated', {}).get('last_inference')
        if last_inf:
            context['prediction'] = last_inf.get('prediction')
            context['confidence'] = last_inf.get('confidence')
            
    out = groq_explain.chat_advisor(messages=messages, context=context)
    return jsonify(out)


@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    f = request.files.get('file')
    if f is None:
        return jsonify({'error': 'No file uploaded'}), 400
    if registry._best is None:
        return jsonify({'error': f'Best model ({BEST}) not available.'}), 400
    dfin = pd.read_csv(f)
    Xs = _prep(dfin)
    try:
        proba = registry.predict_best(Xs)
        # also run RL/FL on first row as a live sample proof
        sample = Xs[:1]
        rl_info = registry.predict_rl_ensemble(sample)
        fl_info = registry.predict_fl(sample)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    preds = proba.argmax(1)
    vc = pd.Series([CLASS_NAMES[p] for p in preds]).value_counts()
    return jsonify({
        'total': int(len(preds)),
        'counts': [{'label': k, 'n': int(v)} for k, v in vc.items()],
        'model': BEST,
        'rl_sample': rl_info,
        'fl_sample': fl_info,
    })


register_learn_routes(app, _learner, _scale_row, CLASS_NAMES, SAVE_DIR, on_feedback=_handle_feedback)


if __name__ == '__main__':
    import groq_explain
    print('\n[OK] AEGIS multi-page IDS starting...')
    print(f'   Best predictor : {BEST}')
    print(f'   RL agents      : {list(registry._rl.keys()) or "(train Section 10 first)"}')
    print(f'   FL global      : {"loaded" if registry._fl else "not yet (run federated/run_rl_federated.bat)"}')
    if registry._fl:
        print(f'   FL pipeline    : {registry._fl.get("pipeline")} ({registry._fl.get("arch")})')
    print(f'   Continual RL   : {"loaded" if _learner.loaded_from else "fresh (no dqn_improved.pt)"} · learning starts OFF')
    print(f'   Groq LLM       : {"ready (" + groq_explain.GROQ_MODEL + ")" if groq_explain.available() else "NO KEY — put GROQ_API_KEY in .env"}')
    print('   Pages:')
    print('     Home     -> http://127.0.0.1:7860/')
    print('     Predict  -> http://127.0.0.1:7860/predict-ui')
    print('     FL       -> http://127.0.0.1:7860/fl')
    print('     RL       -> http://127.0.0.1:7860/rl')
    print('     Settings -> http://127.0.0.1:7860/settings')
    print('')
    app.run(host='127.0.0.1', port=7860, debug=False, use_reloader=False)
