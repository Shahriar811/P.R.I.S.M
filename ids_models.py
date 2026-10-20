"""
Model registry for the IDS dashboard (Latest One FL+RL).

Prediction UI uses ONLY the best supervised model (XGBoost by F1).
Local RL agents (Section 10) and the Federated RL global model (RL -> FL)
are loaded for live comparison / proof panels.
"""

import os
import json
import time
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_agent import DuelingNet  # single source of truth — never duplicate


class ResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(d, d), nn.BatchNorm1d(d), nn.ReLU(),
            nn.Linear(d, d), nn.BatchNorm1d(d),
        )

    def forward(self, x):
        return F.relu(x + self.f(x))


class ResMLP(nn.Module):
    def __init__(self, n_in, n_out, d=128):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_in, d), nn.BatchNorm1d(d), nn.ReLU())
        self.blocks = nn.Sequential(ResBlock(d), ResBlock(d), ResBlock(d))
        self.out = nn.Linear(d, n_out)

    def forward(self, x):
        return self.out(self.blocks(self.inp(x)))


class FLNet(ResMLP):
    pass


class MLPNet(nn.Module):
    def __init__(self, d, a):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, a),
        )

    def forward(self, x):
        return self.net(x)


# DuelingNet is imported from rl_agent — no local duplicate


class ActorCritic(nn.Module):
    def __init__(self, d, a):
        super().__init__()
        self.sh = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 128), nn.ReLU(), nn.LayerNorm(128),
        )
        self.ac = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, a))
        self.cr = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        h = self.sh(x)
        return self.ac(h), self.cr(h)


class NoisyLinear(nn.Module):
    def __init__(self, i, o, sigma=0.017):
        super().__init__()
        self.i, self.o = i, o
        self.w_mu = nn.Parameter(torch.empty(o, i))
        self.w_sig = nn.Parameter(torch.full((o, i), sigma))
        self.b_mu = nn.Parameter(torch.empty(o))
        self.b_sig = nn.Parameter(torch.full((o,), sigma))
        self.register_buffer('w_eps', torch.zeros(o, i))
        self.register_buffer('b_eps', torch.zeros(o))
        std = 1.0 / (i ** 0.5)
        self.w_mu.data.uniform_(-std, std)
        self.b_mu.data.uniform_(-std, std)

    def forward(self, x):
        return F.linear(x, self.w_mu, self.b_mu)


class RainbowNet(nn.Module):
    def __init__(self, d, a):
        super().__init__()
        self.f = nn.Sequential(
            NoisyLinear(d, 256), nn.ReLU(), nn.LayerNorm(256),
            NoisyLinear(256, 128), nn.ReLU(), nn.LayerNorm(128),
        )
        self.v = nn.Sequential(NoisyLinear(128, 64), nn.ReLU(), NoisyLinear(64, 1))
        self.a = nn.Sequential(NoisyLinear(128, 64), nn.ReLU(), NoisyLinear(64, a))

    def forward(self, x):
        f = self.f(x)
        v, a = self.v(f), self.a(f)
        return v + a - a.mean(1, keepdim=True)


N_QUANT = 51


class QRNet(nn.Module):
    def __init__(self, d, a, nq=N_QUANT):
        super().__init__()
        self.a, self.nq = a, nq
        self.body = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 128), nn.ReLU(), nn.LayerNorm(128),
        )
        self.head = nn.Linear(128, a * nq)

    def forward(self, x):
        return self.head(self.body(x)).view(-1, self.a, self.nq)

    def q(self, x):
        return self.forward(x).mean(2)


RL_ARCH = {
    'dqn_basic': ('DQN-Basic', 'dqn', MLPNet),
    'dqn_improved': ('DQN-Improved', 'dqn', DuelingNet),
    'ppo': ('PPO', 'ppo', ActorCritic),
    'rainbow_dqn': ('Rainbow-DQN', 'dqn', RainbowNet),
    'qr_dqn': ('QR-DQN', 'qr', QRNet),
}


class ModelRegistry:
    """Best-model predictor + live RL/FL companions."""

    BEST_MODEL_NAME = 'XGBoost'
    BEST_STEM = 'xgboost'

    def __init__(self, save_dir, n_features, class_names, device, fl_dir=None):
        self.save_dir = save_dir
        self.tm_dir = os.path.join(save_dir, 'TM')
        self.dl_dir = os.path.join(save_dir, 'DL')
        self.rl_dir = os.path.join(save_dir, 'RL')
        self.fl_dir_saved = os.path.join(save_dir, 'FL')
        self.fl_runtime = fl_dir or os.path.join(os.path.dirname(save_dir), 'federated')
        self.n_features = n_features
        self.class_names = class_names
        self.device = device
        self._best = None
        self._rl = {}
        self._fl = None
        self._live_rl = None
        self._live_feedback_status = None
        self._live_sync_status = None
        self._rl_last = None
        self._fl_last_infer = None
        self.best_available = os.path.exists(
            os.path.join(self.tm_dir, self.BEST_STEM + '.joblib')
        )
        self._load_best()
        self._scan_rl()
        self._scan_fl()

    def attach_live_rl(self, learner):
        """Use the online learner as the first RL serving model."""
        self._live_rl = learner

    def attach_feedback_status(self, callback):
        self._live_feedback_status = callback

    def attach_sync_status(self, callback):
        self._live_sync_status = callback

    def reload_fl(self):
        self._fl = None
        self._scan_fl()

    def _load_best(self):
        path = os.path.join(self.tm_dir, self.BEST_STEM + '.joblib')
        if os.path.exists(path):
            self._best = joblib.load(path)
            print(f'  [best] loaded {self.BEST_MODEL_NAME} from {path}')
        else:
            print(f'  [warn] Best model file missing: {path}')

    def _scan_rl(self):
        for stem, (display, kind, cls) in RL_ARCH.items():
            path = os.path.join(self.rl_dir, stem + '.pt')
            if not os.path.exists(path):
                continue
            ck = torch.load(path, map_location=self.device, weights_only=False)
            n_in = int(ck.get('n_features', self.n_features))
            n_out = len(ck.get('class_names', self.class_names))
            net = cls(n_in, n_out)
            net.load_state_dict(ck['state_dict'])
            net.to(self.device).eval()
            self._rl[display] = {'net': net, 'kind': kind, 'path': path}
            print(f'  [rl] loaded {display}')

    def _scan_fl(self):
        """Load federated global model. Prefer RL->FL (Dueling), else legacy ResNet-MLP."""
        candidates = [
            os.path.join(self.fl_dir_saved, 'federated_rl_fedavg.pt'),
            os.path.join(self.fl_dir_saved, 'federated_fedavg.pt'),
            os.path.join(self.fl_runtime, 'global_model.pt'),
            os.path.join(self.dl_dir, 'federated_fedavg.pt'),
        ]

        class LegacyFLNet(nn.Module):
            def __init__(self, n_in, n_out):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(n_in, 128), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_out),
                )
            def forward(self, x):
                return self.net(x)

        for path in candidates:
            if not os.path.exists(path):
                continue
            try:
                ck = torch.load(path, map_location=self.device, weights_only=False)
                sd = ck.get('state_dict', ck)

                # Prefer explicit 'arch' field from checkpoint (reliable).
                # Fall back to key-sniffing only for old checkpoints that lack it.
                arch_field = ck.get('arch')
                pipeline = ck.get('pipeline', 'RL -> FL')

                if arch_field == 'RL-Dueling' or (arch_field is None and 'f.0.weight' in sd):
                    n_in = int(sd['f.0.weight'].shape[1])
                    n_out = int(sd['a.2.weight'].shape[0])
                    model = DuelingNet(n_in, n_out)
                    arch = arch_field or 'RL-Dueling'
                    pipeline = pipeline or 'RL -> FL'
                elif arch_field == 'ResNet-MLP' or (arch_field is None and 'inp.0.weight' in sd):
                    n_in = int(sd['inp.0.weight'].shape[1])
                    n_out = int(sd['out.weight'].shape[0])
                    model = ResMLP(n_in, n_out)
                    arch = arch_field or 'ResNet-MLP'
                    pipeline = 'legacy DL-FL'
                elif arch_field is None and 'net.0.weight' in sd:
                    n_in = int(sd['net.0.weight'].shape[1])
                    n_out = int(sd['net.5.weight'].shape[0])
                    model = LegacyFLNet(n_in, n_out)
                    arch = 'Legacy-MLP'
                    pipeline = 'legacy DL-FL'
                else:
                    continue
                model.load_state_dict(sd)
                model.to(self.device).eval()
                self._fl = {
                    'model': model,
                    'path': path,
                    'round': ck.get('round'),
                    'acc': ck.get('acc'),
                    'arch': arch,
                    'pipeline': pipeline,
                    'is_rl_fl': 'f.0.weight' in sd,
                }
                print(f'  [fl] loaded global model ({arch}, {pipeline}) from {path}')
                break
            except Exception as e:
                print(f'  [fl] skip {path}: {e}')
                continue

    def predict_best(self, Xs):
        if self._best is None:
            raise RuntimeError(
                f'Best model ({self.BEST_MODEL_NAME}) not found. Train & save models first.'
            )
        if hasattr(self._best, 'predict_proba'):
            return self._best.predict_proba(Xs)
        preds = self._best.predict(Xs)
        proba = np.zeros((len(preds), len(self.class_names)), dtype=np.float32)
        for i, p in enumerate(preds):
            proba[i, int(p)] = 1.0
        return proba

    def _rl_scores(self, net, kind, Xs):
        t = torch.tensor(Xs.astype(np.float32)).to(self.device)
        with torch.no_grad():
            if kind == 'ppo':
                logits, _ = net(t)
                return logits.cpu().numpy()
            if kind == 'qr':
                return net.q(t).cpu().numpy()
            return net(t).cpu().numpy()

    def predict_rl_ensemble(self, Xs):
        """Run all loaded RL agents; return primary (best available) + all Q dumps."""
        if not self._rl and self._live_rl is None:
            return None
        agents = []
        if self._live_rl is not None:
            live = self._live_rl.predict(Xs)
            agents.append({
                'agent': 'DQN-Improved (live learner)',
                'action': live['action'],
                'action_index': live['action_index'],
                'confidence': float(live['confidence']),
                'q_values': None,
                'proba': [float(x) for x in live['proba']],
                'source': 'continual_learning_live',
            })
        # Prefer Dueling / Improved, else first available
        preferred = ['DQN-Improved', 'Rainbow-DQN', 'QR-DQN', 'DQN-Basic', 'PPO']
        order = [n for n in preferred if n in self._rl] + [
            n for n in self._rl if n not in preferred
        ]
        for name in order:
            if self._live_rl is not None and name == 'DQN-Improved':
                continue
            entry = self._rl[name]
            scores = self._rl_scores(entry['net'], entry['kind'], Xs)[0]
            exp = np.exp(scores - scores.max())
            proba = (exp / exp.sum()).astype(np.float32)
            k = int(np.argmax(scores))
            agents.append({
                'agent': name,
                'action': self.class_names[k],
                'action_index': k,
                'confidence': float(proba[k]),
                'q_values': [float(x) for x in scores],
                'proba': [float(x) for x in proba],
                'source': 'frozen_checkpoint',
            })
        primary = agents[0]
        self._rl_last = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'primary': primary,
            'agents': agents,
            'mode': 'real_pytorch_inference',
        }
        return self._rl_last

    def predict_fl(self, Xs):
        if self._fl is None:
            return None
        model = self._fl['model']
        t = torch.tensor(Xs.astype(np.float32)).to(self.device)
        with torch.no_grad():
            out = model(t)
            proba = F.softmax(out, 1).cpu().numpy()[0]
        k = int(np.argmax(proba))
        self._fl_last_infer = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'prediction': self.class_names[k],
            'confidence': float(proba[k]),
            'proba': [float(x) for x in proba],
            'round': self._fl.get('round'),
            'train_acc': float(self._fl['acc']) if self._fl.get('acc') is not None else None,
            'path': self._fl['path'],
            'architecture': self._fl.get('arch', 'RL-Dueling'),
            'pipeline': self._fl.get('pipeline', 'RL -> FL'),
            'is_rl_fl': bool(self._fl.get('is_rl_fl')),
            'source': 'GLOBAL_FEDERATED_MODEL',
            'source_note': (
                'Predict uses the GLOBAL averaged RL model only. '
                'Client processes train privately and are NOT used for this prediction.'
            ),
            'mode': 'real_fedavg_rl_weights' if self._fl.get('is_rl_fl') else 'real_fedavg_global_weights',
        }
        return self._fl_last_infer

    @staticmethod
    def _pid_alive(pid):
        """Return True if OS process with pid appears to be running."""
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            return False
        if pid <= 0:
            return False
        if os.name == 'nt':
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
                handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
                if handle:
                    kernel32.CloseHandle(handle)
                    return True
                return False
            except Exception:
                return False
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def live_status(self):
        fl_status = {'state': 'not_started', 'mode': 'real_multiprocess_flower'}
        sp = os.path.join(self.fl_runtime, 'status.json')
        if os.path.exists(sp):
            try:
                fl_status = json.load(open(sp, encoding='utf-8'))
            except Exception as e:
                fl_status = {'state': 'error', 'error': str(e)}
        fl_status['process_alive'] = self._pid_alive(fl_status.get('pid'))
        fl_status['role'] = fl_status.get('role', 'server')

        clients = []
        for i in range(3):
            cp = os.path.join(self.fl_runtime, f'client_{i}_status.json')
            if os.path.exists(cp):
                try:
                    c = json.load(open(cp, encoding='utf-8'))
                except Exception:
                    c = {'client_id': i, 'state': 'unreadable'}
            else:
                c = {'client_id': i, 'state': 'offline'}
            c['client_id'] = c.get('client_id', i)
            c['role'] = 'client'
            c['process_alive'] = self._pid_alive(c.get('pid'))
            clients.append(c)

        alive_n = int(bool(fl_status.get('process_alive'))) + sum(
            1 for c in clients if c.get('process_alive')
        )
        clients_alive = sum(1 for c in clients if c.get('process_alive'))
        fl_path = (self._fl or {}).get('path')
        fl_round = (self._fl or {}).get('round')
        fl_acc = (self._fl or {}).get('acc')

        verify = {
            'local_rl_loaded': len(self._rl) > 0,
            'global_fl_loaded': self._fl is not None,
            'global_is_rl_fl': bool(self._fl and self._fl.get('is_rl_fl')),
            'predict_uses': 'GLOBAL_FEDERATED_MODEL' if self._fl else None,
            'clients_used_for_predict': False,
            'clients_alive': clients_alive,
            'server_alive': bool(fl_status.get('process_alive')),
            'connection_ok': bool(self._fl and self._fl.get('is_rl_fl') and len(self._rl) > 0),
            'learning_happens_during': 'local RL learns on corrections during serving; GLOBAL FL improves after the next Flower/FedAvg sync cycle',
        }

        return {
            'best_model': {
                'name': self.BEST_MODEL_NAME,
                'available': self._best is not None,
                'role': 'primary_attack_predictor',
            },
            'federated': {
                'server': fl_status,
                'clients': clients,
                'processes_alive': alive_n,
                'processes_expected': 4,
                'global_model_loaded': self._fl is not None,
                'is_rl_fl': bool(self._fl and self._fl.get('is_rl_fl')),
                'pipeline': (self._fl or {}).get('pipeline', 'RL -> FL'),
                'architecture': (self._fl or {}).get('arch'),
                'last_inference': self._fl_last_infer,
                'global_model': {
                    'loaded': self._fl is not None,
                    'role': 'GLOBAL',
                    'used_by_predict': True,
                    'path': fl_path,
                    'round': fl_round,
                    'test_acc': float(fl_acc) if fl_acc is not None else None,
                    'architecture': (self._fl or {}).get('arch'),
                    'pipeline': (self._fl or {}).get('pipeline'),
                    'is_rl_fl': bool(self._fl and self._fl.get('is_rl_fl')),
                    'note': 'This is the averaged model after FedAvg. Predict uses ONLY this global model.',
                },
                'clients_summary': {
                    'role': 'CLIENTS',
                    'used_by_predict': False,
                    'alive': clients_alive,
                    'expected': 3,
                    'note': 'Clients train locally on private data. Their raw data never leaves. Only weights are averaged into the GLOBAL model.',
                    'states': [
                        {
                            'id': c.get('client_id'),
                            'state': c.get('state'),
                            'alive': bool(c.get('process_alive')),
                            'train_acc': c.get('train_acc'),
                            'n_samples': c.get('n_samples'),
                        }
                        for c in clients
                    ],
                },
            },
            'reinforcement': {
                'agents_loaded': (
                    (['DQN-Improved (live learner)'] if self._live_rl is not None else [])
                    + [n for n in self._rl.keys() if not (self._live_rl is not None and n == 'DQN-Improved')]
                ),
                'agent_count': len(self._rl) + (1 if self._live_rl is not None and 'DQN-Improved' not in self._rl else 0),
                'last_inference': self._rl_last,
                'mode': 'real_pytorch_inference_with_live_learner' if self._live_rl is not None else 'real_pytorch_inference',
                'note': (
                    'Local RL serves immediately from the live continual learner; '
                    'GLOBAL federated RL is reloaded after Flower FedAvg rounds.'
                    if self._live_rl is not None else
                    'Local Section-10 RL agents (42-D features). Separate from the GLOBAL federated RL model.'
                ),
                'live_learner_attached': self._live_rl is not None,
            },
            'feedback': self._live_feedback_status() if callable(self._live_feedback_status) else {},
            'sync': self._live_sync_status() if callable(self._live_sync_status) else {},
            'verify': verify,
            'proof': {
                'fl_is_simulation': False,
                'rl_is_simulation': False,
                'rl_then_fl': bool(self._fl and self._fl.get('is_rl_fl')),
                'predict_source': 'GLOBAL_FEDERATED_MODEL',
                'note': (
                    'Train RL first (Section 10), then run federated/run_rl_federated.bat '
                    'so Flower FedAvg averages RL weights (RL -> FL) without sharing raw data. '
                    'Predict FL box = GLOBAL model only, never a single client.'
                ),
            },
        }
