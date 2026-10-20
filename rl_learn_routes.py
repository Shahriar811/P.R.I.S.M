"""
rl_learn_routes.py  —  Option A routes for the AEGIS dashboard.

Adds "correct this prediction -> RL agent learns" without modifying the
existing prediction path. Import and call register_learn_routes(app, ...) from
ids_server.py.

Endpoints added:
  POST /rl_learn           {features:[...], true_label:"Bot"}  -> agent learns
  POST /rl_learning_toggle {on:true|false}                     -> switch on/off
  GET  /rl_learn_status                                        -> status json
  POST /rl_learn_save                                          -> persist weights
"""
import os
import numpy as np
from flask import request, jsonify


def register_learn_routes(app, learner, scale_row_fn, class_names, save_dir, on_feedback=None):
    """
    app           : the Flask app
    learner       : a LearnableRLAgent instance
    scale_row_fn  : the server's _scale_row(vals) function
    class_names   : list of class name strings
    save_dir      : where to persist the learned weights
    """
    name_to_idx = {n.lower(): i for i, n in enumerate(class_names)}
    save_path = os.path.join(save_dir, 'RL', 'dqn_improved_learned.pt')

    @app.route('/rl_learn_status')
    def rl_learn_status():
        return jsonify(learner.status())

    @app.route('/rl_learning_toggle', methods=['POST'])
    def rl_learning_toggle():
        d = request.get_json(force=True)
        on = bool(d.get('on', False))
        learner.set_learning(on)
        return jsonify({'learning': learner.learning})

    @app.route('/rl_learn', methods=['POST'])
    def rl_learn():
        d = request.get_json(force=True)
        vals = d.get('features', [])
        true_label = d.get('true_label', None)
        if true_label is None:
            return jsonify({'error': 'true_label required'}), 400
        # accept name or index
        if isinstance(true_label, str) and not true_label.isdigit():
            idx = name_to_idx.get(true_label.lower())
            if idx is None:
                return jsonify({'error': f'unknown class {true_label!r}; '
                                         f'use one of {class_names}'}), 400
        else:
            idx = int(true_label)
        if len(vals) == 0:
            return jsonify({'error': 'features required'}), 400
        try:
            Xs = scale_row_fn(vals)                 # same scaling as /predict
            result = learner.learn_from_correction(Xs, idx)
            if callable(on_feedback):
                extra = on_feedback(
                    Xs, idx, {
                        'true_label': class_names[idx],
                        'mark_new': bool(d.get('mark_new', False)),
                        'predicted': d.get('predicted'),
                    }
                )
                if isinstance(extra, dict):
                    result.update(extra)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        result['status'] = learner.status()
        return jsonify(result)

    @app.route('/rl_learn_save', methods=['POST'])
    def rl_learn_save():
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            learner.save(save_path)
            return jsonify({'saved': True, 'path': save_path})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    print('  [learn] Option A routes registered: /rl_learn, /rl_learning_toggle, '
          '/rl_learn_status, /rl_learn_save')
