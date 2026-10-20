"""
Flower CLIENT for RL -> FL.
Usage:  python rl_client.py <client_id>
Trains a local Dueling RL agent on private shard features; only weights leave.
"""
import sys
import os
import json
import time
import numpy as np
import flwr as fl
from rl_fl_model import (
    make_agent, get_params, set_params, load_npz, train, test, HERE, NUM_CLASSES,
)

cid = int(sys.argv[1]) if len(sys.argv) > 1 else 0
data_path = os.path.join(HERE, 'data', 'client_%d.npz' % cid)
X, y = load_npz(data_path)
feedback_path = os.path.join(HERE, 'feedback', 'client_%d_feedback.npz' % cid)
if os.path.exists(feedback_path):
    try:
        fd = np.load(feedback_path)
        Xf = fd['X'].astype(np.float32)
        yf = fd['y'].astype(np.int64)
        if len(Xf):
            X = np.concatenate([X, Xf], axis=0)
            y = np.concatenate([y, yf], axis=0)
            print('[rl-client %d] merged %d feedback samples from %s' % (cid, len(Xf), feedback_path))
    except Exception as e:
        print('[rl-client %d] could not load feedback shard: %s' % (cid, e))
counts = np.bincount(y.astype(np.int64), minlength=NUM_CLASSES).astype(np.float32)
agent = make_agent(class_counts=counts)
print('[rl-client %d] loaded %d samples from %s' % (cid, len(X), data_path))

CLIENT_STATUS = os.path.join(HERE, 'client_%d_status.json' % cid)


def _write_client(state, extra=None):
    payload = {
        'role': 'client',
        'client_id': cid,
        'state': state,
        'n_samples': int(len(X)),
        'pid': os.getpid(),
        'updated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'mode': 'rl_then_fl_fedavg',
        'architecture': 'RL-Dueling',
    }
    if extra:
        payload.update(extra)
    with open(CLIENT_STATUS, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


_write_client('connecting')


class RLIDSClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return get_params(agent)

    def fit(self, parameters, config):
        set_params(agent, parameters)
        epochs = int(config.get('local_epochs', 1))
        _write_client('local_training', {'local_epochs': epochs})
        # subsample huge shards for speed while still private local train
        if len(X) > 120_000:
            server_round = int(config.get('server_round', 0))
            rng = np.random.default_rng(cid + 7 + server_round)
            idx = rng.choice(len(X), 120_000, replace=False)
            Xs, ys = X[idx], y[idx]
        else:
            Xs, ys = X, y
        train(agent, Xs, ys, epochs=epochs)
        _, acc = test(agent, Xs[: min(20000, len(Xs))], ys[: min(20000, len(ys))])
        _write_client('fit_done', {'train_acc': float(acc)})
        return get_params(agent), len(X), {'train_acc': float(acc)}

    def evaluate(self, parameters, config):
        set_params(agent, parameters)
        loss, acc = test(agent, X[: min(30000, len(X))], y[: min(30000, len(y))])
        _write_client('evaluate_done', {'accuracy': float(acc), 'loss': float(loss)})
        return float(loss), len(X), {'accuracy': float(acc)}


if __name__ == '__main__':
    _write_client('connected')
    fl.client.start_client(
        server_address='127.0.0.1:8080',
        client=RLIDSClient().to_client(),
    )
    _write_client('finished')
    print('[rl-client %d] done' % cid)
