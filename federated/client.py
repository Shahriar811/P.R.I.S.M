"""
Flower Federated Learning CLIENT.
Usage:  python client.py <client_id>
  client_id = 0, 1, or 2   (3 real client processes)
Each client loads its private non-IID shard and trains locally — no raw data leaves the process.
"""
import sys
import os
import json
import time
import flwr as fl
from fl_model import FLNet, get_params, set_params, load_npz, train, test, HERE

cid = int(sys.argv[1]) if len(sys.argv) > 1 else 0
data_path = os.path.join(HERE, 'data', 'client_%d.npz' % cid)
X, y = load_npz(data_path)
model = FLNet()
print('[client %d] loaded %d samples from %s' % (cid, len(X), data_path))

CLIENT_STATUS = os.path.join(HERE, 'client_%d_status.json' % cid)


def _write_client(state, extra=None):
    payload = {
        'role': 'client',
        'client_id': cid,
        'state': state,
        'n_samples': int(len(X)),
        'pid': os.getpid(),
        'updated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'mode': 'real_multiprocess_flower',
    }
    if extra:
        payload.update(extra)
    with open(CLIENT_STATUS, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


_write_client('connecting')


class IDSClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return get_params(model)

    def fit(self, parameters, config):
        set_params(model, parameters)
        epochs = int(config.get('local_epochs', 1))
        _write_client('local_training', {'local_epochs': epochs})
        train(model, X, y, epochs=epochs)
        _, acc = test(model, X, y)
        _write_client('fit_done', {'train_acc': float(acc)})
        return get_params(model), len(X), {'train_acc': float(acc)}

    def evaluate(self, parameters, config):
        set_params(model, parameters)
        loss, acc = test(model, X, y)
        _write_client('evaluate_done', {'accuracy': float(acc), 'loss': float(loss)})
        return float(loss), len(X), {'accuracy': float(acc)}


if __name__ == '__main__':
    _write_client('connected')
    fl.client.start_client(
        server_address='127.0.0.1:8080',
        client=IDSClient().to_client(),
    )
    _write_client('finished')
    print('[client %d] done' % cid)
