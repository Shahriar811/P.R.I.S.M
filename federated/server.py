"""
Flower Federated Learning SERVER (1 main server).
Runs real multi-process FedAvg on port 8080.
Writes status.json after every round for the live web dashboard.
"""
import os
import json
import flwr as fl
import torch
from fl_model import (
    FLNet, get_params, set_params, load_npz, test, write_status, HERE,
)

Xte, yte = load_npz(os.path.join(HERE, 'data', 'test.npz'))
model = FLNet()
GLOBAL_PATH = os.path.join(HERE, 'global_model.pt')
SAVED_FL = os.path.join(os.path.dirname(HERE), 'Saved model', 'FL', 'federated_fedavg.pt')
os.makedirs(os.path.dirname(SAVED_FL), exist_ok=True)

NUM_ROUNDS = int(os.environ.get('FL_ROUNDS', '5'))
MIN_CLIENTS = int(os.environ.get('FL_CLIENTS', '3'))

write_status({
    'role': 'server',
    'state': 'waiting_for_clients',
    'round': 0,
    'num_rounds': NUM_ROUNDS,
    'min_clients': MIN_CLIENTS,
    'clients_expected': MIN_CLIENTS,
    'global_test_acc': None,
    'global_test_loss': None,
    'pid': os.getpid(),
    'address': '0.0.0.0:8080',
})


def evaluate_fn(server_round, parameters, config):
    set_params(model, parameters)
    loss, acc = test(model, Xte, yte)
    print('[server] round %d  global_test_loss=%.4f  global_test_acc=%.4f'
          % (server_round, loss, acc))
    _meta_path = os.path.join(HERE, 'data', 'meta.json')
    with open(_meta_path, encoding='utf-8') as _mf:
        _class_names = json.load(_mf).get('class_names', [])
    ck = {
        'state_dict': model.state_dict(),
        'round': server_round,
        'acc': acc,
        'loss': loss,
        'arch': 'ResNet-MLP',
        'n_features': int(model.inp[0].in_features),
        'class_names': _class_names,
    }
    torch.save(ck, GLOBAL_PATH)
    torch.save(ck, SAVED_FL)
    write_status({
        'role': 'server',
        'state': 'round_complete' if server_round < NUM_ROUNDS else 'finished',
        'round': server_round,
        'num_rounds': NUM_ROUNDS,
        'min_clients': MIN_CLIENTS,
        'clients_expected': MIN_CLIENTS,
        'global_test_acc': float(acc),
        'global_test_loss': float(loss),
        'pid': os.getpid(),
        'address': '0.0.0.0:8080',
        'global_model_path': GLOBAL_PATH,
        'saved_model_path': SAVED_FL,
        'architecture': 'ResNet-MLP (best DL for FedAvg)',
    })
    return loss, {'accuracy': acc}


def fit_config(server_round):
    write_status({
        'role': 'server',
        'state': 'fitting',
        'round': server_round,
        'num_rounds': NUM_ROUNDS,
        'min_clients': MIN_CLIENTS,
        'clients_expected': MIN_CLIENTS,
        'pid': os.getpid(),
        'address': '0.0.0.0:8080',
        'architecture': 'ResNet-MLP (best DL for FedAvg)',
    })
    return {'local_epochs': 2}


def agg(metrics):
    tot = sum(n for n, _ in metrics) or 1
    return {'train_acc': sum(n * m.get('train_acc', 0) for n, m in metrics) / tot}


strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=MIN_CLIENTS,
    min_evaluate_clients=MIN_CLIENTS,
    min_available_clients=MIN_CLIENTS,
    initial_parameters=fl.common.ndarrays_to_parameters(get_params(model)),
    on_fit_config_fn=fit_config,
    evaluate_fn=evaluate_fn,
    fit_metrics_aggregation_fn=agg,
)

if __name__ == '__main__':
    print('FL SERVER starting on 0.0.0.0:8080  (waiting for %d clients)' % MIN_CLIENTS)
    fl.server.start_server(
        server_address='0.0.0.0:8080',
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )
    write_status({
        'role': 'server',
        'state': 'finished',
        'round': NUM_ROUNDS,
        'num_rounds': NUM_ROUNDS,
        'min_clients': MIN_CLIENTS,
        'pid': os.getpid(),
        'global_model_path': GLOBAL_PATH,
        'architecture': 'ResNet-MLP (best DL for FedAvg)',
    })
    print('Federated training complete. Global model saved to', GLOBAL_PATH)
