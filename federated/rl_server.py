"""
Flower SERVER for RL -> FL (FedAvg over Dueling DQN weights).
Port 8080 — launch with run_rl_federated.bat after local RL (Section 10) exists.
"""
import os
import json
import numpy as np
import flwr as fl
import torch
from rl_fl_model import (
    make_agent, get_params, set_params, load_npz, test, write_status, HERE,
    N_FEATURES, NUM_CLASSES,
)

Xte, yte = load_npz(os.path.join(HERE, 'data', 'test.npz'))
agent = make_agent(
    class_counts=np.bincount(yte.astype(np.int64), minlength=NUM_CLASSES).astype(np.float32)
)

GLOBAL_PATH = os.path.join(HERE, 'global_model.pt')
SAVED_FL = os.path.join(os.path.dirname(HERE), 'Saved model', 'FL', 'federated_rl_fedavg.pt')
os.makedirs(os.path.dirname(SAVED_FL), exist_ok=True)

NUM_ROUNDS = int(os.environ.get('FL_ROUNDS', '5'))
MIN_CLIENTS = int(os.environ.get('FL_CLIENTS', '3'))

# Prefer initializing FL from the freshest learned local RL, else baseline Section 10 RL.
_seed_candidates = [
    os.path.join(os.path.dirname(HERE), 'Saved model', 'RL', 'dqn_improved_learned.pt'),
    os.path.join(os.path.dirname(HERE), 'Saved model', 'RL', 'dqn_improved.pt'),
]
for _seed in _seed_candidates:
    if not os.path.exists(_seed):
        continue
    try:
        ck = torch.load(_seed, map_location='cpu', weights_only=False)
        agent.net.load_state_dict(ck['state_dict'], strict=False)
        print('[server] seeded global RL from', _seed)
        break
    except Exception as e:
        print('[server] could not seed from %s: %s' % (_seed, e))

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
    'architecture': 'RL-Dueling (federated)',
    'pipeline': 'RL -> FL',
})


def evaluate_fn(server_round, parameters, config):
    set_params(agent, parameters)
    loss, acc = test(agent, Xte, yte)
    print('[server] round %d  global_test_loss=%.4f  global_test_acc=%.4f'
          % (server_round, loss, acc))
    class_names = []
    meta = os.path.join(HERE, 'data', 'meta.json')
    if os.path.exists(meta):
        with open(meta, encoding='utf-8') as _f:
            class_names = json.load(_f).get('class_names', [])
    ck = {
        'state_dict': agent.net.state_dict(),
        'round': server_round,
        'acc': acc,
        'loss': loss,
        'arch': 'RL-Dueling',
        'n_features': N_FEATURES,
        'state_dim': N_FEATURES,
        'n_actions': NUM_CLASSES,
        'class_names': class_names,
        'mode': 'rl_then_fl',
        'pipeline': 'RL -> FL',
    }
    torch.save(ck, GLOBAL_PATH)
    torch.save(ck, SAVED_FL)
    # NOTE: no longer writing to legacy 'federated_fedavg.pt' to avoid
    # overwrite collisions with the ResNet-MLP FL pipeline (Issue #8).
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
        'architecture': 'RL-Dueling (federated)',
        'pipeline': 'RL -> FL',
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
        'architecture': 'RL-Dueling (federated)',
        'pipeline': 'RL -> FL',
    })
    return {'local_epochs': 1, 'server_round': server_round}


def agg(metrics):
    tot = sum(n for n, _ in metrics) or 1
    return {'train_acc': sum(n * m.get('train_acc', 0) for n, m in metrics) / tot}


strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=MIN_CLIENTS,
    min_evaluate_clients=MIN_CLIENTS,
    min_available_clients=MIN_CLIENTS,
    initial_parameters=fl.common.ndarrays_to_parameters(get_params(agent)),
    on_fit_config_fn=fit_config,
    evaluate_fn=evaluate_fn,
    fit_metrics_aggregation_fn=agg,
)

if __name__ == '__main__':
    print('RL->FL SERVER starting on 0.0.0.0:8080  (waiting for %d clients)' % MIN_CLIENTS)
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
        'architecture': 'RL-Dueling (federated)',
        'pipeline': 'RL -> FL',
    })
    print('Federated RL training complete. Global RL saved to', GLOBAL_PATH)
