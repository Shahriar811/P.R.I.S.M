"""
Federated Learning over an RL agent (RL -> FL).

Each client trains a Dueling DQN on its private flow features (no raw data leaves).
The Flower server FedAvg-averages RL weights into one global continual learner.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

# allow importing project-root rl_agent.py
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rl_agent import RLAgent, DuelingNet  # noqa: E402

META_PATH = os.path.join(HERE, 'data', 'meta.json')
if os.path.exists(META_PATH):
    with open(META_PATH, encoding='utf-8') as _f:
        META = json.load(_f)
    N_FEATURES = int(META['n_features'])
    NUM_CLASSES = int(META['num_classes'])
else:
    N_FEATURES, NUM_CLASSES = 42, 5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
STATUS_PATH = os.path.join(HERE, 'status.json')


def make_agent(class_counts=None):
    """Create a fresh RLAgent.  Flower clients call set_params() immediately
    after, so the random initialisation is discarded — this is the standard
    Flower pattern (each client creates a local model, then overwrites it
    with the global parameters received from the server)."""
    return RLAgent(
        state_dim=N_FEATURES,
        n_actions=NUM_CLASSES,
        class_counts=class_counts,
        device=DEVICE,
    )


def get_params(agent):
    return agent.get_weights()


def set_params(agent, params):
    agent.set_weights(params)


def load_npz(path):
    d = np.load(path)
    return d['X'].astype('float32'), d['y'].astype('int64')


def train(agent, X, y, epochs=1, batch=2048):
    return agent.train_epochs(X, y, epochs=epochs, batch=batch)


def test(agent, X, y, batch=4096):
    """Evaluate the RL agent and return (loss, accuracy).

    The loss is the mean squared error between the Q-values for the selected
    actions and the RL reward signal (correct → +1.0, wrong → −1.0).  This
    gives Flower a meaningful convergence metric instead of the old proxy
    ``1 − accuracy``.
    """
    pred = agent.predict_batch(X)
    y_int = y.astype(np.int64)
    acc = float((pred == y_int).mean()) if len(y_int) else 0.0
    # Compute actual RL reward-based loss
    rewards = np.where(pred == y_int, 1.0, -1.0)
    loss = float(np.mean((rewards - 1.0) ** 2))  # MSE from optimal reward
    return loss, acc


def write_status(payload):
    import time
    payload = dict(payload)
    payload.setdefault('updated_at', time.strftime('%Y-%m-%d %H:%M:%S'))
    payload.setdefault('mode', 'rl_then_fl_fedavg')
    payload.setdefault('architecture', 'RL-Dueling (federated)')
    with open(STATUS_PATH, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
