"""
Offline RL -> FL (in-process FedAvg) for a quick global RL checkpoint.

Use this if you cannot launch the multi-window Flower bat right now.
For the real multi-process demo, prefer: federated/run_rl_federated.bat

  python train_rl_federated_offline.py --rounds 3
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score

import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rl_agent import RLAgent

DATA = os.path.join(ROOT, 'federated', 'data')
SAVE = os.path.join(ROOT, 'Saved model', 'FL')
RL_SEED = os.path.join(ROOT, 'Saved model', 'RL', 'dqn_improved.pt')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--local-epochs', type=int, default=1)
    args = ap.parse_args()
    os.makedirs(SAVE, exist_ok=True)

    meta = json.load(open(os.path.join(DATA, 'meta.json'), encoding='utf-8'))
    n_in, n_out = int(meta['n_features']), int(meta['num_classes'])
    class_names = meta.get('class_names', [])

    test = np.load(os.path.join(DATA, 'test.npz'))
    Xte, yte = test['X'].astype(np.float32), test['y'].astype(np.int64).ravel()

    global_agent = RLAgent(n_in, n_out, device=DEVICE)
    if os.path.exists(RL_SEED):
        ck = torch.load(RL_SEED, map_location=DEVICE, weights_only=False)
        global_agent.net.load_state_dict(ck['state_dict'], strict=False)
        print('Seeded from', RL_SEED)
    g_w = global_agent.get_weights()

    for rnd in range(1, args.rounds + 1):
        local_ws, sizes = [], []
        for cid in range(3):
            d = np.load(os.path.join(DATA, f'client_{cid}.npz'))
            X, y = d['X'].astype(np.float32), d['y'].astype(np.int64).ravel()
            if len(X) > 80000:
                rng = np.random.default_rng(cid + rnd)
                idx = rng.choice(len(X), 80000, replace=False)
                X, y = X[idx], y[idx]
            agent = RLAgent(n_in, n_out, class_counts=np.bincount(y, minlength=n_out), device=DEVICE)
            agent.set_weights(g_w)
            print(f'round {rnd} client {cid}: {len(X)} samples')
            agent.train_epochs(X, y, epochs=args.local_epochs, batch=2048)
            local_ws.append(agent.get_weights())
            sizes.append(len(X))
        total = float(sum(sizes))
        g_w = []
        for layers in zip(*local_ws):
            acc = None
            for w, n in zip(layers, sizes):
                term = w * (n / total)
                acc = term if acc is None else acc + term
            g_w.append(acc)
        global_agent.set_weights(g_w)
        pred = global_agent.predict_batch(Xte)
        acc = accuracy_score(yte, pred)
        print(f'  >> round {rnd} test acc={acc:.4f}')

    out = os.path.join(SAVE, 'federated_rl_fedavg.pt')
    legacy = os.path.join(SAVE, 'federated_fedavg.pt')
    payload = {
        'state_dict': global_agent.net.state_dict(),
        'round': args.rounds,
        'acc': float(accuracy_score(yte, global_agent.predict_batch(Xte))),
        'arch': 'RL-Dueling',
        'n_features': n_in,
        'state_dim': n_in,
        'n_actions': n_out,
        'class_names': class_names,
        'mode': 'rl_then_fl',
        'pipeline': 'RL -> FL',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    torch.save(payload, out)
    torch.save(payload, legacy)
    torch.save(payload, os.path.join(ROOT, 'federated', 'global_model.pt'))
    print('[OK] saved', out)
    print('Restart ids_server.py — Home chip RL->FL should be ON.')


if __name__ == '__main__':
    main()
