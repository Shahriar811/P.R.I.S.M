"""
FL model — ResNet-MLP (best neural architecture from the IDS comparison).
Used by the Flower server and all 3 clients for real FedAvg training.
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
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


class ResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(d, d), nn.BatchNorm1d(d), nn.ReLU(),
            nn.Linear(d, d), nn.BatchNorm1d(d),
        )

    def forward(self, x):
        return F.relu(x + self.f(x))


class FLNet(nn.Module):
    """ResNet-MLP backbone — best DL model for federated weight averaging."""

    def __init__(self, n_in=N_FEATURES, n_out=NUM_CLASSES, d=128):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_in, d), nn.BatchNorm1d(d), nn.ReLU())
        self.blocks = nn.Sequential(ResBlock(d), ResBlock(d), ResBlock(d))
        self.out = nn.Linear(d, n_out)

    def forward(self, x):
        return self.out(self.blocks(self.inp(x)))


def get_params(model):
    return [v.detach().cpu().numpy() for _, v in model.state_dict().items()]


def set_params(model, params):
    sd = OrderedDict(
        (k, torch.tensor(v)) for k, v in zip(model.state_dict().keys(), params)
    )
    model.load_state_dict(sd, strict=True)


def load_npz(path):
    d = np.load(path)
    return d['X'].astype('float32'), d['y'].astype('int64')


def train(model, X, y, epochs=1, batch=256, lr=1e-3):
    model.to(DEVICE).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    Xt, yt = torch.tensor(X), torch.tensor(y)
    n = len(Xt)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            ix = perm[i:i + batch]
            xb, yb = Xt[ix].to(DEVICE), yt[ix].to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()


def test(model, X, y, batch=2048):
    model.to(DEVICE).eval()
    Xt, yt = torch.tensor(X), torch.tensor(y)
    crit = nn.CrossEntropyLoss()
    loss = 0.0
    correct = 0
    n = len(Xt)
    with torch.no_grad():
        for i in range(0, n, batch):
            xb = Xt[i:i + batch].to(DEVICE)
            yb = yt[i:i + batch].to(DEVICE)
            out = model(xb)
            loss += crit(out, yb).item() * len(xb)
            correct += (out.argmax(1) == yb).sum().item()
    return loss / max(n, 1), correct / max(n, 1)


def write_status(payload: dict):
    """Write live FL status so the web dashboard can prove a real run is active."""
    import time
    payload = dict(payload)
    payload['updated_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    payload['mode'] = 'real_multiprocess_flower'
    with open(STATUS_PATH, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
