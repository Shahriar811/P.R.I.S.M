"""
Continual RL agent (Dueling DQN) — used as the model inside Federated Learning.

State  = scaled flow feature vector (n_features,)
Action = attack class

RL is trained first locally; Flower FedAvg then averages agent weights across
clients (RL -> FL) so the global model improves without sharing raw data.

Design notes
~~~~~~~~~~~~
*  **No target network.**  Standard DQN uses a delayed target net to stabilise
   training.  This agent omits it because each network flow is independent
   (no next-state transition), making the problem a contextual *bandit*, not
   a sequential MDP.  The Dueling architecture, LayerNorm, and gradient
   clipping provide sufficient stability for this 1-step reward setting.
*  **1-step bandit reward.**  ``_update()`` computes the reward for the
   selected action in a single step (correct → +class_weight, wrong → −1)
   and regresses Q(s, a) toward that reward via MSE.  This is equivalent to
   cost-sensitive classification through Q-value regression.
"""
from __future__ import annotations

import random
from collections import OrderedDict, deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingNet(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 128), nn.ReLU(), nn.LayerNorm(128),
        )
        self.v = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.a = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_actions))

    def forward(self, x):
        f = self.f(x)
        v, a = self.v(f), self.a(f)
        return v + a - a.mean(1, keepdim=True)


class RLAgent:
    def __init__(
        self,
        state_dim,
        n_actions,
        class_counts=None,
        lr=3e-4,
        replay_capacity=50000,
        replay_batch=384,
        device=None,
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.state_dim = int(state_dim)
        self.n_actions = int(n_actions)
        self.lr = lr
        self.net = DuelingNet(self.state_dim, self.n_actions).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.eps = 0.1
        self._learning = True
        self.replay = deque(maxlen=replay_capacity)
        self.replay_batch = replay_batch

        if class_counts is not None:
            c = np.asarray(class_counts, dtype=np.float32)
            w = c.sum() / (n_actions * np.maximum(c, 1))
        else:
            w = np.ones(n_actions, dtype=np.float32)
        self.class_w = torch.tensor(w, dtype=torch.float32, device=self.device)

    def learning_on(self):
        self._learning = True

    def learning_off(self):
        self._learning = False

    def is_learning(self):
        return self._learning

    def _reward(self, actions, labels):
        return torch.where(
            actions == labels,
            self.class_w[labels],
            torch.tensor(-1.0, device=self.device),
        )

    def act(self, state, explore=False):
        self.net.eval()
        x = torch.as_tensor(np.atleast_2d(state), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q = self.net(x)
            greedy = q.argmax(1)
            if explore:
                rnd = torch.randint(0, self.n_actions, (len(x),), device=self.device)
                m = torch.rand(len(x), device=self.device) < self.eps
                greedy = torch.where(m, rnd, greedy)
            conf = torch.softmax(q, dim=1)
        out = greedy.cpu().numpy()
        conf_np = conf.cpu().numpy()
        if out.shape[0] == 1:
            k = int(out[0])
            return k, float(conf_np[0, k]), q[0].cpu().numpy(), conf_np[0]
        return out, conf_np[np.arange(len(out)), out], q.cpu().numpy(), conf_np

    def _update(self, states, labels):
        """One-step bandit-style Q-learning update.

        Actions are chosen ε-greedily, rewards are +class_w (correct) or −1
        (wrong), and the loss is MSE(Q(s, a), reward).  No next-state
        transition is used because each network flow is independent.
        """
        self.net.train()
        S = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        y = torch.as_tensor(labels, dtype=torch.long, device=self.device)
        q_all = self.net(S)
        with torch.no_grad():
            g = q_all.detach().argmax(1)
            rnd = torch.randint(0, self.n_actions, (len(S),), device=self.device)
            m = torch.rand(len(S), device=self.device) < self.eps
            actions = torch.where(m, rnd, g)
            rewards = self._reward(actions, y)
        qsa = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = F.mse_loss(qsa, rewards)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()
        return float(loss.item())

    def learn_step(self, states, labels):
        return self._update(
            np.asarray(states, dtype=np.float32),
            np.asarray(labels).astype(np.int64),
        )

    def continual_update(self, new_states, new_labels):
        new_states = np.asarray(new_states, dtype=np.float32)
        new_labels = np.asarray(new_labels).astype(np.int64)
        for s, l in zip(new_states, new_labels):
            self.replay.append((s, l))
        if not self._learning:
            return None
        S, y = new_states, new_labels
        if len(self.replay) > self.replay_batch:
            batch = random.sample(self.replay, self.replay_batch)
            rs = np.array([b[0] for b in batch], dtype=np.float32)
            ry = np.array([b[1] for b in batch], dtype=np.int64)
            S = np.concatenate([S, rs], axis=0)
            y = np.concatenate([y, ry], axis=0)
        return self._update(S, y)

    def train_epochs(self, states, labels, epochs=1, batch=2048):
        states = np.asarray(states, dtype=np.float32)
        labels = np.asarray(labels).astype(np.int64)
        n = len(states)
        last = 0.0
        for _ in range(epochs):
            perm = np.random.permutation(n)
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                last = self._update(states[idx], labels[idx])
        return last

    def predict_batch(self, states):
        self.net.eval()
        S = torch.as_tensor(np.asarray(states, dtype=np.float32), device=self.device)
        out = np.empty(len(S), np.int64)
        with torch.no_grad():
            for i in range(0, len(S), 16384):
                out[i:i + 16384] = self.net(S[i:i + 16384]).argmax(1).cpu().numpy()
        return out

    def get_weights(self):
        return [v.detach().cpu().numpy() for _, v in self.net.state_dict().items()]

    def set_weights(self, weights):
        sd = OrderedDict(
            (k, torch.tensor(v)) for k, v in zip(self.net.state_dict().keys(), weights)
        )
        self.net.load_state_dict(sd, strict=True)
        # Reset Adam optimizer: after FedAvg replaces all weights the old
        # momentum / variance estimates are stale and would cause gradient
        # divergence in the first training steps of the next round.
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def save(self, path, meta=None):
        payload = {
            'state_dict': self.net.state_dict(),
            'state_dim': self.state_dim,
            'n_features': self.state_dim,
            'n_actions': self.n_actions,
            'arch': 'RL-Dueling',
            'mode': 'rl_then_fl',
            'class_w': self.class_w.detach().cpu().numpy(),
        }
        if meta:
            payload.update(meta)
        torch.save(payload, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(ck['state_dict'])
        self.net.eval()
        return ck
