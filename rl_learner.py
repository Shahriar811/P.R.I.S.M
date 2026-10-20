"""
rl_learner.py  —  Option A: learnable RL agent for the AEGIS dashboard.

Loads the trained DQN-Improved weights and lets the agent learn from user
corrections on the Predict page ("this prediction was actually X"). Uses a
replay buffer so it doesn't forget old knowledge, and a learning ON/OFF switch.

Kept SEPARATE from ids_models.py so the existing (frozen, inference-only) RL
display path is untouched. This adds learning; it does not modify prediction.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random

from rl_agent import DuelingNet  # single source of truth — never duplicate


class LearnableRLAgent:
    """
    Wraps a DuelingNet that starts from trained weights and can learn online
    from labelled corrections. Cost-sensitive MSE reward (NOT Huber).
    """
    def __init__(self, rl_dir, n_features, class_names, device,
                 class_counts=None, lr=1e-4, replay_capacity=20000,
                 replay_batch=128):
        self.device = device
        self.class_names = class_names
        self.n_actions = len(class_names)
        self.n_features = n_features
        self.net = DuelingNet(n_features, self.n_actions).to(device)
        self.loaded_from = None
        self._load_trained(rl_dir)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.eps = 0.05  # small exploration during replay, aligned with rl_agent

        self.replay = deque(maxlen=replay_capacity)
        self.replay_batch = replay_batch
        self.learning = False           # dashboard switch; starts OFF (safe)

        # Cost-sensitive class weights — same formula as RLAgent so the
        # dashboard learner and FL clients train with identical reward scales.
        if class_counts is not None:
            c = np.asarray(class_counts, dtype=np.float32)
            w = c.sum() / (self.n_actions * np.maximum(c, 1))
        else:
            w = np.ones(self.n_actions, dtype=np.float32)
        self.class_w = torch.tensor(w, dtype=torch.float32, device=device)

        self.n_updates = 0
        self.history = deque(maxlen=200)   # recent (true, was_correct) for a live accuracy read

    def _load_trained(self, rl_dir):
        path = os.path.join(rl_dir, 'dqn_improved.pt')
        if os.path.exists(path):
            ck = torch.load(path, map_location=self.device, weights_only=False)
            sd = ck.get('state_dict', ck)
            try:
                self.net.load_state_dict(sd)
                self.loaded_from = path
                print(f'  [learner] loaded trained DQN-Improved from {path}')
            except Exception as e:
                print(f'  [learner] could not load weights ({e}); starting fresh')
        else:
            print(f'  [learner] no dqn_improved.pt found at {path}; starting fresh')
        self.net.to(self.device)

    # ── switch ──────────────────────────────────────────────────────────────
    def set_learning(self, on: bool):
        self.learning = bool(on)
        return self.learning

    # ── inference (what the agent currently predicts for a flow) ───────────
    def predict(self, Xs):
        self.net.eval()
        t = torch.as_tensor(np.atleast_2d(Xs).astype(np.float32), device=self.device)
        with torch.no_grad():
            q = self.net(t)
            k = int(q.argmax(1)[0].item())
            probs = F.softmax(q, 1)[0].cpu().numpy()
            q_vals = q[0].cpu().numpy()
        return {'action': self.class_names[k], 'action_index': k,
                'confidence': float(probs[k]),
                'q_values': [float(x) for x in q_vals],
                'proba': [float(x) for x in probs]}

    def _reward(self, actions, labels):
        return torch.where(actions == labels, self.class_w[labels],
                           torch.tensor(-1.0, device=self.device))

    # ── learn from ONE correction: (flow features, true class index) ───────
    def learn_from_correction(self, Xs, true_index):
        """
        Called when the user confirms/corrects a prediction. Records the sample
        and, if learning is ON, does a cost-sensitive update plus replay.
        Returns a small dict describing what happened (for the UI).
        """
        x = np.atleast_2d(Xs).astype(np.float32)[0]
        true_index = int(true_index)

        # what did it think before? (for a before/after readout)
        before = self.predict(x)
        self.history.append(true_index == before['action_index'])

        if not self.learning:
            self.replay.append((x, true_index))
            return {'learning': False, 'before': before['action'],
                    'updated': False,
                    'msg': 'Learning is OFF — correction recorded but not trained.'}

        # Sample replay BEFORE appending to avoid double-counting this sample
        S = [x]; y = [true_index]
        if len(self.replay) > 0:
            k = min(self.replay_batch, len(self.replay))
            for xs, ys in random.sample(list(self.replay), k):
                S.append(xs); y.append(ys)
        self.replay.append((x, true_index))  # append AFTER sampling
        S = torch.as_tensor(np.array(S, dtype=np.float32), device=self.device)
        y = torch.as_tensor(np.array(y, dtype=np.int64), device=self.device)

        self.net.train()
        with torch.no_grad():
            actions = self.net(S).argmax(1)
            # force the *true* action to be evaluated for the corrected sample
            actions[0] = true_index
            rewards = self._reward(actions, y)
        qsa = self.net(S).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = F.mse_loss(qsa, rewards)          # MSE, not Huber
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()
        self.n_updates += 1

        after = self.predict(x)
        return {'learning': True, 'updated': True,
                'before': before['action'], 'after': after['action'],
                'true': self.class_names[true_index],
                'now_correct': after['action_index'] == true_index,
                'n_updates': self.n_updates, 'loss': float(loss.item()),
                'msg': f"Learned. Was '{before['action']}', "
                       f"now predicts '{after['action']}' (true: {self.class_names[true_index]})."}

    def status(self):
        recent = list(self.history)
        acc = float(np.mean(recent)) if recent else None
        return {
            'learning': self.learning,
            'n_updates': self.n_updates,
            'replay_size': len(self.replay),
            'recent_accuracy': acc,
            'loaded_from': self.loaded_from,
            'agent': 'DQN-Improved (learnable)',
        }

    def save(self, path):
        torch.save({'state_dict': self.net.state_dict(),
                    'n_features': self.n_features,
                    'class_names': self.class_names}, path)
        return path
