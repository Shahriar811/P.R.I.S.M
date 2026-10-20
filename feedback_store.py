from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np


class FeedbackStore:
    """Persist local continual-learning feedback as a client-side FL shard extension."""

    def __init__(self, federated_dir: str, n_features: int, class_names: List[str], client_id: int = 0):
        self.federated_dir = federated_dir
        self.n_features = int(n_features)
        self.class_names = list(class_names)
        self.client_id = int(client_id)
        self.dir = os.path.join(federated_dir, 'feedback')
        self.data_path = os.path.join(self.dir, f'client_{self.client_id}_feedback.npz')
        self.meta_path = os.path.join(self.dir, 'feedback_meta.json')
        os.makedirs(self.dir, exist_ok=True)

    def _load_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        if not os.path.exists(self.data_path):
            return (
                np.empty((0, self.n_features), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
            )
        d = np.load(self.data_path)
        X = np.asarray(d['X'], dtype=np.float32).reshape(-1, self.n_features)
        y = np.asarray(d['y'], dtype=np.int64).reshape(-1)
        return X, y

    def append(self, Xs: np.ndarray, true_index: int, meta: Optional[Dict] = None) -> Dict:
        x = np.asarray(Xs, dtype=np.float32).reshape(1, self.n_features)
        y = np.asarray([int(true_index)], dtype=np.int64)
        X0, y0 = self._load_arrays()
        X = np.vstack([X0, x])
        yy = np.concatenate([y0, y], axis=0)
        np.savez_compressed(self.data_path, X=X, y=yy)

        payload = self.status()
        payload.update({
            'last_true_label': self.class_names[int(true_index)],
            'last_saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        })
        if meta:
            payload['last_meta'] = meta
        with open(self.meta_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
        return payload

    def status(self) -> Dict:
        X, y = self._load_arrays()
        counts = {}
        if len(y):
            uniq, cnt = np.unique(y, return_counts=True)
            counts = {
                self.class_names[int(k)] if 0 <= int(k) < len(self.class_names) else str(int(k)): int(v)
                for k, v in zip(uniq, cnt)
            }
        meta = {}
        if os.path.exists(self.meta_path):
            try:
                with open(self.meta_path, encoding='utf-8') as f:
                    meta = json.load(f)
            except Exception:
                meta = {}
        return {
            'client_id': self.client_id,
            'samples': int(len(y)),
            'path': self.data_path,
            'class_counts': counts,
            'last_saved_at': meta.get('last_saved_at'),
            'last_true_label': meta.get('last_true_label'),
            'last_meta': meta.get('last_meta'),
        }
