from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from typing import Callable, Dict, Optional


class RLFLSyncManager:
    """Launch one Flower RL->FL cycle and reload the new global model when done."""

    def __init__(self, project_root: str, on_complete: Optional[Callable[[], None]] = None):
        self.project_root = project_root
        self.federated_dir = os.path.join(project_root, 'federated')
        self.on_complete = on_complete
        self._lock = threading.Lock()
        self._thread = None
        self._state = {
            'running': False,
            'last_reason': None,
            'last_started_at': None,
            'last_finished_at': None,
            'last_result': None,
            'last_error': None,
            'cycles': 0,
        }

    def status(self) -> Dict:
        with self._lock:
            return dict(self._state)

    def trigger(self, reason: str = 'manual') -> Dict:
        with self._lock:
            if self._state['running']:
                return {'started': False, 'error': 'Federated RL cycle already running.', **self._state}
            self._state.update({
                'running': True,
                'last_reason': reason,
                'last_started_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'last_error': None,
                'last_result': 'starting',
            })
            self._thread = threading.Thread(target=self._run, args=(reason,), daemon=True)
            self._thread.start()
            return {'started': True, **self._state}

    def _run(self, reason: str):
        env = os.environ.copy()
        py = sys.executable
        logs_dir = os.path.join(self.federated_dir, 'runtime_logs')
        os.makedirs(logs_dir, exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S')
        processes = []
        log_handles = []  # track all file handles for cleanup
        try:
            server_log = open(os.path.join(logs_dir, f'server_{ts}.log'), 'w', encoding='utf-8')
            log_handles.append(server_log)
            processes.append(subprocess.Popen(
                [py, 'rl_server.py'],
                cwd=self.federated_dir,
                env=env,
                stdout=server_log,
                stderr=subprocess.STDOUT,
            ))
            time.sleep(4)
            for cid in range(3):
                log = open(os.path.join(logs_dir, f'client_{cid}_{ts}.log'), 'w', encoding='utf-8')
                log_handles.append(log)
                processes.append(subprocess.Popen(
                    [py, 'rl_client.py', str(cid)],
                    cwd=self.federated_dir,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                ))
                time.sleep(1)

            rcodes = [p.wait() for p in processes]
            ok = all(code == 0 for code in rcodes)
            if ok and callable(self.on_complete):
                self.on_complete()
            with self._lock:
                self._state.update({
                    'running': False,
                    'last_finished_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'last_result': 'completed' if ok else f'failed: {rcodes}',
                    'last_error': None if ok else f'Return codes: {rcodes}',
                    'cycles': int(self._state['cycles']) + 1,
                })
        except Exception as e:
            with self._lock:
                self._state.update({
                    'running': False,
                    'last_finished_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'last_result': 'failed',
                    'last_error': str(e),
                })
        finally:
            # Close all log file handles to release locks (Issue #9)
            for h in log_handles:
                try:
                    h.close()
                except Exception:
                    pass
