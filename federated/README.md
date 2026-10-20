# Federated Learning — RL → FL (FedAvg over RL)

**Current design:** train **RL first** (notebook Section 10), then federate the **RL agent**
across 3 clients with Flower. Clients share **weights only** — not raw flows.

| Role | Count | Process | Address |
|------|-------|---------|---------|
| Main server | 1 | `rl_server.py` | `0.0.0.0:8080` |
| Clients | 3 | `rl_client.py 0/1/2` | `127.0.0.1:8080` |

**Architecture:** RL-Dueling (continual learner).  
Launcher: `run_rl_federated.bat`  
Offline quick train: `python scripts/train_rl_federated_offline.py` from project root.

Legacy ResNet-MLP scripts (`server.py` / `client.py` / `run_federated.bat`) remain for reference only.

---

## Prerequisites

1. Run notebook **Section 10** (local RL) so `Saved model/RL/dqn_improved.pt` exists (recommended seed).
2. Confirm `federated/data/client_*.npz` + `test.npz` exist (from earlier partition step).
3. `pip install flwr torch`

---

## Windows (recommended)

```bat
federated\run_rl_federated.bat
```

This opens **4 terminal windows**: 1 server + 3 clients averaging RL weights.

## Option B — Manual terminals

```bat
cd federated
set FL_ROUNDS=5
set FL_CLIENTS=3
python rl_server.py
```

In three other terminals:

```bat
cd federated
python rl_client.py 0
python rl_client.py 1
python rl_client.py 2
```

Run each client command in its own terminal, using the same activated environment.
On Linux/macOS, use `export FL_ROUNDS=5` and `export FL_CLIENTS=3` in the server
terminal instead of the Windows `set` commands above.

## Option C — From the notebook

Notebook **Section 10B** describes the RL-to-FL launcher. Run it after local RL
training. Section 9.3 and its `RUN_FL` switch launch the legacy ResNet-MLP
experiment, not the current federated RL path.

---

## What “real” looks like

| Evidence | Where |
|----------|--------|
| Separate processes | Task Manager / 4 cmd windows |
| Round logs | `FL-SERVER` window (`global_test_acc=...`) |
| Live status JSON | `federated/status.json` |
| Per-client heartbeats | `federated/client_0_status.json` … `client_2_status.json` |
| Web UI live panel | http://127.0.0.1:7860 → **Live FL / RL Systems** |
| Saved global model | `federated/global_model.pt` and `Saved model/FL/federated_rl_fedavg.pt` |

If `status.json` updates with a new `round` and `updated_at` while clients show `local_training`, the system is **not** a simulation.

---

## After FL finishes

1. The RL server evaluates the global model on the held-out set each round.
   Cell **9.4** belongs to the legacy ResNet pipeline; do not use it to load an RL checkpoint.
2. Start or restart `python ids_server.py`, or wait for its federated-checkpoint
   watcher to reload the new model.
3. Open the dashboard's **Federated FL** page. A loaded checkpoint can remain
   available after all client/server processes have exited.

---

## Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `FL_ROUNDS` | `5` | FedAvg communication rounds |
| `FL_CLIENTS` | `3` | Minimum clients required (must match launched clients) |

## Local development and privacy scope

The default server listens on all interfaces at port 8080 while clients connect
over loopback. Keep federation on a trusted local machine/network; TLS,
authentication, secure aggregation, and differential privacy are not configured.
Sending weights instead of raw shards does not by itself guarantee privacy.
Client training data, feedback, logs, checkpoints, and status files are ignored
by Git. Source code and instructions are published; those local artifacts are not.
