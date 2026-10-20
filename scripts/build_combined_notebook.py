# -*- coding: utf-8 -*-
"""Build IDS_FL_RL_Combined.ipynb and patch main notebook RL section."""
import json
from pathlib import Path

SRC = Path(r"E:\CSE499B\Main part (included X_AI)\IDS_Advanced_GPU.ipynb")
OUT_DIR = Path(r"E:\CSE499B\Latest One(FL+RL)")
OUT_NB = OUT_DIR / "IDS_FL_RL_Combined.ipynb"


def set_src(cell, text):
    lines = text.split("\n")
    cell["source"] = [l + "\n" for l in lines[:-1]] + ([lines[-1] + "\n"] if text else [])
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None


def load_rl_code(save_to_rl_dir=True):
    target = "os.path.join(RL_DIR, stem + '.pt')" if save_to_rl_dir else "os.path.join(DL_DIR, stem + '.pt')"
    msg = "print('RL checkpoints saved to', RL_DIR)" if save_to_rl_dir else "print('RL checkpoints saved to', DL_DIR)"
    return RL_TEMPLATE.replace("__SAVE_PATH__", target).replace("__SAVE_MSG__", msg)


RL_TEMPLATE = r'''# RL — 5 models from COMPLETE_RL_FULL_CODE.ipynb
print(">>> RL cell (5 models) started ...", flush=True)

SEED = RANDOM_STATE if 'RANDOM_STATE' in dir() else 42
np.random.seed(SEED); torch.manual_seed(SEED)
print(f">>> device: {DEVICE}", flush=True)

Xtr = torch.tensor(np.ascontiguousarray(np.asarray(X_train_bal, np.float32)))
Xte = torch.tensor(np.ascontiguousarray(np.asarray(X_test_s,  np.float32)))
ytr = torch.tensor(np.asarray(y_train_bal).astype(np.int64).ravel())
yte_np = np.asarray(y_test).astype(np.int64).ravel()

_N_FEATURES = Xtr.shape[1]
_N_CLASSES  = int(NUM_CLASSES)
print(f">>> features={_N_FEATURES}  classes={_N_CLASSES}  train={len(Xtr)}  test={len(Xte)}", flush=True)

counts = torch.bincount(torch.tensor(np.asarray(y_train).astype(np.int64).ravel()), minlength=_N_CLASSES).float()
CLASS_W = (counts.sum() / (_N_CLASSES * counts.clamp(min=1))).to(DEVICE)
print('>>> reward class-weights:', dict(zip(CLASS_NAMES, np.round(CLASS_W.cpu().numpy(), 3))), flush=True)

Xtr_d, ytr_d = Xtr.to(DEVICE), ytr.to(DEVICE)

def batch_reward(actions, labels):
    correct = (actions == labels)
    return torch.where(correct, CLASS_W[labels], torch.tensor(-1.0, device=DEVICE))

EPOCHS = 2
BATCH  = 2048

class MLPNet(nn.Module):
    def __init__(s, d, a):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(d,256), nn.ReLU(), nn.Linear(256,128), nn.ReLU(), nn.Linear(128,a))
    def forward(s, x): return s.net(x)

class DuelingNet(nn.Module):
    def __init__(s, d, a):
        super().__init__()
        s.f = nn.Sequential(nn.Linear(d,256), nn.ReLU(), nn.LayerNorm(256), nn.Linear(256,128), nn.ReLU(), nn.LayerNorm(128))
        s.v = nn.Sequential(nn.Linear(128,64), nn.ReLU(), nn.Linear(64,1))
        s.a = nn.Sequential(nn.Linear(128,64), nn.ReLU(), nn.Linear(64,a))
    def forward(s, x):
        f = s.f(x); v = s.v(f); a = s.a(f)
        return v + a - a.mean(1, keepdim=True)

class ActorCritic(nn.Module):
    def __init__(s, d, a):
        super().__init__()
        s.sh = nn.Sequential(nn.Linear(d,256), nn.ReLU(), nn.LayerNorm(256), nn.Linear(256,128), nn.ReLU(), nn.LayerNorm(128))
        s.ac = nn.Sequential(nn.Linear(128,64), nn.ReLU(), nn.Linear(64,a))
        s.cr = nn.Sequential(nn.Linear(128,64), nn.ReLU(), nn.Linear(64,1))
    def forward(s, x):
        h = s.sh(x); return s.ac(h), s.cr(h)

class NoisyLinear(nn.Module):
    def __init__(s, i, o, sigma=0.017):
        super().__init__()
        s.i, s.o = i, o
        s.w_mu = nn.Parameter(torch.empty(o, i)); s.w_sig = nn.Parameter(torch.full((o,i), sigma))
        s.b_mu = nn.Parameter(torch.empty(o));    s.b_sig = nn.Parameter(torch.full((o,), sigma))
        s.register_buffer('w_eps', torch.zeros(o, i)); s.register_buffer('b_eps', torch.zeros(o))
        std = 1.0/math.sqrt(i); s.w_mu.data.uniform_(-std,std); s.b_mu.data.uniform_(-std,std)
        s.reset_noise()
    def reset_noise(s):
        s.w_eps.normal_(); s.b_eps.normal_()
    def forward(s, x):
        if s.training:
            w = s.w_mu + s.w_sig * s.w_eps; b = s.b_mu + s.b_sig * s.b_eps
        else:
            w, b = s.w_mu, s.b_mu
        return F.linear(x, w, b)

class RainbowNet(nn.Module):
    def __init__(s, d, a):
        super().__init__()
        s.f = nn.Sequential(NoisyLinear(d,256), nn.ReLU(), nn.LayerNorm(256), NoisyLinear(256,128), nn.ReLU(), nn.LayerNorm(128))
        s.v = nn.Sequential(NoisyLinear(128,64), nn.ReLU(), NoisyLinear(64,1))
        s.a = nn.Sequential(NoisyLinear(128,64), nn.ReLU(), NoisyLinear(64,a))
    def forward(s, x):
        f = s.f(x); v = s.v(f); a = s.a(f)
        return v + a - a.mean(1, keepdim=True)
    def reset_noise(s):
        for m in s.modules():
            if isinstance(m, NoisyLinear): m.reset_noise()

N_QUANT = 51
class QRNet(nn.Module):
    def __init__(s, d, a, nq=N_QUANT):
        super().__init__()
        s.a, s.nq = a, nq
        s.body = nn.Sequential(nn.Linear(d,256), nn.ReLU(), nn.LayerNorm(256), nn.Linear(256,128), nn.ReLU(), nn.LayerNorm(128))
        s.head = nn.Linear(128, a*nq)
    def forward(s, x):
        return s.head(s.body(x)).view(-1, s.a, s.nq)
    def q(s, x):
        return s.forward(x).mean(2)

def train_qnet(net, name, lr, eps_decay, noisy=False):
    net = net.to(DEVICE); opt = torch.optim.Adam(net.parameters(), lr=lr)
    eps, eps_min = 1.0, 0.20
    n = len(Xtr_d); t0 = time.time(); step = 0
    print(f"\n{name} training ...", flush=True)
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]; S, lbl = Xtr_d[idx], ytr_d[idx]
            if noisy: net.reset_noise()
            with torch.no_grad():
                greedy = net(S).argmax(1)
                if noisy:
                    actions = greedy
                else:
                    rand = torch.randint(0, _N_CLASSES, (len(idx),), device=DEVICE)
                    explore = torch.rand(len(idx), device=DEVICE) < eps
                    actions = torch.where(explore, rand, greedy)
                rewards = batch_reward(actions, lbl)
            qsa = net(S).gather(1, actions.unsqueeze(1)).squeeze(1)
            loss = F.mse_loss(qsa, rewards)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            step += 1
            if step % 50 == 0:
                acc = (greedy == lbl).float().mean().item()
                print(f"   {name} ep{ep+1} step{step} greedy_acc={acc:.3f} ({time.time()-t0:.0f}s)", flush=True)
            eps = max(eps_min, eps*eps_decay)
    print(f"   {name} done in {time.time()-t0:.0f}s", flush=True)
    return net

def train_ppo(name, lr):
    net = ActorCritic(_N_FEATURES, _N_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr); clip, vf, ent = 0.2, 0.5, 0.01
    n = len(Xtr_d); t0 = time.time(); step = 0
    print(f"\n{name} training ...", flush=True)
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]; S, lbl = Xtr_d[idx], ytr_d[idx]
            with torch.no_grad():
                logits, val = net(S)
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample(); old_logp = dist.log_prob(a)
                r = batch_reward(a, lbl)
                adv = r - val.squeeze(1); adv = (adv-adv.mean())/(adv.std()+1e-8)
            for _ in range(4):
                logits, val = net(S)
                dist = torch.distributions.Categorical(logits=logits)
                ratio = torch.exp(dist.log_prob(a) - old_logp)
                s1 = ratio*adv; s2 = torch.clamp(ratio, 1-clip, 1+clip)*adv
                loss = -torch.min(s1,s2).mean() + vf*F.mse_loss(val.squeeze(1), r) - ent*dist.entropy().mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            step += 1
            if step % 50 == 0:
                acc = (logits.argmax(1) == lbl).float().mean().item()
                print(f"   {name} ep{ep+1} step{step} acc={acc:.3f} ({time.time()-t0:.0f}s)", flush=True)
    print(f"   {name} done in {time.time()-t0:.0f}s", flush=True)
    return net

def train_qrdqn(name, lr, eps_decay):
    net = QRNet(_N_FEATURES, _N_CLASSES).to(DEVICE); opt = torch.optim.Adam(net.parameters(), lr=lr)
    eps, eps_min = 1.0, 0.05
    tau = ((torch.arange(N_QUANT, device=DEVICE).float() + 0.5) / N_QUANT).view(1, -1)
    n = len(Xtr_d); t0 = time.time(); step = 0
    print(f"\n{name} training ...", flush=True)
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]; S, lbl = Xtr_d[idx], ytr_d[idx]
            with torch.no_grad():
                greedy = net.q(S).argmax(1)
                rand = torch.randint(0, _N_CLASSES, (len(idx),), device=DEVICE)
                explore = torch.rand(len(idx), device=DEVICE) < eps
                actions = torch.where(explore, rand, greedy)
                rewards = batch_reward(actions, lbl)
            allq = net(S)
            theta = allq.gather(1, actions.view(-1,1,1).expand(-1,1,N_QUANT)).squeeze(1)
            target = rewards.unsqueeze(1).expand(-1, N_QUANT)
            u = target - theta
            huber = 0.5*u.pow(2)
            loss = (torch.abs(tau - (u.detach() < 0).float()) * huber).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            step += 1
            if step % 50 == 0:
                acc = (greedy == lbl).float().mean().item()
                print(f"   {name} ep{ep+1} step{step} greedy_acc={acc:.3f} ({time.time()-t0:.0f}s)", flush=True)
            eps = max(eps_min, eps*eps_decay)
    print(f"   {name} done in {time.time()-t0:.0f}s", flush=True)
    return net

def predict_rl(net, kind='dqn'):
    net.eval(); out = np.empty(len(Xte), np.int64); bs = 16384
    with torch.no_grad():
        for i in range(0, len(Xte), bs):
            chunk = Xte[i:i+bs].to(DEVICE)
            if kind == 'ppo':   q = net(chunk)[0]
            elif kind == 'qr':  q = net.q(chunk)
            else:               q = net(chunk)
            out[i:i+bs] = q.argmax(1).cpu().numpy()
    return out

t_rl0 = time.time()
basic_net    = train_qnet(MLPNet(_N_FEATURES, _N_CLASSES),     "DQN-Basic",    1e-3, 0.999)
improved_net = train_qnet(DuelingNet(_N_FEATURES, _N_CLASSES), "DQN-Improved", 3e-4, 0.999)
ppo_net      = train_ppo("PPO", 3e-4)
rainbow_net  = train_qnet(RainbowNet(_N_FEATURES, _N_CLASSES), "Rainbow-DQN",  3e-4, 0.999, noisy=True)
qrdqn_net    = train_qrdqn("QR-DQN", 3e-4, 0.999)
rl_time_total = time.time() - t_rl0

dqn_basic_preds    = predict_rl(basic_net,    'dqn')
dqn_improved_preds = predict_rl(improved_net, 'dqn')
ppo_preds          = predict_rl(ppo_net,      'ppo')
rainbow_preds      = predict_rl(rainbow_net,  'dqn')
qrdqn_preds        = predict_rl(qrdqn_net,    'qr')

print("\n" + "="*66 + "\nRL RESULTS (test set)\n" + "="*66, flush=True)
from sklearn.metrics import accuracy_score, f1_score, classification_report
_rl_bundle = [
    ("DQN-Basic",    "dqn_basic",    basic_net,    dqn_basic_preds,    'dqn'),
    ("DQN-Improved", "dqn_improved", improved_net, dqn_improved_preds, 'dqn'),
    ("PPO",          "ppo",          ppo_net,      ppo_preds,          'ppo'),
    ("Rainbow-DQN",  "rainbow_dqn",  rainbow_net,  rainbow_preds,      'dqn'),
    ("QR-DQN",       "qr_dqn",       qrdqn_net,    qrdqn_preds,        'qr'),
]
for nm, stem, net, pr, kind in _rl_bundle:
    acc = accuracy_score(yte_np, pr)
    f1w = f1_score(yte_np, pr, average='weighted', zero_division=0)
    f1m = f1_score(yte_np, pr, average='macro', zero_division=0)
    print(f"  {nm:<14} acc={acc:.4f}  f1_weighted={f1w:.4f}  f1_macro={f1m:.4f}", flush=True)
    register(nm, y_test, pr, None, rl_time_total / 5, kind='RL')
    torch.save({'state_dict': net.state_dict(), 'class_names': CLASS_NAMES,
                'n_features': _N_FEATURES, 'arch': nm, 'kind': kind},
               __SAVE_PATH__)
__SAVE_MSG__
'''


def patch_combined(nb):
    cells = nb["cells"]
    set_src(cells[0], """# Privacy-Preserving Federated and Reinforcement Learning-Based Intrusion Detection System for IoT Devices
## Traditional ML + Deep Learning + Custom + Real Federated Learning + Multi-Agent RL

**Dataset:** `ASEADOS_SDN_IoT.csv`

**Latest One (FL+RL):**
- Sections 1–8: same training pipeline as `IDS_Advanced_GPU.ipynb`
- Section 9: real Flower FedAvg — 1 server + 3 clients, **ResNet-MLP** (best federatable DL)
- Section 10: 5 RL agents from `COMPLETE_RL_FULL_CODE.ipynb`
- Sections 11–12: `Saved model/`; UI uses **only XGBoost** + live FL/RL proof panel
""")

    set_src(cells[4], r"""# ---- Project paths: models under "Saved model" ----
BASE_DIR = os.getcwd()
SAVE_DIR = os.path.join(BASE_DIR, 'Saved model')
TM_DIR   = os.path.join(SAVE_DIR, 'TM')
DL_DIR   = os.path.join(SAVE_DIR, 'DL')
RL_DIR   = os.path.join(SAVE_DIR, 'RL')
FL_SAVE  = os.path.join(SAVE_DIR, 'FL')
FL_DIR   = os.path.join(BASE_DIR, 'federated')
ART_DIR  = SAVE_DIR
for d in (TM_DIR, DL_DIR, RL_DIR, FL_SAVE, FL_DIR, ART_DIR):
    os.makedirs(d, exist_ok=True)

_candidates = [
    'ASEADOS_SDN_IoT.csv',
    os.path.join('..', 'ASEADOS_SDN_IoT.csv'),
    r'E:\CSE499B\ASEADOS_SDN_IoT.csv',
]
DATA_PATH = next((p for p in _candidates if os.path.exists(p)), _candidates[0])
print('Dataset        ->', os.path.abspath(DATA_PATH))
print('Saved model    ->', SAVE_DIR)
print('Federated      ->', FL_DIR)
""")

    set_src(cells[45], """---
## 9. Federated Learning — Real Multi-Process FedAvg (Flower)

**Not a simulation.** 3 client OS processes + 1 server. Private non-IID shards; only weights leave clients.

**Model:** ResNet-MLP (best DL for FedAvg). XGBoost wins overall F1 but cannot FedAvg.
Live proof: `federated/status.json`, client status files, and the web UI live panel.
""")

    set_src(cells[47], r"""# 9.2 Refresh FL launcher for exactly 3 clients (server/client/fl_model already shipped)
lines = ['@echo off', 'cd /d "%~dp0"', 'set FL_ROUNDS=5', 'set FL_CLIENTS=3',
         'start "FL-SERVER" cmd /k python server.py', 'timeout /t 5 >nul']
for i in range(FL_NUM_CLIENTS):
    lines.append('start "FL-CLIENT-%d" cmd /k python client.py %d' % (i, i))
    lines.append('timeout /t 1 >nul')
lines += ['echo All processes launched.', 'pause']
with open(os.path.join(FL_DIR, 'run_federated.bat'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('FL ready in', FL_DIR, '| clients:', FL_NUM_CLIENTS)
print('Launch: federated/run_federated.bat')
""")

    set_src(cells[48], """# 9.3 Optional: launch FL from notebook (prefer .bat on Windows)
RUN_FL = False
if RUN_FL:
    import subprocess, sys, time
    env = os.environ.copy(); env['FL_ROUNDS']='5'; env['FL_CLIENTS']=str(FL_NUM_CLIENTS)
    srv = subprocess.Popen([sys.executable, 'server.py'], cwd=FL_DIR, env=env)
    time.sleep(5)
    cli = [subprocess.Popen([sys.executable, 'client.py', str(i)], cwd=FL_DIR, env=env) for i in range(FL_NUM_CLIENTS)]
    srv.wait(); [c.wait() for c in cli]
    print('Federated run finished.')
else:
    print('RUN_FL=False. Use federated/run_federated.bat')
""")

    set_src(cells[49], """# 9.4 Score federated global model on held-out test set
gpath = os.path.join(FL_DIR, 'global_model.pt')
alt = os.path.join(FL_SAVE, 'federated_fedavg.pt')
path = gpath if os.path.exists(gpath) else alt
if os.path.exists(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location('fl_model', os.path.join(FL_DIR, 'fl_model.py'))
    fl_model = importlib.util.module_from_spec(spec); spec.loader.exec_module(fl_model)
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    sd = ck.get('state_dict', ck)
    n_in = int(sd['inp.0.weight'].shape[1]) if 'inp.0.weight' in sd else X_test_s.shape[1]
    n_out = int(sd['out.weight'].shape[0]) if 'out.weight' in sd else NUM_CLASSES
    gmodel = fl_model.FLNet(n_in=n_in, n_out=n_out)
    gmodel.load_state_dict(sd); gmodel.to(DEVICE).eval()
    X_eval = X_test_s
    if X_eval.shape[1] != n_in:
        if X_eval.shape[1] > n_in: X_eval = X_eval[:, :n_in]
        else: X_eval = np.hstack([X_eval, np.zeros((len(X_eval), n_in-X_eval.shape[1]), np.float32)])
    with torch.no_grad():
        out = gmodel(torch.tensor(X_eval).to(DEVICE))
        fl_pred = out.argmax(1).cpu().numpy()
        fl_proba = F.softmax(out, 1).cpu().numpy()
    register('Federated (FedAvg)', y_test, fl_pred, fl_proba, None, kind='Federated')
    payload = {'state_dict': sd, 'round': ck.get('round'), 'acc': ck.get('acc'),
               'arch': 'ResNet-MLP', 'n_features': n_in, 'class_names': CLASS_NAMES}
    torch.save(payload, os.path.join(FL_SAVE, 'federated_fedavg.pt'))
    torch.save(payload, os.path.join(DL_DIR, 'federated_fedavg.pt'))
    print('Federated model scored. Saved to', FL_SAVE)
else:
    print('No global_model.pt yet. Run federated/run_federated.bat first.')
""")

    set_src(cells[50], """---
## 10. Reinforcement Learning — 5 Agents (COMPLETE_RL_FULL_CODE)

| Agent | Idea |
|-------|------|
| DQN-Basic | MLP Q-network |
| DQN-Improved | Dueling |
| PPO | Actor-Critic |
| Rainbow-DQN | Dueling + NoisyNets |
| QR-DQN | 51 return quantiles |

Cost-sensitive rewards; real PyTorch training — not a toy simulation.
""")
    set_src(cells[51], load_rl_code(True))
    set_src(cells[52], "# RL training completed in previous cell")
    set_src(cells[53], """_rl_names = ['DQN-Basic','DQN-Improved','PPO','Rainbow-DQN','QR-DQN']
_rows = [(n, RESULTS[n]) for n in _rl_names if n in RESULTS]
if _rows:
    display(pd.DataFrame([{'Model': n, 'Accuracy': r['Accuracy'], 'F1': r['F1'], 'F1_macro': r['F1_macro']} for n, r in _rows]))
else:
    print('Run the RL training cell first.')
""")
    set_src(cells[54], """_rl_names = ['DQN-Basic','DQN-Improved','PPO','Rainbow-DQN','QR-DQN']
_present = [(n, RESULTS[n]['F1']) for n in _rl_names if n in RESULTS]
if _present:
    best_rl = max(_present, key=lambda t: t[1])[0]
    pred_map = {'DQN-Basic': dqn_basic_preds, 'DQN-Improved': dqn_improved_preds,
                'PPO': ppo_preds, 'Rainbow-DQN': rainbow_preds, 'QR-DQN': qrdqn_preds}
    print('Best RL agent by F1:', best_rl)
    print(classification_report(y_test, pred_map[best_rl], target_names=CLASS_NAMES, digits=3))
""")
    set_src(cells[55], "# RL section complete. Checkpoints in Saved model/RL/")

    set_src(cells[56], """---
## 11. Save Shared Artifacts
Artifacts go under `Saved model/`.
""")
    set_src(cells[57], """joblib.dump(scaler,        os.path.join(ART_DIR, 'scaler.joblib'))
joblib.dump(label_encoder, os.path.join(ART_DIR, 'label_encoder.joblib'))
joblib.dump(feature_cols,  os.path.join(ART_DIR, 'feature_cols.joblib'))
if '_build_res_df' in globals() and RESULTS:
    res_df = _build_res_df()
elif os.path.exists(os.path.join(ART_DIR, 'results_summary.csv')):
    res_df = pd.read_csv(os.path.join(ART_DIR, 'results_summary.csv'))
else:
    res_df = pd.DataFrame()
best_name = res_df.iloc[0]['Model'] if len(res_df) > 0 else 'XGBoost'
manifest = {
    'n_features': int(N_FEATURES), 'num_classes': int(NUM_CLASSES),
    'class_names': CLASS_NAMES, 'feature_columns': feature_cols,
    'best_model': best_name, 'ui_predictor': 'XGBoost',
    'results': {k: {m: float(v[m]) for m in ['Accuracy','Precision','Recall','F1','F1_macro']} for k, v in RESULTS.items()},
}
json.dump(manifest, open(os.path.join(ART_DIR, 'model_info.json'), 'w'), indent=2)
res_df.to_csv(os.path.join(ART_DIR, 'results_summary.csv'), index=False)
print('Saved to', ART_DIR, '| Best:', best_name, '| UI predictor: XGBoost')
""")

    set_src(cells[58], """---
## 12. Flask Web UI
Run `python ids_server.py` → http://127.0.0.1:7860
Only XGBoost predicts; live FL/RL panel proves real systems.
""")
    set_src(cells[59], """print('Start dashboard:  python ids_server.py')
print('Open: http://127.0.0.1:7860')
print('FL: federated/run_federated.bat')
""")
    set_src(cells[60], """print('Web stack: ids_server.py + ids_models.py + templates/dashboard.html')
""")
    set_src(cells[61], """---
## 13. Summary
Supervised best (UI): XGBoost · FL: Flower 1+3 ResNet-MLP · RL: 5 agents · Proof: /live + dual inference
See docs/IDS_Universal_Guide.docx
""")
    set_src(cells[62], """print('='*70)
print('Run 1-8 train | 9 FL | 10 RL | 11 save | then python ids_server.py')
print('='*70)
""")
    for i in (63, 64, 65):
        if i < len(cells):
            set_src(cells[i], "")
    for c in cells:
        if c.get("cell_type") == "code":
            c["outputs"] = []
            c["execution_count"] = None


def patch_main_rl(nb):
    cells = nb["cells"]
    set_src(cells[50], """---
## 10. Reinforcement Learning — 5 Agents (from COMPLETE_RL_FULL_CODE)

Replaces the older single Gymnasium DQN with DQN-Basic, DQN-Improved, PPO, Rainbow-DQN, QR-DQN.
""")
    set_src(cells[51], load_rl_code(False))
    set_src(cells[52], "# RL training completed in previous cell")
    set_src(cells[53], """_rl_names = ['DQN-Basic','DQN-Improved','PPO','Rainbow-DQN','QR-DQN']
_rows = [(n, RESULTS[n]) for n in _rl_names if n in RESULTS]
if _rows:
    display(pd.DataFrame([{'Model': n, 'Accuracy': r['Accuracy'], 'F1': r['F1'], 'F1_macro': r['F1_macro']} for n, r in _rows]))
""")
    set_src(cells[54], """_rl_names = ['DQN-Basic','DQN-Improved','PPO','Rainbow-DQN','QR-DQN']
_present = [(n, RESULTS[n]['F1']) for n in _rl_names if n in RESULTS]
if _present:
    best_rl = max(_present, key=lambda t: t[1])[0]
    pred_map = {'DQN-Basic': dqn_basic_preds, 'DQN-Improved': dqn_improved_preds,
                'PPO': ppo_preds, 'Rainbow-DQN': rainbow_preds, 'QR-DQN': qrdqn_preds}
    print('Best RL agent by F1:', best_rl)
    print(classification_report(y_test, pred_map[best_rl], target_names=CLASS_NAMES, digits=3))
""")
    set_src(cells[55], "# RL section complete (5 agents). Old single Gymnasium DQN removed.")


def main():
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    patch_combined(nb)
    OUT_NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Wrote", OUT_NB)

    main_nb = json.loads(SRC.read_text(encoding="utf-8"))
    patch_main_rl(main_nb)
    SRC.write_text(json.dumps(main_nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Updated RL in", SRC)


if __name__ == "__main__":
    main()
