/* Shared site helpers — theme, settings, toast, live helpers */
(function () {
  const KEY = 'aegis_settings_v1';
  const defaults = {
    theme: 'day',
    soundAlert: false,
    saveHist: true,
    alertThreshold: 75,
  };

  function loadSettings() {
    try {
      return { ...defaults, ...JSON.parse(localStorage.getItem(KEY) || '{}') };
    } catch {
      return { ...defaults };
    }
  }

  function saveSettings(next) {
    const merged = { ...loadSettings(), ...next };
    localStorage.setItem(KEY, JSON.stringify(merged));
    applyTheme(merged.theme);
    window.dispatchEvent(new CustomEvent('aegis:settings', { detail: merged }));
    return merged;
  }

  function applyTheme(theme) {
    const t = theme === 'night' ? 'night' : 'day';
    document.documentElement.setAttribute('data-theme', t);
    const chip = document.getElementById('themeChip');
    if (chip) chip.textContent = t === 'night' ? 'Night mode' : 'Day mode';
  }

  function toast(msg, err = false) {
    let el = document.getElementById('toast');
    if (!el) {
      el = document.createElement('div');
      el.id = 'toast';
      el.className = 'toast';
      document.body.appendChild(el);
    }
    el.textContent = msg;
    el.className = 'toast show' + (err ? ' error' : '');
    setTimeout(() => el.classList.remove('show'), 3500);
  }

  let audioCtx = null;
  function unlockAudio() {
    if (!audioCtx) {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      if (!Ctx) return false;
      audioCtx = new Ctx();
    }
    if (audioCtx.state === 'suspended') audioCtx.resume();
    return true;
  }

  function playAlertSound(force) {
    const s = loadSettings();
    if (!force && !s.soundAlert) return;
    if (!unlockAudio()) return;
    const ctx = audioCtx, t0 = ctx.currentTime;
    [880, 660].forEach((freq, i) => {
      const o = ctx.createOscillator(), g = ctx.createGain();
      o.type = 'sine'; o.frequency.value = freq; o.connect(g); g.connect(ctx.destination);
      const st = t0 + i * 0.22;
      g.gain.setValueAtTime(0, st);
      g.gain.linearRampToValueAtTime(0.35, st + 0.02);
      g.gain.exponentialRampToValueAtTime(0.001, st + 0.28);
      o.start(st); o.stop(st + 0.28);
    });
  }

  const HIST_KEY = 'aegis_history_v1';
  function getHistory() {
    try { return JSON.parse(localStorage.getItem(HIST_KEY) || '[]'); } catch { return []; }
  }
  function pushHistory(item) {
    const s = loadSettings();
    if (!s.saveHist) return getHistory();
    const h = getHistory();
    h.unshift(item);
    if (h.length > 40) h.length = 40;
    localStorage.setItem(HIST_KEY, JSON.stringify(h));
    return h;
  }
  function clearHistory() {
    localStorage.removeItem(HIST_KEY);
  }

  const THREAT = {
    Benign: { level: 'low', color: '#1f7a4c', desc: 'Normal network traffic' },
    Bot: { level: 'high', color: '#b42318', desc: 'Botnet command & control' },
    DDoS: { level: 'high', color: '#9f1239', desc: 'Distributed denial of service' },
    Dos: { level: 'high', color: '#c2410c', desc: 'Denial of service attack' },
    Prob: { level: 'med', color: '#b45309', desc: 'Probe / reconnaissance scan' },
  };

  function threatClass(label) {
    const t = THREAT[label];
    if (!t) return 'suspicious';
    return t.level === 'low' ? 'benign' : t.level === 'med' ? 'suspicious' : 'attack';
  }

  function setHb(id, mode) {
    const el = document.getElementById(id);
    if (el) el.className = 'hb ' + (mode || '');
  }

  function pillClass(state, alive) {
    if (alive) return 'running';
    const s = (state || 'offline').toLowerCase().replace(/\s+/g, '_');
    if (['finished', 'round_complete'].includes(s)) return 'finished';
    if (['waiting_for_clients', 'fitting', 'local_training'].includes(s)) return s;
    if (['offline', 'not_started', 'unreadable', 'error'].includes(s)) return 'offline';
    return 'finished';
  }

  /** FL process chip: alive count + last run state (finished is OK — processes exit after rounds). */
  function flProcessSummary(live) {
    const fed = live?.federated || {};
    const alive = fed.processes_alive ?? 0;
    const expected = fed.processes_expected || 4;
    const st = (fed.server?.state || 'not_started').replace(/_/g, ' ');
    const round = fed.server?.round;
    const numRounds = fed.server?.num_rounds;
    if (alive > 0) return `${alive}/${expected} running`;
    if (st === 'finished' && round != null) {
      const rr = numRounds != null ? `${round}/${numRounds}` : String(round);
      return `0/${expected} · last run finished (${rr})`;
    }
    if (st && st !== 'not started') return `0/${expected} · ${st}`;
    return `0/${expected} · not started`;
  }

  function renderProcCard(p) {
    const isServer = p.role === 'server';
    const title = isServer ? 'Flower Server' : ('Client ' + (p.client_id ?? '?'));
    const state = p.state || 'offline';
    const alive = !!p.process_alive;
    const pill = alive ? 'RUNNING' : String(state).toUpperCase();
    const cls = pillClass(state, alive);
    const rows = isServer ? [
      ['PID', p.pid ?? '—'],
      ['State', state],
      ['Round', (p.round != null ? `${p.round}/${p.num_rounds || '?'}` : '—')],
      ['Min clients', p.min_clients ?? '3'],
      ['Updated', p.updated_at || '—'],
    ] : [
      ['PID', p.pid ?? '—'],
      ['State', state],
      ['Samples', p.n_samples != null ? Number(p.n_samples).toLocaleString() : '—'],
      ['Updated', p.updated_at || '—'],
    ];
    return `<div class="proc-card ${isServer ? 'server' : ''} ${alive ? 'alive' : 'dead'}">
      <div class="proc-head">
        <div class="proc-title"><span class="hb ${alive ? 'on' : (state === 'finished' ? 'warn' : 'off')}"></span> ${title}</div>
        <span class="proc-pill ${cls}">${pill}</span>
      </div>
      ${rows.map(r => `<div class="proc-kv">${r[0]} <span>${r[1]}</span></div>`).join('')}
    </div>`;
  }

  async function fetchLive() {
    const r = await fetch('/live');
    return r.json();
  }

  async function fetchInfo() {
    const r = await fetch('/info');
    return r.json();
  }

  function renderFlPanel(d, opts = {}) {
    const s = d.federated?.server || { role: 'server', state: 'not_started' };
    const clients = d.federated?.clients || [];
    while (clients.length < 3) clients.push({ client_id: clients.length, state: 'offline', role: 'client' });
    const procs = [{ ...s, role: 'server' }, ...clients.slice(0, 3).map((c, i) => ({ ...c, role: 'client', client_id: c.client_id ?? i }))];
    const grid = document.getElementById(opts.gridId || 'flProcGrid');
    const meta = document.getElementById(opts.metaId || 'flProcMeta');
    if (grid) grid.innerHTML = procs.map(renderProcCard).join('');
    const aliveN = d.federated?.processes_alive ?? procs.filter(p => p.process_alive).length;
    const expected = d.federated?.processes_expected || 4;
    const st = (s.state || 'not_started');
    if (meta) {
      const summary = flProcessSummary(d);
      const aliveCls = aliveN > 0 ? 'ops-good' : (st === 'finished' ? 'ops-warn' : 'ops-bad');
      meta.innerHTML = `Live OS processes: <strong class="${aliveCls}">${summary}</strong> · Processes exit when FL rounds finish · GLOBAL model stays loaded for Predict`;
    }
    const g = d.federated?.global_model || {};
    if (document.getElementById('flAcc')) {
      const acc = g.test_acc != null ? g.test_acc : s.global_test_acc;
      document.getElementById('flAcc').textContent = acc != null ? (acc * 100).toFixed(2) + '%' : '—';
    }
    if (document.getElementById('flArch')) {
      document.getElementById('flArch').textContent = g.architecture || d.federated?.architecture || s.architecture || '—';
    }
    if (document.getElementById('flGlobal')) {
      document.getElementById('flGlobal').innerHTML = d.federated?.global_model_loaded
        ? '<span class="ops-good">YES — GLOBAL</span>' : '<span class="ops-warn">NO</span>';
    }
    const active = ['fitting', 'round_complete', 'waiting_for_clients', 'finished', 'local_training'].includes(st);
    setHb(opts.hbId || 'flHb', aliveN > 0 ? 'on' : (st === 'finished' ? 'on' : (active ? 'warn' : 'off')));
    return { aliveN, expected, server: s, clients };
  }

  function renderVerifyPanel(live, predict) {
    const el = document.getElementById('verifyGrid');
    if (!el) return;
    const v = live?.verify || {};
    const g = live?.federated?.global_model || {};
    const items = [
      {
        ok: v.local_rl_loaded,
        title: 'Local RL loaded (Section 10)',
        desc: v.local_rl_loaded
          ? `${live.reinforcement?.agent_count || 0}/5 local agents ready`
          : 'Run notebook Section 10 and restart the server',
      },
      {
        ok: v.global_fl_loaded && v.global_is_rl_fl,
        warn: v.global_fl_loaded && !v.global_is_rl_fl,
        title: 'GLOBAL Federated RL loaded (RL→FL)',
        desc: v.global_is_rl_fl
          ? `GLOBAL checkpoint · round ${g.round ?? '—'} · ${(g.architecture || 'RL-Dueling')}`
          : (v.global_fl_loaded
            ? 'A global model is loaded but it is legacy DL-FL — run run_rl_federated.bat'
            : 'Run federated/run_rl_federated.bat or train_rl_federated_offline.py'),
      },
      {
        ok: true,
        title: 'Predict FL box = GLOBAL model (not a client)',
        desc: 'Client 0/1/2 train privately. Predict never reads a single client model.',
      },
      {
        ok: v.connection_ok,
        title: 'RL connected to FL',
        desc: v.connection_ok
          ? 'Local RL serves immediately; saved feedback can trigger Flower so GLOBAL FL catches up'
          : 'Need both local RL agents and an RL-Dueling global FL checkpoint',
      },
      {
        ok: null,
        warn: true,
        title: 'When does learning happen?',
        desc: v.learning_happens_during ||
          'Local RL can learn during serving; GLOBAL FL updates after a triggered federated FedAvg cycle',
      },
    ];
    if (predict?.fl && predict?.rl?.primary) {
      const same = predict.rl.primary.action === predict.fl.prediction;
      items.push({
        ok: true,
        title: 'Last Predict comparison',
        desc: same
          ? `Local & GLOBAL agree on ${predict.fl.prediction} (confidences can differ — different models)`
          : `Local=${predict.rl.primary.action}, GLOBAL=${predict.fl.prediction} (GLOBAL is FedAvg result)`,
      });
    }
    el.innerHTML = items.map(it => {
      const cls = it.ok === true ? 'ok' : (it.warn || it.ok === null ? 'warn' : 'bad');
      const mark = it.ok === true ? 'OK' : (it.warn || it.ok === null ? '!' : 'X');
      return `<div class="verify-item ${cls}"><span class="mark">${mark}</span><div><div class="v-title">${it.title}</div><div class="v-desc">${it.desc}</div></div></div>`;
    }).join('');
  }

  function renderRlPanel(d) {
    const agents = d.reinforcement?.agents_loaded || [];
    if (document.getElementById('rlCount')) document.getElementById('rlCount').textContent = agents.length + '/5';
    if (document.getElementById('rlAgents')) {
      document.getElementById('rlAgents').innerHTML = agents.length
        ? agents.map(a => `<span class="agent-chip">${a}</span>`).join('')
        : '<span class="ops-warn" style="font-size:16px">No RL checkpoints yet — run notebook Section 10</span>';
    }
    const last = d.reinforcement?.last_inference;
    if (last?.primary) {
      if (document.getElementById('rlAction')) document.getElementById('rlAction').textContent = last.primary.action;
      if (document.getElementById('rlAgent')) document.getElementById('rlAgent').textContent = last.primary.agent;
      if (document.getElementById('rlTime')) document.getElementById('rlTime').textContent = last.timestamp;
      if (document.getElementById('rlConf')) document.getElementById('rlConf').textContent = ((last.primary.confidence || 0) * 100).toFixed(1) + '%';
    }
    if (document.getElementById('rlMode')) {
      document.getElementById('rlMode').textContent = d.reinforcement?.mode || 'real_pytorch_inference';
    }
    if (document.getElementById('rlAgentsDetail') && last?.agents) {
      document.getElementById('rlAgentsDetail').innerHTML = last.agents.map(a =>
        `<div class="ops-row"><span>${a.agent}</span><strong>${a.action} (${(a.confidence * 100).toFixed(0)}%)</strong></div>`
      ).join('');
    }
    setHb('rlHb', agents.length ? 'on' : 'off');

    const fed = d.federated || {};
    if (document.getElementById('flRlLoaded')) {
      document.getElementById('flRlLoaded').innerHTML = fed.is_rl_fl
        ? '<span class="ops-good">YES — RL->FL</span>'
        : (fed.global_model_loaded
          ? '<span class="ops-warn">loaded (legacy DL-FL)</span>'
          : '<span class="ops-warn">NO — run run_rl_federated.bat</span>');
    }
    if (document.getElementById('flRlPipeline')) {
      document.getElementById('flRlPipeline').textContent = fed.pipeline || 'RL -> FL';
    }
    if (document.getElementById('flRlLearn')) {
      document.getElementById('flRlLearn').textContent = fed.architecture || '—';
    }
    if (document.getElementById('flRlReplay')) {
      document.getElementById('flRlReplay').textContent = '—';
    }
    const bli = fed.last_inference;
    if (bli && document.getElementById('flRlLast')) {
      document.getElementById('flRlLast').textContent =
        `${bli.prediction} (${((bli.confidence || 0) * 100).toFixed(0)}%)`;
    }
    return agents;
  }

  function formatLlmText(t) {
    return String(t || '')
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/^[-•]\s+(.+)$/gm, '<li>$1</li>')
      .replace(/(<li>.*<\/li>\n?)+/g, m => '<ul>' + m + '</ul>')
      .replace(/\n{2,}/g, '</p><p>')
      .replace(/\n/g, '<br>');
  }

  async function askLlm(body, targetId, methodId) {
    const el = document.getElementById(targetId || 'llmText');
    const mid = document.getElementById(methodId || 'llmMethod');
    if (el) el.innerHTML = '<span style="color:var(--muted)">Generating explanation…</span>';
    try {
      const r = await fetch('/explain_llm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body || {}),
      });
      const j = await r.json();
      if (!r.ok || j.error) {
        if (el) el.innerHTML = '<span style="color:var(--danger,#b33)">' + (j.error || 'Explanation failed') + '</span>';
        return null;
      }
      if (mid) mid.textContent = '';
      if (el) el.innerHTML = '<p>' + formatLlmText(j.explanation) + '</p>';
      return j;
    } catch (e) {
      if (el) el.innerHTML = '<span style="color:var(--muted)">Unavailable: ' + e.message + '</span>';
      return null;
    }
  }

  function getRole() {
    return localStorage.getItem('prism_role');
  }

  function setRole(role) {
    if (role !== 'admin' && role !== 'user') role = 'admin';
    localStorage.setItem('prism_role', role);
    applyRole(role);
    const modal = document.getElementById('roleModalOverlay');
    if (modal) modal.classList.add('hidden');
    toast('Switched to ' + (role === 'admin' ? 'Administrator' : 'Standard User') + ' mode');
  }

  function applyRole(role) {
    const r = role || getRole() || 'admin';
    document.body.setAttribute('data-role', r);
    document.documentElement.setAttribute('data-role', r);
    const chip = document.getElementById('roleChip');
    if (chip) chip.textContent = 'Role: ' + (r === 'admin' ? 'Admin 👑' : 'User 👤');
    
    const path = window.location.pathname;
    if (r === 'user' && (path === '/fl' || path === '/rl' || path === '/about' || path === '/poster')) {
      window.location.href = '/predict-ui';
    }
  }

  function promptRole() {
    const modal = document.getElementById('roleModalOverlay');
    if (modal) modal.classList.remove('hidden');
  }

  // boot theme and role immediately
  applyTheme(loadSettings().theme);
  
  document.addEventListener('DOMContentLoaded', () => {
    const r = getRole();
    if (!r) {
      promptRole();
    } else {
      applyRole(r);
    }
  });

  window.Aegis = {
    loadSettings, saveSettings, applyTheme, toast, unlockAudio, playAlertSound,
    getHistory, pushHistory, clearHistory, THREAT, threatClass, setHb,
    fetchLive, fetchInfo, renderFlPanel, renderRlPanel, renderProcCard, renderVerifyPanel,
    formatLlmText, askLlm, flProcessSummary,
    getRole, setRole, applyRole, promptRole,
  };
  window.Prism = window.Aegis;
})();
