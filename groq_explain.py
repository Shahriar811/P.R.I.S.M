"""
Groq LLM helper for plain-language IDS explanations (GPT-OSS 120B).
Modes: predict, status, fl, rl, history, class, batch, learn
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_dotenv():
    path = os.path.join(HERE, '.env')
    if not os.path.exists(path):
        return
    try:
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                k, v = line.split('=', 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass


_load_dotenv()

GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '').strip()
GROQ_MODEL = os.environ.get('GROQ_MODEL', 'openai/gpt-oss-120b').strip()
GROQ_DISPLAY_MODEL = os.environ.get('GROQ_DISPLAY_MODEL', 'GPT-OSS 120B').strip()

SYSTEM = (
    "You are an experienced cybersecurity AI analyst for the P.R.I.S.M threat detection platform. "
    "STRICT SCOPE: You MUST ONLY answer questions and provide explanations directly related to cybersecurity, cyber threats, network telemetry anomalies, attack vectors (DDoS, DoS, Botnets, Probing, Malware), and model predictions. "
    "If asked anything outside of cybersecurity and network defense, politely and firmly refuse to answer. "
    "Provide clear, conversational, and human-like explanations written in 1 to 2 small, well-structured paragraphs without markdown headers or bullet lists. Speak naturally and authoritatively like a human security expert."
)


def available() -> bool:
    return True


def _fallback_explanation(user_prompt: str) -> str:
    lines = [line.strip() for line in user_prompt.split('\n') if line.strip()]
    pred = 'Flow'
    conf = 'n/a'
    model = 'XGBoost'
    feats = []
    
    for l in lines:
        if l.startswith(('- Predicted class:', '- Predicted classification:')):
            pred = l.split(':', 1)[1].strip()
        elif l.startswith(('- Confidence:', '- Model confidence:')):
            conf = l.split(':', 1)[1].strip()
        elif l.startswith(('- Model:', '- Primary model:')):
            model = l.split(':', 1)[1].strip()
        elif l.startswith('- ') and 'impact=' in l:
            feats.append(l[2:])
            
    if 'Explain this' in user_prompt:
        if pred == 'Benign':
            return (
                f"The {model} classifier labeled this flow as Benign with {conf} confidence. "
                "The feature attributions show how the input influenced that classification. "
                "This model output does not prove that the traffic is harmless; review it alongside surrounding network telemetry."
            )
        else:
            feat_summary = f"Feature signals including {', '.join([f.split(':')[0] for f in feats[:2]])} contributed to this classification." if feats else "No feature attributions were supplied for this explanation."
            return (
                f"The {model} classifier labeled this flow as {pred} with {conf} confidence. "
                f"{feat_summary} Review the prediction and compare the local RL and global federated model outputs before taking action."
            )

    return (
        "P.R.I.S.M is using its built-in explanation mode. "
        "Check the model readiness indicators and FL/RL monitor pages for the actual operational state. "
        "This fallback does not independently verify model accuracy, active training, or network safety."
    )


def _chat(user_prompt: str, max_tokens: int = 450) -> Dict[str, Any]:
    api_key = os.environ.get('GROQ_API_KEY', '').strip()
    model = os.environ.get('GROQ_MODEL', 'openai/gpt-oss-120b').strip()
    display_model = os.environ.get('GROQ_DISPLAY_MODEL', 'GPT-OSS 120B').strip()

    if api_key:
        try:
            from groq import Groq
            client = Groq(api_key=api_key)

            kwargs = {
                'model': model,
                'messages': [
                    {'role': 'system', 'content': SYSTEM},
                    {'role': 'user', 'content': user_prompt},
                ],
                'temperature': 1,
                'max_completion_tokens': max(max_tokens, 2048),
                'top_p': 1,
                'reasoning_effort': 'medium',
                'stream': False,
                'stop': None,
            }

            completion = client.chat.completions.create(**kwargs)
            text = completion.choices[0].message.content if completion.choices else ''
            if text:
                return {
                    'ok': True,
                    'model': display_model,
                    'explanation': text.strip(),
                    'provider': 'Groq',
                }
        except Exception:
            pass

    # Built-in Engine Fallback
    return {
        'ok': True,
        'model': 'Built-in Engine',
        'explanation': _fallback_explanation(user_prompt),
        'provider': 'P.R.I.S.M XAI',
    }


def explain_prediction(
    prediction: str,
    confidence: float,
    model_name: str,
    contributions: List[Dict[str, Any]],
    rl_action: Optional[str] = None,
    fl_action: Optional[str] = None,
    class_proba: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    top_lines = []
    for c in (contributions or [])[:6]:
        impact = float(c.get('impact', 0))
        direction = 'increases likelihood' if impact > 0 else 'decreases likelihood'
        top_lines.append(
            f"- {c.get('feature')}: impact={impact:+.3f} ({direction}), measured value={c.get('input')}"
        )
    proba_txt = ''
    if class_proba:
        proba_txt = ', '.join(f'{k}={v * 100:.1f}%' for k, v in class_proba.items())

    prompt = f"""Explain this threat detection result to a network operator in 1 to 2 short, human-like paragraphs without headers or bullet points.

- Predicted classification: {prediction}
- Model confidence: {confidence * 100:.1f}%
- Primary model: {model_name}
- Class probability breakdown: {proba_txt or 'n/a'}
- Local RL Agent decision: {rl_action or 'n/a'}
- Federated FL (Global) decision: {fl_action or 'n/a'}

Top contributing SHAP feature signals:
{chr(10).join(top_lines) if top_lines else '(none provided)'}

Write a natural, flowing explanation: (1) what the traffic classification means in human terms, (2) how the top feature metrics influenced the decision, and (3) whether the local RL and federated global models agree with the result."""
    return _chat(prompt)


def explain_status(live: Dict[str, Any], info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    fed = live.get('federated') or {}
    rl = live.get('reinforcement') or {}
    verify = live.get('verify') or {}
    g = fed.get('global_model') or {}
    prompt = f"""Write a friendly 1-2 paragraph briefing on the current operational status of the P.R.I.S.M security system for an operator, without headers or bullets.

System state:
- Primary supervised model: {(info or {}).get('best_model', 'XGBoost')}
- Hardware device: {(info or {}).get('device', 'n/a')}
- Local RL agents active: {rl.get('agent_count', 0)}/5 -> {rl.get('agents_loaded')}
- Global Federated Model loaded: {fed.get('global_model_loaded')} ({g.get('architecture') or fed.get('architecture')})
- RL-to-FL pipeline active: {fed.get('is_rl_fl')}
- Current FedAvg Round: {g.get('round')}
- Active Flower client processes: {fed.get('processes_alive')}/{fed.get('processes_expected', 4)}
- Local RL and Federated connection verified: {verify.get('connection_ok')}

Explain concisely what components are online and ready to protect incoming IoT traffic."""
    return _chat(prompt, max_tokens=350)


def explain_fl(live: Dict[str, Any]) -> Dict[str, Any]:
    fed = live.get('federated') or {}
    g = fed.get('global_model') or {}
    cs = fed.get('clients_summary') or {}
    last = fed.get('last_inference') or {}
    prompt = f"""Explain how the Federated Learning pipeline works in this system in 1-2 conversational paragraphs without bullet points or headers.

Facts:
- Architecture: {g.get('architecture') or fed.get('architecture')}
- Global model status: {fed.get('global_model_loaded')}
- Round: {g.get('round')}
- Global test accuracy: {g.get('test_acc')}
- Edge clients online: {cs.get('alive')}/{cs.get('expected', 3)}
- Privacy guarantee: Raw flow packets remain entirely local on edge devices; only parameter weights are aggregated via FedAvg over gRPC.

Explain clearly why edge privacy is preserved and how global knowledge updates without data sharing."""
    return _chat(prompt, max_tokens=400)


def explain_rl(live: Dict[str, Any], learn: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    rl = live.get('reinforcement') or {}
    fed = live.get('federated') or {}
    last = rl.get('last_inference') or {}
    primary = last.get('primary') or {}
    learn = learn or {}
    prompt = f"""Explain the Reinforcement Learning and Continual Learning features in 1-2 natural paragraphs without headers or bullet points.

Facts:
- Active local RL agents: {rl.get('agent_count', 0)}/5 -> {rl.get('agents_loaded')}
- Primary agent action: {primary.get('action')} ({primary.get('agent')}) with {((primary.get('confidence') or 0) * 100):.1f}% confidence
- Continual learning state: {learn.get('learning')}
- Total replay updates: {learn.get('n_updates')}
- Recent feedback accuracy: {learn.get('recent_accuracy')}

Explain how the Dueling DQN adapts to live traffic and how operator feedback continually refines edge detection."""
    return _chat(prompt, max_tokens=400)


def explain_history_item(item: Dict[str, Any]) -> Dict[str, Any]:
    prompt = f"""Explain this historical detection entry in 1 short, natural paragraph without headers or bullet points.

- Recorded classification: {item.get('pred')}
- Confidence score: {item.get('conf')}%
- Supervised model: {item.get('model')}
- Local RL verdict: {item.get('rl') or 'n/a'}
- Federated FL verdict: {item.get('fl') or 'n/a'}
- Timestamp: {item.get('time')}

Provide a concise, human security takeaway on what happened and whether the models agreed."""
    return _chat(prompt, max_tokens=250)


def explain_attack_class(label: str, desc: Optional[str] = None) -> Dict[str, Any]:
    prompt = f"""Explain the cyber attack class '{label}' to a student or network operator in 1 short, engaging paragraph without headers or lists.
Context hint: {desc or 'n/a'}

Explain what this attack looks like in practice on IoT networks, why it is dangerous, and how standard systems detect it."""
    return _chat(prompt, max_tokens=250)


def explain_batch(counts: List[Dict[str, Any]], total: int, model: str = 'XGBoost') -> Dict[str, Any]:
    lines = ', '.join(f"{c.get('label')}: {c.get('n')}" for c in (counts or []))
    prompt = f"""Summarize this batch CSV traffic analysis in 1-2 conversational paragraphs without bullet points or headers.

- Detection engine: {model}
- Total network records analyzed: {total}
- Breakdown of classes: {lines}

Give an executive summary of the overall risk level, highlight the dominant threat patterns, and provide actionable next steps."""
    return _chat(prompt, max_tokens=300)


def explain_learn_event(
    true_label: str,
    predicted: Optional[str] = None,
    updated: bool = False,
    learn_status: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    s = learn_status or {}
    prompt = f"""Explain a continual learning correction event in 1 short, human paragraph without headers or lists.

- Correct ground truth label provided by operator: {true_label}
- Original model prediction: {predicted or 'n/a'}
- Neural weights updated on edge: {updated}
- Total replay buffer updates: {s.get('n_updates')}

Explain how this real-time human feedback immediately helps the local RL agent adapt without waiting for offline retraining."""
    return _chat(prompt, max_tokens=250)


ADVISOR_SYSTEM = (
    "You are P.R.I.S.M AI Security Advisor, a specialized cybersecurity and incident response expert.\n\n"
    "STRICT SCOPE AND DOMAIN GUARDRAILS:\n"
    "1. You MUST ONLY answer questions strictly related to cybersecurity, cyber attacks (such as DDoS, DoS, Botnets, Malware, Intrusion, Reconnaissance/Probing), "
    "threat prevention, firewall/SDN security flow rules, network hardening, and incident response playbooks for operators and organizations (what to do if attacked/compromised).\n"
    "2. You MUST STRICTLY REFUSE any questions or requests that are irrelevant or outside of cybersecurity, network defense, attack mitigation, or incident response "
    "(such as cooking, non-security general programming, creative writing, entertainment, weather, sports, trivia, personal advice, politics, or casual chat).\n"
    "3. When an irrelevant or non-cybersecurity query is received, respond ONLY with:\n"
    "\"I am strictly designated as a cybersecurity and incident response advisor for the P.R.I.S.M platform. "
    "I can only assist with cybersecurity queries, cyber attack analysis, threat prevention, mitigation strategies, and incident response procedures.\"\n"
    "4. When answering valid cybersecurity questions, provide actionable, practical, and structured advice covering: "
    "(1) Immediate Containment Actions for the Operator (isolation, triage, verification), "
    "(2) Organizational & SOC Policies (firewall/SDN rules, ISP scrubbing, escalation, enterprise guidelines), and "
    "(3) Long-Term Hardening & Prevention Tips. Stay authoritative, concise, and professional."
)


def chat_advisor(
    messages: List[Dict[str, str]],
    context: Optional[Dict[str, Any]] = None,
    max_tokens: int = 750,
) -> Dict[str, Any]:
    api_key = os.environ.get('GROQ_API_KEY', '').strip()
    model = os.environ.get('GROQ_MODEL', 'openai/gpt-oss-120b').strip()
    display_model = os.environ.get('GROQ_DISPLAY_MODEL', 'GPT-OSS 120B').strip()

    sys_prompt = ADVISOR_SYSTEM
    if context:
        threat = context.get('prediction', 'Unknown')
        conf = context.get('confidence')
        conf_str = f" ({conf*100:.1f}%)" if conf is not None else ""
        sys_prompt += f"\n\nCurrent Detection Context:\n- Detected Threat: {threat}{conf_str}\n- Primary Detector: {context.get('model', 'XGBoost')}"
        if context.get('rl_action'):
            sys_prompt += f"\n- Local RL Agent Decision: {context.get('rl_action')}"
        if context.get('fl_action'):
            sys_prompt += f"\n- Federated Global FL Decision: {context.get('fl_action')}"

    conv = [{'role': 'system', 'content': sys_prompt}]
    for m in (messages or []):
        role = m.get('role', 'user')
        if role not in ('user', 'assistant', 'system'):
            role = 'user'
        content = str(m.get('content', '')).strip()
        if content:
            conv.append({'role': role, 'content': content})

    if api_key:
        try:
            from groq import Groq
            client = Groq(api_key=api_key)
            kwargs = {
                'model': model,
                'messages': conv,
                'temperature': 1,
                'max_completion_tokens': max(max_tokens, 2048),
                'top_p': 1,
                'reasoning_effort': 'medium',
                'stream': False,
                'stop': None,
            }
            completion = client.chat.completions.create(**kwargs)
            text = completion.choices[0].message.content if completion.choices else ''
            if text:
                return {
                    'ok': True,
                    'model': display_model,
                    'provider': 'Groq',
                    'response': text.strip(),
                }
        except Exception as e:
            print(f"[Advisor Chat Error] {e}")

    # Fallback response
    threat = (context or {}).get('prediction', 'Cyber Threat')
    fallback_text = (
        f"### Security Advisory Playbook: Dealing with {threat}\n\n"
        "**1. Immediate Actions for the Operator:**\n"
        f"- **Isolate Endpoint:** Disconnect or quarantine the affected IoT subnet/device to prevent lateral movement.\n"
        "- **Verify Telemetry:** Cross-check flow packet rates and packet lengths to confirm whether traffic is anomalous.\n"
        "- **Credential Reset:** Immediately change default credentials and rotate SSH/API keys on edge nodes.\n\n"
        "**2. Organizational & SOC Guidelines:**\n"
        "- **Firewall & ACLs:** Implement rate-limiting and drop ingress traffic from suspicious source IP ranges.\n"
        "- **Incident Escalation:** Log the alert in your SIEM/ticket queue and notify the security operations team.\n"
        "- **Upstream Scrubbing:** For volumetric DDoS/DoS, coordinate with upstream ISP/carrier mitigation services.\n\n"
        "**3. Long-Term Hardening:**\n"
        "- Enforce strict network segmentation (VLANs), automated firmware patching, and continuous federated model synchronization."
    )
    return {
        'ok': True,
        'model': 'Built-in Advisor',
        'provider': 'P.R.I.S.M Advisor',
        'response': fallback_text,
    }

