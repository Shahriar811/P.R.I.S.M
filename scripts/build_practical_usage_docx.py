# -*- coding: utf-8 -*-
r"""
build_practical_usage_docx.py
=============================
Builds a comprehensive, publication-quality Word document (.docx) for:
"Chapter 7: Practical Usage & Dashboard Application"

Target destination:
C:\Users\mdsha\Downloads\Programs\Practical_Usage_and_Dashboard_Application.docx
"""

import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

TARGET_DIR = r"C:\Users\mdsha\Downloads\Programs"
os.makedirs(TARGET_DIR, exist_ok=True)

# Color Palette (Executive Navy Theme)
COLOR_PRIMARY = RGBColor(31, 78, 121)    # #1F4E79 (Deep Navy)
COLOR_SECONDARY = RGBColor(46, 117, 182) # #2E75B6 (Steel Blue)
COLOR_TEXT = RGBColor(51, 51, 51)        # #333333 (Charcoal)
COLOR_MUTED = RGBColor(102, 102, 102)    # #666666 (Muted Gray)

HEX_PRIMARY = "1F4E79"
HEX_SECONDARY = "2E75B6"
HEX_LIGHT_BG = "F2F4F8"
HEX_BORDER = "CCCCCC"

def set_cell_background(cell, hex_color):
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=90, bottom=90, left=130, right=130):
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = parse_xml(f'''
        <w:tcMar {nsdecls("w")}>
            <w:top w:w="{top}" w:type="dxa"/>
            <w:bottom w:w="{bottom}" w:type="dxa"/>
            <w:left w:w="{left}" w:type="dxa"/>
            <w:right w:w="{right}" w:type="dxa"/>
        </w:tcMar>
    ''')
    tcPr.append(tcMar)

def format_paragraph(p, space_before=0, space_after=4.5, line_spacing=1.15):
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = line_spacing

def add_header_footer(doc):
    section = doc.sections[0]
    section.top_margin = Inches(0.85)
    section.bottom_margin = Inches(0.85)
    section.left_margin = Inches(0.85)
    section.right_margin = Inches(0.85)
    
    header = section.header
    hp = header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    hrun = hp.add_run("CSE499B Senior Design Project | Chapter 7: Practical Usage & Dashboard Application")
    hrun.font.name = 'Calibri'
    hrun.font.size = Pt(8.5)
    hrun.font.color.rgb = COLOR_MUTED
    
    footer = section.footer
    fp = footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    frun = fp.add_run("Department of Electrical and Computer Engineering — North South University")
    frun.font.name = 'Calibri'
    frun.font.size = Pt(8.5)
    frun.font.color.rgb = COLOR_MUTED

def add_title_block(doc):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_paragraph(p, space_before=4, space_after=2)
    r = p.add_run("CHAPTER 7: PRACTICAL USAGE & DASHBOARD APPLICATION")
    r.font.name = 'Calibri'
    r.font.size = Pt(18.5)
    r.font.bold = True
    r.font.color.rgb = COLOR_PRIMARY

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_paragraph(p_sub, space_before=0, space_after=12)
    r_sub = p_sub.add_run("End-to-End Operational Deployment, Multi-Page Security Dashboard, Real-Time Network Sniffing, Dual-Layer Explainable AI & RESTful Cloud Ingress\nSenior Design Capstone Project Report (CSE499B)")
    r_sub.font.name = 'Calibri'
    r_sub.font.size = Pt(10.5)
    r_sub.font.italic = True
    r_sub.font.color.rgb = COLOR_SECONDARY

def add_heading_1(doc, text):
    p = doc.add_paragraph()
    format_paragraph(p, space_before=13, space_after=5)
    r = p.add_run(text)
    r.font.name = 'Calibri'
    r.font.size = Pt(13.5)
    r.font.bold = True
    r.font.color.rgb = COLOR_PRIMARY
    return p

def add_heading_2(doc, text):
    p = doc.add_paragraph()
    format_paragraph(p, space_before=9, space_after=3.5)
    r = p.add_run(text)
    r.font.name = 'Calibri'
    r.font.size = Pt(11.5)
    r.font.bold = True
    r.font.color.rgb = COLOR_SECONDARY
    return p

def add_heading_3(doc, text):
    p = doc.add_paragraph()
    format_paragraph(p, space_before=7, space_after=2.5)
    r = p.add_run(text)
    r.font.name = 'Calibri'
    r.font.size = Pt(10.5)
    r.font.bold = True
    r.font.color.rgb = COLOR_TEXT
    return p

def add_body_p(doc, text, bold_prefix=None):
    p = doc.add_paragraph()
    format_paragraph(p, space_before=0, space_after=4.5, line_spacing=1.15)
    if bold_prefix:
        r0 = p.add_run(bold_prefix)
        r0.font.name = 'Calibri'
        r0.font.size = Pt(10)
        r0.font.bold = True
        r0.font.color.rgb = COLOR_TEXT
    r1 = p.add_run(text)
    r1.font.name = 'Calibri'
    r1.font.size = Pt(10)
    r1.font.color.rgb = COLOR_TEXT
    return p

def add_bullet(doc, text, bold_prefix=None):
    p = doc.add_paragraph(style='List Bullet')
    format_paragraph(p, space_before=0, space_after=2.5, line_spacing=1.15)
    if bold_prefix:
        r0 = p.add_run(bold_prefix)
        r0.font.name = 'Calibri'
        r0.font.size = Pt(10)
        r0.font.bold = True
        r0.font.color.rgb = COLOR_TEXT
    r1 = p.add_run(text)
    r1.font.name = 'Calibri'
    r1.font.size = Pt(10)
    r1.font.color.rgb = COLOR_TEXT
    return p

def add_callout_box(doc, text, title="KEY OPERATIONAL HIGHLIGHT"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.8)
    set_cell_background(cell, HEX_LIGHT_BG)
    set_cell_margins(cell, top=90, bottom=90, left=140, right=140)
    
    tcPr = cell._element.get_or_add_tcPr()
    tcBorders = parse_xml(f'''
        <w:tcBorders {nsdecls("w")}>
            <w:top w:val="none"/>
            <w:left w:val="single" w:sz="24" w:space="0" w:color="{HEX_PRIMARY}"/>
            <w:bottom w:val="none"/>
            <w:right w:val="none"/>
        </w:tcBorders>
    ''')
    tcPr.append(tcBorders)
    
    p = cell.paragraphs[0]
    format_paragraph(p, space_before=1.5, space_after=2.5)
    r_title = p.add_run(f"[{title}]\n")
    r_title.font.name = 'Calibri'
    r_title.font.size = Pt(9.5)
    r_title.font.bold = True
    r_title.font.color.rgb = COLOR_PRIMARY
    
    r_text = p.add_run(text)
    r_text.font.name = 'Calibri'
    r_text.font.size = Pt(9)
    r_text.font.italic = True
    r_text.font.color.rgb = COLOR_TEXT
    
    p_spacer = doc.add_paragraph()
    format_paragraph(p_spacer, space_before=0, space_after=3.5)

def add_styled_table(doc, headers, data, col_widths=None, alignment_list=None):
    tbl = doc.add_table(rows=len(data) + 1, cols=len(headers))
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    # Format Header
    hdr_cells = tbl.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        set_cell_background(hdr_cells[i], HEX_PRIMARY)
        set_cell_margins(hdr_cells[i], top=70, bottom=70, left=90, right=90)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        format_paragraph(p, space_before=1.5, space_after=1.5)
        for r in p.runs:
            r.font.name = 'Calibri'
            r.font.size = Pt(9)
            r.font.bold = True
            r.font.color.rgb = RGBColor(255, 255, 255)
            
    # Format Data Rows
    for row_idx, row_data in enumerate(data):
        row_cells = tbl.rows[row_idx + 1].cells
        bg_hex = HEX_LIGHT_BG if row_idx % 2 == 1 else "FFFFFF"
        for col_idx, cell_value in enumerate(row_data):
            row_cells[col_idx].text = str(cell_value)
            set_cell_background(row_cells[col_idx], bg_hex)
            set_cell_margins(row_cells[col_idx], top=55, bottom=55, left=90, right=90)
            p = row_cells[col_idx].paragraphs[0]
            format_paragraph(p, space_before=1.5, space_after=1.5)
            
            if alignment_list and col_idx < len(alignment_list):
                p.alignment = alignment_list[col_idx]
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT if col_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
                
            for r in p.runs:
                r.font.name = 'Calibri'
                r.font.size = Pt(8.5)
                r.font.color.rgb = COLOR_TEXT
                
    if col_widths:
        for row in tbl.rows:
            for idx, w in enumerate(col_widths):
                row.cells[idx].width = Inches(w)
                
    p_spacer = doc.add_paragraph()
    format_paragraph(p_spacer, space_before=0, space_after=5)

def generate_practical_usage_document():
    doc = docx.Document()
    add_header_footer(doc)
    add_title_block(doc)
    
    # ---------------------------------------------------------
    # 7.1 Historical Evolution: CSE499A vs CSE499B Dashboard
    # ---------------------------------------------------------
    add_heading_1(doc, "7.1 Evolution of the Operational Security Dashboard (499A to 499B)")
    add_body_p(doc, "In the CSE499A research phase, intrusion detection evaluation was primarily delivered through static Jupyter notebook scripts, paired with a minimal FastAPI prototype providing only two basic functions: simple CSV uploading and raw tabular text output. While sufficient for offline baseline verification, the 499A implementation lacked interactive telemetry controls, real-time alert mechanisms, explainable AI explanations, and production-grade federated monitoring.")
    
    add_body_p(doc, "The CSE499B project extension completely reimagines the operational interface into an enterprise-ready, multi-page Security Operations Center (SOC) dashboard. Built with Flask, Jinja2 templating, dynamic CSS glassmorphism, client-side asynchronous JavaScript, and RESTful API backends, the new dashboard transforms complex machine learning, federated averaging, and reinforcement learning pipelines into an intuitive, real-time command center for cybersecurity administrators.")

    # Table 7.1: Evolution Comparison
    headers_t71 = ["Operational Dimension", "CSE499A Foundation Baseline", "CSE499B Production Dashboard System", "Operational Benefit"]
    data_t71 = [
        ["Web Framework", "FastAPI (Basic 2-route backend)", "Flask + Jinja2 + Asynchronous REST API", "Full multi-page responsive web dashboard"],
        ["User Interface Pages", "Single static HTML page", "7 Dedicated Functional View Pages", "Specialized views for Predict, FL, RL, History, Settings"],
        ["Inference Serving", "Single-model raw prediction", "Tri-Model Real-Time Consensus Cards", "Side-by-side XGBoost, Local RL, and Global FL output"],
        ["Explainable AI (XAI)", "None (Black-box predictions)", "Dual-Layer XAI (SHAP + Groq Llama 3.3 70B)", "Quantitative feature attribution + Plain-English threat text"],
        ["FL Monitoring", "None (Silent in-memory loop)", "Live Flower Server & Client Monitor (/fl)", "Real-time process status, round counters, accuracy curves"],
        ["Continual Learning", "Static offline models", "Interactive Teach/Correction Panel (/rl)", "Analyst feedback updates replay buffers & triggers FL"],
        ["Traffic Ingestion", "Manual CSV file upload only", "Live Packet Sniffer (Scapy) + CSV Batch", "Continuous live TCP/UDP packet flow inspection"],
        ["Remote Deployment", "Localhost binding only", "ngrok Secure HTTPS Tunneling (TLS v3)", "Instant public demonstration URL with encrypted ingress"]
    ]
    align_t71 = [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT]
    add_styled_table(doc, headers_t71, data_t71, [1.3, 1.6, 1.8, 2.1], align_t71)

    add_callout_box(doc, "CSE499B transitions the user interface from a rudimentary offline demonstration into an enterprise-grade SOC platform, enabling real-time tri-model inference, plain-English LLM threat explainability, live Scapy packet sniffing, and continuous human-in-the-loop federated model synchronization.", "OPERATIONAL PARADIGM SHIFT")

    # ---------------------------------------------------------
    # 7.2 End-to-End System Deployment Architecture
    # ---------------------------------------------------------
    add_heading_1(doc, "7.2 End-to-End System Deployment Architecture")
    add_body_p(doc, "The operational deployment architecture is structured across four decoupled, high-performance layers ensuring scalability, fault tolerance, and zero operational downtime:")
    
    add_bullet(doc, "Flask handles interactive server-side template rendering, dynamic UI routes, and client sessions on Port 7860. A parallel FastAPI service operates on Port 8000 to deliver high-throughput asynchronous RESTful JSON serialization.", "1. Dual-Server Web Application Tier (Port 7860 / 8000): ")
    add_bullet(doc, "Operates via federated/rl_server.py on TCP Port 8080. Accepts gRPC connections from multi-process edge clients (Client 0, 1, 2), coordinates FedAvg weight averaging, updates global_model.pt, and logs live round telemetry to status.json.", "2. Decentralized Federated Aggregator (gRPC Port 8080): ")
    add_bullet(doc, "Operates via live_sniffer.py, utilizing Scapy to capture live raw network packets from physical/virtual adapters, computes 42 flow statistics across 5-second tumbling time windows, scales features via scaler.joblib, and posts flows directly to the inference engine.", "3. Real-Time Network Packet Sniffer Engine: ")
    add_bullet(doc, "Automated via expose_ngrok.ps1, creating a secure TLS v3 encrypted ingress tunnel (e.g., https://xxxx.ngrok-free.app) forwarding remote requests directly to the local dashboard, allowing external demonstration across isolated enterprise subnets.", "4. Secure Public Cloud Ingress Tunnel (ngrok): ")

    # ---------------------------------------------------------
    # 7.3 Multi-Page Dashboard UI/UX Design & Module Breakdown
    # ---------------------------------------------------------
    add_heading_1(doc, "7.3 Multi-Page Dashboard UI/UX Design & Functional Modules")
    add_body_p(doc, "The frontend application is engineered with a modern, glassmorphic UI aesthetic featuring cohesive typography (Calibri/Inter), dynamic theme toggling (Light / Dark mode), responsive grid layouts, and asynchronous AJAX polling. The interface is organized into seven specialized navigation modules:")

    add_heading_2(doc, "7.3.1 Home Overview Page (Route: /)")
    add_body_p(doc, "Serves as the executive health dashboard for security analysts. It displays active system status indicators, loaded neural network model inventories (XGBoost, Dueling DQN, FL-ResNet-MLP), total evaluated network requests, server hardware metrics (CPU/RAM/GPU VRAM), and high-level attack distribution summaries.")

    add_heading_2(doc, "7.3.2 Predict UI Page (Route: /predict-ui)")
    add_body_p(doc, "The primary real-time intrusion inspection workstation. It provides two flexible input modes: (1) manual 42-feature comma-separated text insertion, and (2) an automated 'Sample Flow' button that draws class-balanced ground-truth flow records directly from test.npz. Key functional components include:")
    add_bullet(doc, "Displays independent prediction boxes for Supervised XGBoost, Local Dueling DQN, and Federated Global Model, showing predicted class names, class probabilities, and confidence percentages.", "Tri-Model Consensus Cards: ")
    add_bullet(doc, "Renders horizontal diverging bar charts of the top 8 numerical features driving the classification decision (e.g., Flow Duration, Fwd Pkt Len Max), color-coded by positive (+) and negative (-) attribution forces.", "Interactive SHAP Feature Chart: ")
    add_bullet(doc, "An interactive pop-up modal that queries the Groq Llama 3.3 70B API to generate a 3-bullet plain-English threat summary, severity assessment, and actionable mitigation guide.", "Groq LLM Threat Explanation: ")

    add_heading_2(doc, "7.3.3 Federated FL Control & Monitoring Page (Route: /fl)")
    add_body_p(doc, "Provides real-time visibility into the Flower federated learning infrastructure. It reads federated/status.json and client status heartbeats every 3 seconds, rendering active communication round progress bars, quorum indicators (Client 0, 1, 2 status: connecting -> local_training -> fit_done), global test loss/accuracy curves, and a manual 'Trigger FL Round' button for on-demand aggregation.")

    add_heading_2(doc, "7.3.4 Reinforcement Learning & Continual Teach Page (Route: /rl)")
    add_body_p(doc, "Visualizes the inner workings of the autonomous RL agents. It plots Q-value distributions across all 5 action classes (Benign, DoS, DDoS, Bot, Prob), primary policy confidence, and experience replay buffer capacity (0 to 10,000 transitions). Crucially, it features the 'Continual Teach / Correction' panel: when an analyst identifies a misclassified flow, they can select the correct ground-truth label and submit feedback, which instantly updates FeedbackStore across all client nodes.")

    add_heading_2(doc, "7.3.5 History & Incident Audit Page (Route: /history)")
    add_body_p(doc, "Maintains a persistent client-side localStorage audit trail of all inspected flow records. Analysts can filter past detections by attack category, search by timestamp, review model agreement/disagreement flags, and export formatted CSV reports for forensic incident documentation.")

    add_heading_2(doc, "7.3.6 System Settings Page (Route: /settings)")
    add_body_p(doc, "Allows system administrators to calibrate operational thresholds in real time without restarting backend services: adjusting Novelty Confidence Cutoffs (default: tau = 0.70), toggling Automatic FL Round Triggers on model disagreement, configuring Replay Buffer batch sizes, and securely updating Groq API authentication keys.")

    add_heading_2(doc, "7.3.7 CSV Batch Upload Page (Route: /predict_csv)")
    add_body_p(doc, "Designed for bulk forensic analysis. Analysts can drag-and-drop large network flow capture CSV files (containing thousands of records). The engine processes rows in parallel chunks, generating aggregated multi-class attack breakdowns, high-risk flow alerts, and downloadable annotated prediction summaries.")

    # ---------------------------------------------------------
    # 7.4 Tri-Model Consensus & Real-Time Alert Engine
    # ---------------------------------------------------------
    add_heading_1(doc, "7.4 Tri-Model Consensus & Real-Time Alert Monitoring Engine")
    add_body_p(doc, "Rather than relying on a single fallible classifier, CSE499B deploys a multi-model consensus architecture that cross-references three distinct analytical paradigms for every network flow:")

    # Table 7.2: Tri-Model Architecture
    headers_t72 = ["Model Layer", "Active Model / Architecture", "Analytical Role in Pipeline", "Operational Decision Logic"]
    data_t72 = [
        ["Layer 1: Supervised", "XGBoost Classifier", "High-accuracy static baseline (84.42% accuracy)", "Provides primary classification and class probability vector"],
        ["Layer 2: Autonomous RL", "Dueling DQN Agent", "Adaptive continual learner (Cost-sensitive policy)", "Evaluates Q-value advantage; flags concept drift / novel attack paths"],
        ["Layer 3: Federated FL", "Global FL-ResNet-MLP / RL", "Decentralized consensus (Privacy-preserved)", "Ensures cross-organization threat validation without raw data sharing"]
    ]
    align_t72 = [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT]
    add_styled_table(doc, headers_t72, data_t72, [1.4, 1.6, 1.8, 2.0], align_t72)

    add_heading_2(doc, "Novelty & Model Disagreement Detection Algorithm:")
    add_body_p(doc, "The system continuously evaluates incoming flows against two anomaly criteria: (1) Low Confidence Flag: prediction confidence < 0.70, and (2) Model Disagreement Flag: Supervised XGBoost, Local RL, or Global FL assign conflicting attack classes (e.g., XGBoost predicts DoS while Local RL flags Bot). When triggered, the flow is marked as a 'Novel Threat Candidate', generating immediate high-priority visual alerts on the dashboard and routing the sample to the analyst review queue.")

    # ---------------------------------------------------------
    # 7.5 Dual-Layer Explainable AI (XAI) & Generative Threat Synthesis
    # ---------------------------------------------------------
    add_heading_1(doc, "7.5 Dual-Layer Explainable AI (XAI) & Generative Threat Synthesis")
    add_body_p(doc, "To eliminate the traditional 'black-box' opacity of machine learning in cybersecurity, CSE499B integrates a dual-layer XAI framework combining quantitative feature attribution with generative natural language synthesis:")

    add_bullet(doc, "For tree models, exact Shapley values are calculated via shap.TreeExplainer. For neural networks and fallback models, normalized feature deviation impact vectors phi_i = I_i * (x_i - mu_i) are computed, extracting the top 8 positive and negative network features influencing the decision.", "Layer 1 — SHAP Quantitative Attribution: ")
    add_bullet(doc, "Operating via groq_explain.py, feature impact vectors, model predictions, confidence scores, and agreement states are compiled into a structured prompt sent to Groq Cloud running Llama 3.3 70B Versatile (llama-3.3-70b-versatile). The LLM returns a structured 3-bullet briefing explaining: (1) Threat Nature, (2) Critical Network Signals, and (3) Recommended Mitigation.", "Layer 2 — Generative LLM Threat Briefing: ")
    add_bullet(doc, "To guarantee 100% operational availability during network isolation or when Groq API keys are unconfigured, the system incorporates the Aegis XAI Rule-Based Engine. It parses feature attribution vectors deterministically, delivering structured natural-language reports with zero external API dependencies.", "Resilient Offline Fallback Engine: ")

    # Table 7.3: XAI Modes
    headers_t73 = ["XAI Mode", "Backend Function", "Context Supplied to Engine", "Operational Security Benefit"]
    data_t73 = [
        ["Predict Mode", "explain_prediction()", "Class, confidence, top 8 SHAP features, RL/FL votes", "Explains why an individual packet flow was flagged as malicious"],
        ["Status Mode", "explain_status()", "System health, active models, hardware utilization", "Generates plain-English executive briefings on SOC readiness"],
        ["FL Mode", "explain_fl()", "Round count, loss trajectory, client quorum states", "Explains federated convergence progress to non-technical admins"],
        ["RL Mode", "explain_rl()", "Q-values, action choices, replay buffer fill state", "Interprets autonomous agent exploration and learning milestones"],
        ["Batch Mode", "explain_batch()", "Total flows, attack class distribution, risk counts", "Summarizes large CSV network captures for forensic incident logs"]
    ]
    align_t73 = [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT]
    add_styled_table(doc, headers_t73, data_t73, [1.1, 1.5, 2.0, 2.2], align_t73)

    # ---------------------------------------------------------
    # 7.6 Complete REST API Specification
    # ---------------------------------------------------------
    add_heading_1(doc, "7.6 Complete REST API Specification")
    add_body_p(doc, "The backend exposes a comprehensive RESTful API enabling programmatic integration with external Security Information and Event Management (SIEM) systems, SDN controllers, and firewall automation scripts:")

    # Table 7.4: REST API Endpoints
    headers_t74 = ["HTTP Route", "Method", "Request Payload", "Response Structure & System Function"]
    data_t74 = [
        ["/info", "GET", "None", "Returns system health, active models, feature list, and hardware details"],
        ["/live", "GET", "None", "Returns live status.json, active FL rounds, RL heartbeats, and model modes"],
        ["/predict", "POST", "JSON: {features: [42 floats]}", "Multi-model inference (XGBoost, RL, FL), confidence, and novelty flags"],
        ["/explain", "POST", "JSON: {features: [42 floats]}", "Computes Layer 1 SHAP feature contributions and base expectation values"],
        ["/explain_llm", "POST", "JSON: {mode: 'predict', ...}", "Queries Groq Llama 3.3 70B for Layer 2 natural language threat explanation"],
        ["/predict_csv", "POST", "Form-Data: CSV file upload", "Batch flow analysis, multi-class distribution counts, risk breakdown"],
        ["/learn/status", "GET", "None", "Returns continual learning metrics, feedback count, replay buffer size"],
        ["/learn/teach", "POST", "JSON: {features, correct_label}", "Captures human correction, stores feedback, and updates replay buffer"],
        ["/simulate_client_attack", "POST", "JSON: {features: [42 floats]}", "Simulates attack on edge node, updates local RL, triggers Flower FL sync"]
    ]
    align_t74 = [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT]
    add_styled_table(doc, headers_t74, data_t74, [1.4, 0.8, 1.9, 2.7], align_t74)

    # ---------------------------------------------------------
    # 7.7 Technology Stack & Implementation Specifications
    # ---------------------------------------------------------
    add_heading_1(doc, "7.7 Technology Stack & Production Specifications")
    add_body_p(doc, "The production system integrates modern, industry-standard frameworks across machine learning, federated networking, backend web serving, and explainability:")

    # Table 7.5: Tech Stack
    headers_t75 = ["Technology / Library", "Version", "Architectural Role & Functional Responsibility"]
    data_t75 = [
        ["Python", "3.10 / 3.11 / 3.13", "Core programming runtime for model training, inference, and web serving"],
        ["PyTorch", ">= 2.0.0", "Deep learning engine for ResNet-MLP, Dueling DQN, PPO, and QR-DQN networks"],
        ["Flower (flwr)", ">= 1.5.0", "Multi-process Federated Learning orchestration and gRPC parameter exchange"],
        ["Flask & Jinja2", ">= 2.3.0 / >= 3.1.0", "Multi-page web dashboard server, dynamic HTML templating, and REST routes"],
        ["FastAPI & Uvicorn", ">= 0.110.0 / >= 0.27.0", "High-throughput asynchronous JSON REST API inference engine"],
        ["Groq Cloud SDK", "Latest", "Ultra-fast LPU inference serving Llama 3.3 70B Versatile for XAI briefings"],
        ["Llama 3.3 70B", "Versatile (Meta)", "Generative foundation model formatting plain-English threat summaries"],
        ["SHAP", ">= 0.42.0", "Game-theoretic Shapley feature attribution for tabular flow explainability"],
        ["Scapy", ">= 2.5.0", "Low-level packet capturing and 42-feature flow reconstruction in live_sniffer.py"],
        ["Scikit-Learn & joblib", ">= 1.2.0 / >= 1.3.0", "Data preprocessing, MinMaxScaler scaling, and model artifact serialization"],
        ["ngrok", "v3 (TLS Encryption)", "Secure HTTPS ingress tunneling for encrypted public access and remote demos"],
        ["CSS Glassmorphism", "Custom Vanilla CSS", "Modern responsive UI design system with dark/light mode toggle"]
    ]
    align_t75 = [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT]
    add_styled_table(doc, headers_t75, data_t75, [1.8, 1.4, 3.6], align_t75)

    # ---------------------------------------------------------
    # 7.8 Summary & Operational Impact
    # ---------------------------------------------------------
    add_heading_1(doc, "7.8 Summary & Practical Operational Impact")
    add_bullet(doc, "Eliminates static model decay by enabling live packet sniffing, tri-model consensus validation, and seamless human-in-the-loop federated re-synchronization across distributed edge gateways.", "1. Real-Time Adaptive Security: ")
    add_bullet(doc, "Bridges the gap between raw statistical machine learning vectors and non-technical security decision-makers through quantitative SHAP charts and generative Groq LLM threat summaries.", "2. Actionable Threat Explainability: ")
    add_bullet(doc, "Combines lightweight local edge footprints (< 2 MB memory per model, sub-millisecond inference) with encrypted ngrok cloud tunneling, providing a complete blueprint for next-generation Cyber-Physical System defense.", "3. Enterprise-Ready Deployment: ")

    # Save logic with graceful fallback
    candidates = [
        os.path.join(TARGET_DIR, "Practical_Usage_and_Dashboard_Application.docx"),
        os.path.join(TARGET_DIR, "Chapter_7_Practical_Usage_and_Dashboard_Application.docx")
    ]
    saved_paths = []
    for p in candidates:
        try:
            doc.save(p)
            saved_paths.append(p)
            print(f"Document successfully created and saved to: {p}")
        except PermissionError:
            print(f"Notice: {p} is currently open in another program.")
        except Exception as e:
            print(f"Warning: Could not save to {p}: {e}")

    if not saved_paths:
        fallback = os.path.join(TARGET_DIR, "Chapter_7_Practical_Usage_v2.docx")
        doc.save(fallback)
        print(f"Document saved to fallback path: {fallback}")

if __name__ == "__main__":
    generate_practical_usage_document()
