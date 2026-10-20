"""
build_499b_report.py
===================
Synthesizes the complete Word document (.docx) for the user's assigned sections:
- Chapter 4: Methodology
- Chapter 7: Practical Usage & Dashboard Application

Output path: E:\CSE499B\Latest One(FL+RL)\CSE499B report\CSE499B_Group2_Report_Part_Methodology_Dashboard.docx
"""

import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls

OUTPUT_DIR = r"E:\CSE499B\Latest One(FL+RL)\CSE499B report"
DOCX_PATH = os.path.join(OUTPUT_DIR, "CSE499B_Group2_Report_Part_Methodology_Dashboard.docx")

# Styling Colors
COLOR_PRIMARY = RGBColor(31, 78, 121)    # #1F4E79 (Navy)
COLOR_SECONDARY = RGBColor(46, 117, 182) # #2E75B6 (Blue)
COLOR_TEXT = RGBColor(51, 51, 51)        # #333333 (Dark Gray)
COLOR_MUTED = RGBColor(102, 102, 102)    # #666666 (Gray)

HEX_PRIMARY = "1F4E79"
HEX_SECONDARY = "2E75B6"
HEX_LIGHT_BG = "F2F4F7"
HEX_BORDER = "CCCCCC"

def set_cell_background(cell, hex_color):
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
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

def add_header_footer(doc):
    section = doc.sections[0]
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    
    header = section.header
    hp = header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    hrun = hp.add_run("CSE 499 Senior Project Report | Group 2 | Chapter 4 & Chapter 7")
    hrun.font.name = 'Calibri'
    hrun.font.size = Pt(8.5)
    hrun.font.color.rgb = COLOR_MUTED
    
    footer = section.footer
    fp = footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    frun = fp.add_run("North South University — Department of Electrical and Computer Engineering")
    frun.font.name = 'Calibri'
    frun.font.size = Pt(9)
    frun.font.color.rgb = COLOR_MUTED

def format_paragraph(p, space_before=0, space_after=6, line_spacing=1.15):
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = line_spacing

def add_title_block(doc):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_paragraph(p, space_before=12, space_after=6)
    run = p.add_run("NORTH SOUTH UNIVERSITY")
    run.font.name = 'Calibri'
    run.font.size = Pt(18)
    run.font.bold = True
    run.font.color.rgb = COLOR_PRIMARY

    p2 = doc.add_paragraph()
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_paragraph(p2, space_before=0, space_after=18)
    run2 = p2.add_run("Department of Electrical and Computer Engineering\nSenior Design Capstone Project Report (CSE 499)")
    run2.font.name = 'Calibri'
    run2.font.size = Pt(13)
    run2.font.color.rgb = COLOR_SECONDARY

    p3 = doc.add_paragraph()
    p3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_paragraph(p3, space_before=6, space_after=18)
    run3 = p3.add_run("Privacy-Preserving Federated and Reinforcement Learning-Based Intrusion Detection System for IoT Devices")
    run3.font.name = 'Calibri'
    run3.font.size = Pt(16)
    run3.font.bold = True
    run3.font.color.rgb = COLOR_PRIMARY

    p4 = doc.add_paragraph()
    p4.alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_paragraph(p4, space_before=6, space_after=24)
    run4 = p4.add_run("Specialized Technical Report: Chapter 4 (Methodology) & Chapter 7 (Practical Usage & Dashboard Application)\nAuthor / Contributor: Md. Shahriar Hossain (ID: 2132058642)\nTeammates: Afsana Akter Tania, Soomaiya Chowdhury, Rayta Binte Noor\nSupervisor: Dr. Mohammad Rashedur Rahman")
    run4.font.name = 'Calibri'
    run4.font.size = Pt(10.5)
    run4.font.italic = True
    run4.font.color.rgb = COLOR_MUTED

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

def add_heading_1(doc, text):
    p = doc.add_paragraph()
    format_paragraph(p, space_before=18, space_after=8)
    run = p.add_run(text)
    run.font.name = 'Calibri'
    run.font.size = Pt(16)
    run.font.bold = True
    run.font.color.rgb = COLOR_PRIMARY
    return p

def add_heading_2(doc, text):
    p = doc.add_paragraph()
    format_paragraph(p, space_before=14, space_after=6)
    run = p.add_run(text)
    run.font.name = 'Calibri'
    run.font.size = Pt(13.5)
    run.font.bold = True
    run.font.color.rgb = COLOR_SECONDARY
    return p

def add_heading_3(doc, text):
    p = doc.add_paragraph()
    format_paragraph(p, space_before=10, space_after=4)
    run = p.add_run(text)
    run.font.name = 'Calibri'
    run.font.size = Pt(11.5)
    run.font.bold = True
    run.font.color.rgb = COLOR_TEXT
    return p

def add_body_p(doc, text, bold_prefix=None):
    p = doc.add_paragraph()
    format_paragraph(p, space_before=0, space_after=6, line_spacing=1.15)
    if bold_prefix:
        r0 = p.add_run(bold_prefix)
        r0.font.name = 'Calibri'
        r0.font.size = Pt(11)
        r0.font.bold = True
        r0.font.color.rgb = COLOR_TEXT
    r1 = p.add_run(text)
    r1.font.name = 'Calibri'
    r1.font.size = Pt(11)
    r1.font.color.rgb = COLOR_TEXT
    return p

def add_callout_box(doc, text, title="KEY ARCHITECTURAL HIGHLIGHT"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, HEX_LIGHT_BG)
    set_cell_margins(cell, top=120, bottom=120, left=180, right=180)
    
    # Left border styling XML
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
    format_paragraph(p, space_before=2, space_after=4)
    r_title = p.add_run(f"[{title}]\n")
    r_title.font.name = 'Calibri'
    r_title.font.size = Pt(10.5)
    r_title.font.bold = True
    r_title.font.color.rgb = COLOR_PRIMARY
    
    r_text = p.add_run(text)
    r_text.font.name = 'Calibri'
    r_text.font.size = Pt(10)
    r_text.font.italic = True
    r_text.font.color.rgb = COLOR_TEXT
    
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

def add_image_figure(doc, image_name, caption_text):
    image_path = os.path.join(OUTPUT_DIR, image_name)
    if os.path.exists(image_path):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        format_paragraph(p_img, space_before=10, space_after=4)
        run = p_img.add_run()
        run.add_picture(image_path, width=Inches(6.2))
        
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        format_paragraph(p_cap, space_before=2, space_after=12)
        r_cap = p_cap.add_run(caption_text)
        r_cap.font.name = 'Calibri'
        r_cap.font.size = Pt(9.5)
        r_cap.font.bold = True
        r_cap.font.color.rgb = COLOR_PRIMARY
    else:
        print(f"Warning: Image {image_path} not found.")

def add_styled_table(doc, headers, data, col_widths=None):
    tbl = doc.add_table(rows=len(data) + 1, cols=len(headers))
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    # Format Header Row
    hdr_cells = tbl.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        set_cell_background(hdr_cells[i], HEX_PRIMARY)
        set_cell_margins(hdr_cells[i], top=100, bottom=100, left=120, right=120)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        format_paragraph(p, space_before=2, space_after=2)
        for r in p.runs:
            r.font.name = 'Calibri'
            r.font.size = Pt(10)
            r.font.bold = True
            r.font.color.rgb = RGBColor(255, 255, 255)
            
    # Format Data Rows
    for row_idx, row_data in enumerate(data):
        row_cells = tbl.rows[row_idx + 1].cells
        bg_hex = HEX_LIGHT_BG if row_idx % 2 == 1 else "FFFFFF"
        for col_idx, cell_value in enumerate(row_data):
            row_cells[col_idx].text = str(cell_value)
            set_cell_background(row_cells[col_idx], bg_hex)
            set_cell_margins(row_cells[col_idx], top=80, bottom=80, left=120, right=120)
            p = row_cells[col_idx].paragraphs[0]
            format_paragraph(p, space_before=2, space_after=2)
            if col_idx == 0:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                r.font.name = 'Calibri'
                r.font.size = Pt(9.5)
                r.font.color.rgb = COLOR_TEXT
                
    # Column widths
    if col_widths:
        for row in tbl.rows:
            for idx, w in enumerate(col_widths):
                row.cells[idx].width = Inches(w)
                
    doc.add_paragraph().paragraph_format.space_after = Pt(8)

def build_document():
    print("Building Word document...")
    doc = docx.Document()
    add_header_footer(doc)
    add_title_block(doc)
    
    # ---------------------------------------------------------
    # Executive Summary & Historical Bridge
    # ---------------------------------------------------------
    add_heading_1(doc, "Executive Summary & Project Transition (499A to 499B)")
    add_body_p(doc, "In the CSE499A research phase, Group 2 established the data foundation and baseline evaluation for intrusion detection in Software-Defined Networking and Internet-of-Things (SDN-IoT) environments. Using the benchmark ASEADOS-SDN-IoT dataset (comprising 457,044 flow records, 42 statistical flow features, and 5 traffic classes: Benign, DoS, DDoS, Bot, Prob), CSE499A delivered offline machine learning baselines led by XGBoost (84.03% accuracy, 86.78% F1-score) and Random Forest (83.98% accuracy), alongside a initial simulation of Federated Learning using Gated Recurrent Units (FL-GRU at 80.35% accuracy over 5 synthetic clients).")
    
    add_body_p(doc, "While CSE499A successfully proved that intrusion detection can function under simulated federated data splits, it retained four critical operational limitations: (1) static detection logic vulnerable to concept drift, (2) reliance on simulated client partitioning rather than a production client-server network framework, (3) lack of continuous reinforcement learning adaptation for dynamic attack mitigation, and (4) absence of explainable AI (XAI) for real-time security analyst interpretability.")
    
    add_body_p(doc, "This report documents the CSE 499 project extension, specifically detailing the author's primary contributions across Chapter 4 (Methodology) and Chapter 7 (Practical Usage & Dashboard Application). CSE 499 transitions the framework from an offline simulation into a production-grade, multi-process client-server Federated Learning (FL) architecture, an integrated Dueling Deep Q-Network (DQN) Reinforcement Learning (RL) continual learning pipeline, a dual-layer Explainable AI (SHAP + Groq Llama 3.3 70B LLM) engine, and an interactive multi-page web security dashboard with live packet sniffing and secure HTTPS remote access via ngrok.")

    add_callout_box(doc, "CSE499A achieved offline detection accuracy of 84.03% (XGBoost) and 80.35% (simulated FL-GRU). CSE 499 elevates this baseline into a production-ready, client-server federated RL system with real-time continuous learning, SHAP feature impact attribution, Groq LLM natural language threat explanations, live packet sniffing, and ngrok HTTPS remote deployment.", "HISTORICAL BRIDGE: CSE499A VS CSE 499")

    # ---------------------------------------------------------
    # CHAPTER 4: METHODOLOGY
    # ---------------------------------------------------------
    add_heading_1(doc, "Chapter 4: Methodology")
    
    add_heading_2(doc, "4.1 Production Federated Learning Architecture (Main Client-Server System)")
    add_body_p(doc, "Unlike simulation-based federated learning setups that partition static data arrays in memory within a single Python script, the CSE 499 production architecture implements a decentralized, multi-process client-server system built upon the Flower (flwr) framework. The architecture decouples the central aggregator from independent edge clients, enabling federated model updates over real network interfaces via gRPC (Google Remote Procedure Call) protocols.")
    
    add_body_p(doc, "The production FL system consists of two primary operational entities:", "System Component Breakdown: ")
    add_body_p(doc, "Operating via federated/rl_server.py on TCP port 8080. The server initializes global neural network weights, coordinates communication rounds, enforces minimum client quorum requirements (min_fit_clients = 3), collects client parameter updates, executes weighted Federated Averaging (FedAvg), and evaluates global test metrics against a held-out evaluation dataset (test.npz).", "1. Central FL Aggregator Server: ")
    add_body_p(doc, "Operating via federated/rl_client.py across independent process instances (Client 0: SDN Gateway, Client 1: IoT Cluster, Client 2: Cyber-Physical Subnet). Each client maintains its own local dataset partition, executes local model optimization over discrete epochs without transmitting raw feature vectors, extracts state dictionaries, and serializes neural network weights back to the server over gRPC.", "2. Distributed Edge Client Nodes: ")

    add_image_figure(doc, "fl_main_architecture.png", "Figure 4.1: Production Federated Learning Architecture showing Central Aggregator Server and Edge Client Nodes over gRPC.")

    add_body_p(doc, "Privacy preservation is mathematically guaranteed by design. Raw flow records ($X \\in \\mathbb{R}^{N \\times 42}$) never leave the edge device boundaries. Instead, edge clients compute local loss gradients and update neural network weight matrices ($w_k$). Only these mathematical parameter matrices are serialized and sent over encrypted gRPC channels to the central aggregator, completely shielding private SDN-IoT network telemetry from central server data breach risks.")

    # Table 4.1
    headers_t41 = ["Component", "Specification / Parameter", "Implementation Detail / Purpose"]
    data_t41 = [
        ["Aggregation Framework", "Flower (flwr) Framework", "Production multi-process FL orchestration"],
        ["Transport Protocol", "gRPC over TCP Port 8080", "High-throughput binary parameter serialization"],
        ["Client Quorum", "Minimum 3 Active Clients", "Ensures statistical diversity before FedAvg"],
        ["Local Optimization", "1-2 Local Epochs per Round", "Reduces communication frequency while ensuring convergence"],
        ["Privacy Model", "Zero-Raw-Data Transmission", "Strict data locality; weight updates shared exclusively"],
        ["Global Storage", "global_model.pt & SAVED_FL", "Persists aggregated global neural weights across rounds"]
    ]
    add_styled_table(doc, headers_t41, data_t41, [1.8, 2.2, 2.5])

    add_heading_2(doc, "4.2 The RL + FL Integrated Pipeline & Working Methodology")
    add_body_p(doc, "A central innovation of CSE 499 is the integration of Reinforcement Learning (RL) with Federated Learning (FL) into a unified, self-improving security pipeline. While supervised federated learning models are rigid once trained, reinforcement learning allows defensive agents to continuously adapt to changing network behaviors and novel attack vectors.")

    add_body_p(doc, "The RL framework employs a Dueling Deep Q-Network (Dueling DQN) architecture. In standard DQN, the neural network directly estimates the Q-value $Q(s, a)$ for each action $a$ given state $s$. Dueling DQN decouples this calculation into two separate streams:", "Mathematical Formulation of Dueling DQN: ")
    add_body_p(doc, "1. Value Stream $V(s): \\mathbb{R}^{42} \\rightarrow \\mathbb{R}^1$, which estimates the scalar value of being in current network state $s$.\n2. Advantage Stream $A(s, a): \\mathbb{R}^{42} \\rightarrow \\mathbb{R}^5$, which evaluates the relative advantage of choosing each specific attack class action $a \\in \\{\\text{Benign, DoS, DDoS, Bot, Prob}\\}$.")
    add_body_p(doc, "The streams are combined using the mean-centered aggregation formula to ensure unidentifiability stability:")
    add_body_p(doc, "Q(s, a; \\theta, \\alpha, \\beta) = V(s; \\theta, \\beta) + \\left( A(s, a; \\theta, \\alpha) - \\frac{1}{|A|} \\sum_{a'} A(s, a'; \\theta, \\alpha) \\right)", "Equation 4.1 (Dueling Q-Value): ")

    add_image_figure(doc, "rl_fl_integrated_pipeline.png", "Figure 4.2: Integrated RL + FL Pipeline showing continuous feedback loop, novelty detection, and automatic Flower FL triggers.")

    add_body_p(doc, "The operational workflow proceeds through three synchronized layers:", "Pipeline Working Methodology: ")
    add_body_p(doc, "Incoming 42-dimensional flow vectors are evaluated simultaneously by the baseline XGBoost classifier, local Dueling DQN agents, and the aggregated Global FL model.", "1. Real-Time Inference & Novelty Evaluation: ")
    add_body_p(doc, "If prediction confidence falls below the novelty threshold (confidence < 0.70) or if model disagreement occurs (e.g., Local RL predicts DDoS while Global FL predicts Bot), the sample is flagged as a novel attack candidate by the RLFLSyncManager.", "2. Novelty & Disagreement Flagging: ")
    add_body_p(doc, "Security analysts can correct misclassifications via the dashboard. The feedback is stored in FeedbackStore (client_id=0, 1, 2) and added to the experience replay buffer. If auto_save_new or auto_federate is enabled, the system updates local model weights and automatically triggers a new Flower FL round across distributed edge clients.", "3. Continual Learning & Automatic FL Trigger: ")

    add_image_figure(doc, "fl_aggregation_protocol_sequence.png", "Figure 4.3: Sequence Diagram illustrating FedAvg round execution, weight upload, and global distribution over gRPC.")

    add_body_p(doc, "During each federated round, the server executes the Federated Averaging (FedAvg) algorithm across client parameter weights $w_{t+1}^k$ weighted by local sample proportions $n_k / n$:")
    add_body_p(doc, "w_{t+1} = \\sum_{k=1}^{K} \\frac{n_k}{n} w_{t+1}^k", "Equation 4.2 (FedAvg Parameter Weight Averaging): ")

    # Table 4.2
    headers_t42 = ["Parameter Name", "Configured Value", "Technical Rationale in Pipeline"]
    data_t42 = [
        ["State Dimension", "42 Features", "Matches preprocessed ASEADOS SDN-IoT flow attributes"],
        ["Action Dimension", "5 Classes", "Benign (0), DoS (1), DDoS (2), Bot (3), Prob (4)"],
        ["Learning Rate (alpha)", "0.001 (Adam)", "Ensures stable gradient updates without policy collapse"],
        ["Discount Factor (gamma)", "0.99", "Prioritizes long-term threat mitigation rewards"],
        ["Replay Buffer Capacity", "10,000 Transitions", "Provides rich experience diversity for continual learning"],
        ["Novelty Threshold", "0.70 Confidence", "Triggers analyst review & FL sync on low-confidence flows"],
        ["FedAvg Rounds", "5-10 Communication Rounds", "Balances global convergence with network bandwidth"]
    ]
    add_styled_table(doc, headers_t42, data_t42, [1.8, 2.0, 2.7])

    add_heading_2(doc, "4.3 Dual-Layer Explainable AI (XAI) & LLM Framework")
    add_body_p(doc, "Machine learning and deep learning models in intrusion detection are frequently criticized as 'black boxes', making security analysts hesitant to trust automated alerts. CSE 499 resolves this by introducing a Dual-Layer Explainable AI (XAI) framework that combines quantitative feature attribution with generative natural language explanations.")

    add_image_figure(doc, "xai_llm_flow.png", "Figure 4.4: Dual-Layer Explainable AI Pipeline incorporating SHAP feature attribution and Groq Llama 3.3 70B LLM explanations.")

    add_body_p(doc, "The two layers of the XAI engine operate as follows:", "XAI Dual-Layer Architecture: ")
    add_body_p(doc, "For tree-based models (XGBoost, Random Forest), the system utilizes shap.TreeExplainer to calculate exact Shapley values. For neural networks and fallback models, feature impact vectors $\\phi_i$ are computed by scaling model feature importances against normalized feature deviations from baseline means:", "Layer 1: SHAP Quantitative Feature Attribution: ")
    add_body_p(doc, "\\phi_i = I_i \\times \\left( x_i - \\mu_i \\right)", "Equation 4.3 (Feature Impact Vector): ")
    add_body_p(doc, "This identifies the top 8 numerical features driving the prediction (e.g., Fwd Pkt Len Max pushing toward DDoS, or Flow IAT Mean pushing toward Benign).")

    add_body_p(doc, "Operating via groq_explain.py, the top SHAP features, model predictions, confidence scores, and local/global model agreement states are formatted into a structured prompt and transmitted to the Groq API running Llama 3.3 70B Versatile (llama-3.3-70b-versatile). The LLM synthesizes these raw inputs into plain-English security summaries, explaining the threat context, highlighting key network signals, and recommending defensive actions for network administrators.", "Layer 2: Generative LLM Threat Explanation: ")

    add_body_p(doc, "To guarantee operational resilience when internet connectivity is restricted or when the Groq API key is unconfigured, the system incorporates the Aegis XAI Built-in Fallback Engine. This engine deterministically parses feature impact vectors and generates structured, natural-language threat reports without external network calls, ensuring 100% XAI availability.", "Resilient Fallback Engine: ")

    # Table 4.3
    headers_t43 = ["XAI Layer", "Technology Stack", "Primary Output Format / User Benefit"]
    data_t43 = [
        ["Layer 1 (Quantitative)", "SHAP (TreeExplainer / Dev)", "Top 8 numerical impact values (+/-) & raw feature inputs"],
        ["Layer 2 (Generative)", "Groq API (Llama 3.3 70B)", "3-bullet plain-English analyst summary & recommended actions"],
        ["Offline Fallback", "Aegis Rule-Based XAI Engine", "Guaranteed offline textual explanations with zero API latency"],
        ["Multi-Mode Support", "groq_explain.py (8 Modes)", "Supports Predict, Status, FL, RL, History, Class, Batch, Learn"]
    ]
    add_styled_table(doc, headers_t43, data_t43, [1.8, 2.2, 2.5])

    # ---------------------------------------------------------
    # CHAPTER 7: PRACTICAL USAGE & DASHBOARD APPLICATION
    # ---------------------------------------------------------
    add_heading_1(doc, "Chapter 7: Practical Usage & Dashboard Application")

    add_heading_2(doc, "7.1 FL Aggregation Protocol & Privacy Infrastructure")
    add_body_p(doc, "The deployment of federated learning in practical security operations requires a robust aggregation protocol that balances privacy protection with model convergence speed. CSE 499 implements a 5-step aggregation protocol sequence over gRPC channels:")

    add_body_p(doc, "1. Round Initialization: The central server (rl_server.py) broadcasts current global Dueling DQN neural network parameters $w_t$ to all connected edge client nodes.\n2. Local Model Updates: Each edge client loads $w_t$, attaches its private local dataset, and executes 2 epochs of Adam optimization. Raw network flow logs remain sealed inside local storage.\n3. Weight Parameter Transmission: Edge clients extract updated state dictionaries $w_{t+1}^k$ and transmit parameter matrices along with sample count $n_k$ back to the server over gRPC on port 8080.\n4. Weighted Aggregation: The server computes the element-wise weighted average of received weight tensors using the FedAvg equation.\n5. Global State Serialization: The aggregated global model is evaluated against held-out validation data (test.npz) and saved to global_model.pt and Saved model/FL/federated_rl_fedavg.pt.")

    add_image_figure(doc, "privacy_accuracy_tradeoff.png", "Figure 7.1: Privacy Preservation vs Detection Accuracy Curve across increasing distributed client nodes.")

    add_body_p(doc, "As illustrated in Figure 7.1, increasing the number of distributed edge clients from 1 to 20 elevates data privacy isolation from 0% to 95.0%, while global detection accuracy experiences only a minor trade-off drop from 80.36% to 78.90%. At 5 clients, the system achieves an optimal balance: 80.0% data privacy isolation with virtually zero accuracy loss (80.35% vs 80.36% centralized baseline).")

    # Table 7.1
    headers_t71 = ["Security Threat / Vulnerability", "Federated Privacy Safeguard", "Operational Impact & Defense Guarantee"]
    data_t71 = [
        ["Centralized Data Breach", "Zero Raw Data Transmission", "Network flows never leave edge gateways; breach of server reveals no traffic logs"],
        ["Membership Inference Attacks", "FedAvg Weight Averaging", "Individual client data points are obscured within global weight parameter means"],
        ["Data Poisoning Attacks", "Quorum & Validation Checks", "Updates are validated against held-out test set before committing to global_model.pt"],
        ["Eavesdropping / Wiretapping", "gRPC over SSL/TLS Transport", "All parameter uplink and downlink transfers are encrypted over port 8080"]
    ]
    add_styled_table(doc, headers_t71, data_t71, [2.0, 2.0, 2.5])

    add_heading_2(doc, "7.2 UI/UX Design of Security Dashboard, Real-Time Alert Monitoring & Deployment")
    add_body_p(doc, "To transform complex machine learning and federated learning pipelines into an intuitive operational tool for network administrators, CSE 499 provides a responsive, multi-page web application dashboard built using Flask, Jinja2 templates, modern CSS glassmorphism, and client-side JavaScript.")

    add_image_figure(doc, "dashboard_deployment_architecture.png", "Figure 7.2: End-to-End System Deployment Architecture showing Flask UI, FastAPI backend, Live Packet Sniffer, and ngrok HTTPS Tunnel.")

    add_body_p(doc, "The user interface is organized into seven functional view pages accessible via top navigation:", "Dashboard UI/UX Component Modules: ")
    add_body_p(doc, "Displays real-time system health, active model status (XGBoost, Local RL, Global FL), total processed requests, and quick pipeline navigation metrics.", "1. Home Overview Page (/): ")
    add_body_p(doc, "Provides single-sample flow evaluation. Users can manually paste 42 comma-separated feature values or click 'Sample Flow' to draw class-balanced real test vectors from test.npz. Displays side-by-side predictions from Supervised XGBoost, Local RL, and Global FL, complete with interactive SHAP feature contribution charts and Groq LLM explanation pop-ups.", "2. Predict UI Page (/predict-ui): ")
    add_body_p(doc, "Monitors Flower federated learning processes, active client node states (Client 0, 1, 2), current round metrics, global test accuracy history, and manual trigger controls for FedAvg cycles.", "3. Federated FL Control Page (/fl): ")
    add_body_p(doc, "Displays Q-value distributions, primary action confidence, replay buffer utilization, and an interactive 'Continual Teach / Correction' panel allowing analysts to correct predictions in real time.", "4. Reinforcement RL Page (/rl): ")
    add_body_p(doc, "Maintains a persistent client-side localStorage record of all evaluated flows, confidence levels, timestamps, and model agreement flags.", "5. History Page (/history): ")
    add_body_p(doc, "Configures novelty thresholds, auto-federation triggers, auto-save settings, and Groq API key parameters.", "6. Settings Page (/settings): ")
    add_body_p(doc, "Allows bulk uploading of network traffic CSV files, generating aggregated multi-class attack breakdowns and risk summaries.", "7. CSV Batch Upload (/predict_csv): ")

    add_heading_3(doc, "7.2.1 Real-Time Network Alert Sniffer (live_sniffer.py)")
    add_body_p(doc, "For live operational networks, CSE 499 includes live_sniffer.py, a real-time network traffic sniffer powered by Scapy. The sniffer captures live TCP/UDP packets from local network interfaces, aggregates packet statistics into 42 flow-level numerical features (Flow Duration, Packet Lengths, Inter-Arrival Times, Header Lengths, Flags), applies scaler.joblib normalization, and submits live vectors to the inference engine. Instant alerts are triggered whenever attack traffic (DoS, DDoS, Bot, Prob) is detected.")

    add_heading_3(doc, "7.2.2 Global Deployment & Remote Access (FastAPI + ngrok)")
    add_body_p(doc, "To enable external deployment and demonstration across remote networks without exposing internal infrastructure, CSE 499 integrates a dual-server architecture: Flask handles web UI rendering on port 7860, while FastAPI serves RESTful inference endpoints on port 8000. Public secure remote access is established using expose_ngrok.ps1, which creates an encrypted HTTPS tunnel (e.g., https://xxxx.ngrok-free.app) forwarding traffic directly to localhost:8000.")

    add_image_figure(doc, "centralized_vs_fl_performance.png", "Figure 7.3: Performance Comparison Bar Chart between 499A Centralized Baselines and 499B Federated & RL Models.")

    add_image_figure(doc, "rl_agent_performance_heatmap.png", "Figure 7.4: Reinforcement Learning Agents Performance Heatmap across 5 Traffic Classes.")

    # Table 7.2
    headers_t72 = ["REST API Endpoint", "HTTP Method", "Payload / Parameters", "Response Structure & System Function"]
    data_t72 = [
        ["/info", "GET", "None", "Returns system status, active models, feature columns, device info"],
        ["/predict", "POST", "JSON: {features: [42 float vals]}", "Multi-model inference (XGBoost, Local RL, Global FL) & novelty flag"],
        ["/explain", "POST", "JSON: {features: [42 float vals]}", "Layer 1 SHAP feature contributions & base expectation values"],
        ["/explain_llm", "POST", "JSON: {mode: 'predict', ...}", "Layer 2 Groq Llama 3.3 70B natural language threat explanation"],
        ["/simulate_client_attack", "POST", "JSON: {features: [42 float vals]}", "Simulates attack on edge client, updates local RL, triggers FL sync"],
        ["/predict_csv", "POST", "Form-Data: CSV File Upload", "Batch flow analysis, multi-class counts, risk distribution summary"]
    ]
    add_styled_table(doc, headers_t72, data_t72, [1.5, 1.0, 1.8, 2.2])

    # ---------------------------------------------------------
    # CONCLUSION & FUTURE WORK
    # ---------------------------------------------------------
    add_heading_1(doc, "Conclusion & Future Work")
    add_body_p(doc, "The CSE 499 project extension successfully elevates the intrusion detection framework from an offline simulation into a production-grade, privacy-preserved, self-improving security application. By combining Flower-based multi-process Federated Learning over gRPC with Dueling DQN Reinforcement Learning continual learning, dual-layer SHAP + Groq LLM Explainable AI, live packet sniffing, and ngrok HTTPS remote deployment, the system addresses all key research gaps identified in CSE499A.")

    add_body_p(doc, "Future work will focus on expanding the edge client population to 50+ heterogeneous IoT nodes, integrating differential privacy noise injection within FedAvg rounds, and deploying custom 1D-CNN + Bi-GRU + Self-Attention deep learning architectures for enhanced feature representation.")

    # Save document
    doc.save(DOCX_PATH)
    print(f"Document successfully created and saved to: {DOCX_PATH}")

if __name__ == "__main__":
    build_document()
