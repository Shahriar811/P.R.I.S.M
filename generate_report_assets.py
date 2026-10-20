"""
generate_report_assets.py
=========================
Renders 8 high-resolution (300 DPI) visual assets for the CSE499B Project Report:
1. fl_main_architecture.png
2. rl_fl_integrated_pipeline.png
3. xai_llm_flow.png
4. fl_aggregation_protocol_sequence.png
5. dashboard_deployment_architecture.png
6. centralized_vs_fl_performance.png
7. rl_agent_performance_heatmap.png
8. privacy_accuracy_tradeoff.png

Saves all images to: E:\CSE499B\Latest One(FL+RL)\CSE499B report\
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns

OUTPUT_DIR = r"E:\CSE499B\Latest One(FL+RL)\CSE499B report"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Common styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8

def draw_box(ax, x, y, w, h, text, bg_color="#1F4E79", text_color="white", fontsize=10, fontweight="bold", boxstyle="round,pad=0.5", linestyle="-", edgecolor=None):
    if edgecolor is None:
        edgecolor = bg_color
    p = patches.FancyBboxPatch((x, y), w, h, boxstyle=boxstyle, facecolor=bg_color, edgecolor=edgecolor, linewidth=1.5, linestyle=linestyle)
    ax.add_patch(p)
    ax.text(x + w/2., y + h/2., text, color=text_color, fontsize=fontsize, fontweight=fontweight, ha='center', va='center', multialignment='center', wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, label="", text_color="#333333", arrow_color="#1F4E79", linestyle="-"):
    ax.annotate(
        label,
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=arrow_color, lw=2, linestyle=linestyle),
        fontsize=8, fontweight="bold", color=text_color, ha='center', va='center',
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8) if label else None
    )

# ---------------------------------------------------------
# Figure 1: Production FL Client-Server Architecture
# ---------------------------------------------------------
def create_fl_main_architecture():
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Title
    ax.text(50, 95, "Production Federated Learning Architecture (Client-Server over gRPC)", fontsize=14, fontweight='bold', color='#1F4E79', ha='center')
    
    # Central Server Box
    draw_box(ax, 30, 68, 40, 20, "FL Global Aggregator Server\n(Flower Framework / rl_server.py)\n• FedAvg Weight Aggregator\n• Global Model State (global_model.pt)\n• gRPC Listener on Port 8080", bg_color="#1F4E79", fontsize=10)
    
    # Client Boxes
    draw_box(ax, 5, 20, 26, 32, "Edge Client Node 0\n(SDN Gateway A)\n\n• Local Dataset (42 features)\n• Local Dueling DQN Net\n• Local Epoch Training\n• Weight Extractor", bg_color="#2E75B6", fontsize=9)
    draw_box(ax, 37, 20, 26, 32, "Edge Client Node 1\n(IoT Cluster B)\n\n• Local Dataset (42 features)\n• Local Dueling DQN Net\n• Local Epoch Training\n• Weight Extractor", bg_color="#2E75B6", fontsize=9)
    draw_box(ax, 69, 20, 26, 32, "Edge Client Node 2\n(CPS Subnet C)\n\n• Local Dataset (42 features)\n• Local Dueling DQN Net\n• Local Epoch Training\n• Weight Extractor", bg_color="#2E75B6", fontsize=9)
    
    # Arrows (Bi-directional parameter exchange)
    draw_arrow(ax, 50, 68, 18, 52, label="gRPC (Weights Uplink / Downlink)", arrow_color="#5B9BD5")
    draw_arrow(ax, 50, 68, 50, 52, label="gRPC (Weights Uplink / Downlink)", arrow_color="#5B9BD5")
    draw_arrow(ax, 50, 68, 82, 52, label="gRPC (Weights Uplink / Downlink)", arrow_color="#5B9BD5")
    
    # Privacy Shield Box at Bottom
    draw_box(ax, 10, 3, 80, 10, "Privacy Preservation Layer: Raw packet flows remain local on edge clients.\nOnly neural network weights (Dueling DQN parameters) are transmitted across gRPC.", bg_color="#E7E6E6", text_color="#1F4E79", fontsize=9, linestyle="--", edgecolor="#1F4E79")
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fl_main_architecture.png")
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {path}")

# ---------------------------------------------------------
# Figure 2: Integrated RL + FL Pipeline
# ---------------------------------------------------------
def create_rl_fl_integrated_pipeline():
    fig, ax = plt.subplots(figsize=(13, 7.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    ax.text(50, 95, "Integrated RL + FL Pipeline & Continual Learning Orchestration", fontsize=14, fontweight='bold', color='#1F4E79', ha='center')
    
    # Pipeline components
    draw_box(ax, 3, 55, 20, 30, "Network Flow Stream\n(42 Numerical Features)\n\n• Live Sniffer / CSV Upload\n• MinMaxScaler Scaling\n• Label Encoder Alignment", bg_color="#264653", fontsize=9)
    draw_box(ax, 27, 55, 22, 30, "Local Dueling DQN RL Agent\n(LearnableRLAgent)\n\n• Value & Advantage Streams\n• Q-value Action Selection\n• Replay Buffer Update", bg_color="#2A9D8F", fontsize=9)
    draw_box(ax, 53, 55, 20, 30, "Novelty & Disagreement\nEvaluator\n\n• Threshold < 0.70 check\n• Supervised vs RL vs FL\n  Disagreement Detection", bg_color="#E76F51", fontsize=9)
    draw_box(ax, 77, 55, 20, 30, "Feedback & Replay Store\n(FeedbackStore)\n\n• Human Correction Capture\n• Multi-Client Broadcast\n• Continuous Retraining", bg_color="#F4A261", fontsize=9)
    
    # Sync Manager & Flower FL Server at Bottom
    draw_box(ax, 15, 10, 32, 28, "RLFLSyncManager Orchestrator\n\n• Monitors Novelty Triggers\n• Spawns Multi-Process FL\n• Controls Subprocess Execution", bg_color="#1F4E79", fontsize=9)
    draw_box(ax, 53, 10, 42, 28, "Flower Federated FL Round (FedAvg)\n\n• 3 Distributed Edge Clients Retrain\n• Parameter Weights Averaged over Port 8080\n• Updates Global RL-FL Model (global_model.pt)", bg_color="#1D3557", fontsize=9)
    
    # Arrows
    draw_arrow(ax, 23, 70, 27, 70, label="Scale")
    draw_arrow(ax, 49, 70, 53, 70, label="Q-values")
    draw_arrow(ax, 73, 70, 77, 70, label="Flag Novel")
    draw_arrow(ax, 87, 55, 31, 38, label="Auto-Trigger Sync", arrow_color="#E76F51")
    draw_arrow(ax, 47, 24, 53, 24, label="Spawn FL Round", arrow_color="#2A9D8F")
    draw_arrow(ax, 74, 38, 38, 55, label="Global Model Broadcast", arrow_color="#2A9D8F")
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "rl_fl_integrated_pipeline.png")
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {path}")

# ---------------------------------------------------------
# Figure 3: Dual-Layer XAI Flow
# ---------------------------------------------------------
def create_xai_llm_flow():
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    ax.text(50, 95, "Dual-Layer Explainable AI (SHAP + Groq Llama 3.3 70B LLM)", fontsize=14, fontweight='bold', color='#1F4E79', ha='center')
    
    # Boxes
    draw_box(ax, 5, 55, 24, 28, "Input Network Record\n(42 Features)\n\nE.g.: Flow Duration, Packet Length, Fwd Pkt/s, IAT Mean", bg_color="#34495E", fontsize=9)
    draw_box(ax, 38, 70, 24, 20, "Model Inference Engine\n\n• XGBoost / Dueling DQN\n• Multi-Class Confidence", bg_color="#2980B9", fontsize=9)
    draw_box(ax, 38, 40, 24, 20, "Layer 1: SHAP Attribution\n\n• TreeExplainer / Feature Dev\n• Top 8 Feature Impacts (±)", bg_color="#16A085", fontsize=9)
    draw_box(ax, 70, 48, 26, 34, "Layer 2: Groq LLM Engine\n(Llama 3.3 70B Versatile)\n\n• Formats Plain-English Text\n• Threat Severity & Context\n• Model Agreement Note\n• Mitigation Recommendation", bg_color="#8E44AD", fontsize=9)
    
    # Fallback box
    draw_box(ax, 20, 5, 60, 20, "Aegis XAI Built-in Fallback Engine\n\nProvides deterministic structured natural-language explanations when Groq API key is unconfigured or offline, ensuring zero operational downtime.", bg_color="#ECF0F1", text_color="#2C3E50", fontsize=9, linestyle="--", edgecolor="#8E44AD")
    
    # Arrows
    draw_arrow(ax, 29, 69, 38, 79, label="Predict")
    draw_arrow(ax, 29, 69, 38, 50, label="Calculate SHAP")
    draw_arrow(ax, 62, 80, 70, 68, label="Class & Conf")
    draw_arrow(ax, 62, 50, 70, 60, label="Feature Impacts")
    draw_arrow(ax, 50, 40, 50, 25, label="API Offline Fallback", arrow_color="#E74C3C", linestyle="--")
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "xai_llm_flow.png")
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {path}")

# ---------------------------------------------------------
# Figure 4: FL Aggregation Protocol Sequence Diagram
# ---------------------------------------------------------
def create_fl_aggregation_protocol_sequence():
    fig, ax = plt.subplots(figsize=(12, 7.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    ax.text(50, 96, "Sequence Diagram: FedAvg Protocol Over gRPC (Port 8080)", fontsize=14, fontweight='bold', color='#1F4E79', ha='center')
    
    # Lifelines
    clients = ["Central FL Server\n(rl_server.py)", "Edge Client 0\n(SDN Node)", "Edge Client 1\n(IoT Node)", "Edge Client 2\n(CPS Node)"]
    xs = [15, 40, 65, 90]
    
    for x, c in zip(xs, clients):
        draw_box(ax, x-10, 84, 20, 8, c, bg_color="#1F4E79" if x==15 else "#2E75B6", fontsize=9)
        ax.plot([x, x], [10, 84], color="#999999", linestyle="--", lw=1.5)
    
    # Steps
    y = 76
    ax.annotate("1. Initialize Round & Broadcast Parameters $w_t$", xy=(40, y), xytext=(15, y), arrowprops=dict(arrowstyle="->", color="#27AE60", lw=1.5), fontsize=8, fontweight="bold")
    ax.annotate("", xy=(65, y), xytext=(15, y), arrowprops=dict(arrowstyle="->", color="#27AE60", lw=1.5))
    ax.annotate("", xy=(90, y), xytext=(15, y), arrowprops=dict(arrowstyle="->", color="#27AE60", lw=1.5))
    
    y = 64
    for x in [40, 65, 90]:
        draw_box(ax, x-7, y-4, 14, 8, "Local Training\n(2 Epochs)", bg_color="#F39C12", fontsize=7)
        
    y = 48
    ax.annotate("2. Return Local Weights $w_{t+1}^k$ & Sample Count $n_k$", xy=(15, y), xytext=(40, y), arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.5), fontsize=8, fontweight="bold")
    ax.annotate("", xy=(15, y-2), xytext=(65, y-2), arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.5))
    ax.annotate("", xy=(15, y-4), xytext=(90, y-4), arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.5))
    
    y = 34
    draw_box(ax, 5, y-4, 20, 8, "Execute FedAvg:\n$w_{t+1} = \sum \\frac{n_k}{n} w_{t+1}^k$", bg_color="#8E44AD", fontsize=7)
    
    y = 20
    ax.annotate("3. Save Global State (global_model.pt) & Notify Clients", xy=(40, y), xytext=(15, y), arrowprops=dict(arrowstyle="->", color="#27AE60", lw=1.5), fontsize=8, fontweight="bold")
    ax.annotate("", xy=(65, y), xytext=(15, y), arrowprops=dict(arrowstyle="->", color="#27AE60", lw=1.5))
    ax.annotate("", xy=(90, y), xytext=(15, y), arrowprops=dict(arrowstyle="->", color="#27AE60", lw=1.5))
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fl_aggregation_protocol_sequence.png")
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {path}")

# ---------------------------------------------------------
# Figure 5: Dashboard & Deployment Architecture
# ---------------------------------------------------------
def create_dashboard_deployment_architecture():
    fig, ax = plt.subplots(figsize=(13, 7.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    ax.text(50, 95, "End-to-End System Deployment Architecture (Flask UI + FastAPI + Live Sniffer + ngrok)", fontsize=14, fontweight='bold', color='#1F4E79', ha='center')
    
    # Client Browser
    draw_box(ax, 3, 50, 22, 38, "User & Admin Clients\n(Web Browsers / Remote)\n\n• Multi-Page Flask UI\n• Home, Predict, FL, RL\n• Theme Toggle (Dark/Light)\n• CSV Upload / Live Stats", bg_color="#2C3E50", fontsize=9)
    
    # Deployment Host Server
    draw_box(ax, 30, 42, 42, 46, "Host Workstation / Edge Gateway (Localhost:7860)\n\n", bg_color="#ECF0F1", text_color="#2C3E50", fontsize=10, linestyle="-", edgecolor="#2C3E50")
    draw_box(ax, 32, 64, 18, 20, "Flask Multi-Page UI\n(ids_server.py)\n\n• Web Routes\n• Template Engine", bg_color="#2980B9", fontsize=8)
    draw_box(ax, 52, 64, 18, 20, "FastAPI Service\n(Port 8000)\n\n• REST Endpoints\n• JSON Serialization", bg_color="#27AE60", fontsize=8)
    draw_box(ax, 32, 46, 38, 15, "Model Registry & Artifact Store\nSaved model/ (scaler.joblib, label_encoder.joblib, global_model.pt)", bg_color="#7F8C8D", fontsize=8)
    
    # Packet Sniffer & ngrok
    draw_box(ax, 77, 65, 20, 23, "Live Packet Sniffer\n(live_sniffer.py)\n\n• Scapy Flow Extraction\n• 42-Feature Normalizer\n• Real-Time Alert Engine", bg_color="#D35400", fontsize=8)
    draw_box(ax, 77, 28, 20, 23, "ngrok HTTPS Tunnel\n(expose_ngrok.ps1)\n\n• Public Secure URL\n• Encrypted Ingress\n• Remote Admin Access", bg_color="#8E44AD", fontsize=8)
    
    # Arrows
    draw_arrow(ax, 25, 70, 32, 70, label="HTTP GET/POST")
    draw_arrow(ax, 25, 38, 77, 38, label="HTTPS Tunnel Traffic", arrow_color="#8E44AD")
    draw_arrow(ax, 77, 75, 70, 75, label="Stream Flows", arrow_color="#D35400")
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "dashboard_deployment_architecture.png")
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {path}")

# ---------------------------------------------------------
# Figure 6: Centralized vs FL Performance Comparison
# ---------------------------------------------------------
def create_centralized_vs_fl_performance():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    models = ['XGBoost\n(499A Central)', 'Random Forest\n(499A Central)', 'GRU Central\n(499A Baseline)', 'DNN Central\n(499A Baseline)', 'FL-GRU\n(5 Clients)', 'FL-Dueling DQN\n(499B Proposed)']
    acc = [84.03, 83.98, 80.36, 80.36, 80.35, 82.15]
    f1 = [86.78, 86.64, 84.62, 84.15, 84.11, 85.30]
    
    x = np.arange(len(models))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, acc, width, label='Accuracy (%)', color='#1F4E79')
    rects2 = ax.bar(x + width/2, f1, width, label='F1-Score (%)', color='#2E75B6')
    
    ax.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax.set_title('Performance Comparison: 499A Centralized Baselines vs 499B Federated & RL Models', fontsize=12, fontweight='bold', color='#1F4E79', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylim(70, 95)
    ax.legend(frameon=True, facecolor='white', edgecolor='#CCCCCC')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Value labels
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.2f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.2f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
        
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "centralized_vs_fl_performance.png")
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {path}")

# ---------------------------------------------------------
# Figure 7: RL Agent Performance Heatmap
# ---------------------------------------------------------
def create_rl_agent_performance_heatmap():
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    
    agents = ['Basic DQN', 'DQN Improved', 'PPO', 'Rainbow DQN', 'QR-DQN']
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Bot Recall', 'Prob Recall']
    
    data = np.array([
        [93.50, 46.08, 64.46, 50.86, 39.88, 66.39],
        [94.06, 48.54, 60.81, 52.64, 22.22, 57.58],
        [94.70, 43.34, 58.34, 47.57,  0.00, 60.56],
        [94.66, 41.32, 46.51, 40.92,  0.00,  6.47],
        [91.39, 43.40, 66.99, 43.88, 89.11, 64.11]
    ])
    
    sns.heatmap(data, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=metrics, yticklabels=agents, cbar_kws={'label': 'Percentage (%)'}, ax=ax, linewidths=0.5)
    ax.set_title('Reinforcement Learning Agents Performance Heatmap Across Evaluation Metrics', fontsize=11, fontweight='bold', color='#1F4E79', pad=12)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "rl_agent_performance_heatmap.png")
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {path}")

# ---------------------------------------------------------
# Figure 8: Privacy vs Accuracy Tradeoff Curve
# ---------------------------------------------------------
def create_privacy_accuracy_tradeoff():
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
    
    clients = [1, 3, 5, 10, 20]
    central_acc = [80.36] * len(clients)
    fl_acc = [80.36, 80.35, 80.30, 79.85, 78.90]
    privacy_gain = [0, 66.7, 80.0, 90.0, 95.0]
    
    ax.plot(clients, fl_acc, 'o-', color='#1F4E79', lw=2.5, label='FL-GRU Accuracy (%)')
    ax.plot(clients, central_acc, '--', color='#C0392B', lw=2, label='Centralized GRU Accuracy (80.36%)')
    
    ax2 = ax.twinx()
    ax2.plot(clients, privacy_gain, 's:', color='#27AE60', lw=2, label='Data Privacy Isolation (%)')
    ax2.set_ylabel('Data Privacy Isolation (%)', color='#27AE60', fontsize=10, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#27AE60')
    
    ax.set_xlabel('Number of Distributed Client Nodes (K)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Model Accuracy (%)', color='#1F4E79', fontsize=10, fontweight='bold')
    ax.set_title('Privacy Preservation vs Detection Performance Trade-Off Curve', fontsize=11, fontweight='bold', color='#1F4E79', pad=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='center right', frameon=True, facecolor='white')
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "privacy_accuracy_tradeoff.png")
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {path}")

if __name__ == "__main__":
    print("Generating report assets...")
    create_fl_main_architecture()
    create_rl_fl_integrated_pipeline()
    create_xai_llm_flow()
    create_fl_aggregation_protocol_sequence()
    create_dashboard_deployment_architecture()
    create_centralized_vs_fl_performance()
    create_rl_agent_performance_heatmap()
    create_privacy_accuracy_tradeoff()
    print("All report assets generated successfully!")
