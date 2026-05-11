"""
AgriTalk Visualization 06 — 3-Year PhD Roadmap Timeline
=========================================================
From proposal (p.5), "Research Roadmap" section:

Year 1 (Oct'26–Sep'27, UCBLyon1):
  Q1: Literature survey, AgroNLP corpus construction
  Q2: LLM domain adaptation (LoRA/QLoRA), Gazebo sim env
  Q3: C1 conformal calibration, Gazebo tests
  Q4: Write EMNLP/ACL 2027 submission

Year 2 (Oct'27–Sep'28, UCBLyon1 + UniMI + ProBayes):
  Q1: Streaming TSGA architecture (Kafka/Spark), C3 prototype
  Q2: BVF attribution adapter (C2), prelim operator study
  Q3: Milan farm field tests, seasonal drift experiments
  Q4: Write C&E Agriculture + VLDB 2028 submissions

Year 3 (Oct'28–Sep'29, Both + CFL):
  Q1: Full CTEF operator study (N=30+, conditions A/B/C)
  Q2: Cross-farm CFL (C4), MetaFlow federated replay
  Q3: Thesis writing, CTEF analysis
  Q4: Thesis submission, ACL/FAccT 2029 submissions

Datasets: AgroNLP→PANGAEA/ACRE (Y1) → USDA-ARS-AgAID/UniMI sprayer (Y2) → federated (Y3)
Milestones: papers (EMNLP'27, C&E'28, VLDB'28, ACL'29, FAccT'29) + 5 system checkpoints
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

os.makedirs("visualizations/html", exist_ok=True)

# ── Timeline data ─────────────────────────────────────────────────────────────
# Quarters: 1–12 (Q1Y1 through Q4Y3)
Q = list(range(1, 13))

MILESTONES = [
    # (quarter, type, label, contribution, url_tooltip)
    (4,  "paper",     "EMNLP/ACL'27\n(C1 conformal NLU)",             "C1", "#2ecc71"),
    (6,  "system",    "C3 Streaming\nprototype (Kafka+Spark)",          "C3", "#3498db"),
    (7,  "study",     "Prelim operator\nstudy (N=15)",                  "C2", "#9b59b6"),
    (8,  "paper",     "C&E Agriculture'28\n(C3 streaming IoRT)",        "C3", "#3498db"),
    (8,  "paper",     "VLDB'28\n(TSGA temporal grounding)",             "C3", "#3498db"),
    (9,  "system",    "Full CTEF\nstudy (N=30+)",                       "C4", "#e67e22"),
    (10, "system",    "Cross-farm CFL\n(C4 federated replay)",          "C4", "#e67e22"),
    (11, "paper",     "Thesis\nsubmission draft",                        "ALL","#f39c12"),
    (12, "paper",     "ACL'29 (C2 BVF)\n+ FAccT'29 (C4 CTEF)",         "C2+C4","#9b59b6"),
]

PHASES = [
    # (q_start, q_end, contribution, label, color)
    (1,  4,  "C1", "C1: Conformal NLU\n(RAPS calibration, domain adapt)", "#2ecc71"),
    (5,  8,  "C3", "C3: TSGA Streaming\n(Kafka, failure boundary)", "#3498db"),
    (3,  8,  "C2", "C2: BVF Attribution\n(5-method, operator study)", "#9b59b6"),
    (9,  12, "C4", "C4: CTEF Evaluation\n(N=30 study, CFL replay)", "#e67e22"),
    (1,  12, "Inf", "Responsible AI\n(throughout: audit, GDPR, V5)", "#f39c12"),
]

DATASETS = [
    (1,  "AgroNLP corpus\n(construct Y1)",        "#8b949e"),
    (2,  "PANGAEA / ACRE\n(NDVI, satellite Y1)",  "#8b949e"),
    (5,  "USDA-ARS-AgAID\n(transfer Y2)",         "#6e7681"),
    (5,  "UniMI Sprayer Records\n(Milan field Y2)","#6e7681"),
    (6,  "ProBayes farm logs\n(partner Y2)",       "#6e7681"),
    (9,  "Federated corpus\n(Lyon+Milan+CFL Y3)",  "#484f58"),
]

INSTITUTIONS = [
    (1, 4,  "UCBLyon1",           "#2ecc71"),
    (5, 8,  "UCBLyon1 + UniMI\n+ ProBayes", "#3498db"),
    (9, 12, "UCBLyon1 + UniMI\n+ CFL",     "#e67e22"),
]

# ── Build figure ──────────────────────────────────────────────────────────────
fig = make_subplots(
    rows=3, cols=1,
    specs=[[{"type": "scatter"}], [{"type": "scatter"}], [{"type": "scatter"}]],
    subplot_titles=[
        "PhD Research Roadmap: Phases & Contributions",
        "Dataset Availability & Deployment Checkpoints",
        "Publication Targets & Risk Milestones",
    ],
    vertical_spacing=0.10,
    row_heights=[0.40, 0.30, 0.30],
)

# ── Panel 1: Phases ────────────────────────────────────────────────────────────
q_labels = [
    "Q1\nY1", "Q2\nY1", "Q3\nY1", "Q4\nY1",
    "Q1\nY2", "Q2\nY2", "Q3\nY2", "Q4\nY2",
    "Q1\nY3", "Q2\nY3", "Q3\nY3", "Q4\nY3",
]

y_positions = {"C1": 5, "C2": 3, "C3": 4, "C4": 2, "Inf": 1}
y_labels = {1: "Responsible AI", 2: "C4: CTEF", 3: "C2: Attribution",
            4: "C3: Streaming", 5: "C1: Conformal NLU"}

for q_start, q_end, contrib, label, color in PHASES:
    fig.add_shape(
        type="rect",
        x0=q_start - 0.5, x1=q_end + 0.5,
        y0=y_positions[contrib] - 0.38, y1=y_positions[contrib] + 0.38,
        fillcolor=color, opacity=0.30,
        line=dict(color=color, width=2),
        row=1, col=1,
    )
    # Short label centered on bar
    fig.add_annotation(
        x=(q_start + q_end) / 2, y=y_positions[contrib],
        text=label, font=dict(size=8, color=color),
        showarrow=False, row=1, col=1,
    )

# Year separators
for y_sep, y_label in [(4.5, "Year 2\nOct'27"), (8.5, "Year 3\nOct'28")]:
    fig.add_vline(x=y_sep, line_dash="dash", line_color="rgba(255,255,255,0.3)",
                  annotation_text=y_label, annotation_position="top",
                  annotation_font=dict(size=9, color="#8b949e"), row=1, col=1)

# Year 1 start label
fig.add_annotation(x=2, y=5.6, text="Year 1: Oct'26\n(UCBLyon1)",
                   font=dict(size=9, color="#8b949e"), showarrow=False, row=1, col=1)

# Milestone markers on top panel
milestone_symbols = {"paper": "star", "system": "diamond", "study": "circle"}
for q, mtype, label, contrib, color in MILESTONES:
    y_pos = y_positions.get(contrib.split("+")[0], 3)
    fig.add_trace(go.Scatter(
        x=[q], y=[y_pos + 0.5],
        mode="markers",
        marker=dict(size=12, color=color, symbol=milestone_symbols[mtype],
                    line=dict(color="white", width=1.5)),
        name=label.replace("\n", " "),
        hovertemplate=f"<b>{label.replace(chr(10), ' ')}</b><br>Q{q} ({contrib})<extra></extra>",
        showlegend=False,
    ), row=1, col=1)

# ── Panel 2: Datasets ─────────────────────────────────────────────────────────
for d_idx, (q_avail, label, color) in enumerate(DATASETS):
    y_val = (d_idx % 3) * 0.3 + 0.5
    fig.add_trace(go.Scatter(
        x=[q_avail], y=[y_val],
        mode="markers+text",
        text=[label.split("\n")[0]],
        textposition="top center",
        textfont=dict(size=8, color=color),
        marker=dict(size=10, color=color, symbol="circle",
                    line=dict(color="white", width=1)),
        name=label.replace("\n", " "),
        hovertemplate=f"<b>{label.replace(chr(10), ' ')}</b><br>Available Q{q_avail}<extra></extra>",
        showlegend=False,
    ), row=2, col=1)

# Institution bands
inst_y = 1.8
for q_s, q_e, inst_label, color in INSTITUTIONS:
    fig.add_shape(type="rect", x0=q_s-0.5, x1=q_e+0.5,
                  y0=inst_y-0.2, y1=inst_y+0.2,
                  fillcolor=color, opacity=0.2,
                  line=dict(color=color, width=1), row=2, col=1)
    fig.add_annotation(x=(q_s+q_e)/2, y=inst_y, text=inst_label,
                       font=dict(size=8, color=color), showarrow=False, row=2, col=1)

# Edge deployment milestones
for q_dep, dep_label, color in [(4, "Gazebo sim Y1", "#2ecc71"),
                                   (7, "Real robot\nMilan field Y2", "#e67e22"),
                                   (10, "Edge Jetson\ndeployment Y3", "#3498db")]:
    fig.add_trace(go.Scatter(
        x=[q_dep], y=[1.2],
        mode="markers+text",
        text=[dep_label.split("\n")[0]],
        textposition="bottom center",
        textfont=dict(size=8, color=color),
        marker=dict(size=12, color=color, symbol="triangle-up",
                    line=dict(color="white", width=1.5)),
        hovertemplate=f"<b>{dep_label.replace(chr(10), ' ')}</b><extra></extra>",
        showlegend=False,
    ), row=2, col=1)

# ── Panel 3: Publication timeline ─────────────────────────────────────────────
pubs = [
    (4,  "EMNLP/ACL 2027",          "C1",     "#2ecc71", "star", 0.8),
    (8,  "C&E Agriculture 2028",    "C3",     "#3498db", "star", 0.5),
    (8,  "VLDB 2028",               "C3",     "#3498db", "star", 1.1),
    (11, "Thesis draft",             "ALL",    "#f39c12", "diamond", 0.8),
    (12, "ACL 2029",                "C2",     "#9b59b6", "star", 0.5),
    (12, "FAccT 2029",              "C4",     "#e67e22", "star", 1.1),
]
risks = [
    (3,  "Data access\nrisk (mitigated: PANGAEA)", "#8b949e"),
    (6,  "Farm access\nrisk (mitigated: ProBayes MOU)", "#8b949e"),
    (10, "Study recruitment\nrisk (mitigated: UCBLyon1 pool)", "#8b949e"),
]

for q, pub_label, contrib, color, symbol, y_pos in pubs:
    fig.add_trace(go.Scatter(
        x=[q], y=[y_pos],
        mode="markers+text",
        text=[pub_label],
        textposition="top center",
        textfont=dict(size=9, color=color, family="Arial Bold"),
        marker=dict(size=14, color=color, symbol=symbol,
                    line=dict(color="white", width=2)),
        name=pub_label,
        hovertemplate=f"<b>{pub_label}</b><br>Contribution: {contrib}<br>Target: Q{q}<extra></extra>",
        showlegend=True,
    ), row=3, col=1)

for q_risk, risk_label, color in risks:
    fig.add_trace(go.Scatter(
        x=[q_risk], y=[0.2],
        mode="markers+text",
        text=[risk_label.split("\n")[0]],
        textposition="bottom center",
        textfont=dict(size=8, color=color),
        marker=dict(size=8, color="rgba(231,76,60,0.6)", symbol="x",
                    line=dict(color="#e74c3c", width=1.5)),
        hovertemplate=f"<b>Risk: {risk_label.replace(chr(10), ' ')}</b><extra></extra>",
        showlegend=False,
    ), row=3, col=1)

# Year separators for panels 2 and 3
for row_n in [2, 3]:
    for y_sep in [4.5, 8.5]:
        fig.add_vline(x=y_sep, line_dash="dot", line_color="rgba(255,255,255,0.2)",
                      row=row_n, col=1)

# ── Layout ────────────────────────────────────────────────────────────────────
fig.update_xaxes(
    tickvals=list(range(1, 13)),
    ticktext=q_labels,
    range=[0.3, 12.7],
)
fig.update_yaxes(
    showticklabels=True,
    tickvals=list(y_labels.keys()),
    ticktext=list(y_labels.values()),
    row=1, col=1,
)
fig.update_yaxes(showticklabels=False, range=[0, 2.2], row=2, col=1)
fig.update_yaxes(showticklabels=False, range=[0, 1.5], row=3, col=1)

fig.update_layout(
    title=dict(
        text=(
            "<b>AgriTalk — 3-Year PhD Roadmap: Oct 2026 → Sep 2029</b><br>"
            "<sup>C1 (Conformal NLU) → C2 (BVF Attribution) → C3 (Streaming TSGA) → C4 (CTEF Trust Evaluation) · "
            "UCBLyon1 / UniMI / ProBayes / CFL institutions · "
            "5 publications: EMNLP'27, C&E'28, VLDB'28, ACL'29, FAccT'29</sup>"
        ),
        x=0.5, xanchor="center", font=dict(size=14),
    ),
    legend=dict(x=0.01, y=0.02, bgcolor="rgba(13,17,23,0.85)",
                bordercolor="#30363d", borderwidth=1, font=dict(size=9)),
    height=820,
    template="plotly_dark",
    paper_bgcolor="#0d1117",
    font=dict(family="Inter, Arial", color="#e6edf3"),
    margin=dict(l=100, r=40, t=120, b=60),
    annotations=[
        dict(
            text=(
                "Top: Contribution phases with overlaps reflecting integration (C2 builds on C1; C4 requires C1+C2+C3) · "
                "Middle: Datasets introduced per year with institutional partnerships · "
                "Bottom: Publication targets (★) and mitigated risks (×)"
            ),
            x=0.5, y=-0.03, xref="paper", yref="paper",
            showarrow=False, font=dict(size=9, color="#8b949e"),
        )
    ],
)

out = "visualizations/html/06_phd_roadmap_timeline.html"
fig.write_html(out, include_plotlyjs="cdn", full_html=True)
print(f"✅ Saved: {out}")
