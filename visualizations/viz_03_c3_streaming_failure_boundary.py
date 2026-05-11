"""
AgriTalk Visualization 03 — C3/RQ3: Temporal Streaming Grounding Failure Boundary
====================================================================================
From proposal (p.4): "Failure boundary experiments: sensor dropout (10–50%),
telemetry lag (>5 min), GPS/vision conflict — characterising connectivity-limited
rural farm risk rather than assuming reliable IoT."

"TSGA maintains a time-indexed field-state register: Kafka topics (sensor.raw 15s;
ndvi.updates per mission; weather 10min; telemetry 1s; spray.events immutable)
with Avro schema registry; Spark Structured Streaming applies 15-min rolling aggregates."

3 panels:
1. 3D failure surface: sensor_dropout × telemetry_lag → grounding_recall
   (shows the cliff edge where both degrade simultaneously)
2. Kafka topic latency timeline (5 topics with different cadences)
3. Atomic snapshot freshness: V2 staleness verifier decisions over time

Key research insight: rural farm connectivity is intermittent — the system must
remain safe even when 30-50% of sensors drop out, which existing systems cannot handle.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

os.makedirs("visualizations/html", exist_ok=True)
rng = np.random.default_rng(42)

# ── Failure surface parameters ────────────────────────────────────────────────
dropout_pct   = np.linspace(0, 50, 35)     # sensor dropout 0–50%
lag_minutes   = np.linspace(0, 30, 30)     # telemetry lag 0–30 min
D, L = np.meshgrid(dropout_pct, lag_minutes)

# Grounding recall: degrades with both dropout and lag
# With redundancy (TSGA aggregation): graceful degradation until cliff
# Cliff: dropout > 30% AND lag > 10 min → collapse
recall_base = 1.0 - (D/100)**1.4 * 0.75 - (L/30)**2.2 * 0.40
# Cliff effect (both degrade together)
cliff_penalty = np.where((D > 30) & (L > 10), 0.25 * (D - 30)/20 * (L - 10)/20, 0)
grounding_recall = np.clip(recall_base - cliff_penalty, 0.05, 1.0)

# V2 Staleness verifier threshold (alert when recall < 0.70)
stale_threshold = 0.70
grounding_alert = grounding_recall < stale_threshold

# Latency to atomic snapshot (milliseconds)
latency_ms = 150 + 80 * (D/50)**2 + 200 * (L/30)**1.5 + rng.normal(0, 10, D.shape)
latency_ms = np.clip(latency_ms, 80, 1500)

print(f"Recall at (0% dropout, 0 lag):    {grounding_recall[0, 0]:.3f}")
print(f"Recall at (30% dropout, 10 lag):  {grounding_recall[np.argmin(abs(lag_minutes-10)), np.argmin(abs(dropout_pct-30))]:.3f}")
print(f"Recall at (50% dropout, 20 lag):  {grounding_recall[np.argmin(abs(lag_minutes-20)), -1]:.3f}")
print(f"Cells with V2 staleness alert:    {grounding_alert.mean():.1%}")

# ── Kafka topic simulation over 2 hours ──────────────────────────────────────
sim_duration_min = 120
t_min = np.linspace(0, sim_duration_min, 1000)

KAFKA_TOPICS = {
    "sensor.raw (15s)":         {"period": 0.25, "dropout_prob": 0.05, "color": "#2ecc71",  "symbol": "circle"},
    "ndvi.updates (per mission)":{"period": 20,   "dropout_prob": 0.15, "color": "#3498db",  "symbol": "diamond"},
    "weather (10min)":           {"period": 10,   "dropout_prob": 0.08, "color": "#9b59b6",  "symbol": "square"},
    "robot.telemetry (1s)":      {"period": 1/60, "dropout_prob": 0.03, "color": "#e67e22",  "symbol": "triangle-up"},
    "spray.events (immutable)":  {"period": 8,    "dropout_prob": 0.00, "color": "#e74c3c",  "symbol": "x"},
}

# Generate event times for each topic
topic_events = {}
for topic_name, cfg in KAFKA_TOPICS.items():
    events = []
    t = 0
    while t < sim_duration_min:
        t += cfg["period"]
        if rng.random() > cfg["dropout_prob"]:
            events.append(t)
    topic_events[topic_name] = (events, cfg)

# ── Snapshot freshness over time ──────────────────────────────────────────────
# Simulate field-state register freshness as a composite score
t_snapshot = np.linspace(0, sim_duration_min, 500)
# Freshness decays between events, resets on new event
freshness = np.ones(len(t_snapshot))
for i in range(1, len(t_snapshot)):
    dt = t_snapshot[i] - t_snapshot[i-1]
    freshness[i] = freshness[i-1] * np.exp(-0.05 * dt)  # exponential decay
    # Simulate sensor event refreshing the register
    if rng.random() < 0.15:  # sensor event every ~7min average
        freshness[i] = min(freshness[i] + rng.uniform(0.3, 0.6), 1.0)

# V2 staleness alerts (freshness < 0.5)
staleness_alerts = t_snapshot[freshness < 0.50]

# ── Build figure ──────────────────────────────────────────────────────────────
fig = make_subplots(
    rows=2, cols=2,
    specs=[
        [{"type": "surface", "colspan": 2}, None],
        [{"type": "scatter"}, {"type": "scatter"}],
    ],
    subplot_titles=[
        "Grounding Recall Failure Surface — Dropout × Telemetry Lag (RQ3 core)",
        "Kafka Topic Event Timeline (2-hour farm session)",
        "Field-State Register Freshness & V2 Staleness Alerts",
    ],
    row_heights=[0.55, 0.45],
    vertical_spacing=0.10,
    horizontal_spacing=0.10,
)

# ── Panel 1: 3D failure surface ───────────────────────────────────────────────
fig.add_trace(go.Surface(
    x=D, y=L, z=grounding_recall,
    colorscale="RdYlGn",
    cmin=0.0, cmax=1.0,
    name="Grounding Recall",
    showscale=True,
    colorbar=dict(x=1.01, title="Recall", len=0.45, y=0.77,
                  tickvals=[0, 0.25, 0.50, 0.70, 0.85, 1.0],
                  ticktext=["0", "0.25", "0.50", "0.70⚠", "0.85", "1.0"]),
    hovertemplate=(
        "Dropout: %{x:.0f}%<br>"
        "Lag: %{y:.1f} min<br>"
        "Recall: %{z:.3f}<extra>TSGA Grounding</extra>"
    ),
    opacity=0.88,
    lighting=dict(ambient=0.7, diffuse=0.9, roughness=0.5),
), row=1, col=1)

# V2 staleness threshold plane
fig.add_trace(go.Surface(
    x=D, y=L, z=np.full_like(D, stale_threshold),
    colorscale=[[0, "rgba(231,76,60,0.22)"], [1, "rgba(231,76,60,0.22)"]],
    showscale=False, opacity=0.35, name="V2 Staleness Alert (recall<0.70)",
    hoverinfo="skip",
), row=1, col=1)

# Safe operating zone contour marker
fig.add_trace(go.Scatter3d(
    x=[10, 10, 20, 20, 10],
    y=[0, 5, 5, 0, 0],
    z=[grounding_recall[0,2]]*5,
    mode="lines",
    line=dict(color="#2ecc71", width=3),
    name="Safe operating zone",
    hoverinfo="skip",
), row=1, col=1)

# ── Panel 2: Kafka timeline ───────────────────────────────────────────────────
y_offset = 0
for topic_name, (events, cfg) in list(topic_events.items())[::-1]:
    if not events:
        continue
    # Event markers
    fig.add_trace(go.Scatter(
        x=events[:200],
        y=[y_offset] * min(len(events), 200),
        mode="markers",
        name=topic_name,
        marker=dict(
            size=6 if "telemetry" not in topic_name else 3,
            color=cfg["color"],
            symbol=cfg["symbol"],
            opacity=0.8,
        ),
        hovertemplate=f"<b>{topic_name}</b><br>t = %{{x:.1f}} min<extra></extra>",
    ), row=2, col=1)
    y_offset += 1

# Highlight simulated dropout period (30-50 min) — add_vrect replaced for plotly 6.x compat
fig.add_trace(go.Scatter(
    x=[30, 30, 50, 50, 30],
    y=[0, len(KAFKA_TOPICS)-0.5, len(KAFKA_TOPICS)-0.5, 0, 0],
    fill="toself", fillcolor="rgba(231,76,60,0.12)",
    line=dict(color="rgba(0,0,0,0)"),
    mode="lines", showlegend=False, hoverinfo="skip", name="dropout_period",
), row=2, col=1)

# ── Panel 3: Freshness & alerts ───────────────────────────────────────────────
fig.add_trace(go.Scatter(
    x=t_snapshot, y=freshness,
    mode="lines",
    name="Register freshness",
    line=dict(color="#3498db", width=2),
    fill="tozeroy", fillcolor="rgba(52,152,219,0.12)",
    hovertemplate="t = %{x:.1f} min<br>Freshness: %{y:.3f}<extra></extra>",
), row=2, col=2)

# Staleness threshold line (add_hline replaced for plotly 6.x compat)
fig.add_trace(go.Scatter(
    x=[t_snapshot[0], t_snapshot[-1]], y=[0.50, 0.50],
    mode="lines", name="V2 alert threshold (0.50)",
    line=dict(dash="dash", color="rgba(231,76,60,0.7)", width=1.5),
    showlegend=True, hoverinfo="skip",
), row=2, col=2)

# Alert markers
if len(staleness_alerts) > 0:
    fig.add_trace(go.Scatter(
        x=staleness_alerts,
        y=[0.50] * len(staleness_alerts),
        mode="markers",
        name="V2 Staleness Alert",
        marker=dict(size=7, color="#e74c3c", symbol="triangle-up"),
        hovertemplate="V2 Alert at t=%{x:.1f} min<extra></extra>",
    ), row=2, col=2)

fig.update_layout(
    scene=dict(
        xaxis_title="Sensor Dropout (%)",
        yaxis_title="Telemetry Lag (minutes)",
        zaxis_title="Grounding Recall",
        camera=dict(eye=dict(x=1.7, y=-1.7, z=1.3)),
        bgcolor="#0d1117",
    ),
    xaxis=dict(title="Time (minutes)", range=[0, sim_duration_min]),
    yaxis=dict(title="Kafka Topic", tickvals=list(range(5)),
               ticktext=[t.split("(")[0].strip() for t in list(topic_events.keys())[::-1]]),
    xaxis2=dict(title="Time (minutes)", range=[0, sim_duration_min]),
    yaxis2=dict(title="Register Freshness Score", range=[0, 1.05]),
    title=dict(
        text=(
            "<b>AgriTalk C3/RQ3 — Temporal Streaming Grounding Failure Boundary Analysis</b><br>"
            "<sup>TSGA: 5 Kafka topics · Spark 15-min rolling aggregates · "
            "V2 Staleness Verifier · Rural farm connectivity characterisation</sup>"
        ),
        x=0.5, xanchor="center", font=dict(size=14),
    ),
    legend=dict(x=0.50, y=0.43, bgcolor="rgba(13,17,23,0.85)",
                bordercolor="#30363d", borderwidth=1, font=dict(size=9)),
    height=820,
    template="plotly_dark",
    paper_bgcolor="#0d1117",
    font=dict(family="Inter, Arial", color="#e6edf3"),
    margin=dict(l=20, r=60, t=120, b=40),
    annotations=[
        dict(
            text=(
                "Top: Recall cliff at dropout>30% & lag>10min — novel rural IoRT failure characterisation (RQ3) · "
                "Bottom-left: 5 Kafka topics with different cadences (sensor.raw 15s → weather 10min) · "
                "Bottom-right: V2 Staleness Verifier triggers alerts when field-state register freshness falls below threshold"
            ),
            x=0.5, y=-0.03, xref="paper", yref="paper",
            showarrow=False, font=dict(size=9, color="#8b949e"),
        )
    ],
)

out = "visualizations/html/03_c3_streaming_failure_boundary.html"
fig.write_html(out, include_plotlyjs="cdn", full_html=True)
print(f"✅ Saved: {out}")
