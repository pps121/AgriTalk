"""
generate_gantt.py — AgriTalk PhD 36-Month GANTT Chart
Generates a high-quality matplotlib GANTT chart for the GreenFieldData PhD-L proposal.
Exports: gantt_chart.png and gantt_chart.pdf

Usage: python generate_gantt.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from datetime import datetime
import numpy as np

# ── Colour palette ──────────────────────────────────────────────────────────
COLOURS = {
    "c1": "#006699",      # C1 — Calibrated Intent Compiler (CIC)
    "c2": "#228B22",      # C2 — Temporal Streaming Aggregator (TSCA)
    "c3": "#6428A0",      # C3 — Faithful Explanation System (FAES)
    "c4": "#C05A00",      # C4 — Reproducible Infrastructure (RAAI)
    "corpus": "#20B2AA",  # Corpus / data collection
    "thesis": "#555555",  # Thesis writing
    "sim": "#AA6622",     # Simulation / robots
    "review": "#8888BB",  # Literature review
}

# ── Task definitions ─────────────────────────────────────────────────────────
# (label, start_month, duration_months, colour_key, year_group)
# Months are 0-indexed from Oct 2026 (Month 0) to Sep 2029 (Month 35)
TASKS = [
    # Year 1 — UCBLyon1
    ("Literature review & architecture",   0,  4, "review",  "Y1"),
    ("Corpus collection (agronomist)",     1,  5, "corpus",  "Y1"),
    ("Metaflow IoT pipeline — C4",         3,  4, "c4",      "Y1"),
    ("LLM fine-tuning (intent)",           4,  5, "c1",      "Y1"),
    ("Conformal calibration — C1",         6,  5, "c1",      "Y1"),

    # Year 2 — UniMI + ProBayes
    ("ProBayes secondment: calibration",  12,  6, "c1",      "Y2"),
    ("Streaming TSCA — C2",               14,  6, "c2",      "Y2"),
    ("Gazebo + ROS2 integration",         16,  5, "sim",     "Y2"),
    ("HITL user study",                   18,  6, "c3",      "Y2"),

    # Year 3 — Both institutions
    ("Faithful explanation — C3",         24,  5, "c3",      "Y3"),
    ("Cross-farm generalisation study",   27,  4, "c2",      "Y3"),
    ("Thesis writing",                    28,  7, "thesis",  "Y3"),
]

# ── Milestone definitions ────────────────────────────────────────────────────
# (label, month, marker)
MILESTONES = [
    ("Paper 1\nCIC / ACL 2027",       10, "D"),
    ("Paper 2\nTSCA / VLDB 2028",     20, "D"),
    ("Paper 3\nFAES / FAccT 2029",    29, "D"),
    ("Paper 4\nSystem / MLSys 2029",  32, "D"),
    ("PhD\nDefence",                  35, "*"),
]

YEAR_GROUPS = {
    "Y1": (0, 12,  "#E8F0F8", "Year 1 — UCBLyon1, Lyon"),
    "Y2": (12, 24, "#E8F5E8", "Year 2 — UniMI + ProBayes"),
    "Y3": (24, 36, "#F3ECF8", "Year 3 — Both Institutions"),
}

# ── Month labels ─────────────────────────────────────────────────────────────
def month_label(m):
    months = ["Oct","Nov","Dec","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep"]
    years  = [2026, 2026, 2026,
               2027, 2027, 2027, 2027, 2027, 2027, 2027, 2027, 2027,
               2027, 2028, 2028, 2028, 2028, 2028, 2028, 2028, 2028, 2028,
               2027, 2028,
               2028, 2029, 2029, 2029, 2029, 2029, 2029, 2029, 2029, 2029,
               2028, 2029]
    # simple lookup
    month_idx = m % 12
    year_offset = m // 12
    base_years = [2026, 2027, 2028]
    yr = base_years[year_offset] if year_offset < 3 else 2029
    return f"{months[month_idx]}\n'{str(yr)[2:]}"

MONTH_NAMES = ["Oct","Nov","Dec","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep"] * 3
MONTH_YEARS = (["'26"] * 3 + ["'27"] * 12 + ["'28"] * 12 + ["'29"] * 9)[:36]

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(22, 10))
ax.set_xlim(-0.5, 36.5)
ax.set_ylim(-1.5, len(TASKS) + 0.5)

# Background bands per year
for yg, (start, end, color, label) in YEAR_GROUPS.items():
    ax.axvspan(start - 0.5, end - 0.5, color=color, alpha=0.7, zorder=0)
    ax.text((start + end) / 2, len(TASKS) + 0.25, label,
            ha="center", va="bottom", fontsize=10, fontweight="bold",
            color={"Y1": "#003366", "Y2": "#1A5C1A", "Y3": "#4A1680"}[yg])

# Year boundary lines
for x in [11.5, 23.5]:
    ax.axvline(x, color="white", linewidth=2.5, zorder=1)
    ax.axvline(x, color="#AAAAAA", linewidth=0.8, linestyle="--", zorder=2)

# Task bars
bar_height = 0.55
for i, (label, start, dur, ck, yg) in enumerate(TASKS):
    y = len(TASKS) - 1 - i
    rect = FancyBboxPatch(
        (start - 0.5 + 0.05, y - bar_height / 2),
        dur - 0.10, bar_height,
        boxstyle="round,pad=0.05",
        facecolor=COLOURS[ck], edgecolor="white", linewidth=0.8,
        zorder=3
    )
    ax.add_patch(rect)
    # Label inside bar if wide enough, else to the right
    if dur >= 3:
        ax.text(start - 0.5 + dur / 2, y, label,
                ha="center", va="center", fontsize=7.5, color="white",
                fontweight="bold", zorder=4)
    else:
        ax.text(start - 0.5 + dur + 0.15, y, label,
                ha="left", va="center", fontsize=7.5, color=COLOURS[ck],
                fontweight="bold", zorder=4)

# Milestones
for (mlabel, m, marker) in MILESTONES:
    ax.plot(m - 0.5, -0.8, marker=marker,
            markersize=14 if marker == "*" else 11,
            color="#CC2200" if marker == "*" else "#0055AA",
            zorder=5, markeredgecolor="white", markeredgewidth=1.0)
    ax.text(m - 0.5, -1.35, mlabel,
            ha="center", va="top", fontsize=6.5,
            color="#CC2200" if marker == "*" else "#0055AA",
            fontweight="bold")

# X-axis: month ticks
ax.set_xticks(range(36))
ax.set_xticklabels(
    [f"{mn}\n{yr}" for mn, yr in zip(MONTH_NAMES, MONTH_YEARS)],
    fontsize=6.5
)

# Y-axis: task labels
ax.set_yticks(range(len(TASKS)))
ax.set_yticklabels(
    [t[0] for t in reversed(TASKS)],
    fontsize=8.5
)

ax.set_xlabel("Month (Oct 2026 – Sep 2029)", fontsize=9, labelpad=4)
ax.grid(axis="x", color="white", linewidth=0.6, zorder=1)
ax.set_facecolor("#F8F8F8")
fig.patch.set_facecolor("white")

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=COLOURS["c1"],     label="C1 — Calibrated Intent Compiler"),
    mpatches.Patch(facecolor=COLOURS["c2"],     label="C2 — Temporal Streaming Aggregator"),
    mpatches.Patch(facecolor=COLOURS["c3"],     label="C3 — Faithful Explanation System"),
    mpatches.Patch(facecolor=COLOURS["c4"],     label="C4 — Reproducible Infrastructure"),
    mpatches.Patch(facecolor=COLOURS["corpus"], label="Corpus / Data collection"),
    mpatches.Patch(facecolor=COLOURS["sim"],    label="Simulation / Robot integration"),
    mpatches.Patch(facecolor=COLOURS["thesis"], label="Thesis writing"),
    plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#0055AA",
               markersize=8, label="Paper milestone"),
    plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#CC2200",
               markersize=10, label="PhD Defence"),
]
ax.legend(handles=legend_items, loc="lower right", bbox_to_anchor=(1.0, -0.02),
          ncol=3, fontsize=7.5, framealpha=0.9, edgecolor="#AAAAAA")

ax.set_title(
    "AgriTalk PhD-L — 36-Month Research Roadmap\n"
    "Calibrated Intent Compilation and Faithful Explanation for Human-Supervised Agricultural Robotic Systems",
    fontsize=11, fontweight="bold", pad=10
)

plt.tight_layout(rect=[0, 0.06, 1, 1])

# ── Save ──────────────────────────────────────────────────────────────────────
import os
out_dir = os.path.dirname(os.path.abspath(__file__))
for fmt in ["png", "pdf"]:
    out_path = os.path.join(out_dir, f"gantt_chart.{fmt}")
    plt.savefig(out_path, dpi=180 if fmt == "png" else None,
                bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_path}")

plt.close()
