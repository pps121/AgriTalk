"""
AgriTalk — Run All Proposal-Aligned Visualizations
====================================================
Generates 7 interactive HTML visualizations from the AgriTalk PhD proposal:
  01 — C1/RQ1: Conformal seasonal drift + HITL ablation + ABORT recall
  02 — C2/RQ2: BVF attribution Kendall tau + operator trust study
  03 — C3/RQ3: Streaming failure boundary (Kafka/TSGA)
  04 — C4/RQ4: CTEF trust evaluation + edge deployment tiers
  05 — Full evaluation dashboard + PhD significance landscape
  06 — 3-year PhD roadmap timeline
  07 — C1 conformal coverage surface (alternative view)

Usage:
  cd /Volumes/Research/Agriculture-PhD
  python visualizations/run_all_visualizations.py

Output: visualizations/html/*.html (open in browser, fully interactive 3D)
"""

import subprocess
import sys
import os
import time

SCRIPTS = [
    ("C1/RQ1: Conformal Seasonal Drift",        "visualizations/viz_01_c1_conformal_seasonal_drift.py"),
    ("C2/RQ2: BVF Attribution & Trust",         "visualizations/viz_02_c2_bvf_attribution_trust.py"),
    ("C3/RQ3: Streaming Failure Boundary",      "visualizations/viz_03_c3_streaming_failure_boundary.py"),
    ("C4/RQ4: CTEF Trust & Deployment",         "visualizations/viz_04_c4_trust_deployment.py"),
    ("Full Evaluation Dashboard",                "visualizations/viz_05_full_evaluation_dashboard.py"),
    ("PhD Roadmap Timeline",                     "visualizations/viz_06_phd_roadmap_timeline.py"),
    ("C1 Conformal Coverage Surface",            "visualizations/viz_07_c1_coverage_surface.py"),
]


def main():
    os.makedirs("visualizations/html", exist_ok=True)
    print("=" * 60)
    print("  AgriTalk: 7 proposal-aligned 3D visualizations")
    print("  GreenFieldData PhD-L | Partha Pratim Saha | 2026")
    print("=" * 60)

    ok_list, fail_list = [], []
    t0_total = time.time()

    for label, script in SCRIPTS:
        print(f"\n▶ {label}")
        if not os.path.exists(script):
            print(f"  ✗ NOT FOUND: {script}")
            fail_list.append((label, "script not found"))
            continue

        t0 = time.time()
        result = subprocess.run(
            [sys.executable, script],
            capture_output=True, text=True,
        )
        elapsed = time.time() - t0

        if result.returncode == 0:
            print(f"  ✓ Done in {elapsed:.1f}s")
            for line in result.stdout.strip().splitlines()[-3:]:
                print(f"    {line}")
            ok_list.append(label)
        else:
            print(f"  ✗ FAILED (exit {result.returncode}) in {elapsed:.1f}s")
            for line in result.stderr.strip().splitlines()[-5:]:
                print(f"    STDERR: {line}")
            fail_list.append((label, result.stderr.strip()[-200:]))

    total = time.time() - t0_total
    print("\n" + "=" * 60)
    print(f"  {len(ok_list)} succeeded, {len(fail_list)} failed in {total:.1f}s")

    if ok_list:
        print("\n✅ Saved to visualizations/html/")
        for f in sorted(os.listdir("visualizations/html")):
            if f.endswith(".html"):
                print(f"   open visualizations/html/{f}")
    if fail_list:
        print("\n✗ Failed:")
        for lbl, err in fail_list:
            print(f"   • {lbl}: {err[:80]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
