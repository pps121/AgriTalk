"""
inspect_run.py -- MetaFlow Run Inspector for AgriTalk Demo
==========================================================
Run this AFTER metaflow_demo.py to explore MetaFlow's versioning and
artifact inspection capabilities -- core to the V2 Validation and
V3 Versioning concepts in the interview.

Usage:
  python inspect_run.py                  # list all runs
  python inspect_run.py <run_id>         # inspect a specific run
  python inspect_run.py compare          # compare two most recent runs

This demonstrates:
  - How a supervisor in Lyon can inspect a run from Milan
  - How every metric in a paper is traceable to an artifact
  - How seasonal calibration drift can be detected

Prerequisites:
  pip install metaflow pandas
  Run metaflow_demo.py at least once first.
"""

import sys
import pandas as pd
from metaflow import Flow, Run


def list_all_runs():
    """List all historical runs with key metrics."""
    print("\n" + "="*65)
    print("  AgriTalkDemoFlow -- All Historical Runs")
    print("="*65)

    try:
        flow = Flow("AgriTalkDemoFlow")
    except Exception as e:
        print(f"  No runs found. Run: python metaflow_demo.py run first.")
        return

    rows = []
    for run in flow.runs():
        if run.successful:
            try:
                rows.append({
                    "Run ID":       run.id,
                    "Season":       run.data.season,
                    "Alpha":        run.data.alpha,
                    "Coverage":     f"{run.data.calibration['coverage']:.3f}",
                    "Macro-F1":     f"{run.data.metrics['macro_f1']:.3f}",
                    "ECE (after)":  f"{run.data.metrics['ece_after']:.3f}",
                    "HITL Rate":    f"{run.data.calibration['set_size_geq2']:.1%}",
                    "Spray Safe":   run.data.field_context["spray_safe"],
                    "Policy":       run.data.hitl_policy_result["policy_status"][:10],
                })
            except AttributeError:
                pass

    if rows:
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
    else:
        print("  No completed runs found.")

    print()


def inspect_run(run_id: str):
    """Deep-inspect a specific run -- demonstrates V2 Validation."""
    print("\n" + "="*65)
    print(f"  Inspecting Run: {run_id}")
    print("="*65)

    try:
        run = Run(f"AgriTalkDemoFlow/{run_id}")
    except Exception as e:
        print(f"  Run not found: {run_id}")
        print(f"  Error: {e}")
        return

    if not run.successful:
        print("  Run did not complete successfully.")
        return

    print(f"\n  SEASON:  {run.data.season}")
    print(f"  ALPHA:   {run.data.alpha}")

    print(f"\n  [C1] CONFORMAL CALIBRATION:")
    cal = run.data.calibration
    print(f"    Coverage achieved:      {cal['coverage']:.4f}  (target: {1-run.data.alpha:.4f})")
    print(f"    Calibration threshold:  {cal['threshold']:.4f}")
    print(f"    Calibration set size:   {cal['n_cal']}")
    print(f"    Test set size:          {cal['n_test']}")
    print(f"    Mean pred-set size:     {cal['set_size_mean']:.2f}")
    print(f"    -> Singleton (proceed): {cal['set_size_1']:.1%}")
    print(f"    -> Size>=2 (HITL):      {cal['set_size_geq2']:.1%}")
    print(f"    -> All classes (reject):{cal['set_size_all']:.1%}")

    print(f"\n  [EVALUATION]:")
    m = run.data.metrics
    print(f"    Macro-F1:             {m['macro_f1']:.4f}")
    print(f"    ECE (softmax base):   {m['ece_before']:.4f}")
    print(f"    ECE (post-conformal): {m['ece_after']:.4f}")
    print(f"    ECE improvement:      {m['ece_before']-m['ece_after']:.4f} "
          f"({(m['ece_before']-m['ece_after'])/m['ece_before']:.0%})")
    print(f"    ABORT recall:         {m['abort_recall']:.4f}  (safety-critical)")

    print(f"\n  [FIELD CONTEXT] (C3 atomic snapshot):")
    fc = run.data.field_context
    for k, v in fc.items():
        print(f"    {k:25s}: {v}")

    print(f"\n  [HITL POLICY]:")
    hp = run.data.hitl_policy_result
    print(f"    Status:    {hp['policy_status']}")
    print(f"    HITL rate: {hp['hitl_rate']:.1%}")
    print(f"    Spray:     {'ALLOWED' if hp['spray_allowed'] else 'BLOCKED'} -- {hp['spray_reason']}")

    print(f"\n  [STEP ARTIFACTS] -- V2: supervisor can verify each step independently:")
    for step in run:
        task = list(step.tasks())[0] if list(step.tasks()) else None
        if task:
            print(f"    {step.id:30s} artifacts: {list(task)}")

    print()


def compare_recent_runs():
    """
    Compare the two most recent runs -- demonstrates seasonal replay concept.
    V3: detect calibration drift across seasons without re-running experiments.
    """
    print("\n" + "="*65)
    print("  Seasonal Calibration Drift Analysis (V3 Versioning)")
    print("="*65)

    try:
        flow = Flow("AgriTalkDemoFlow")
        recent_runs = [r for r in flow.runs() if r.successful][:2]
    except Exception:
        print("  Need at least 2 completed runs. Run:")
        print("    python metaflow_demo.py run")
        print("    python metaflow_demo.py run --season milan_autumn_2028")
        return

    if len(recent_runs) < 2:
        print(f"  Only {len(recent_runs)} run(s) found. Need 2.")
        print("  Try: python metaflow_demo.py run --season milan_autumn_2028")
        return

    r1, r2 = recent_runs[0], recent_runs[1]
    print(f"\n  Run A (ID: {r1.id}) | Season: {r1.data.season}")
    print(f"  Run B (ID: {r2.id}) | Season: {r2.data.season}")

    print(f"\n  {'Metric':<30} {'Run A':>12} {'Run B':>12} {'Delta':>10}")
    print(f"  {'-'*64}")

    metrics = [
        ("Coverage",       r1.data.calibration["coverage"],         r2.data.calibration["coverage"]),
        ("ECE (after)",    r1.data.metrics["ece_after"],             r2.data.metrics["ece_after"]),
        ("Macro-F1",       r1.data.metrics["macro_f1"],              r2.data.metrics["macro_f1"]),
        ("ABORT recall",   r1.data.metrics["abort_recall"],          r2.data.metrics["abort_recall"]),
        ("HITL rate",      r1.data.calibration["set_size_geq2"],     r2.data.calibration["set_size_geq2"]),
        ("Mean wind (m/s)",r1.data.field_context["mean_wind_ms"],    r2.data.field_context["mean_wind_ms"]),
        ("Stress zones",   r1.data.field_context["n_stress_zones"],  r2.data.field_context["n_stress_zones"]),
    ]

    for name, v1, v2 in metrics:
        delta = v2 - v1
        flag = "  <-- DRIFT DETECTED" if abs(delta) > 0.03 else ""
        print(f"  {name:<30} {v1:>12.4f} {v2:>12.4f} {delta:>+10.4f}{flag}")

    print(f"\n  Seasonal replay interpretation:")
    if abs(r1.data.calibration["coverage"] - r2.data.calibration["coverage"]) > 0.02:
        print(f"  --> Coverage drift detected (>{2}%) between seasons.")
        print(f"  --> This triggers adaptive recalibration in the full system.")
    else:
        print(f"  --> Coverage stable across seasons. Calibration transferable.")

    print()


# ===========================================================================
if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        list_all_runs()
    elif args[0] == "compare":
        compare_recent_runs()
    else:
        inspect_run(args[0])
