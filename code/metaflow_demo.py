"""
AgriTalk MetaFlow Demo
======================
Run this script BEFORE your interview to get a feel for MetaFlow fully.

What this demonstrates:
  - Multi-step DAG with self.next()
  - Fan-out (parallel steps) and fan-in (merge)
  - Parameters passed from CLI, tracked per run
  - Artifact versioning: every self.X is saved automatically
  - Resuming from a specific step
  - Inspecting historical runs and artifacts

Prerequisites:
  pip install metaflow pandas numpy scikit-learn

Quick run:
  python metaflow_demo.py run
  python metaflow_demo.py run --help
  python metaflow_demo.py run --season milan_autumn_2028
  python metaflow_demo.py run --alpha 0.10

After running, inspect:
  python inspect_runs.py  (see bottom of this file for instructions)
"""

import numpy as np
import pandas as pd
from metaflow import FlowSpec, step, Parameter, current


# ===========================================================================
# Simulated agricultural data loading helpers
# (In production these would connect to Kafka, NDVI drone data, etc.)
# ===========================================================================

# Simulate IoT sensor data: temperature, wind speed, soil moisture, with seasonal drift
def simulate_sensor_data(season: str, n_rows: int = 500) -> pd.DataFrame:
    """
    Simulate IoT sensor readings for a given season.
    Real version: reads from Kafka topic 'agritalk.sensor.raw'
    """
    np.random.seed(hash(season) % 2**31)

    # Season-specific base values (non-stationarity simulation)
    base_temp = {"lyon_spring_2027": 15.0, "milan_autumn_2028": 12.0,
                 "lyon_winter_2027": 5.0}.get(season, 14.0)
    base_wind = {"lyon_spring_2027": 2.1, "milan_autumn_2028": 3.5,
                 "lyon_winter_2027": 4.0}.get(season, 2.5)

    return pd.DataFrame({
        "timestamp": pd.date_range("2027-04-01", periods=n_rows, freq="15s"),
        "temperature_c": np.random.normal(base_temp, 2.0, n_rows),
        "wind_speed_ms": np.abs(np.random.normal(base_wind, 0.8, n_rows)),
        "soil_moisture_pct": np.random.uniform(20, 80, n_rows),
        "sensor_ok": np.random.choice([True, True, True, False], n_rows),  # 25% dropout
    })

# simulate NDVI data: 12 zones, NDVI < 0.35 = crop stress
# Real version: reads from Kafka topic 'agritalk.ndvi.updates' (per drone mission).
# NDVI < 0.35 indicates crop stress zones needing precision intervention.
# Note: this simulates seasonal drift in NDVI as well, to reflect real-world conditions.
# The function below is an alternative to the MONTH_NAMES/MONTH_YEARS approach in generate_gantt.py, providing a more direct mapping from month index to label.
def month_label(m: int) -> str:
    """
    Convert month index (0-35) to label like "Oct '26".
    This simulates seasonal drift in NDVI as well, to reflect real-world conditions.
    """
    # Map month index to month name and year
    months = ["Oct","Nov","Dec","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep"] * 3
    # why * 3? because we have 36 months (3 years) and this repeats the month names for each year
    month_idx = m % 12
    year_offset = m // 12
    base_years = [2026, 2027, 2028]
    yr = base_years[year_offset] if year_offset < 3 else 2029
    return f"{months[month_idx]} '{str(yr)[2:]}"    

# Simulate NDVI drone readings per field zone.
# Real version: reads from Kafka topic 'agritalk.ndvi.updates' (per mission).
# NDVI < 0.35 indicates crop stress zones needing precision intervention.   
# Note: this simulates seasonal drift in NDVI as well, to reflect real-world conditions.
def simulate_ndvi_data(season: str, n_zones: int = 12) -> pd.DataFrame:
    """
    Simulate NDVI readings per field zone.
    Real version: reads from Kafka topic 'agritalk.ndvi.updates'
    NDVI < 0.35 indicates crop stress zones needing precision intervention.
    """
    np.random.seed(hash(season + "_ndvi") % 2**31)
    base_ndvi = {"lyon_spring_2027": 0.52, "milan_autumn_2028": 0.44}.get(season, 0.50)

    return pd.DataFrame({
        "zone_id": [f"Z{i:02d}" for i in range(n_zones)],
        "ndvi": np.clip(np.random.normal(base_ndvi, 0.12, n_zones), 0.1, 0.9),
        "stress_detected": lambda df: df["ndvi"] < 0.35,
    })

#data quality check: flag dropout, stale sensors, conflicting readings
# V1 Verifier: sensor freshness and dropout detection. 
# In production, this would be more complex and might trigger alerts or HITL routing.
# Note: this is a simplified version for demonstration purposes. 
# In a real system, would have more sophisticated checks and possibly integrate with monitoring/alerting systems.
# The function below is a simple implementation that checks for sensor dropout and prints a warning if dropout exceeds 30%. 
# In a real system, this could be expanded to include checks for stale data (e.g., timestamps), conflicting readings (e.g., sudden spikes), and integration with alerting systems to notify operators or trigger HITL routing.
# The key point is that this function represents a V1 Verifier step that can be iterated on without re-running the entire pipeline, 
# demonstrating MetaFlow's velocity and versioning benefits. 
# Supervisors can inspect the raw and cleaned data artifacts from this step in any run to understand the impact of data quality on downstream calibration and policy decisions.
def dq_check(df: pd.DataFrame) -> pd.DataFrame:
    """
    Data quality check: flag dropout, stale sensors, conflicting readings.
    V1 Verifier: sensor freshness and dropout detection.
    """
    clean = df[df["sensor_ok"]].copy()
    n_dropped = len(df) - len(clean)
    dropout_pct = n_dropped / len(df) * 100
    print(f"    [V1 DQ] Rows before: {len(df)}, after: {len(clean)}, "
          f"dropout: {dropout_pct:.1f}%")
    if dropout_pct > 30:
        print("    [V1 ALERT] Dropout > 30% -- triggering mandatory HITL for next command")
    return clean

# Simulate NL command corpus with intent labels
# intent classes: 
#   COMMAND, 
#   QUERY, 
#   ABORT_MISSION, 
#   REPORT, 
#   DIAGNOSE, 
#   CLARIFY, 
#   CONFIRM, 
#   DELEGATE 
# Real version: from AgroNLP corpus and agronomist interviews.
def simulate_nl_commands(season: str, n: int = 200):
    """
    Simulate NL command corpus with intent labels.
    Real version: from AgroNLP corpus and agronomist interviews.
    """
    np.random.seed(hash(season + "_cmds") % 2**31)

    templates = [
        ("Spray more pesticide on the east section",           "COMMAND"),
        ("What is the current NDVI in zone Z03?",              "QUERY"),
        ("Stop all spraying operations immediately",           "ABORT_MISSION"),
        ("Generate a spray event report for today",            "REPORT"),
        ("Is wind speed safe for spraying right now?",         "QUERY"),
        ("Adjust dosage by 15% in zones with NDVI below 0.35","COMMAND"),
        ("Confirm the pesticide formulation for wheat rust",   "CLARIFY"),
        ("Diagnose why zone Z07 shows abnormal moisture",      "DIAGNOSE"),
    ]

    indices = np.random.choice(len(templates), size=n)
    commands = [templates[i][0] for i in indices]
    intents  = [templates[i][1] for i in indices]

    # Add noise to make classification non-trivial
    noise_phrases = ["please", "quickly", "now", "for the northern area"]
    commands_noisy = []
    for cmd in commands:
        if np.random.random() < 0.3:
            noise = np.random.choice(noise_phrases)
            cmd = f"{cmd}, {noise}"
        commands_noisy.append(cmd)

    return pd.DataFrame({"command": commands_noisy, "intent": intents})

# conformal calibration simulation: generates coverage and set size stats
# Real version: calibrates a fine-tuned agricultural LLM with conformal sets.
# Key concepts:
#   - alpha = miscoverage rate target (default 0.05 = 95% coverage)
#   - coverage = fraction of test examples where true label is in prediction set
#   - set_size_1 = fraction of singletons (high confidence -> proceed without HITL)
#   - set_size_geq2 = fraction requiring HITL (ambiguous or uncertain)
#   - set_size_all = fraction rejected (no valid intent found)
def simulate_conformal_calibration(
    commands_df: pd.DataFrame,
    alpha: float = 0.05,
    random_seed: int = 42,
) -> dict:
    """
    Simulated conformal prediction calibration.
    Real version: calibrates a fine-tuned agricultural LLM with conformal sets.

    Returns:
        dict with coverage, set sizes, and calibration threshold.

    Key concepts:
      - alpha = miscoverage rate target (default 0.05 = 95% coverage)
      - coverage = fraction of test examples where true label is in prediction set
      - set_size_1 = fraction of singletons (high confidence -> proceed without HITL)
      - set_size_geq2 = fraction requiring HITL (ambiguous or uncertain)
      - set_size_all = fraction rejected (no valid intent found)
    """
    from sklearn.model_selection import train_test_split

    np.random.seed(random_seed)
    n = len(commands_df)

    # Simulate softmax confidence scores (deliberately overconfident for contrast)
    n_classes = 8  # intent classes
    raw_scores = np.random.dirichlet([1.2] * n_classes, size=n)

    # Nonconformity scores: 1 - score of true class
    true_classes = pd.Categorical(commands_df["intent"]).codes
    nonconf_scores = 1.0 - raw_scores[np.arange(n), true_classes]

    # Split calibration / test
    cal_idx, test_idx = train_test_split(np.arange(n), test_size=0.3, random_state=42)
    cal_scores = nonconf_scores[cal_idx]

    # Conformal threshold: (1-alpha) quantile of calibration scores
    quantile = np.quantile(cal_scores, 1 - alpha)

    # Prediction sets on test data
    test_scores = raw_scores[test_idx]
    prediction_sets = [
        [j for j in range(n_classes) if (1 - test_scores[i, j]) <= quantile]
        for i in range(len(test_idx))
    ]

    set_sizes = [len(s) for s in prediction_sets]
    true_test_classes = true_classes[test_idx]
    covered = sum(
        tc in ps for tc, ps in zip(true_test_classes, prediction_sets)
    )
    coverage = covered / len(test_idx)

    return {
        "alpha":          alpha,
        "threshold":      float(quantile),
        "coverage":       float(coverage),
        "n_cal":          len(cal_idx),
        "n_test":         len(test_idx),
        "set_size_mean":  float(np.mean(set_sizes)),
        "set_size_1":     float(np.mean([s == 1 for s in set_sizes])),
        "set_size_geq2":  float(np.mean([s >= 2 for s in set_sizes])),
        "set_size_all":   float(np.mean([s == n_classes for s in set_sizes])),
        # Interpretation:
        #   set_size_1    -> proceed without HITL (confident singleton)
        #   set_size_geq2 -> trigger HITL (ambiguous)
        #   set_size_all  -> reject (no valid prediction)
    }

# Simulate evaluation metrics.
# Real version: uses AgroNLP test set and full fine-tuned LLM.
# Key metrics:
#   - macro_f1: intent classification performance
#   - ece_before: expected calibration error before conformal calibration
#   - ece_after: expected calibration error after conformal calibration
#   - abort_recall: recall for ABORT_MISSION class (safety-critical)
#   - n_eval: number of test examples evaluated
#   - coverage: achieved coverage from calibration step
#   - hitl_rate: fraction of examples triggering HITL (set_size_geq2 from calibration)
# Note: this is a simulation with random values in realistic ranges. 
# In a real system, these would be computed from the model's predictions on the test set, 
# using the conformal prediction sets to adjust confidence and compute metrics accordingly.
# The function below simulates evaluation metrics for the intent classifier, demonstrating the expected improvements from conformal calibration and the safety-critical recall for the ABORT_MISSION class.
# This function is meant to be called after the simulate_conformal_calibration step, 
# using its output to inform the evaluation metrics, particularly the hitl_rate which is derived from the set_size_geq2 statistic.
def evaluate_intent_classifier(calibration: dict, commands_df: pd.DataFrame) -> dict:
    """
    Simulate evaluation metrics.
    Real version: uses AgroNLP test set and full fine-tuned LLM.
    """
    np.random.seed(99)
    n = len(commands_df)
    intents = commands_df["intent"].unique()

    macro_f1 = float(np.random.uniform(0.68, 0.82))  # realistic range
    ece = float(np.random.uniform(0.04, 0.12))
    abort_recall = float(np.random.uniform(0.88, 0.97))  # ABORT is safety-critical

    return {
        "macro_f1":      macro_f1,
        "ece_before":    0.18,  # softmax baseline (Guo et al. 2017)
        "ece_after":     ece,   # after conformal calibration
        "abort_recall":  abort_recall,
        "n_eval":        n,
        "coverage":      calibration["coverage"],
        "hitl_rate":     calibration["set_size_geq2"],  # fraction triggering HITL
    }


# ===========================================================================
# The MetaFlow Pipeline
# ===========================================================================
    # Demonstrates :
    #   multi-step DAG, 
    #   fan-out/fan-in, 
    #   parameters, 
    #   artifact versioning, 
    #   resuming, 
    #   inspecting historical runs.
# Each step's self.X assignments are automatically versioned artifacts: inspectable from any run
# Fan-out steps (ingest_sensors, fetch_ndvi) run in parallel; their outputs are merged in join_context.
# Parameters (season, alpha, n_commands) are passed from CLI and tracked per run.
# Resuming: you can re-run the pipeline from any step, 
#   e.g. `python metaflow_demo.py run --start-step calibrate` to skip data loading and directly test calibration with different alpha.
# Inspecting historical runs: after running, use `python inspect_runs.py` to see all runs, their parameters, and artifacts like calibration results and evaluation metrics.
# This pipeline is a simplified demo of the AgriTalk calibrated intent pipeline: showcasing MetaFlow's capabilities for building and iterating on complex data pipelines with versioned artifacts and easy inspection.
# In a real implementation, each step would involve more complex logic, integration with real data sources (Kafka, drone data), and the actual model training and evaluation.
# It allows : rapid iteration (V1 Velocity), deep inspection and validation (V2 Validation), and robust versioning of all artifacts (V3 Versioning) in the context of agricultural AI research.

class AgriTalkDemoFlow(FlowSpec):
    """
    AgriTalk Calibrated Intent Pipeline (Demo)
    -------------------------------------------
    Demonstrates MetaFlow's 3 Vs for agricultural AI:
      V1 Velocity:   Iterate on calibrate step without re-ingesting sensors
      V2 Validation: Supervisors inspect identical artifact state
      V3 Versioning: Every run is immutably tagged; seasonal replay enabled

    PhD contributions targeted:
      C1 (RQ1): Conformal calibration -> HITL policy (steps: calibrate, policy)
      C4:       Metaflow infrastructure (this pipeline itself)

    Usage:
      python metaflow_demo.py run
      python metaflow_demo.py run --season milan_autumn_2028 --alpha 0.10
      python metaflow_demo.py run --start-step calibrate  # skip re-ingestion
    """

    season = Parameter(
        "season",
        help="Farm-season tag: e.g. lyon_spring_2027, milan_autumn_2028",
        default="lyon_spring_2027",
        type=str,
    )

    alpha = Parameter(
        "alpha",
        help="Conformal miscoverage rate (default 0.05 = 95% coverage guarantee)",
        default=0.05,
        type=float,
    )

    n_commands = Parameter(
        "n_commands",
        help="Number of NL commands to simulate",
        default=300,
        type=int,
    )

    # ------------------------------------------------------------------
    # STEP 1: start
    # ------------------------------------------------------------------
    @step
    def start(self):
        """
        Initialize run. In production: load config, validate DB connections.
        MetaFlow concept: self.X assignments here become versioned artifacts.
        """
        print(f"\n{'='*60}")
        print(f"  AgriTalkDemoFlow | Run ID: {current.run_id}")
        print(f"  Season: {self.season} | Alpha: {self.alpha}")
        print(f"{'='*60}\n")

        # These are versioned artifacts -- inspectable from any other run later
        self.run_id   = current.run_id
        self.flow_name = current.flow_name

        # Fan-out: run sensor ingestion and NDVI fetch in parallel
        # Fetch NDVI drone readings per field zone. 
        # Real version: reads from Kafka topic 'agritalk.ndvi.updates' (per mission). 
        # NDVI < 0.35 = crop stress zone.
        self.next(self.ingest_sensors, self.fetch_ndvi)

    # ------------------------------------------------------------------
    # STEP 2a: ingest_sensors  (parallel branch 1)
    # ------------------------------------------------------------------
    @step
    def ingest_sensors(self):
        """
        Ingest IoT sensor readings.
        Real version: reads from Kafka topic 'agritalk.sensor.raw' (15-sec interval).
        MetaFlow concept: @conda decorator here would pin kafka-python version.
        """
        print("  [B1] Ingesting sensor data from field IoT...")
        raw_df = simulate_sensor_data(self.season, n_rows=600)
        print(f"    Raw rows: {len(raw_df)}")

        # Data quality check (V1 Verifier)
        self.sensor_df = dq_check(raw_df)
        self.sensor_dropout_pct = (1 - len(self.sensor_df) / len(raw_df)) * 100

        self.next(self.join_context)

    # ------------------------------------------------------------------
    # STEP 2b: fetch_ndvi  (parallel branch 2)
    # ------------------------------------------------------------------
    @step
    def fetch_ndvi(self):
        """
        Fetch NDVI drone readings per field zone.
        Real version: reads from Kafka topic 'agritalk.ndvi.updates' (per mission).
        NDVI < 0.35 = crop stress zone.
        """
        print("  [B1] Fetching NDVI drone data...")
        ndvi_df = simulate_ndvi_data(self.season, n_zones=12)
        # Compute stress zones: NDVI < 0.35 = precision intervention needed
        ndvi_df["stress"] = ndvi_df["ndvi"] < 0.35

        self.ndvi_df     = ndvi_df
        self.stress_zones = ndvi_df[ndvi_df["stress"]]["zone_id"].tolist()
        print(f"    NDVI zones total: {len(ndvi_df)}, stress zones: {len(self.stress_zones)}")
        print(f"    Stress zones: {self.stress_zones}")

        self.next(self.join_context)

    # ------------------------------------------------------------------
    # STEP 3: join_context  (fan-in)
    # ------------------------------------------------------------------
    @step
    def join_context(self, inputs):
        """
        Merge sensor and NDVI data into atomic field-state context snapshot.
        MetaFlow concept: fan-in step receives merged 'inputs' from all branches.
        This is the C3 field-state register -- atomic at query receipt time.
        """
        print("  [B2] Building atomic field-state context snapshot...")

        # Merge artifacts from parallel branches
        self.merge_artifacts(inputs, include=["season", "alpha", "n_commands"])

        sensor = inputs.ingest_sensors.sensor_df
        ndvi   = inputs.fetch_ndvi.ndvi_df

        # Atomic snapshot: summarize current field state
        self.field_context = {
            "season":            self.season,
            "n_sensor_readings": len(sensor),
            "mean_wind_ms":      float(sensor["wind_speed_ms"].mean()),
            "mean_temp_c":       float(sensor["temperature_c"].mean()),
            "mean_soil_moisture":float(sensor["soil_moisture_pct"].mean()),
            "n_ndvi_zones":      len(ndvi),
            "n_stress_zones":    int((ndvi["ndvi"] < 0.35).sum()),
            "min_ndvi":          float(ndvi["ndvi"].min()),
            "max_ndvi":          float(ndvi["ndvi"].max()),
            # Spray safety check: wind > 4 m/s triggers spray ban
            "spray_safe":        float(sensor["wind_speed_ms"].mean()) < 4.0,
        }

        print(f"    Wind: {self.field_context['mean_wind_ms']:.2f} m/s | "
              f"Spray safe: {self.field_context['spray_safe']}")
        print(f"    Stress zones (NDVI<0.35): {self.field_context['n_stress_zones']}")

        self.next(self.load_commands)

    # ------------------------------------------------------------------
    # STEP 4: load_commands
    # ------------------------------------------------------------------
    @step
    def load_commands(self):
        """
        Load NL command corpus.
        Real version: AgroNLP corpus from agronomist interviews + field annotation.
        MetaFlow concept: self.commands_df is a versioned artifact -- supervisor
        can inspect it as: Run('AgriTalkDemoFlow/{id}').data.commands_df
        """
        print("  [B3] Loading NL command corpus...")
        self.commands_df = simulate_nl_commands(self.season, n=self.n_commands)
        print(f"    Commands loaded: {len(self.commands_df)}")
        print(f"    Intent distribution:\n"
              f"{self.commands_df['intent'].value_counts().to_string()}")

        self.next(self.calibrate)

    # ------------------------------------------------------------------
    # STEP 5: calibrate  (C1 -- the key research step)
    # ------------------------------------------------------------------
    # In production: @batch(cpu=8, memory=16000) for cloud execution
    @step
    def calibrate(self):
        """
        Conformal prediction calibration.
        This is C1 (RQ1): replace softmax overconfidence with
        statistically guaranteed coverage sets.

        Key formula: P(y in C(x)) >= 1 - alpha
          where alpha = miscoverage rate (default 0.05 = 95% coverage).

        MetaFlow benefit: THIS step can be re-run without re-running steps 1-4.
          python metaflow_demo.py run --start-step calibrate
        """
        print(f"\n  [C1/RQ1] Conformal calibration (alpha={self.alpha})...")
        self.calibration = simulate_conformal_calibration(
            self.commands_df,
            alpha=self.alpha,
            random_seed=42,
        )

        print(f"\n    Calibration results:")
        print(f"      Coverage achieved:      {self.calibration['coverage']:.3f} "
              f"(target: {1 - self.alpha:.3f})")
        print(f"      Mean prediction set size: {self.calibration['set_size_mean']:.2f}")
        print(f"      Singleton (->proceed):  {self.calibration['set_size_1']:.1%}")
        print(f"      Size>=2  (->HITL):      {self.calibration['set_size_geq2']:.1%}")
        print(f"      All classes (->reject): {self.calibration['set_size_all']:.1%}")
        print(f"\n    [Interpretation]")
        print(f"      {self.calibration['set_size_1']:.1%} of commands: proceed autonomously")
        print(f"      {self.calibration['set_size_geq2']:.1%} of commands: route to human HITL")

        self.next(self.hitl_policy)

    # ------------------------------------------------------------------
    # STEP 6: hitl_policy  (C1 -- HITL gate logic)
    # ------------------------------------------------------------------
    @step
    def hitl_policy(self):
        """
        Derive HITL trigger policy from calibration results.
        5 conditions that trigger HITL:
          1. Action type in {SPRAY, ABORT, DOSAGE_CHANGE, ZONE_OVERRIDE, ...}
          2. |C(x)| >= 2  (conformal non-singleton)
          3. Any entity KB-unverified
          4. Upstream DQ check failed (stale/missing sensor)
          5. User role = OBSERVER
        """
        print("\n  [C1] Deriving HITL trigger policy...")

        hitl_rate = self.calibration["set_size_geq2"]
        spray_safe = self.field_context["spray_safe"]

        # Policy: map coverage to HITL aggressiveness
        if self.calibration["coverage"] >= (1 - self.alpha):
            policy_status = "NOMINAL"
        elif self.calibration["coverage"] >= (1 - self.alpha - 0.02):
            policy_status = "MARGINAL (consider recalibration)"
        else:
            policy_status = "DEGRADED (mandatory HITL for ALL critical actions)"

        self.hitl_policy_result = {
            "policy_status":  policy_status,
            "hitl_rate":      hitl_rate,
            "spray_allowed":  spray_safe,
            "spray_reason":   "Wind speed safe" if spray_safe else "Wind speed too high",
            "critical_actions_require_hitl": [
                "SPRAY", "ABORT_MISSION", "DOSAGE_CHANGE",
                "ZONE_OVERRIDE", "HARVEST_EARLY", "IRRIGATION_OVERRIDE",
                "PESTICIDE_SWITCH", "EMERGENCY_STOP",
            ],
            "intent_classes": [
                "COMMAND", "ABORT_MISSION", "QUERY", "REPORT",
                "DIAGNOSE", "CLARIFY", "CONFIRM", "DELEGATE",
            ],
        }

        print(f"    Policy status:  {policy_status}")
        print(f"    HITL rate:      {hitl_rate:.1%}")
        print(f"    Spray allowed:  {spray_safe} ({self.hitl_policy_result['spray_reason']})")

        self.next(self.evaluate)

    # ------------------------------------------------------------------
    # STEP 7: evaluate
    # ------------------------------------------------------------------
    @step
    def evaluate(self):
        """
        Evaluate intent classification with calibrated uncertainty.
        All metrics are versioned artifacts -- traceable in every paper.
        """
        print("\n  [Evaluation] Running evaluation metrics...")
        self.metrics = evaluate_intent_classifier(self.calibration, self.commands_df)

        print(f"\n    Evaluation results:")
        print(f"      Macro-F1:       {self.metrics['macro_f1']:.3f}")
        print(f"      ECE (before):   {self.metrics['ece_before']:.3f}  (softmax baseline)")
        print(f"      ECE (after):    {self.metrics['ece_after']:.3f}  (post-conformal)")
        print(f"      ABORT recall:   {self.metrics['abort_recall']:.3f}  (safety-critical)")
        print(f"      HITL trigger:   {self.metrics['hitl_rate']:.1%}")

        ece_improvement = (self.metrics["ece_before"] - self.metrics["ece_after"])
        print(f"\n    ECE improvement: {ece_improvement:.3f} ({ece_improvement/self.metrics['ece_before']:.0%} better than softmax baseline)")

        self.next(self.end)

    # ------------------------------------------------------------------
    # STEP 8: end
    # ------------------------------------------------------------------
    @step
    def end(self):
        """
        Final summary. All self.X are now versioned artifacts.
        MetaFlow concept: access any artifact after the run:

          from metaflow import Run
          r = Run("AgriTalkDemoFlow/<run_id>")
          print(r.data.metrics)
          print(r.data.calibration)
          print(r.data.field_context)
          print(r.data.hitl_policy_result)
        """
        print(f"\n{'='*60}")
        print(f"  RUN COMPLETE | ID: {self.run_id}")
        print(f"  Season: {self.season}")
        print(f"{'='*60}")
        print(f"\n  SUMMARY:")
        print(f"    Coverage:     {self.calibration['coverage']:.3f} "
              f"(target >= {1 - self.alpha:.3f})")
        print(f"    Macro-F1:     {self.metrics['macro_f1']:.3f}")
        print(f"    ABORT recall: {self.metrics['abort_recall']:.3f}")
        print(f"    HITL rate:    {self.calibration['set_size_geq2']:.1%}")
        print(f"    Spray safe:   {self.field_context['spray_safe']}")
        print(f"    Policy:       {self.hitl_policy_result['policy_status']}")
        print(f"\n  ARTIFACT INSPECTION (run after this completes):")
        print(f"    python inspect_run.py {self.run_id}")
        print(f"{'='*60}\n")


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    AgriTalkDemoFlow()
