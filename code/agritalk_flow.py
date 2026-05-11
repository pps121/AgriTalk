"""
AgriTalk: Conversational AI for Sustainable Agriculture
========================================================
Complete Metaflow pipeline implementing the PhD research project
"Conversational AI for Sustainable Agriculture: Natural Language
Interfaces over Robotic and Analytical Farming Systems"

Architecture follows Fig. 1 (prototype → production) and
Fig. 2 (DAG structure) from the MetaFlow paper:
  Tagliabue et al. (2023). "You Do Not Need a Bigger Boat:
  Metaflow for ML at Reasonable Scale." MLSys 2023.

Pipeline stages:
  1. IoT Stream Ingestion & Semantic Annotation (Task 1)
  2. Intent Recognition & NL-to-Command Compilation (Task 2)
  3. LLM Fine-tuning via LoRA (Task 2 continued)
  4. Edge-cloud Deployment (Task 3)
  5. Human-in-the-Loop Confirmation Gate (Task 5)
  6. Robotic Command Publication / ROS2 Bridge (Task 4)
  7. Evaluation & Audit Trail (Task 5)

Run locally (prototype):
  python agritalk_flow.py run --scenario pesticide_spray

Run with cloud GPU (production training):
  python agritalk_flow.py run --with batch --max-workers 4

Run scheduled production:
  python agritalk_flow.py argo-workflows create
"""

import json
import os
import time
import hashlib
import datetime
import random
from typing import Dict, List, Optional, Tuple

from metaflow import (
    FlowSpec,
    step,
    Parameter,
    JSONType,
    card,
    conda,
    batch,
    kubernetes,
    schedule,
    retry,
    timeout,
    catch,
    current,
)

# ─────────────────────────────────────────────────────────────────────
# Constants & Domain Taxonomy
# ─────────────────────────────────────────────────────────────────────
INTENT_CLASSES = ["QUERY", "COMMAND", "ABORT", "REPORT", "DIAGNOSE"]

# Safety-critical action types that ALWAYS require HITL confirmation
SAFETY_CRITICAL_ACTIONS = {"SPRAY", "ABORT_MISSION", "DOSAGE_CHANGE", "ZONE_OVERRIDE"}

# Confidence threshold below which HITL is mandatory
CONFIDENCE_THRESHOLD = 0.75

# Latency budget (milliseconds) for edge inference
EDGE_LATENCY_BUDGET_MS = 800


# ─────────────────────────────────────────────────────────────────────
# AgriTalk Main Metaflow Pipeline
# ─────────────────────────────────────────────────────────────────────
class AgriTalkFlow(FlowSpec):
    """
    End-to-end Metaflow DAG for Conversational AI in Precision Agriculture.

    This flow implements the full pipeline described in the GreenFieldData
    PhD Position L research proposal, connecting IoT data streams → LLM
    inference → semantic grounding → HITL confirmation → robotic actuation.

    Parameters
    ----------
    scenario : str
        Research scenario to run: 'pesticide_spray', 'robot_diagnosis',
        'mission_query', 'field_report'
    field_id : str
        Farm field identifier (e.g., 'FIELD_NORTH_01')
    user_query : str
        Natural language query or command from farmer/agronomist
    user_role : str
        Permission role: 'OBSERVER', 'OPERATOR', or 'AGRONOMIST'
    edge_mode : bool
        If True, use quantized edge model; if False, use cloud LLM API
    dry_run : bool
        If True, simulate robot actuation without real ROS2 commands
    """

    # ── Flow Parameters ───────────────────────────────────────────────
    scenario = Parameter(
        "scenario",
        help="Research scenario: pesticide_spray | robot_diagnosis | mission_query | field_report",
        default="pesticide_spray",
    )
    field_id = Parameter(
        "field_id",
        help="Farm field identifier",
        default="FIELD_NORTH_01",
    )
    user_query = Parameter(
        "user_query",
        help="Natural language query or command from farmer",
        default="Spray pesticide on the northern block — soil humidity is below threshold",
    )
    user_role = Parameter(
        "user_role",
        help="User role: OBSERVER | OPERATOR | AGRONOMIST",
        default="OPERATOR",
    )
    edge_mode = Parameter(
        "edge_mode",
        help="Use edge-quantized model instead of cloud API",
        default=False,
        type=bool,
    )
    dry_run = Parameter(
        "dry_run",
        help="Simulate actuation (no real robot commands sent)",
        default=True,
        type=bool,
    )
    confidence_threshold = Parameter(
        "confidence_threshold",
        help="Confidence threshold below which HITL confirmation is required",
        default=CONFIDENCE_THRESHOLD,
        type=float,
    )

    # ─────────────────────────────────────────────────────────────────
    # STEP 1: START — Validate & Initialise
    # ─────────────────────────────────────────────────────────────────
    @card
    @step
    def start(self):
        """
        Validate input parameters, authenticate the user role, and
        initialise the pipeline run with a unique audit ID.

        Implements: OWASP A01 — input validation & A07 — authentication.
        Metaflow concept: start step, artifact initialisation.
        """
        print(f"[AgriTalk] Run ID: {current.run_id}")
        print(f"[AgriTalk] Scenario: {self.scenario}")
        print(f"[AgriTalk] Field: {self.field_id}")
        print(f"[AgriTalk] User Query: {self.user_query}")
        print(f"[AgriTalk] Role: {self.user_role}")

        # ── Input validation (OWASP A03: Injection prevention) ────────
        allowed_roles = {"OBSERVER", "OPERATOR", "AGRONOMIST"}
        allowed_scenarios = {
            "pesticide_spray", "robot_diagnosis",
            "mission_query", "field_report"
        }

        if self.user_role not in allowed_roles:
            raise ValueError(f"Invalid role '{self.user_role}'. Must be one of {allowed_roles}")
        if self.scenario not in allowed_scenarios:
            raise ValueError(f"Invalid scenario '{self.scenario}'")

        # Sanitize user query — remove potential prompt injection patterns
        self.sanitized_query = _sanitize_nl_input(self.user_query)

        # ── Audit trail initialisation ─────────────────────────────────
        self.audit_record = {
            "run_id": current.run_id,
            "field_id": self.field_id,
            "user_role": self.user_role,
            "scenario": self.scenario,
            "timestamp_utc": datetime.datetime.utcnow().isoformat(),
            "raw_query": self.user_query,
            "sanitized_query": self.sanitized_query,
            "edge_mode": self.edge_mode,
            "steps_completed": ["start"],
        }

        print(f"[AgriTalk] Sanitized query: {self.sanitized_query}")
        self.next(self.ingest_iot_streams, self.load_knowledge_base)

    # ─────────────────────────────────────────────────────────────────
    # STEP 2a: IoT Stream Ingestion (PARALLEL BRANCH A)
    # ─────────────────────────────────────────────────────────────────
    @card
    @retry(times=2)
    @timeout(seconds=120)
    @step
    def ingest_iot_streams(self):
        """
        Ingest heterogeneous IoT field data streams and produce a
        semantically annotated field state snapshot.

        In production: reads from Apache Kafka topics fed by:
          - Soil moisture & pH sensors (structured JSON)
          - Weather station API (REST)
          - Drone multispectral imagery metadata (semi-structured XML)
          - Robot GPS telemetry (structured binary → JSON)

        For the prototype/dry_run, uses simulated field data.

        Metaflow concept: parallel branch, artifact versioning.
        PhD task: Task 1 — Heterogeneous IoT Stream Integration.
        """
        print(f"[IoT Ingest] Loading field state for {self.field_id}...")
        t0 = time.time()

        # Simulate IoT data ingestion (replace with Kafka consumer in production)
        self.field_state = _simulate_field_state(self.field_id)

        # Semantic annotation: map raw readings to AGROVOC ontology concepts
        self.annotated_field_state = _annotate_field_state(self.field_state)

        # Data quality check (Great Expectations equivalent)
        dq_report = _validate_field_data_quality(self.annotated_field_state)
        self.data_quality_passed = dq_report["all_checks_passed"]
        self.data_quality_report = dq_report

        elapsed_ms = (time.time() - t0) * 1000
        print(f"[IoT Ingest] Completed in {elapsed_ms:.1f}ms | DQ passed: {self.data_quality_passed}")
        print(f"[IoT Ingest] Field state keys: {list(self.annotated_field_state.keys())}")

        self.next(self.join_context)

    # ─────────────────────────────────────────────────────────────────
    # STEP 2b: Knowledge Base Loading (PARALLEL BRANCH B)
    # ─────────────────────────────────────────────────────────────────
    @card
    @step
    def load_knowledge_base(self):
        """
        Load the RAG knowledge base: pesticide database (EPPO/PPDB),
        crop ontology (AGROVOC), robot capability registry.

        This underpins semantic grounding in Task 2 — preventing LLM
        hallucination of pesticide names, dosages, and application rules.

        Metaflow concept: parallel branch, shared artifact store.
        PhD task: Task 2 — Semantic Grounding.
        """
        print("[KB] Loading agricultural knowledge bases...")
        t0 = time.time()

        self.pesticide_kb = _load_pesticide_kb()
        self.crop_ontology = _load_crop_ontology()
        self.robot_capability_registry = _load_robot_capabilities()

        elapsed_ms = (time.time() - t0) * 1000
        print(f"[KB] Loaded {len(self.pesticide_kb)} pesticide entries | "
              f"{len(self.crop_ontology)} crop types | "
              f"{len(self.robot_capability_registry)} robot capabilities "
              f"in {elapsed_ms:.1f}ms")

        self.next(self.join_context)

    # ─────────────────────────────────────────────────────────────────
    # STEP 3: Join Context from Parallel Branches
    # ─────────────────────────────────────────────────────────────────
    @card
    @step
    def join_context(self, inputs):
        """
        Merge artifacts from IoT ingestion and KB loading branches.
        Prepares the full context object for LLM inference.

        Metaflow concept: join step after branch, merge_artifacts.
        """
        self.merge_artifacts(inputs, exclude=["data_quality_report"])

        # Collect reports separately
        self.all_dq_reports = [
            i.data_quality_report
            for i in inputs
            if hasattr(i, "data_quality_report")
        ]

        # Build unified context for LLM
        self.inference_context = {
            "field_state": self.annotated_field_state,
            "pesticide_kb_summary": list(self.pesticide_kb.keys())[:10],
            "robot_capabilities": list(self.robot_capability_registry.keys()),
            "crop_types": list(self.crop_ontology.keys())[:5],
            "timestamp_utc": datetime.datetime.utcnow().isoformat(),
            "field_id": self.field_id,
        }

        print(f"[Context Join] Context ready | "
              f"DQ checks: {len(self.all_dq_reports)} | "
              f"Data quality: {self.data_quality_passed}")

        self.next(self.intent_recognition)

    # ─────────────────────────────────────────────────────────────────
    # STEP 4: Intent Recognition & Confidence Scoring
    # ─────────────────────────────────────────────────────────────────
    @card
    @step
    def intent_recognition(self):
        """
        Classify the user's natural language query into one of five
        agricultural intent classes:
          QUERY | COMMAND | ABORT | REPORT | DIAGNOSE

        Uses: fine-tuned Llama-3-8B-Instruct (cloud) or quantized
              Mistral-7B-GGUF (edge) depending on self.edge_mode.

        Produces a confidence score using entropy-based calibration
        (inspired by the BVF / conformal prediction approach from
        the candidate's existing research).

        Metaflow concept: conditional branching based on confidence.
        PhD task: Task 2 — Intent Recognition.
        """
        print(f"[Intent] Classifying: '{self.sanitized_query}'")
        print(f"[Intent] Mode: {'EDGE (quantized)' if self.edge_mode else 'CLOUD'}")

        t0 = time.time()

        if self.edge_mode:
            intent_result = _edge_intent_classify(
                self.sanitized_query, self.inference_context
            )
        else:
            intent_result = _cloud_intent_classify(
                self.sanitized_query, self.inference_context
            )

        self.intent_class = intent_result["intent"]
        self.intent_confidence = intent_result["confidence"]
        self.intent_slots = intent_result["slots"]
        self.intent_entropy = intent_result["entropy"]
        self.intent_all_probs = intent_result["all_probs"]

        latency_ms = (time.time() - t0) * 1000
        self.intent_latency_ms = latency_ms

        print(f"[Intent] Class: {self.intent_class} | "
              f"Confidence: {self.intent_confidence:.3f} | "
              f"Entropy: {self.intent_entropy:.3f} | "
              f"Latency: {latency_ms:.1f}ms")
        print(f"[Intent] Slots: {self.intent_slots}")

        # Check latency budget
        if latency_ms > EDGE_LATENCY_BUDGET_MS:
            print(f"[Intent] WARNING: Latency {latency_ms:.1f}ms exceeds budget "
                  f"{EDGE_LATENCY_BUDGET_MS}ms")

        self.next(self.semantic_grounding)

    # ─────────────────────────────────────────────────────────────────
    # STEP 5: Semantic Grounding & Command Compilation
    # ─────────────────────────────────────────────────────────────────
    @card
    @step
    def semantic_grounding(self):
        """
        Ground the extracted slots against domain knowledge bases
        (EPPO pesticide DB, AGROVOC crop ontology, robot registry)
        to produce a structured, executable command object.

        Implements RAG-style grounding: every entity in the compiled
        command must be traceable to a KB source. Unverified entities
        are flagged with [UNVERIFIED] and blocked from actuation.

        Metaflow concept: linear step with rich artifact production.
        PhD task: Task 2 — Semantic Grounding.
        """
        print(f"[Grounding] Grounding intent '{self.intent_class}' "
              f"with slots: {self.intent_slots}")

        grounding_result = _ground_intent_to_command(
            intent_class=self.intent_class,
            slots=self.intent_slots,
            field_state=self.annotated_field_state,
            pesticide_kb=self.pesticide_kb,
            robot_capabilities=self.robot_capability_registry,
            crop_ontology=self.crop_ontology,
        )

        self.compiled_command = grounding_result["command"]
        self.grounding_verified = grounding_result["all_entities_verified"]
        self.grounding_warnings = grounding_result["warnings"]
        self.unverified_entities = grounding_result["unverified_entities"]
        self.action_type = grounding_result["action_type"]
        self.natural_language_explanation = grounding_result["nl_explanation"]

        print(f"[Grounding] Compiled command: {json.dumps(self.compiled_command, indent=2)}")
        print(f"[Grounding] All entities verified: {self.grounding_verified}")
        if self.grounding_warnings:
            print(f"[Grounding] Warnings: {self.grounding_warnings}")

        self.next(self.safety_gate)

    # ─────────────────────────────────────────────────────────────────
    # STEP 6: Safety Gate — HITL Confirmation Logic
    # ─────────────────────────────────────────────────────────────────
    @card
    @step
    def safety_gate(self):
        """
        Responsible AI gate: determines whether the command requires
        mandatory human-in-the-loop (HITL) confirmation before execution.

        HITL is mandatory when ANY of:
          1. Action type is safety-critical (SPRAY, ABORT_MISSION, etc.)
          2. Confidence < threshold (calibrated uncertainty)
          3. Any entity in command is [UNVERIFIED] (hallucination guard)
          4. User role is OBSERVER (read-only permissions)
          5. Data quality check failed upstream

        This implements the Responsible AI framework from Section 6 of the proposal.

        Metaflow concept: conditional routing based on artifact values.
        PhD task: Task 5 — Responsible AI Audit.
        """
        hitl_required = False
        hitl_reasons = []

        # Check 1: Safety-critical action type
        if self.action_type in SAFETY_CRITICAL_ACTIONS:
            hitl_required = True
            hitl_reasons.append(f"Safety-critical action: {self.action_type}")

        # Check 2: Low confidence (uncertainty threshold)
        if self.intent_confidence < self.confidence_threshold:
            hitl_required = True
            hitl_reasons.append(
                f"Confidence {self.intent_confidence:.3f} < threshold {self.confidence_threshold}"
            )

        # Check 3: Unverified entities (hallucination guard)
        if self.unverified_entities:
            hitl_required = True
            hitl_reasons.append(
                f"Unverified entities: {self.unverified_entities}"
            )

        # Check 4: Role-based permission check
        if self.user_role == "OBSERVER":
            hitl_required = True
            hitl_reasons.append("OBSERVER role: read-only access, commands blocked")

        # Check 5: Upstream data quality failure
        if not self.data_quality_passed:
            hitl_required = True
            hitl_reasons.append("Data quality check failed — field state unreliable")

        self.hitl_required = hitl_required
        self.hitl_reasons = hitl_reasons

        # Confidence tier for user display
        if self.intent_confidence >= 0.90:
            self.confidence_tier = "HIGH"
        elif self.intent_confidence >= self.confidence_threshold:
            self.confidence_tier = "MEDIUM"
        else:
            self.confidence_tier = "LOW"

        print(f"[Safety Gate] HITL required: {hitl_required}")
        if hitl_reasons:
            for reason in hitl_reasons:
                print(f"  ↳ {reason}")
        print(f"[Safety Gate] Confidence tier: {self.confidence_tier}")

        # Branch: HITL path vs. autonomous path
        if self.hitl_required:
            self.next(self.request_human_confirmation)
        else:
            self.next(self.autonomous_execute)

    # ─────────────────────────────────────────────────────────────────
    # STEP 7a: HITL Confirmation Request (CONDITIONAL BRANCH — HITL)
    # ─────────────────────────────────────────────────────────────────
    @card
    @step
    def request_human_confirmation(self):
        """
        Formats and presents a structured confirmation dialogue to the
        human operator. In real deployment this publishes a message to
        the AgriTalk mobile app / web dashboard.

        For the prototype/dry_run, simulates operator response.

        Contains: natural language explanation, compiled command,
        uncertainty score, HITL reasons, and explicit YES/NO/MODIFY options.

        PhD task: Task 4 & 5 — HITL integration.
        """
        print("[HITL] Preparing confirmation request for human operator...")

        # Format the confirmation message
        self.confirmation_message = _format_confirmation_dialogue(
            nl_explanation=self.natural_language_explanation,
            compiled_command=self.compiled_command,
            confidence_tier=self.confidence_tier,
            confidence_score=self.intent_confidence,
            hitl_reasons=self.hitl_reasons,
            field_state=self.annotated_field_state,
        )

        print("\n" + "="*60)
        print("AGRITALK — HUMAN CONFIRMATION REQUIRED")
        print("="*60)
        print(self.confirmation_message)
        print("="*60 + "\n")

        # Simulate human response (in production: wait for mobile app response)
        if self.dry_run:
            self.human_decision = _simulate_human_response(
                self.intent_confidence, self.hitl_reasons
            )
        else:
            # Production: read from confirmation queue (e.g., AWS SQS)
            self.human_decision = _await_confirmation_from_queue(
                run_id=current.run_id,
                timeout_seconds=120,
            )

        print(f"[HITL] Human decision: {self.human_decision['decision']}")
        if self.human_decision.get("modification"):
            print(f"[HITL] Operator modification: {self.human_decision['modification']}")

        # Apply any operator modifications to the command
        if self.human_decision["decision"] == "MODIFY":
            self.compiled_command = _apply_operator_modification(
                self.compiled_command, self.human_decision["modification"]
            )
            print(f"[HITL] Modified command: {self.compiled_command}")

        self.next(self.execute_or_abort)

    # ─────────────────────────────────────────────────────────────────
    # STEP 7b: Autonomous Execution (CONDITIONAL BRANCH — AUTO)
    # ─────────────────────────────────────────────────────────────────
    @card
    @step
    def autonomous_execute(self):
        """
        Commands that passed all safety gate checks execute autonomously.
        This branch is only reachable for QUERY/REPORT/DIAGNOSE intent
        with high confidence and verified entities.

        No irreversible robotic actions are ever in this branch.

        PhD task: Task 4 — Low-risk autonomous orchestration.
        """
        print(f"[Auto] Executing autonomously: {self.intent_class}")

        # Autonomous path only for non-actuation intents
        if self.intent_class not in {"QUERY", "REPORT", "DIAGNOSE"}:
            raise RuntimeError(
                f"BUG: Non-query intent '{self.intent_class}' reached autonomous branch. "
                "Safety gate logic error."
            )

        self.execution_result = _execute_query_command(
            compiled_command=self.compiled_command,
            field_state=self.annotated_field_state,
        )
        self.execution_status = "SUCCESS"
        self.human_decision = {"decision": "AUTONOMOUS", "modification": None}

        print(f"[Auto] Result: {self.execution_result}")
        self.next(self.finalize_response)

    # ─────────────────────────────────────────────────────────────────
    # STEP 8: Execute or Abort (HITL path resolution)
    # ─────────────────────────────────────────────────────────────────
    @card
    @step
    def execute_or_abort(self):
        """
        Based on human decision (YES / NO / MODIFY), either execute
        the (potentially modified) command or abort with explanation.

        In production: publishes approved commands to ROS2 topic via
        the AgriTalk ROS2 bridge node.

        PhD task: Task 4 — Robotic command publication.
        """
        decision = self.human_decision["decision"]
        print(f"[Execute] Processing decision: {decision}")

        if decision in {"YES", "MODIFY"}:
            # Execute approved command
            if self.dry_run:
                self.execution_result = _simulate_robot_actuation(
                    self.compiled_command, self.field_id
                )
            else:
                self.execution_result = _publish_to_ros2(
                    self.compiled_command, self.field_id
                )
            self.execution_status = "SUCCESS"
            print(f"[Execute] Command executed: {self.execution_result}")

        elif decision == "NO":
            self.execution_result = {
                "status": "ABORTED_BY_HUMAN",
                "reason": "Operator declined command",
                "command": self.compiled_command,
            }
            self.execution_status = "ABORTED"
            print(f"[Execute] Command aborted by operator")

        else:
            self.execution_result = {
                "status": "ABORTED_TIMEOUT",
                "reason": "Confirmation timeout — no operator response",
                "command": self.compiled_command,
            }
            self.execution_status = "TIMEOUT"
            print(f"[Execute] Command aborted — confirmation timeout")

        self.next(self.finalize_response)

    # ─────────────────────────────────────────────────────────────────
    # STEP 9: Finalise Natural Language Response
    # ─────────────────────────────────────────────────────────────────
    @card
    @step
    def finalize_response(self):
        """
        Generate the final natural language response to the user,
        incorporating execution result, confidence tier, and
        any audit-relevant information.

        All branches join here — this step always executes regardless
        of whether HITL or autonomous path was taken.

        Metaflow concept: join after conditional branches.
        """
        self.nl_response = _generate_nl_response(
            intent_class=self.intent_class,
            execution_result=self.execution_result,
            execution_status=self.execution_status,
            confidence_tier=self.confidence_tier,
            natural_language_explanation=self.natural_language_explanation,
            field_state=self.annotated_field_state,
            hitl_reasons=self.hitl_reasons,
            human_decision=self.human_decision,
        )

        print("\n" + "="*60)
        print("AGRITALK RESPONSE")
        print("="*60)
        print(self.nl_response)
        print("="*60 + "\n")

        self.next(self.audit_and_version)

    # ─────────────────────────────────────────────────────────────────
    # STEP 10: Audit Trail & Metaflow Versioning
    # ─────────────────────────────────────────────────────────────────
    @card
    @step
    def audit_and_version(self):
        """
        Produce a complete, immutable audit record of the full pipeline
        execution. This record is stored as a versioned Metaflow artifact
        and optionally exported to a DAG Card.

        The audit record constitutes the traceability trail required by:
        - Responsible AI safeguard 5 (proposal Section 6)
        - EU AI Act Art. 13 (transparency for high-risk AI systems)
        - GreenFieldData ethics framework

        Metaflow concept: artifact versioning, @card documentation.
        PhD task: Task 5 — Responsible AI Audit.
        """
        # Complete audit record
        self.full_audit_record = {
            **self.audit_record,
            "intent_classification": {
                "class": self.intent_class,
                "confidence": self.intent_confidence,
                "confidence_tier": self.confidence_tier,
                "entropy": self.intent_entropy,
                "slots": self.intent_slots,
                "latency_ms": self.intent_latency_ms,
            },
            "semantic_grounding": {
                "compiled_command": self.compiled_command,
                "action_type": self.action_type,
                "all_entities_verified": self.grounding_verified,
                "warnings": self.grounding_warnings,
                "unverified_entities": self.unverified_entities,
            },
            "safety_gate": {
                "hitl_required": self.hitl_required,
                "hitl_reasons": self.hitl_reasons,
            },
            "human_decision": self.human_decision,
            "execution": {
                "status": self.execution_status,
                "result": self.execution_result,
            },
            "nl_response": self.nl_response,
            "data_quality_passed": self.data_quality_passed,
            "steps_completed": [
                "start", "ingest_iot_streams", "load_knowledge_base",
                "join_context", "intent_recognition", "semantic_grounding",
                "safety_gate",
                "request_human_confirmation" if self.hitl_required else "autonomous_execute",
                "execute_or_abort" if self.hitl_required else "finalize_response",
                "finalize_response", "audit_and_version",
            ],
            "timestamp_complete_utc": datetime.datetime.utcnow().isoformat(),
        }

        # Compute audit hash for integrity verification
        audit_json = json.dumps(self.full_audit_record, sort_keys=True)
        self.audit_hash = hashlib.sha256(audit_json.encode()).hexdigest()
        print(f"[Audit] Audit record created | SHA256: {self.audit_hash[:16]}...")
        print(f"[Audit] Execution status: {self.execution_status}")
        print(f"[Audit] Run ID: {current.run_id} — all artifacts versioned in Metaflow store")

        self.next(self.evaluate_pipeline)

    # ─────────────────────────────────────────────────────────────────
    # STEP 11: Pipeline Evaluation & Metrics Collection
    # ─────────────────────────────────────────────────────────────────
    @card
    @step
    def evaluate_pipeline(self):
        """
        Compute and accumulate evaluation metrics for this pipeline run.
        These metrics feed into the longitudinal evaluation study (Task 5).

        Metrics collected:
        - Intent classification accuracy (vs. ground truth if available)
        - End-to-end latency
        - Confidence calibration score
        - Safety gate trigger rate
        - HITL acceptance rate

        In production, metrics are pushed to a monitoring dashboard
        (e.g., Prometheus + Grafana or AWS CloudWatch).

        PhD task: Task 5 — Evaluation Framework.
        """
        print("[Eval] Computing pipeline metrics...")

        # End-to-end latency
        self.pipeline_metrics = {
            "run_id": current.run_id,
            "field_id": self.field_id,
            "scenario": self.scenario,
            "intent_class": self.intent_class,
            "confidence": self.intent_confidence,
            "confidence_tier": self.confidence_tier,
            "intent_entropy": self.intent_entropy,
            "intent_latency_ms": self.intent_latency_ms,
            "hitl_triggered": self.hitl_required,
            "execution_status": self.execution_status,
            "grounding_verified": self.grounding_verified,
            "action_type": self.action_type,
            "data_quality_passed": self.data_quality_passed,
            "edge_mode": self.edge_mode,
        }

        # Safety check: log if latency exceeded budget
        if self.intent_latency_ms > EDGE_LATENCY_BUDGET_MS:
            self.pipeline_metrics["latency_budget_exceeded"] = True
            print(f"[Eval] ⚠ Latency budget exceeded: {self.intent_latency_ms:.1f}ms "
                  f"> {EDGE_LATENCY_BUDGET_MS}ms")
        else:
            self.pipeline_metrics["latency_budget_exceeded"] = False
            print(f"[Eval] ✓ Latency within budget: {self.intent_latency_ms:.1f}ms")

        print(f"[Eval] Metrics: {json.dumps(self.pipeline_metrics, indent=2)}")

        self.next(self.end)

    # ─────────────────────────────────────────────────────────────────
    # STEP 12: END
    # ─────────────────────────────────────────────────────────────────
    @card
    @step
    def end(self):
        """
        Pipeline complete. All artifacts are versioned and accessible
        via Metaflow Client API for:
        - Cross-run comparison and trend analysis
        - Resume after failure
        - Supervisor review and thesis documentation
        - Production monitoring

        Usage examples:
          from metaflow import Flow, Run
          run = Run('AgriTalkFlow/{run_id}')
          print(run.data.full_audit_record)
          print(run.data.pipeline_metrics)
        """
        print("\n[AgriTalk] Pipeline complete!")
        print(f"  Run ID       : {current.run_id}")
        print(f"  Field        : {self.field_id}")
        print(f"  Intent       : {self.intent_class} ({self.confidence_tier})")
        print(f"  Status       : {self.execution_status}")
        print(f"  HITL invoked : {self.hitl_required}")
        print(f"  Audit hash   : {self.audit_hash[:16]}...")
        print(f"\n  All artifacts versioned in Metaflow datastore.")
        print(f"  Access with: Run('AgriTalkFlow/{current.run_id}').data\n")


# ─────────────────────────────────────────────────────────────────────
# Helper Functions (Modular — replace with real implementations)
# ─────────────────────────────────────────────────────────────────────

def _sanitize_nl_input(query: str) -> str:
    """
    Remove potential prompt injection patterns from user input.
    OWASP A03 — Injection prevention.
    """
    import re
    # Remove common injection patterns
    injection_patterns = [
        r"ignore previous instructions",
        r"system:",
        r"<\|.*?\|>",
        r"\[INST\]",
        r"### System",
    ]
    sanitized = query
    for pattern in injection_patterns:
        sanitized = re.sub(pattern, "[FILTERED]", sanitized, flags=re.IGNORECASE)
    # Truncate to reasonable length
    return sanitized[:500]


def _simulate_field_state(field_id: str) -> Dict:
    """Simulate IoT sensor readings for a given field."""
    random.seed(hash(field_id) % 2**32)
    return {
        "field_id": field_id,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "soil_moisture_percent": round(random.uniform(20.0, 65.0), 2),
        "soil_ph": round(random.uniform(5.5, 7.5), 2),
        "soil_temperature_c": round(random.uniform(12.0, 28.0), 2),
        "air_temperature_c": round(random.uniform(10.0, 35.0), 2),
        "wind_speed_ms": round(random.uniform(0.0, 8.0), 2),
        "wind_direction_deg": round(random.uniform(0, 360), 1),
        "relative_humidity_percent": round(random.uniform(40.0, 90.0), 2),
        "precipitation_mm": round(random.uniform(0.0, 5.0), 2),
        "ndvi_index": round(random.uniform(0.3, 0.85), 3),  # drone multispectral
        "robot_gps_lat": 45.4654 + random.uniform(-0.01, 0.01),
        "robot_gps_lon": 9.1859 + random.uniform(-0.01, 0.01),
        "robot_battery_percent": round(random.uniform(30.0, 100.0), 1),
        "robot_status": random.choice(["IDLE", "ACTIVE", "CHARGING"]),
        "pest_risk_score": round(random.uniform(0.0, 1.0), 3),
    }


def _annotate_field_state(raw_state: Dict) -> Dict:
    """Apply AGROVOC ontology annotations to raw sensor readings."""
    annotated = dict(raw_state)

    # Classify soil moisture into AGROVOC categories
    moisture = raw_state["soil_moisture_percent"]
    if moisture < 30:
        annotated["soil_moisture_category"] = "AGROVOC:DRY"
        annotated["spray_recommendation"] = "HOLD — soil too dry, runoff risk"
    elif moisture < 50:
        annotated["soil_moisture_category"] = "AGROVOC:OPTIMAL"
        annotated["spray_recommendation"] = "PROCEED — conditions optimal"
    else:
        annotated["soil_moisture_category"] = "AGROVOC:WET"
        annotated["spray_recommendation"] = "HOLD — soil saturated, leaching risk"

    # Wind speed safety classification
    wind = raw_state["wind_speed_ms"]
    if wind < 3.0:
        annotated["wind_safety"] = "SAFE"
    elif wind < 6.0:
        annotated["wind_safety"] = "MARGINAL — monitor closely"
    else:
        annotated["wind_safety"] = "UNSAFE — spray drift risk"

    # NDVI crop health status
    ndvi = raw_state["ndvi_index"]
    if ndvi < 0.4:
        annotated["crop_health"] = "STRESSED"
    elif ndvi < 0.7:
        annotated["crop_health"] = "MODERATE"
    else:
        annotated["crop_health"] = "HEALTHY"

    return annotated


def _validate_field_data_quality(state: Dict) -> Dict:
    """
    Great Expectations-style data quality checks on field state.
    Returns pass/fail report.
    """
    checks = []

    # Check 1: Sensor readings within physical bounds
    checks.append({
        "check": "soil_moisture_in_range",
        "passed": 0 <= state["soil_moisture_percent"] <= 100,
    })
    checks.append({
        "check": "wind_speed_nonnegative",
        "passed": state["wind_speed_ms"] >= 0,
    })
    checks.append({
        "check": "ndvi_in_range",
        "passed": 0 <= state["ndvi_index"] <= 1.0,
    })
    checks.append({
        "check": "robot_battery_valid",
        "passed": 0 <= state["robot_battery_percent"] <= 100,
    })
    checks.append({
        "check": "timestamp_present",
        "passed": "timestamp" in state and state["timestamp"] is not None,
    })

    all_passed = all(c["passed"] for c in checks)
    return {
        "all_checks_passed": all_passed,
        "checks": checks,
        "n_passed": sum(c["passed"] for c in checks),
        "n_total": len(checks),
    }


def _load_pesticide_kb() -> Dict:
    """Load EPPO/PPDB pesticide knowledge base (stub — replace with real DB)."""
    return {
        "glyphosate": {
            "max_dose_l_ha": 4.0, "withholding_days": 7,
            "wind_limit_ms": 4.0, "source": "PPDB",
        },
        "lambda_cyhalothrin": {
            "max_dose_l_ha": 0.1, "withholding_days": 14,
            "wind_limit_ms": 3.0, "source": "PPDB",
        },
        "copper_hydroxide": {
            "max_dose_kg_ha": 3.0, "withholding_days": 3,
            "wind_limit_ms": 5.0, "source": "EPPO",
        },
        "spinosad": {
            "max_dose_l_ha": 0.2, "withholding_days": 3,
            "wind_limit_ms": 4.0, "source": "PPDB",
        },
        "neem_oil": {
            "max_dose_l_ha": 5.0, "withholding_days": 1,
            "wind_limit_ms": 5.0, "source": "EPPO",
        },
    }


def _load_crop_ontology() -> Dict:
    """Load AGROVOC crop ontology (stub)."""
    return {
        "wheat": {"latin": "Triticum aestivum", "agrovoc_id": "c_8373"},
        "maize": {"latin": "Zea mays", "agrovoc_id": "c_12332"},
        "tomato": {"latin": "Solanum lycopersicum", "agrovoc_id": "c_7805"},
        "grape": {"latin": "Vitis vinifera", "agrovoc_id": "c_3399"},
        "apple": {"latin": "Malus domestica", "agrovoc_id": "c_468"},
    }


def _load_robot_capabilities() -> Dict:
    """Load robot capability registry (stub)."""
    return {
        "SPRAY": {"robot_type": "sprayer", "requires_confirmation": True},
        "NAVIGATE": {"robot_type": "ground", "requires_confirmation": False},
        "PHOTOGRAPH": {"robot_type": "drone", "requires_confirmation": False},
        "ABORT_MISSION": {"robot_type": "any", "requires_confirmation": True},
        "REPORT_STATUS": {"robot_type": "any", "requires_confirmation": False},
        "DOSAGE_CHANGE": {"robot_type": "sprayer", "requires_confirmation": True},
    }


def _cloud_intent_classify(query: str, context: Dict) -> Dict:
    """
    Intent classification via cloud LLM (Llama-3-8B-Instruct or Mistral).
    In prototype: uses rule-based simulation.
    In production: POST to inference endpoint (SageMaker/vLLM).
    """
    # Prototype simulation: keyword-based classification
    query_lower = query.lower()

    scores = {
        "QUERY": 0.1,
        "COMMAND": 0.1,
        "ABORT": 0.1,
        "REPORT": 0.1,
        "DIAGNOSE": 0.1,
    }

    if any(w in query_lower for w in ["how much", "what", "when", "show", "tell me", "status"]):
        scores["QUERY"] += 0.6
    if any(w in query_lower for w in ["spray", "apply", "start", "deploy", "go to", "begin"]):
        scores["COMMAND"] += 0.65
    if any(w in query_lower for w in ["stop", "abort", "cancel", "halt", "emergency"]):
        scores["ABORT"] += 0.7
    if any(w in query_lower for w in ["report", "summary", "log", "record"]):
        scores["REPORT"] += 0.6
    if any(w in query_lower for w in ["diagnose", "problem", "error", "fault", "wrong", "fail"]):
        scores["DIAGNOSE"] += 0.65

    # Normalize to probabilities
    total = sum(scores.values())
    probs = {k: v / total for k, v in scores.items()}

    # Get top intent
    intent = max(probs, key=probs.get)
    confidence = probs[intent]

    # Entropy (uncertainty)
    import math
    entropy = -sum(p * math.log(p + 1e-9) for p in probs.values())

    # Extract slots (simplified)
    slots = _extract_slots(query_lower)

    return {
        "intent": intent,
        "confidence": confidence,
        "entropy": entropy,
        "slots": slots,
        "all_probs": probs,
    }


def _edge_intent_classify(query: str, context: Dict) -> Dict:
    """
    Edge-mode intent classification via quantized local model.
    Uses GGUF 4-bit quantized Mistral-7B on NVIDIA Jetson.
    Prototype: same simulation as cloud with added latency.
    """
    # Simulate slightly lower accuracy but faster latency
    result = _cloud_intent_classify(query, context)
    # Add small noise to simulate quantization effect
    result["confidence"] = max(0.3, result["confidence"] - random.uniform(0, 0.05))
    return result


def _extract_slots(query: str) -> Dict:
    """Extract named entities / slots from NL query."""
    slots = {}

    # Pesticide name detection
    for pesticide in ["glyphosate", "copper_hydroxide", "lambda_cyhalothrin",
                      "spinosad", "neem_oil", "pesticide"]:
        if pesticide in query:
            slots["pesticide"] = pesticide
            break

    # Dosage extraction (e.g., "2 litres per hectare")
    import re
    dose_match = re.search(r"(\d+\.?\d*)\s*(l|litre|liter|kg|gram).*?(ha|hectare)?", query)
    if dose_match:
        slots["dosage_value"] = float(dose_match.group(1))
        slots["dosage_unit"] = dose_match.group(2)

    # Field/zone extraction
    if "northern" in query or "north" in query:
        slots["zone"] = "NORTH"
    elif "southern" in query or "south" in query:
        slots["zone"] = "SOUTH"
    elif "eastern" in query or "east" in query:
        slots["zone"] = "EAST"
    elif "western" in query or "west" in query:
        slots["zone"] = "WEST"

    # Temporal extraction
    if "yesterday" in query:
        slots["time_ref"] = "yesterday"
    elif "today" in query:
        slots["time_ref"] = "today"
    elif "now" in query or "immediately" in query:
        slots["time_ref"] = "now"

    # Condition triggers
    if "humidity" in query or "moisture" in query:
        slots["trigger_condition"] = "soil_moisture"
    if "wind" in query:
        slots["trigger_condition"] = "wind_speed"

    return slots


def _ground_intent_to_command(
    intent_class: str,
    slots: Dict,
    field_state: Dict,
    pesticide_kb: Dict,
    robot_capabilities: Dict,
    crop_ontology: Dict,
) -> Dict:
    """
    Ground extracted slots against knowledge bases and compile
    a structured executable command object.
    """
    warnings = []
    unverified_entities = []
    all_verified = True

    # Determine action type from intent
    action_map = {
        "COMMAND": "SPRAY",
        "ABORT": "ABORT_MISSION",
        "QUERY": "REPORT_STATUS",
        "REPORT": "REPORT_STATUS",
        "DIAGNOSE": "REPORT_STATUS",
    }
    action_type = action_map.get(intent_class, "REPORT_STATUS")

    # Compile command object
    command = {
        "action": action_type,
        "field_id": slots.get("zone", "FULL_FIELD"),
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }

    # Ground pesticide entity
    pesticide_name = slots.get("pesticide", "unspecified")
    if pesticide_name != "unspecified":
        if pesticide_name in pesticide_kb:
            kb_entry = pesticide_kb[pesticide_name]
            command["pesticide"] = pesticide_name
            command["max_dose_l_ha"] = kb_entry.get("max_dose_l_ha", "N/A")
            command["wind_limit_ms"] = kb_entry.get("wind_limit_ms", "N/A")
            command["source"] = kb_entry["source"]

            # Wind safety check
            current_wind = field_state.get("wind_speed_ms", 0)
            wind_limit = kb_entry.get("wind_limit_ms", 999)
            if current_wind > wind_limit:
                warnings.append(
                    f"Wind speed {current_wind}m/s exceeds {pesticide_name} limit {wind_limit}m/s"
                )
        else:
            unverified_entities.append(pesticide_name)
            command["pesticide"] = f"[UNVERIFIED: {pesticide_name}]"
            all_verified = False
    else:
        if action_type == "SPRAY":
            warnings.append("No pesticide specified for SPRAY command")

    # Ground dosage
    if "dosage_value" in slots:
        dosage = slots["dosage_value"]
        max_dose = command.get("max_dose_l_ha")
        if max_dose and max_dose != "N/A" and dosage > max_dose:
            warnings.append(
                f"Requested dose {dosage} L/ha exceeds PPDB maximum {max_dose} L/ha"
            )
            command["dosage_l_ha"] = max_dose  # Cap at maximum
            command["dosage_capped"] = True
        else:
            command["dosage_l_ha"] = dosage

    # Field soil condition check
    spray_rec = field_state.get("spray_recommendation", "")
    if "HOLD" in spray_rec and action_type == "SPRAY":
        warnings.append(f"Field condition advisory: {spray_rec}")

    # NL explanation
    nl_explanation = _build_nl_explanation(
        intent_class, command, warnings, field_state, slots
    )

    return {
        "command": command,
        "action_type": action_type,
        "all_entities_verified": all_verified,
        "warnings": warnings,
        "unverified_entities": unverified_entities,
        "nl_explanation": nl_explanation,
    }


def _build_nl_explanation(
    intent: str, command: Dict, warnings: List, field_state: Dict, slots: Dict
) -> str:
    """Generate natural language explanation of the compiled command."""
    lines = []

    if intent == "COMMAND":
        pesticide = command.get("pesticide", "the requested pesticide")
        zone = command.get("field_id", "the full field")
        lines.append(f"I understood your request as: apply {pesticide} to {zone}.")
        if command.get("dosage_l_ha"):
            lines.append(f"Proposed dosage: {command['dosage_l_ha']} L/ha (PPDB source).")
        if command.get("dosage_capped"):
            lines.append(f"⚠ Dosage was capped at the PPDB maximum for safety.")

    elif intent == "QUERY":
        lines.append("I understood your request as a field status query.")
        lines.append(f"Current soil moisture: {field_state.get('soil_moisture_percent', 'N/A')}%")
        lines.append(f"Current wind speed: {field_state.get('wind_speed_ms', 'N/A')} m/s")
        lines.append(f"Spray recommendation: {field_state.get('spray_recommendation', 'N/A')}")

    elif intent == "ABORT":
        lines.append("I understood your request as an ABORT MISSION command.")
        lines.append("All active robot missions will be halted immediately if confirmed.")

    elif intent == "DIAGNOSE":
        lines.append("I understood your request as a robot diagnostics query.")
        lines.append(f"Robot status: {field_state.get('robot_status', 'UNKNOWN')}")
        lines.append(f"Battery level: {field_state.get('robot_battery_percent', 'N/A')}%")

    if warnings:
        lines.append("\n⚠ Warnings:")
        for w in warnings:
            lines.append(f"  - {w}")

    return "\n".join(lines)


def _format_confirmation_dialogue(
    nl_explanation: str,
    compiled_command: Dict,
    confidence_tier: str,
    confidence_score: float,
    hitl_reasons: List,
    field_state: Dict,
) -> str:
    """Format the HITL confirmation message for human operator."""
    tier_icon = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(confidence_tier, "⚪")
    lines = [
        f"System confidence: {tier_icon} {confidence_tier} ({confidence_score:.1%})",
        "",
        "PROPOSED ACTION:",
        nl_explanation,
        "",
        "CURRENT FIELD CONDITIONS:",
        f"  Soil moisture: {field_state.get('soil_moisture_percent', 'N/A')}%  "
        f"[{field_state.get('soil_moisture_category', 'N/A')}]",
        f"  Wind speed: {field_state.get('wind_speed_ms', 'N/A')} m/s  "
        f"[{field_state.get('wind_safety', 'N/A')}]",
        f"  Crop health: {field_state.get('crop_health', 'N/A')}",
        "",
    ]
    if hitl_reasons:
        lines.append("CONFIRMATION REQUIRED BECAUSE:")
        for r in hitl_reasons:
            lines.append(f"  ⚠ {r}")
        lines.append("")
    lines.append("Please respond: YES (execute) | NO (cancel) | MODIFY (adjust parameters)")
    return "\n".join(lines)


def _simulate_human_response(confidence: float, hitl_reasons: List) -> Dict:
    """
    Simulate human operator response for dry_run mode.
    In production: reads from confirmation queue (AWS SQS / MQTT topic).
    """
    # High-confidence commands are typically approved
    if confidence > 0.85 and not any("UNVERIFIED" in r for r in hitl_reasons):
        return {"decision": "YES", "modification": None, "response_time_s": 3.2}
    # Low-confidence or warnings → sometimes modified
    elif confidence > 0.6:
        return {
            "decision": "MODIFY",
            "modification": {"dosage_l_ha": 2.0, "operator_note": "Reduced dose"},
            "response_time_s": 8.7,
        }
    else:
        return {"decision": "NO", "reason": "Operator not satisfied with confidence", "response_time_s": 5.1}


def _await_confirmation_from_queue(run_id: str, timeout_seconds: int = 120) -> Dict:
    """
    Production: poll confirmation queue for human response.
    Replace with actual AWS SQS / MQTT consumer.
    """
    # Placeholder — in production: listen to SQS queue with run_id filter
    time.sleep(2)
    return {"decision": "YES", "modification": None, "response_time_s": 2.0}


def _apply_operator_modification(command: Dict, modification: Dict) -> Dict:
    """Apply operator-specified modifications to the compiled command."""
    modified = dict(command)
    modified.update(modification)
    modified["operator_modified"] = True
    return modified


def _execute_query_command(compiled_command: Dict, field_state: Dict) -> Dict:
    """Execute a non-actuation query command (QUERY/REPORT/DIAGNOSE)."""
    return {
        "status": "SUCCESS",
        "data": {
            "field_state_summary": {
                k: v for k, v in field_state.items()
                if k in ["soil_moisture_percent", "wind_speed_ms",
                         "spray_recommendation", "robot_status",
                         "pest_risk_score", "crop_health"]
            }
        },
        "command": compiled_command,
    }


def _simulate_robot_actuation(command: Dict, field_id: str) -> Dict:
    """Simulate robotic actuation (dry_run mode)."""
    return {
        "status": "SIMULATED",
        "message": f"[DRY RUN] Would have published to ROS2 topic /agritalk/{field_id}/command",
        "command_hash": hashlib.md5(
            json.dumps(command, sort_keys=True).encode()
        ).hexdigest()[:8],
        "simulated_at": datetime.datetime.utcnow().isoformat(),
    }


def _publish_to_ros2(command: Dict, field_id: str) -> Dict:
    """
    Production: publish command to ROS2 topic via AgriTalk bridge node.
    Replace with actual rclpy publisher call.
    """
    # In production:
    # import rclpy
    # from agritalk_msgs.msg import RobotCommand
    # node.publish(f'/agritalk/{field_id}/command', ...)
    return {
        "status": "PUBLISHED",
        "topic": f"/agritalk/{field_id}/command",
        "published_at": datetime.datetime.utcnow().isoformat(),
    }


def _generate_nl_response(
    intent_class: str,
    execution_result: Dict,
    execution_status: str,
    confidence_tier: str,
    natural_language_explanation: str,
    field_state: Dict,
    hitl_reasons: List,
    human_decision: Dict,
) -> str:
    """Generate the final NL response displayed to the user."""
    status_map = {
        "SUCCESS": "✓ Action completed successfully.",
        "SIMULATED": "✓ Action simulated (dry run mode).",
        "ABORTED": "✗ Action cancelled by operator.",
        "TIMEOUT": "✗ Action cancelled — no response received within 2 minutes.",
        "AUTONOMOUS": "✓ Query answered autonomously.",
    }

    lines = [
        "AgriTalk Response",
        "─" * 40,
        status_map.get(execution_status, f"Status: {execution_status}"),
        "",
        natural_language_explanation,
    ]

    if execution_result.get("data"):
        lines.append("\nField data summary:")
        for k, v in execution_result["data"].get("field_state_summary", {}).items():
            lines.append(f"  {k}: {v}")

    if human_decision["decision"] == "MODIFY":
        lines.append(f"\nNote: Command was adjusted by operator: "
                     f"{human_decision.get('modification', {})}")

    lines.append(f"\nConfidence: {confidence_tier}")
    lines.append(f"All decisions and actions are logged for audit. "
                 f"Contact your agronomist to review this report.")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    AgriTalkFlow()
