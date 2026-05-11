"""
AgriTalk LLM Fine-tuning Flow (Task 2)
=======================================
Metaflow DAG for fine-tuning LLMs (Llama-3, Mistral) on the
agricultural intent corpus using LoRA/PEFT adapters.

Pipeline:
  1. Load & validate training corpus
  2. Parallel: tokenise + base model checks
  3. LoRA fine-tuning (GPU via @batch)
  4. Evaluation on held-out set
  5. Model registration & artifact storage
  6. Edge model quantisation (GGUF 4-bit)

Connects to: agritalk_flow.py (intent_recognition step)

Run locally (prototype / data check):
  python agritalk_finetune_flow.py run --model_name mistral-7b --dry_run True

Run with cloud GPU (production):
  python agritalk_finetune_flow.py run --with batch --max-workers 2
"""

import json
import os
import datetime
import random
from typing import Dict, List, Optional

from metaflow import (
    FlowSpec,
    step,
    Parameter,
    card,
    batch,
    conda,
    retry,
    timeout,
    current,
)

INTENT_CLASSES = ["QUERY", "COMMAND", "ABORT", "REPORT", "DIAGNOSE"]

# Target evaluation metrics
TARGET_F1 = 0.92
TARGET_CALIBRATION_ECE = 0.05


class AgriTalkFinetuneFlow(FlowSpec):
    """
    Metaflow DAG for fine-tuning agricultural NL intent classifiers.
    Implements Task 2 (Intent Recognition) and Task 3 (Metaflow MLOps Pipeline)
    from the GreenFieldData PhD proposal.

    This flow follows Fig. 1 of the MetaFlow paper: the same code runs
    locally for quick prototyping and on cloud GPU for full training.
    """

    model_name = Parameter(
        "model_name",
        help="Base model: mistral-7b | llama-3-8b | llama-3-8b-instruct",
        default="mistral-7b",
    )
    lora_rank = Parameter(
        "lora_rank",
        help="LoRA rank for PEFT fine-tuning",
        default=16,
        type=int,
    )
    lora_alpha = Parameter(
        "lora_alpha",
        help="LoRA alpha scaling factor",
        default=32,
        type=int,
    )
    epochs = Parameter(
        "epochs",
        help="Number of training epochs",
        default=3,
        type=int,
    )
    batch_size = Parameter(
        "batch_size",
        help="Training batch size per device",
        default=4,
        type=int,
    )
    learning_rate = Parameter(
        "learning_rate",
        help="Learning rate",
        default=2e-4,
        type=float,
    )
    dry_run = Parameter(
        "dry_run",
        help="If True, skip actual training; validate data pipeline only",
        default=True,
        type=bool,
    )
    corpus_path = Parameter(
        "corpus_path",
        help="Path to agricultural dialogue corpus (JSONL)",
        default="data/agritalk_corpus.jsonl",
    )
    experiment_tag = Parameter(
        "experiment_tag",
        help="Tag for this experiment run (e.g., 'baseline', 'v2_rag_augmented')",
        default="baseline",
    )

    # ─────────────────────────────────────────────────────────────────
    @card
    @step
    def start(self):
        """Validate parameters and initialise experiment tracking."""
        print(f"[Finetune] Experiment: {self.experiment_tag}")
        print(f"[Finetune] Model: {self.model_name} | LoRA rank: {self.lora_rank}")
        print(f"[Finetune] Epochs: {self.epochs} | LR: {self.learning_rate}")
        print(f"[Finetune] Dry run: {self.dry_run}")

        allowed_models = {"mistral-7b", "llama-3-8b", "llama-3-8b-instruct"}
        if self.model_name not in allowed_models:
            raise ValueError(f"Unknown model '{self.model_name}'")

        self.experiment_id = f"{self.experiment_tag}_{current.run_id}"
        self.start_time = datetime.datetime.utcnow().isoformat()

        self.next(self.load_corpus, self.validate_base_model)

    # ─────────────────────────────────────────────────────────────────
    @card
    @step
    def load_corpus(self):
        """
        Load, validate, and split the agricultural dialogue corpus.

        Corpus format (JSONL):
        {"text": "Spray glyphosate on north field", "label": "COMMAND",
         "slots": {"pesticide": "glyphosate", "zone": "NORTH"}, "split": "train"}

        Implements Task 2 corpus construction methodology.
        """
        print(f"[Corpus] Loading from: {self.corpus_path}")

        # Generate synthetic corpus for prototype (replace with real corpus)
        corpus = _generate_synthetic_corpus(n_samples=500)

        # Split 80/10/10
        random.shuffle(corpus)
        n = len(corpus)
        self.train_samples = corpus[:int(0.8 * n)]
        self.val_samples = corpus[int(0.8 * n):int(0.9 * n)]
        self.test_samples = corpus[int(0.9 * n):]

        # Class distribution check
        from collections import Counter
        train_dist = Counter(s["label"] for s in self.train_samples)
        print(f"[Corpus] Train: {len(self.train_samples)} | "
              f"Val: {len(self.val_samples)} | Test: {len(self.test_samples)}")
        print(f"[Corpus] Class distribution: {dict(train_dist)}")

        # Validate all samples have required fields
        for sample in corpus:
            assert "text" in sample, "Missing 'text' field"
            assert "label" in sample, "Missing 'label' field"
            assert sample["label"] in INTENT_CLASSES, \
                f"Unknown label: {sample['label']}"

        self.corpus_stats = {
            "total": n,
            "train": len(self.train_samples),
            "val": len(self.val_samples),
            "test": len(self.test_samples),
            "class_distribution": dict(train_dist),
        }
        print("[Corpus] Validation passed ✓")

        self.next(self.join_prep)

    # ─────────────────────────────────────────────────────────────────
    @card
    @step
    def validate_base_model(self):
        """
        Validate base model availability and compute baseline accuracy
        (zero-shot performance before fine-tuning).
        """
        print(f"[BaseModel] Validating {self.model_name}...")

        # Simulate baseline zero-shot performance
        self.baseline_metrics = _simulate_baseline_metrics(self.model_name)
        print(f"[BaseModel] Zero-shot F1: {self.baseline_metrics['macro_f1']:.3f}")
        print(f"[BaseModel] Per-class: {self.baseline_metrics['per_class_f1']}")

        self.next(self.join_prep)

    # ─────────────────────────────────────────────────────────────────
    @card
    @step
    def join_prep(self, inputs):
        """Join corpus loading and base model validation."""
        self.merge_artifacts(inputs, exclude=["baseline_metrics"])
        self.baseline_metrics = [
            i.baseline_metrics for i in inputs
            if hasattr(i, "baseline_metrics")
        ][0]

        print(f"[Prep] Ready to train on {self.corpus_stats['train']} samples")
        print(f"[Prep] Baseline macro-F1: {self.baseline_metrics['macro_f1']:.3f}")

        self.next(self.finetune_model)

    # ─────────────────────────────────────────────────────────────────
    @card
    @retry(times=1)
    # In production: @batch(gpu=1, memory=32000, image='pytorch/pytorch:2.3-cuda12.1')
    @step
    def finetune_model(self):
        """
        LoRA fine-tuning of the base LLM on the agricultural corpus.

        Production: runs on @batch GPU instance (AWS p3.2xlarge / p4d).
        Prototype: simulates training metrics.

        Uses PEFT/LoRA (Hu et al., 2022) for parameter-efficient adaptation.
        References: Hu et al. ICLR 2022; Alpaca (Taori et al., 2023).

        Metaflow concept: @batch decorator for cloud GPU compute.
        PhD task: Task 2 fine-tuning; Task 3 MLOps.
        """
        print(f"[Finetune] Starting LoRA fine-tuning...")
        print(f"  Model: {self.model_name}")
        print(f"  LoRA rank={self.lora_rank}, alpha={self.lora_alpha}")
        print(f"  Epochs: {self.epochs} | Batch size: {self.batch_size} | LR: {self.learning_rate}")
        print(f"  Training samples: {len(self.train_samples)}")

        if self.dry_run:
            print("[Finetune] DRY RUN — simulating training...")
            # Simulate training loop
            self.training_history = []
            for epoch in range(self.epochs):
                # Simulate improving loss
                train_loss = 1.5 - 0.4 * epoch + random.uniform(-0.05, 0.05)
                val_loss = 1.6 - 0.35 * epoch + random.uniform(-0.05, 0.08)
                val_f1 = 0.65 + 0.10 * epoch + random.uniform(-0.02, 0.02)
                self.training_history.append({
                    "epoch": epoch + 1,
                    "train_loss": round(train_loss, 4),
                    "val_loss": round(val_loss, 4),
                    "val_macro_f1": round(val_f1, 4),
                })
                print(f"  Epoch {epoch+1}/{self.epochs}: "
                      f"train_loss={train_loss:.4f} | "
                      f"val_loss={val_loss:.4f} | "
                      f"val_F1={val_f1:.4f}")

            # Simulate model checkpoint path
            self.model_checkpoint_path = (
                f"s3://agritalk-models/{self.model_name}/"
                f"lora_r{self.lora_rank}_e{self.epochs}_"
                f"{self.experiment_id}/checkpoint_final"
            )
            self.training_config = {
                "model_name": self.model_name,
                "lora_rank": self.lora_rank,
                "lora_alpha": self.lora_alpha,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
                "trainable_params": f"{self.lora_rank * 2 * 4 * 4096 * 1e-6:.1f}M",
            }
        else:
            # Production: real fine-tuning with HuggingFace + PEFT
            self.training_history, self.model_checkpoint_path, self.training_config = (
                _run_real_lora_finetuning(
                    model_name=self.model_name,
                    train_samples=self.train_samples,
                    val_samples=self.val_samples,
                    lora_rank=self.lora_rank,
                    lora_alpha=self.lora_alpha,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    learning_rate=self.learning_rate,
                )
            )

        print(f"[Finetune] Training complete!")
        print(f"[Finetune] Checkpoint: {self.model_checkpoint_path}")

        self.next(self.evaluate_model)

    # ─────────────────────────────────────────────────────────────────
    @card
    @step
    def evaluate_model(self):
        """
        Evaluate fine-tuned model on held-out test set.
        Computes: macro-F1, per-class F1, Expected Calibration Error (ECE),
        and confusion matrix.

        Also runs safety-specific evaluation:
        - ABORT/safety-critical recall (must be ≥ 0.99)
        - Confidence calibration (ECE < 0.05)

        PhD task: Task 5 — Evaluation Framework.
        """
        print(f"[Eval] Evaluating on {len(self.test_samples)} test samples...")

        # Simulate evaluation metrics (replace with real inference)
        self.test_metrics = _simulate_evaluation_metrics(
            self.model_name, self.lora_rank, self.epochs
        )

        print(f"[Eval] Macro-F1: {self.test_metrics['macro_f1']:.4f} "
              f"(target: ≥ {TARGET_F1})")
        print(f"[Eval] ECE: {self.test_metrics['ece']:.4f} "
              f"(target: < {TARGET_CALIBRATION_ECE})")
        print(f"[Eval] ABORT recall: {self.test_metrics['per_class_recall']['ABORT']:.4f} "
              f"(target: ≥ 0.99)")
        print(f"[Eval] Per-class F1: {self.test_metrics['per_class_f1']}")

        # Improvement over baseline
        baseline_f1 = self.baseline_metrics["macro_f1"]
        self.f1_improvement = self.test_metrics["macro_f1"] - baseline_f1
        print(f"[Eval] F1 improvement over zero-shot baseline: "
              f"+{self.f1_improvement:.4f}")

        # Safety check: ABORT recall must be near-perfect
        abort_recall = self.test_metrics["per_class_recall"]["ABORT"]
        if abort_recall < 0.99:
            print(f"[Eval] ⚠ WARNING: ABORT recall {abort_recall:.4f} < 0.99 "
                  f"— model may miss emergency stop commands!")
        else:
            print(f"[Eval] ✓ ABORT recall {abort_recall:.4f} ≥ 0.99")

        self.next(self.register_model)

    # ─────────────────────────────────────────────────────────────────
    @card
    @step
    def register_model(self):
        """
        Register the fine-tuned model as a versioned Metaflow artifact.
        Also triggers edge quantisation if metrics pass thresholds.

        This implements the Metaflow versioning principle:
        every model is uniquely tagged with run_id and experiment_tag.

        PhD task: Task 3 — Metaflow MLOps Pipeline.
        """
        # Check if model meets quality bar before registration
        f1_ok = self.test_metrics["macro_f1"] >= TARGET_F1
        ece_ok = self.test_metrics["ece"] < TARGET_CALIBRATION_ECE

        self.model_registered = f1_ok and ece_ok
        self.model_record = {
            "model_name": self.model_name,
            "experiment_id": self.experiment_id,
            "run_id": current.run_id,
            "checkpoint_path": self.model_checkpoint_path,
            "training_config": self.training_config,
            "test_metrics": self.test_metrics,
            "baseline_metrics": self.baseline_metrics,
            "f1_improvement": self.f1_improvement,
            "registered": self.model_registered,
            "registration_timestamp": datetime.datetime.utcnow().isoformat(),
            "tags": {
                "experiment": self.experiment_tag,
                "model": self.model_name,
                "lora_rank": self.lora_rank,
            },
        }

        if self.model_registered:
            print(f"[Register] ✓ Model registered! F1={self.test_metrics['macro_f1']:.4f} "
                  f"ECE={self.test_metrics['ece']:.4f}")
        else:
            print(f"[Register] ✗ Model NOT registered: "
                  f"F1 ok={f1_ok}, ECE ok={ece_ok}")
            print(f"  Run more epochs or collect more training data")

        self.next(self.quantize_for_edge)

    # ─────────────────────────────────────────────────────────────────
    @card
    @step
    def quantize_for_edge(self):
        """
        Quantise the registered model to 4-bit GGUF format for
        deployment on NVIDIA Jetson Orin / Raspberry Pi 4 field stations.

        Target: <800ms end-to-end latency at edge.
        Tool: llama.cpp GGUF quantisation.

        PhD task: Task 3 — Edge deployment.
        """
        if not self.model_registered:
            print("[Quant] Skipping quantisation — model not registered")
            self.quantized_model_path = None
            self.edge_metrics = None
        else:
            print(f"[Quant] Quantising {self.model_name} to GGUF 4-bit...")

            # Simulate quantisation
            self.quantized_model_path = (
                self.model_checkpoint_path.replace("checkpoint_final", "gguf_q4_km")
            )
            # Simulate edge latency benchmarks
            self.edge_metrics = {
                "model_size_gb": 4.1,  # ~4.1GB for 7B 4-bit
                "intent_latency_ms": {
                    "jetson_orin": 142,
                    "raspberry_pi_4": 890,  # over budget — needs optimisation
                    "laptop_cpu": 340,
                },
                "intent_accuracy_vs_full": 0.97,  # minor accuracy degradation
                "memory_gb": 4.1,
            }

            print(f"[Quant] GGUF path: {self.quantized_model_path}")
            print(f"[Quant] Edge latencies: {self.edge_metrics['intent_latency_ms']}")

            # Flag if RPi latency exceeds budget
            if self.edge_metrics["intent_latency_ms"]["raspberry_pi_4"] > 800:
                print(f"[Quant] ⚠ RPi4 latency {self.edge_metrics['intent_latency_ms']['raspberry_pi_4']}ms "
                      f"> 800ms budget — consider Jetson Orin instead")

        self.next(self.end)

    # ─────────────────────────────────────────────────────────────────
    @card
    @step
    def end(self):
        """
        Fine-tuning pipeline complete. Model artifacts versioned in Metaflow store.

        Access results:
          from metaflow import Flow
          run = Flow('AgriTalkFinetuneFlow').latest_run
          print(run.data.model_record)
          print(run.data.test_metrics)
        """
        print("\n[Finetune] Pipeline complete!")
        print(f"  Experiment  : {self.experiment_id}")
        print(f"  Model       : {self.model_name} (LoRA r={self.lora_rank})")
        print(f"  Macro-F1    : {self.test_metrics['macro_f1']:.4f}")
        print(f"  ECE         : {self.test_metrics['ece']:.4f}")
        print(f"  Registered  : {self.model_registered}")
        if self.quantized_model_path:
            print(f"  Edge model  : {self.quantized_model_path}")
        print(f"\n  Artifacts versioned: run ID {current.run_id}")


# ─────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────

def _generate_synthetic_corpus(n_samples: int = 500) -> List[Dict]:
    """Generate a synthetic agricultural dialogue corpus for prototyping."""
    templates = {
        "QUERY": [
            "What is the current soil moisture in {zone}?",
            "Show me the humidity readings for the last 24 hours",
            "What was sprayed on the {zone} block yesterday?",
            "How much pesticide is left in the tank?",
            "What is the robot battery level?",
            "Check wind speed before spraying",
        ],
        "COMMAND": [
            "Spray {pesticide} on the {zone} field",
            "Apply {pesticide} at {dosage} litres per hectare",
            "Start spraying the {zone} block now",
            "Begin the scheduled spray mission",
            "Deploy the sprayer to {zone}",
        ],
        "ABORT": [
            "Stop all spray operations immediately",
            "Abort the current mission",
            "Emergency stop",
            "Halt the robot now",
            "Cancel the spraying — wind is too strong",
        ],
        "REPORT": [
            "Generate a summary report for today",
            "Log the spraying activity for this morning",
            "Record the NDVI readings for audit",
            "Create a field report for the agronomist",
        ],
        "DIAGNOSE": [
            "The robot is not responding, what is wrong?",
            "Diagnose the sprayer malfunction",
            "Why did the mission fail last night?",
            "Check if the GPS signal is working",
        ],
    }

    zone_options = ["north", "south", "east", "west", "central"]
    pesticide_options = ["glyphosate", "copper hydroxide", "spinosad", "neem oil"]
    dosage_options = ["1.5", "2", "2.5", "3"]

    corpus = []
    for _ in range(n_samples):
        label = random.choice(INTENT_CLASSES)
        template = random.choice(templates[label])
        text = template.format(
            zone=random.choice(zone_options),
            pesticide=random.choice(pesticide_options),
            dosage=random.choice(dosage_options),
        )
        corpus.append({
            "text": text,
            "label": label,
            "slots": {},
            "source": "synthetic",
        })

    return corpus


def _simulate_baseline_metrics(model_name: str) -> Dict:
    """Simulate zero-shot baseline metrics."""
    # Larger models have better zero-shot performance
    base_f1 = {"mistral-7b": 0.51, "llama-3-8b": 0.54, "llama-3-8b-instruct": 0.63}
    f1 = base_f1.get(model_name, 0.50) + random.uniform(-0.02, 0.02)
    return {
        "macro_f1": round(f1, 4),
        "per_class_f1": {c: round(f1 + random.uniform(-0.15, 0.15), 4) for c in INTENT_CLASSES},
    }


def _simulate_evaluation_metrics(model_name: str, lora_rank: int, epochs: int) -> Dict:
    """Simulate evaluation metrics after fine-tuning."""
    # Fine-tuning significantly improves over baseline
    base_f1 = {"mistral-7b": 0.89, "llama-3-8b": 0.91, "llama-3-8b-instruct": 0.93}
    f1 = base_f1.get(model_name, 0.88)
    f1 += (lora_rank - 8) * 0.002  # Higher rank → slightly better
    f1 += min(epochs - 1, 2) * 0.015  # More epochs → better (up to 3)
    f1 = min(f1 + random.uniform(-0.01, 0.01), 0.99)

    per_class_f1 = {c: round(f1 + random.uniform(-0.04, 0.04), 4) for c in INTENT_CLASSES}
    # ABORT should have very high recall — safety critical
    per_class_recall = {c: round(f1 + random.uniform(-0.02, 0.02), 4) for c in INTENT_CLASSES}
    per_class_recall["ABORT"] = round(0.98 + random.uniform(0, 0.015), 4)

    return {
        "macro_f1": round(f1, 4),
        "per_class_f1": per_class_f1,
        "per_class_recall": per_class_recall,
        "ece": round(random.uniform(0.02, 0.06), 4),
        "accuracy": round(f1 + 0.02, 4),
        "n_test_samples": 50,
    }


def _run_real_lora_finetuning(
    model_name: str,
    train_samples: List,
    val_samples: List,
    lora_rank: int,
    lora_alpha: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
):
    """
    Production fine-tuning using HuggingFace Transformers + PEFT.
    Uncomment when running on actual GPU infrastructure.
    """
    # from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
    # from peft import get_peft_model, LoraConfig, TaskType
    # import torch
    #
    # model_id_map = {
    #     "mistral-7b": "mistralai/Mistral-7B-v0.1",
    #     "llama-3-8b": "meta-llama/Meta-Llama-3-8B",
    #     "llama-3-8b-instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    # }
    # model_id = model_id_map[model_name]
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     model_id, num_labels=5, torch_dtype=torch.bfloat16
    # )
    # lora_config = LoraConfig(
    #     task_type=TaskType.SEQ_CLS,
    #     r=lora_rank,
    #     lora_alpha=lora_alpha,
    #     target_modules=["q_proj", "v_proj"],
    #     lora_dropout=0.1,
    # )
    # model = get_peft_model(model, lora_config)
    # ... training loop ...
    raise NotImplementedError("Set dry_run=True for prototype or implement GPU training")


if __name__ == "__main__":
    AgriTalkFinetuneFlow()
