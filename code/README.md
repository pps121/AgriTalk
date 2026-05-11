# AgriTalk — Codebase README

**GreenFieldData MSCA Joint Doctorate — Position L**  
Candidate: Partha Pratim Saha  
Research: *Calibrated Intent Compilation and Faithful Explanation for Human-Supervised Agricultural Robotic Systems*

---

## Project Structure

```
Agriculture-PhD/
├── proposal/
│   ├── proposal_main.tex          # Main proposal (9 sections, C1–C4 framework)
│   ├── proposal_refs.bib          # BibTeX references (26 entries)
│   └── proposal_main_v1_backup.tex
├── slides/
│   └── interview_slides.tex       # Beamer 12-slide interview presentation
├── code/
│   ├── agritalk_flow.py           # Main AgriTalk Metaflow DAG (12 steps)
│   ├── agritalk_finetune_flow.py  # LLM fine-tuning Metaflow DAG (9 steps)
│   ├── generate_gantt.py          # 36-month GANTT chart (matplotlib)
│   └── README.md                  # This file
└── requirements.txt
```

---

## Quick Start

### 1. Install Dependencies

```bash
# Create a virtual environment (recommended)
python3 -m venv agritalk_env
source agritalk_env/bin/activate   # Linux/macOS

pip install -r requirements.txt
```

> **Note on ROS2**: `rclpy` is NOT pip-installable. Install via rosdep:
> `sudo apt install ros-humble-rclpy` (Ubuntu + ROS Humble).  
> For local development without ROS2, all scripts run in `dry_run=True` mode.

### 2. Install Metaflow

```bash
pip install metaflow>=2.10.0
# Optionally configure AWS credentials for @batch GPU runs:
metaflow configure aws
```

---

## Running the Main Pipeline: `agritalk_flow.py`

The AgriTalk pipeline is a 12-step Metaflow DAG covering:
B1 (IoT ingestion) → B2 (streaming context) → B3 (NL intent) → B4 (KB grounding) → B5 (HITL/robot).

### Run Scenarios

```bash
cd code/

# Default demo: precision pesticide spray command
python agritalk_flow.py run \
  --scenario pesticide_spray \
  --user_query "Spray more pesticide on the struggling section near the east boundary." \
  --user_role OPERATOR \
  --dry_run True

# Edge mode (simulates Jetson Orin Mistral-7B 4-bit inference)
python agritalk_flow.py run \
  --scenario pesticide_spray \
  --edge_mode True \
  --dry_run True

# Adjust conformal coverage threshold (default alpha=0.05 -> threshold=0.95)
python agritalk_flow.py run \
  --scenario pesticide_spray \
  --confidence_threshold 0.90 \
  --dry_run True

# Production mode (requires real Kafka + LLM endpoint)
python agritalk_flow.py run \
  --scenario pesticide_spray \
  --dry_run False
```

### Pipeline Steps

| Step | Function | Research Contribution |
|------|----------|-----------------------|
| `start` | Input validation, audit init | Security / OWASP A01–A07 |
| `ingest_iot_streams` | Kafka IoT ingestion, AGROVOC annotation | C2 (B1–B2) |
| `load_knowledge_base` | EPPO/PPDB/AGROVOC KB loading | C3 (B4) |
| `join_context` | Merge streaming + KB into inference context | C2 |
| `intent_recognition` | 5-class LLM intent + conformal confidence | C1 (B3) |
| `semantic_grounding` | RAG-style slot grounding vs KB | C3 (B4) |
| `safety_gate` | 5-rule HITL trigger check | C1 (B5) |
| `request_human_confirmation` | HITL dialogue, YES/NO/MODIFY | C1, C3 |
| `autonomous_execute` | Only QUERY/REPORT/DIAGNOSE | C1 |
| `execute_or_abort` | Process HITL response, ROS2 publish | C1 (B5) |
| `finalize_response` | NL response generation | C3 |
| `audit_and_version` | Immutable audit + SHA-256 hash | C4 |

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--scenario` | `pesticide_spray` | Scenario type |
| `--user_query` | (see code) | Natural language command |
| `--user_role` | `OPERATOR` | `OBSERVER`, `OPERATOR`, `AGRONOMIST` |
| `--edge_mode` | `False` | Use edge model (Mistral-7B GGUF) |
| `--dry_run` | `True` | Simulate (no real Kafka/ROS2) |
| `--confidence_threshold` | `0.95` | Conformal coverage target (1-alpha) |
| `--field_id` | `FIELD_001` | Field identifier |

---

## Running LLM Fine-Tuning: `agritalk_finetune_flow.py`

9-step Metaflow DAG for LoRA/PEFT fine-tuning of Llama-3-8B or Mistral-7B on agricultural intent corpus.

```bash
cd code/

# Dry run (simulates training; no GPU required)
python agritalk_finetune_flow.py run \
  --model_name mistral-7b \
  --lora_rank 16 \
  --lora_alpha 32 \
  --epochs 3 \
  --dry_run True \
  --experiment_tag conformal_v1

# View training metrics from last run
python agritalk_finetune_flow.py dump \
  --max-runs 1 evaluate_model

# Production: GPU required (uses @batch decorator -> AWS GPU endpoint)
# Set: metaflow configure aws  first
python agritalk_finetune_flow.py run \
  --model_name llama-3-8b-instruct \
  --dry_run False
```

### Fine-Tuning Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_name` | `mistral-7b` | Base model identifier |
| `--lora_rank` | `16` | LoRA rank (r) |
| `--lora_alpha` | `32` | LoRA alpha (scaling) |
| `--epochs` | `3` | Training epochs |
| `--batch_size` | `4` | Per-device batch size |
| `--learning_rate` | `2e-4` | Learning rate |
| `--dry_run` | `True` | Simulate (no real training) |
| `--corpus_path` | `None` | Path to JSONL corpus (auto-generates synthetic if None) |
| `--experiment_tag` | `baseline` | Tag for tracking |

### Registration Gate

The flow only registers a model if: `macro_F1 >= 0.92` AND `ECE < 0.05` AND `abort_recall >= 0.99`.

---

## Generating the GANTT Chart

```bash
cd code/
python generate_gantt.py
# Outputs: gantt_chart.png, gantt_chart.pdf
```

---

## Compiling the Proposal PDF

Requires: `pdflatex`, `bibtex`, and these LaTeX packages:
`geometry`, `times`, `setspace`, `titlesec`, `enumitem`, `booktabs`, `xcolor`, `hyperref`, `cite`,
`amsmath`, `pgfgantt`, `fancyhdr`, `mdframed`, `tabularx`.

```bash
cd proposal/

# Full compile sequence (run twice for cross-references)
pdflatex proposal_main.tex
bibtex proposal_main
pdflatex proposal_main.tex
pdflatex proposal_main.tex

# Rename for submission
cp proposal_main.pdf ../Saha_Partha.pdf
echo "Submission PDF ready: Saha_Partha.pdf"
```

---

## Compiling the Interview Slides

```bash
cd slides/

pdflatex interview_slides.tex
pdflatex interview_slides.tex   # twice for GANTT labels
```

---

## Architecture Overview

```
                    ┌─────────────────────────────────────────┐
                    │           AGRITALK SYSTEM                │
                    │                                          │
NL Command ──────► B3: Language→Intent (CIC, C1)               │
                    │       Conformal prediction sets           │
                    │       5-class: COMMAND/ABORT/QUERY/...   │
                    │                   │                       │
                    │   ┌───────────────▼──────────────────┐   │
                    │   │  B4: Intent→Command (FAES, C3)   │   │
                    │   │  EPPO/PPDB/AGROVOC KB grounding  │   │
                    │   │  Attribution-grounded explanation │   │
                    │   └───────────────┬──────────────────┘   │
                    │                   │                       │
                    │   ┌───────────────▼──────────────────┐   │
IoT Sensors ──────► B2: Streaming Context (TSCA, C2)        │   │
(Kafka/Spark)       │   15-min rolling mean / 48hr weather  │   │
                    │   Atomic snapshot at query time       │   │
                    │   └───────────────────────────────────┘   │
                    │                   │                       │
                    │   ┌───────────────▼──────────────────┐   │
                    │   │  B5: HITL Gate → Robot (C1, C3)  │   │
                    │   │  5 rule-auditable trigger conds  │   │
                    │   │  YES/NO/MODIFY → ROS2 publish    │   │
                    │   │  ABORT path: <800ms (Jetson)      │   │
                    │   └──────────────────────────────────┘   │
                    └─────────────────────────────────────────┘
                           Metaflow DAG (C4) orchestrates all steps
```

---

## Security Notes (OWASP A01–A07)

- **A01 Broken Access Control**: JWT role enforcement (OBSERVER cannot execute commands)
- **A03 Injection**: NL input sanitized (`_sanitize_nl_input`) before LLM processing
- **A07 Auth Failures**: JWT expiry + role validation on every pipeline entry
- All actuator commands require KB verification + HITL confirmation for critical actions
- Audit logs are SHA-256 hashed for integrity

---

## Local Development (Without Kafka/ROS2)

All components run in `dry_run=True` mode with simulated:
- Field IoT state (NDVI, soil moisture, wind, temperature, humidity)
- Knowledge base (pesticide list, crop ontology, robot capabilities)
- LLM intent classification (rule-based fallback)
- HITL confirmation (auto-approved for QUERY intent, prompted for SPRAY)
- ROS2 publication (logged to console)

No external services required for local prototyping.

---

## Citation

If you use this codebase, please cite:

```bibtex
@phdthesis{saha_agritalk_2029,
  author = {Saha, Partha Pratim},
  title  = {Calibrated Intent Compilation and Faithful Explanation 
             for Human-Supervised Agricultural Robotic Systems},
  school = {Universit{\'e} Claude Bernard Lyon 1 \& Universit{\`a} degli Studi di Milano},
  year   = {2029},
  note   = {GreenFieldData MSCA Joint Doctorate, Position L}
}
```
