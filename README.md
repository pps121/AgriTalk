# AgriTalk: Calibrated Natural Language Interfaces for Agricultural Robotics

**PhD Position L — GreenFieldData Project (ANR)**  
Candidate: Partha Pratim Saha  
Supervisors: Dr. Genoveva Vargas-Solar (CNRS, LIG) · Prof. Roberto Oberti (UniMI)  
Institutions: UCBLyon1 · UniMI · ProBayes · CFL · CNRS  
Timeline: October 2026 – September 2029

---

## What is this?

This repository contains the full materials for the AgriTalk PhD proposal, submitted to the GreenFieldData PhD-L competition. AgriTalk proposes **calibrated natural-language control interfaces for agricultural spray robots**, addressing the fundamental bottleneck that prevents non-expert farm operators from safely directing intelligent IoRT (Internet of Robotic Things) systems in precision agriculture.

The proposal targets the IoRT agricultural market projected at **$11.9B by 2030** (19.3% CAGR), where the control interface remains the unsolved problem.

---

## Research Overview

### The Problem

Precision agriculture robots (e.g., UCBLyon1/ProBayes spray robots, UniMI autonomous platforms) require expert programming. Farm operators speak natural language. Existing NLI approaches for agriculture provide no formal safety guarantee, no mechanistic explanability, and no streaming grounding — the three pillars AgriTalk addresses.

### The 4 Contributions

| # | Name | Research Question | Target Venue |
|---|------|-------------------|--------------|
| **C1** | Conformal NLU calibration (RAPS) | RQ1: Does conformal calibration maintain 95% coverage under seasonal distribution shifts while keeping HITL rate ≤25%? | EMNLP/ACL 2027 |
| **C2** | Belief Vector Field (BVF) attribution | RQ2: Does BVF attribution achieve Kendall τ>0.5 with IG/LRP on safety-critical intents, and does operator trust gap ∆(C−B)<0? | ACL 2029 |
| **C3** | Temporal Streaming Grounding Architecture (TSGA) | RQ3: Under sensor dropout (10–50%) and telemetry lag (>5 min), does TSGA maintain grounding recall? | C&E Agriculture / VLDB 2028 |
| **C4** | Conformal Trust Evaluation Framework (CTEF) | RQ4: Does BVF explanation (condition C) achieve better trust calibration than CoT (B) and no-explanation (A)? | FAccT 2029 |

### The 8 Intent Classes

`SPRAY` · `ABORT` · `DOSAGE_CHANGE` · `QUERY` · `MONITOR` · `SCHEDULE` · `ZONE_OVERRIDE` · `EMERGENCY_STOP`

> **ABORT recall is maximised** — a missed ABORT intent (Type-II error) is catastrophic.

### Safety Architecture (5 verifiable layers)

| Layer | Name | Guarantee |
|-------|------|-----------|
| V1 | Input sanitiser | Strips adversarial payloads |
| V2 | Staleness verifier | Rejects stale field-state (freshness < 0.5) |
| V3 | Conformal predictor (RAPS) | P(y∈C(x)) ≥ 1−α; triggers HITL when \|C\|≥2 |
| V4 | Attribution sufficiency gate | Requires Kendall τ(IG,BVF)>0.5 before actuation |
| V5 | Non-bypassable HITL for ABORT | Always routes ABORT/EMERGENCY_STOP to human |

---

## Repository Structure

```
Agriculture-PhD/
├── proposal/                        # LaTeX source (v6, APA natbib)
│   ├── proposal_main.tex            # 6-page proposal
│   └── proposal_refs.bib            # 36 APA-compatible citations
├── code/
│   ├── metaflow_demo.py             # 8-step MetaFlow pipeline (demo)
│   └── inspect_run.py               # MetaFlow run inspection tool
├── visualizations/                  # 7 proposal-aligned 3D visualizations
│   ├── viz_01_c1_conformal_seasonal_drift.py
│   ├── viz_02_c2_bvf_attribution_trust.py
│   ├── viz_03_c3_streaming_failure_boundary.py
│   ├── viz_04_c4_trust_deployment.py
│   ├── viz_05_full_evaluation_dashboard.py
│   ├── viz_06_phd_roadmap_timeline.py
│   ├── viz_07_c1_coverage_surface.py
│   ├── run_all_visualizations.py    # runner
│   └── html/                        # Generated interactive HTML (git-ignored for size)
├── slides/                          # Presentation materials
├── Partha-workPlan-L.pdf            # Candidate's 6-page work plan
├── ResearchProposal_GreenFieldData-PhDL-WorkPlan.pdf  # Professor's assignment
├── AgriTalk_ResearchProposal.pdf    # Full compiled proposal
└── requirements.txt
```

---

## 7 Interactive Visualizations

All visualizations are **fully interactive 3D HTML** — drag, zoom, hover for exact values. Generated from proposal-specific parameters.

| # | File | Content | Contribution |
|---|------|---------|--------------|
| 01 | `01_c1_conformal_seasonal_drift.html` | Coverage surface α×drift + HITL ablation + ABORT recall | C1/RQ1 |
| 02 | `02_c2_bvf_attribution_trust.html` | Kendall τ heatmap + trust study ∆ + BVF layer trajectory | C2/RQ2 |
| 03 | `03_c3_streaming_failure_boundary.html` | Grounding recall failure surface + Kafka timeline + freshness | C3/RQ3 |
| 04 | `04_c4_trust_deployment.html` | Tier latency (Jetson/cloud) + CTEF trust evolution + artifact lineage | C4/RQ4 |
| 05 | `05_full_evaluation_dashboard.html` | All 8 metrics × 3 seasons + PhD significance landscape | All |
| 06 | `06_phd_roadmap_timeline.html` | 3-year PhD timeline (Q1Y1→Q4Y3): phases, datasets, publications | All |
| 07 | `07_c1_conformal_coverage_surface.html` | RAPS coverage surface + HITL/ABORT policy comparison | C1/RQ1 |

### Quick start

```bash
pip install plotly numpy pandas scipy scikit-learn
cd /path/to/Agriculture-PhD
python visualizations/run_all_visualizations.py
# then open visualizations/html/*.html in any browser
```

---

## Evaluation Targets (from proposal Table)

| Metric | Target | Baseline |
|--------|--------|----------|
| Macro-F1 | > softmax baseline | 0.741 (softmax) |
| ECE | < 0.04 | 0.142 (softmax) |
| ABORT recall | ≥ 0.90 | 0.831 (softmax, Milan Y2) |
| Coverage P(y∈C(x)) | ≥ 0.95 | n/a (softmax has no guarantee) |
| Kendall τ (IG vs BVF) | > 0.50 for all pairs | — |
| Trust gap ∆(C−B) | < 0 (BVF better than CoT) | — |
| NASA-TLX (BVF vs CoT) | non-inferior | — |
| Edge P95 latency (Jetson) | < 800ms | — |

---

## Datasets

| Year | Dataset | Source |
|------|---------|--------|
| Y1 | AgroNLP corpus (constructed) | UCBLyon1 / IRSTEA |
| Y1 | PANGAEA (satellite, NDVI) | pangaea.de |
| Y1 | ACRE (EU AgRI competition) | ACRE consortium |
| Y1 | OpenWeather Lyon 2020–2026 | openweathermap.org |
| Y2 | USDA-ARS-AgAID | usda.gov |
| Y2 | UniMI spray robot records | UniMI lab |
| Y2 | ProBayes farm logs | ProBayes SAS (partner) |
| Y2 | EPPO/PPDB (pest database) | eppo.int |
| Y3 | Federated corpus (Lyon+Milan+CFL) | all partners |

---

## Responsible AI

Five embedded principles:
1. **Safety-by-architecture**: V5 HITL is non-bypassable — no LLM output can circumvent ABORT/EMERGENCY_STOP escalation
2. **Explainability before actuation**: V4 attribution gate requires mechanistic explanation for any SPRAY/DOSAGE_CHANGE
3. **GDPR / federated learning**: CFL (C4) keeps farm data on-premise; only model updates are shared
4. **Failure transparency**: V2 Staleness Verifier surfaces sensor dropout to operators with explicit staleness scores
5. **Operator autonomy**: Trust calibration study (RQ4) measures whether explanations empower or mislead operators

---

## Deployment Architecture

```
Operator NL input
       ↓
[V1: Input sanitiser]
       ↓
[AgriTalk LLM — fine-tuned, domain-adapted]
       ↓
[V2: Staleness verifier] ← Kafka field-state register (TSGA)
       ↓
[V3: Conformal predictor RAPS] → |C|≥2 → HITL
       ↓
[V4: Attribution gate (BVF + τ check)]
       ↓
[V5: ABORT/EMERGENCY_STOP non-bypassable HITL]
       ↓
Robot actuation
```

**Edge**: NVIDIA Jetson AGX Orin (8-bit quantized, P95 < 800ms)  
**Infra**: MetaFlow (artifact versioning) + Kafka/Spark (streaming) + Azure/k8s (cloud)

---

*Candidate: Partha Pratim Saha | partha.saha@ens.ucbl.fr*  
*This repository accompanies the PhD application to GreenFieldData Position L, UCBLyon1 / CNRS, October 2026.*
