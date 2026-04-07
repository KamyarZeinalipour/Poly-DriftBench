# 🔬 Poly-DriftBench: Multilingual Context Drift Benchmark

<p align="center">
  <b>Tokenizer Fertility as a Hidden Confounder in Multilingual Instruction Following</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Languages-EN%20%7C%20IT%20%7C%20ES%20%7C%20FR%20%7C%20DE-blue" alt="Languages">
  <img src="https://img.shields.io/badge/Models-12-orange" alt="Models">
  <img src="https://img.shields.io/badge/Experiments-13-red" alt="Experiments">
  <img src="https://img.shields.io/badge/Conversations-4%2C500-green" alt="Conversations">
  <img src="https://img.shields.io/badge/Pipeline-v6--full--experiment-purple" alt="Pipeline">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>

---

## 📖 Overview

**Poly-DriftBench** is a research framework for studying how Large Language Models degrade in instruction-following ability as conversations grow longer — and how this degradation **accelerates in non-English languages** due to **token fertility** (more subword tokens per word).

### The Token Squeeze Hypothesis

> *Languages with higher Token Fertility Ratio (TFR) consume more of the model's context window for the same semantic content, causing earlier onset of instruction-following drift. German (TFR ≈ 1.56×) drifts faster than English (TFR = 1.0×) because the system prompt becomes a smaller fraction of the total context.*

### Key Contributions

1. **Poly-DriftBench** — A parallel corpus of 4,500 DDM-constrained conversations across 5 languages and 3 length tiers
2. **Drift Decay Model (DDM)** — A 4-level constraint evaluation framework with continuous scoring, AUC, half-life, and bootstrap CIs
3. **13 experiments** spanning GPU inference, analytical, and mechanistic analysis across 12 open-source models
4. **Token Squeeze Proof** — Paraphrastic control experiment isolating tokenizer fertility as the causal factor

---

## 🧪 The 13-Experiment Pipeline

### GPU-Heavy Experiments (Model Inference)

| # | Experiment | Description | Key Metric |
|---|-----------|-------------|------------|
| 1 | **Token Fertility Profiling** | Compute TFR(L) = tokens(L) / tokens(EN) for all model × language pairs | TFR per language |
| 2 | **Baseline Drift Measurement** | Run inference on all conversations, evaluate DDM decay curves | DOP, AUC, τ½ |
| 3 | **Paraphrastic Control** | Expand English to match non-EN token counts via CRI, compare drift | Drift curve overlap |
| 5 | **SPAR Attention Analysis** | Extract attention weights, compute System Prompt Attention Ratio | SPAR decay curve |
| 6 | **System Prompt Re-injection** | Re-inject instructions at turns 15/30/50, measure DDM recovery | Recovery boost |
| 7 | **Context Budget Analysis** | Track DDM vs context window utilization (%), not turn number | Critical utilization % |
| 8 | **Perplexity at Drift Onset** | Measure model certainty when instructions start to degrade | Confident vs confused drift |

### Analytical Experiments (Post-Processing)

| # | Experiment | Description | Statistical Test |
|---|-----------|-------------|-----------------|
| 4 | **Regression Analysis** | Fit DOP = β₀ + β₁ × TFR + ε | OLS, p-value on β₁ |
| 9 | **Drift Velocity** | Rate of DDM decay (ΔDDM/Δturn), rolling window | ANOVA across languages |
| 10 | **Cross-Model Consistency** | Do all 12 models rank languages in the same drift order? | Kendall's W concordance |
| 11 | **Tier Effect Analysis** | Compare drift across short/medium/long tiers | Kruskal-Wallis, Cohen's d |
| 12 | **Per-Level Failure Ordering** | Which DDM constraint (L1–L4) fails first per language? | Chi-squared independence |

### Mechanistic Experiment

| # | Experiment | Description | Key Metric |
|---|-----------|-------------|------------|
| 13 | **Token Position Analysis** | System prompt's relative position in context as conversations grow | Ratio at DOP |

---

## 📊 The DDM (Drift Decay Model)

### 4 Constraint Levels

| Level | Constraint | What It Measures | Scoring |
|-------|-----------|-----------------|---------|
| **L1** | `[SYS_ACK: ACTIVE]` canary tag | Basic instruction retention | Binary (with/without brackets) |
| **L2** | Numbered bullet points (1. 2. 3.) | Format compliance | Continuous (0–1, partial credit) |
| **L3** | Forbidden word ban ("however" + per-language lists) | Lexical constraint adherence | Binary per language |
| **L4** | `[Source: ...]` or "According to..." citations | Citation retention | Binary with strict mode |

### Enhanced Metrics

| Metric | Definition | Use |
|--------|-----------|-----|
| **DDM Score** | Mean of L1–L4 per turn (0.0–1.0) | Turn-level compliance |
| **DOP** (Drift Onset Point) | First turn where DDM < 1.0 | When drift starts |
| **sDOP** (Sustained DOP) | First turn where DDM stays below 1.0 for 3+ turns | Robust onset detection |
| **τ½** (Half-Life) | Turn where DDM first drops ≤ 0.5 | Severity measure |
| **AUC** | Area Under the DDM curve (0–1) | Overall conversation quality |
| **Recovery Rate** | % of turns that improve after a decline | Model self-correction ability |
| **95% Bootstrap CI** | Confidence intervals on all aggregated metrics | Statistical rigor |

### L3 Forbidden Words (Per-Language)

| Language | Forbidden Words |
|----------|----------------|
| EN | however |
| IT | tuttavia, comunque, però |
| ES | sin embargo, no obstante |
| FR | cependant, toutefois, néanmoins |
| DE | jedoch, allerdings, dennoch |

---

## 📁 Dataset: 3-Tier Length Strategy

| Tier | Conversations | Turns/Conv | ~Context Length | Purpose |
|------|:---:|:---:|:---:|------------|
| **Short** | 25 | 10–15 | ~5K–8K tokens | Control — minimal drift expected |
| **Medium** | 25 | 30–50 | ~12K–20K tokens | Drift onset zone |
| **Long** | 25 | 80–120 | ~30K–50K tokens | Deep drift — maximum effect |

**Total: 75 conversations × 5 languages = 375 parallel conversation sets**  
**Total inference: 12 models × 375 = 4,500 evaluated conversations**

### 9 Conversation Domains
IT Troubleshooting · Legal Document Review · Customer Support · Travel Planning · Medical Consultation · Financial Advisory · Academic Tutoring · Recipe Instruction · Real Estate · Insurance Claims

---

## 🤖 12 Models Under Evaluation

| Model | Parameters | Architecture | Context Window |
|-------|:---------:|:----------:|:-------------:|
| LLaMA 3.1 8B Instruct | 8B | GQA | 131K |
| LLaMA 3.2 3B Instruct | 3B | GQA | 131K |
| LLaMA 3.2 1B Instruct | 1B | GQA | 131K |
| Mistral 7B Instruct v0.3 | 7B | SWA | 32K |
| Mistral Nemo 12B | 12B | SWA | 131K |
| Mistral Small 24B | 24B | SWA | 32K |
| Qwen 2.5 3B Instruct | 3B | GQA | 32K |
| Qwen 2.5 7B Instruct | 7B | GQA | 32K |
| Qwen 2.5 14B Instruct | 14B | GQA | 32K |
| Qwen 2.5 32B Instruct | 32B | GQA | 32K |
| Gemma 2 9B IT | 9B | GQA | 8K |
| Phi 3.5 Mini | 3.8B | GQA | 128K |

---

## 🏗️ Architecture

### Multi-Agent Data Generation Pipeline

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│     Scenario     │────▶│       User       │◀───▶│    Assistant     │
│    Architect     │     │    Simulator     │     │    Simulator     │
│  (plans arcs)    │     │  (personality)   │     │  (DDM-compliant) │
└──────────────────┘     └────────┬─────────┘     └────────┬─────────┘
                                 │                         │
                    ┌────────────▼─────────────────────────▼──┐
                    │     Rule-Based Validator (L1-L4)         │
                    │   + Quality Auditor (LLM-based scoring)  │
                    └────────────────────┬─────────────────────┘
                                         │
         ┌───────────────────────────────▼──────────────────────────────┐
         │              3-Agent Translation Pipeline                    │
         │  ┌────────────┐   ┌──────────────┐   ┌───────────────────┐  │
         │  │ Translator │──▶│   Reviewer    │──▶│ Back-Translator   │  │
         │  │ (initial)  │   │ (format/nat.) │   │ (semantic verify) │  │
         │  └────────────┘   └──────────────┘   └───────────────────┘  │
         │              + Rule-Based Format Validator                   │
         └─────────────────────────────────────────────────────────────┘
```

### Experiment Pipeline Architecture

```
Phase 1: Token Fertility (CPU)          Phase 2: GPU Inference
┌─────────────────────┐                 ┌──────────────────────────┐
│  Exp 1: Fertility   │                 │  Exp 2: Drift Baseline   │
│  12 models × 5 langs│                 │  ┌──────────┐            │
│  TFR computation    │     ┌──────────▶│  │ Model    │ inference  │
└─────────┬───────────┘     │           │  │ Manager  │──▶ DDM     │
          │                 │           │  └──────────┘   Evaluate  │
          │    ┌────────────┘           │  Also: Exp 7, 8, 13      │
          │    │ GPU 2 + GPU 3          └───────────┬──────────────┘
          │    │ parallel                           │
          ▼    │                        ┌───────────▼──────────────┐
Phase 3: Analytical                     │  Exp 3: Paraphrastic     │
┌─────────────────────┐                 │  Exp 5: SPAR Attention   │
│  Exp 4:  Regression │                 │  Exp 6: Re-injection     │
│  Exp 9:  Velocity   │                 └──────────────────────────┘
│  Exp 10: Cross-Model│
│  Exp 11: Tier Effect│
│  Exp 12: Level Order│
└─────────────────────┘
```

---

## 🚀 Quick Start

### 1. Setup

```bash
git clone https://github.com/KamyarZeinalipour/Poly-DriftBench.git
cd Poly-DriftBench

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# For GPU experiments
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install accelerate
```

### 2. Run Experiments

```bash
# Dry run (2 conversations per tier — verify pipeline)
CUDA_VISIBLE_DEVICES=0 python scripts/run_full_experiment.py --dry-run

# Full single-GPU experiment
CUDA_VISIBLE_DEVICES=0 python scripts/run_full_experiment.py

# Multi-GPU (2x speedup — split models across GPUs)
tmux new-session -d -s gpu2 "PYTHONPATH=. python scripts/run_multi_gpu.py --gpu 2"
tmux new-session -d -s gpu3 "PYTHONPATH=. python scripts/run_multi_gpu.py --gpu 3"

# After both GPUs finish — merge results and run analytical experiments
python scripts/merge_and_analyze.py

# Run specific experiments only
python scripts/run_full_experiment.py --experiments 9 10 11 12

# Analytical-only (skip GPU experiments, use existing data)
python scripts/run_full_experiment.py --skip-gpu
```

### 3. Generate Data (if needed)

```bash
# Full production (100 conversations, 3-tier, 5 languages)
python scripts/run_production.py --parallel 3 --output data/production

# Quick test
python -m src.cli produce --num 2 --output data/test
```

### 4. Human Annotation

```bash
cd annotation_ui && bash start.sh
# Accessible via Cloudflare tunnel
```

---

## 📁 Project Structure

```
Poly-DriftBench/
├── configs/
│   └── default.yaml                    # Models, languages, experiment config
├── src/
│   ├── cli.py                          # CLI entry point (produce, fertility, drift, run-all)
│   ├── data_gen/
│   │   ├── agents.py                   # 9 specialized agents + PipelineStats
│   │   ├── pipeline.py                 # DataFactory orchestrator
│   │   ├── validators.py              # Rule-based DDM + translation validators
│   │   └── seed_generator.py          # Domain templates + seed generation
│   ├── evaluation/
│   │   └── ddm.py                      # DDM scoring (L1–L4, AUC, τ½, sDOP, CI95)
│   ├── experiments/
│   │   ├── runner.py                   # Master 13-experiment orchestrator
│   │   ├── inference.py                # GPU model manager + conversation inference
│   │   ├── exp6_reinjection.py         # Exp 6: System prompt re-injection
│   │   ├── exp7_context_budget.py      # Exp 7: Context budget analysis
│   │   ├── exp8_perplexity.py          # Exp 8: Perplexity at drift onset
│   │   ├── exp9_drift_velocity.py      # Exp 9: Drift velocity analysis
│   │   ├── exp10_cross_model.py        # Exp 10: Cross-model consistency
│   │   ├── exp11_tier_effect.py        # Exp 11: Tier effect analysis
│   │   ├── exp12_level_ordering.py     # Exp 12: Per-level failure ordering
│   │   └── exp13_token_position.py     # Exp 13: Token position analysis
│   ├── tokenizer/
│   │   └── fertility.py                # Token Fertility Ratio computation
│   ├── expansion/
│   │   └── strategies.py               # Paraphrastic expansion (BTE, CPI, CRI)
│   ├── attention/
│   │   └── spar.py                     # SPAR attention analysis module
│   └── visualization/
│       └── plots.py                    # Result visualization
├── scripts/
│   ├── run_full_experiment.py          # Single-GPU full pipeline
│   ├── run_multi_gpu.py                # Multi-GPU parallel experiment
│   ├── merge_and_analyze.py            # Merge GPU results + analytical experiments
│   └── run_production.py              # Data generation pipeline
├── annotation_ui/
│   ├── app.py                          # Flask annotation server
│   ├── start.sh                        # Server + Cloudflare tunnel
│   └── templates/                      # Login, guidelines, annotation pages
├── data/
│   └── production/
│       ├── short/parallel/{en,it,es,fr,de}/    # 25 × 5 short conversations
│       ├── medium/parallel/{en,it,es,fr,de}/   # 25 × 5 medium conversations
│       └── long/parallel/{en,it,es,fr,de}/     # 25 × 5 long conversations
├── results/                            # Experiment outputs
│   ├── fertility/                      # TFR ratios (CSV, JSON)
│   ├── drift_curves/                   # Per-turn DDM scores, summaries
│   ├── paraphrastic/                   # Expansion control results
│   ├── attention_maps/                 # SPAR profiles (JSON)
│   ├── reinjection/                    # Re-injection recovery
│   ├── context_budget/                 # Context utilization analysis
│   ├── perplexity/                     # PPL at drift onset
│   ├── regression/                     # TFR → DOP regression
│   ├── drift_velocity/                 # Velocity analysis + ANOVA
│   ├── cross_model/                    # Kendall's W + pairwise τ
│   ├── tier_effect/                    # Short vs Medium vs Long
│   ├── level_ordering/                 # L1–L4 failure cascade
│   └── token_position/                # System prompt ratio analysis
├── requirements.txt
└── README.md
```

---

## 📈 Preliminary Results

### Token Fertility Ratios (Exp 1)

| Language | Mean TFR | Overhead | Highest Model | Lowest Model |
|----------|:--------:|:--------:|:-------------:|:------------:|
| English | 1.000 | — | — | — |
| Spanish | 1.373 | +37.3% | Qwen 2.5 (1.37) | Gemma 2 (1.12) |
| Italian | 1.525 | +52.5% | LLaMA 3.1 (1.53) | Gemma 2 (1.23) |
| French | 1.496 | +49.6% | LLaMA 3.1 (1.51) | Gemma 2 (1.29) |
| German | 1.563 | +56.3% | LLaMA 3.1 (1.58) | Gemma 2 (1.25) |

### Drift Measurement (Exp 2 — In Progress)

| Signal | Observation |
|--------|-------------|
| L1 (Canary) | Most frequently retained — easiest constraint |
| L3 (Forbidden words) | Drops earliest — models use "however" by turn 2 |
| L4 (Citations) | Drops alongside L3 — models stop citing sources |
| Short tier | Minimal cross-lingual difference (control ✓) |
| Medium tier | Italian shows lower AUC than English (0.694 vs 0.713) |

---

## 🔧 Environment & Infrastructure

### Hardware
- 4× NVIDIA RTX A6000 (49 GB each)
- Multi-GPU parallelism: models split across GPUs

### Software
- Python 3.10+
- PyTorch 2.5+ with CUDA 12.1
- Transformers (HuggingFace) with offline mode
- Models cached locally at `HF_HUB_CACHE`

### Key Environment Variables
```bash
export HF_HUB_CACHE=/path/to/hf_cache/hub
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=2,3
export PYTHONPATH=/path/to/Poly-DriftBench
```

---

## 📝 Human Evaluation Plan

### Validation Tasks

| Task | Samples | Annotators | Purpose |
|------|---------|-----------|---------|
| **Translation Quality** | 50 × 5 langs | 1 native/lang | Verify parallel corpus equivalence |
| **DDM Agreement** | 100 responses | 3 annotators | Human-DDM score correlation |
| **Drift Impact** | 50 early/late pairs | 3 annotators | Does DDM decline = worse quality? |

### Annotation Dimensions
- Naturalness (1-5)
- User Realism (1-5)
- Coherence (1-5)
- DDM Compliance — L1/L2/L3/L4 (binary each)
- Overall Quality (1-5)

### Inter-Annotator Agreement
Target: Krippendorff's α ≥ 0.80 for all constraint levels.

---

## 📊 Statistical Methods

| Test | Used In | Purpose |
|------|---------|---------|
| OLS Regression | Exp 4 | DOP = β₀ + β₁ × TFR + ε |
| One-way ANOVA | Exp 9 | Drift velocity differences across languages |
| Kendall's W | Exp 10 | Cross-model ranking concordance |
| Kendall's τ | Exp 10 | Pairwise model rank correlation |
| Kruskal-Wallis | Exp 11 | Non-parametric tier effect |
| Cohen's d | Exp 11 | Effect size between tiers |
| Chi-squared | Exp 12 | Failure ordering × language independence |
| Pearson r | Exp 8, 13 | DDM-perplexity / DDM-position correlation |
| Bootstrap CI | All | 95% confidence intervals (n=1000) |

---

## 📚 Citation

```bibtex
@article{zeinalipour2026polydriftbench,
  title={Poly-DriftBench: Tokenizer Fertility as a Hidden Confounder 
         in Multilingual Instruction Following},
  author={Zeinalipour, Kamyar},
  journal={Proceedings of EMNLP},
  year={2026}
}
```

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- **DeepSeek** — V3 model used for data generation and translation
- **Meta, Mistral AI, Alibaba, Google, Microsoft** — Open-source models used for evaluation
- Built with the multi-agent orchestration paradigm for synthetic data quality assurance
