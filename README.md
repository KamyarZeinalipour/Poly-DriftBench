# 🔬 Poly-DriftBench: Multilingual Context Drift Benchmark

<p align="center">
  <b>Tokenizer Fertility as a Hidden Confounder in Multilingual Instruction Following</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Languages-EN%20%7C%20IT%20%7C%20ES%20%7C%20FR%20%7C%20DE-blue" alt="Languages">
  <img src="https://img.shields.io/badge/Models-13-orange" alt="Models">
  <img src="https://img.shields.io/badge/Experiments-18-red" alt="Experiments">
  <img src="https://img.shields.io/badge/Conversations-4%2C500-green" alt="Conversations">
  <img src="https://img.shields.io/badge/Pipeline-v8--quality--hardened-purple" alt="Pipeline">
  <img src="https://img.shields.io/badge/Validators-16%20checks-brightgreen" alt="Validators">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>

---

## 📖 Overview

**Poly-DriftBench** is a research framework for studying how Large Language Models degrade in instruction-following ability as conversations grow longer — and how this degradation **accelerates in non-English languages** due to **token fertility** (more subword tokens per word).

### The Token Squeeze Hypothesis

> *Languages with higher Token Fertility Ratio (TFR) consume more of the model's context window for the same semantic content, causing earlier onset of instruction-following drift. German (TFR ≈ 1.56×) drifts faster than English (TFR = 1.0×) because the system prompt becomes a smaller fraction of the total context.*

### Key Contributions

1. **Poly-DriftBench** — A parallel corpus of 4,500 DDM-constrained conversations across 5 languages and 3 length tiers
2. **Drift Decay Model (DDM) v2** — A 5-level constraint evaluation framework (L1-L5) with continuous scoring, AUC, half-life, and bootstrap CIs
3. **Gold-Context Decomposition** — Novel Oracle scaffolding that decomposes `Drift = Pure Forgetting + Cascade Damage`
4. **Dual-Track Validation** — Static (CAT-D) + Dynamic (DeepSeek agentic) evaluation proving drift isn't a benchmark artifact
5. **15 experiments** spanning GPU inference, analytical, mechanistic, and validation analysis across 12 open-source models
6. **Token Squeeze Proof** — Paraphrastic control experiment isolating tokenizer fertility as the causal factor

### 6 Features Only We Have (vs. 8 Related Works)

| # | Feature | What It Proves |
|---|---|---|
| 1 | **Gold-Context Decomposition** | Separates "model forgets" from "one mistake causes more" |
| 2 | **Dual-Track Validation** | Proves benchmark measures real drift, not artifacts |
| 3 | **120-Turn Drift Curves** | 6× longer than closest competitor (EvolIF ~20 turns) |
| 4 | **L5 Dynamic State Constraint** | Separates memory (pattern matching) from computation |
| 5 | **Cross-Lingual Drift × Token Fertility** | First link between language "cost" and instruction decay |
| 6 | **Kaplan-Meier Survival Curves** | Medical-grade statistics applied to IF for first time |

> See [`docs/paper_comparison/`](docs/paper_comparison/) for full paper-by-paper comparison tables.

---

## 🧪 The 18-Experiment Pipeline

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
| 14 | **Gold-Context Scaffolding** | Compare free-form vs gold-context drift → decompose forgetting vs cascade | Δ(DOP) |
| 15 | **Static vs Dynamic Delta** | Compare static (CAT-D) vs dynamic (DeepSeek) user simulation | DOP_static ≈ DOP_dynamic |
| 16 | **🧠 Thought-Action Dissonance** | Parse `<think>` blocks in reasoning models → separate thought DDM vs output DDM | Dissonance score |
| 17 | **🔬 Diagnostic Probe** | Inject rule-recitation probes at DOP → Memory Eviction vs Behavioral Laziness | Rule recall % |
| 18 | **🌊 Information Shockwave** | Inject massive text blocks mid-conversation → cognitive load resilience | DDM drop & recovery |

### Analytical Experiments (Post-Processing)

| # | Experiment | Description | Statistical Test |
|---|-----------|-------------|-----------------|
| 4 | **Regression Analysis** | Fit DOP = β₀ + β₁ × TFR + ε | OLS, p-value on β₁ |
| 9 | **Drift Velocity** | Rate of DDM decay (ΔDDM/Δturn), rolling window | ANOVA across languages |
| 10 | **Cross-Model Consistency** | Do all 13 models rank languages in the same drift order? | Kendall's W concordance |
| 11 | **Tier Effect Analysis** | Compare drift across short/medium/long tiers | Kruskal-Wallis, Cohen's d |
| 12 | **Per-Level Failure Ordering** | Which DDM constraint (L1–L5) fails first per language? | Chi-squared independence |

### Mechanistic Experiment

| # | Experiment | Description | Key Metric |
|---|-----------|-------------|------------|
| 13 | **Token Position Analysis** | System prompt's relative position in context as conversations grow | Ratio at DOP |

---

## 📊 The DDM v2 (Drift Decay Model)

### 5 Constraint Levels

| Level | Constraint | What It Measures | Scoring | Type |
|-------|-----------|-----------------|---------|------|
| **L1** | `[SYS_ACK: ACTIVE]` canary tag | Basic instruction retention | Binary | Static |
| **L2** | Numbered bullet points (1. 2. 3.) | Format compliance | Continuous (0–1) | Static |
| **L3** | Forbidden word ban ("however" + per-language) | Lexical constraint adherence | Binary | Static |
| **L4** | `[Source: ...]` citation | Citation retention | Binary | Static |
| **L5** | `[Turn: N]` counter (must increment) | **Dynamic state-tracking** | Binary | **Dynamic** |

> L1-L4 test static pattern-matching. L5 tests active computation — the model must track and increment a counter. This separates "memory" from "reasoning."

### Enhanced Metrics

| Metric | Definition | Use |
|--------|-----------|-----|
| **DDM Score** | Weighted mean of L1–L5 per turn (0.0–1.0) | Turn-level compliance |
| **DOP** (Drift Onset Point) | First turn where DDM < 1.0 | When drift starts |
| **DOP_tokens** | Context token count at DOP | Verbosity-normalized onset |
| **sDOP** (Sustained DOP) | First turn where DDM stays below 1.0 for 3+ turns | Robust onset detection |
| **τ½** (Half-Life) | Turn where DDM first drops ≤ 0.5 | Severity measure |
| **AUC** | Area Under the DDM curve (0–1) | Overall conversation quality |
| **Recovery Rate** | % of turns that improve after a decline | Model self-correction ability |
| **95% Bootstrap CI** | Confidence intervals on all aggregated metrics | Statistical rigor |
| **🏆 PDRI** | Poly-Drift Resilience Index (0–100) | **Single leaderboard score** |

### 🏆 PDRI — Poly-Drift Resilience Index

A single **0–100** score for leaderboard ranking, designed so that every benchmark headline is one number:

```
PDRI = 100 × (0.40·AUC + 0.30·DOP_norm + 0.15·Cascade_Resist + 0.15·Recovery)
```

| Component | Weight | What It Measures |
|---|---|---|
| **AUC** | 40% | Total compliance across all turns |
| **DOP_norm** | 30% | Later drift onset = higher score (DOP/total_turns) |
| **Cascade Resistance** | 15% | How spread out are per-level failures (independent > correlated) |
| **Recovery Rate** | 15% | Self-correction ability after drift |

| PDRI Range | Interpretation |
|---|---|
| **90–100** | Near-perfect: no or minimal drift |
| **70–89** | Strong: late onset, good recovery |
| **40–69** | Moderate: mid-conversation drift |
| **10–39** | Weak: early collapse, poor recovery |
| **0–9** | Total failure: immediate and permanent drift |

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
**Total inference: 13 models × 375 = 4,875 evaluated conversations**

### 10 Conversation Domains (5 Easy / 3 Medium / 2 Hard)

| Difficulty | Domains | State-Tracking Complexity |
|---|---|---|
| 🟢 Easy (5) | Daily Life Tips · Cooking · Pet Care · Entertainment · Home & Garden | Stateless — each turn independent |
| 🟡 Medium (3) | Gift Shopping · Fitness · Study Tips | Shallow dependency — references prior turns |
| 🔴 Hard (2) | Travel Planning · Event Planning | Deep dependency — branching decisions across turns |

> Difficulty is defined by **conversation complexity** (state-tracking), NOT content knowledge. All topics are trivially easy to answer.

---

## 🤖 13 Models Under Evaluation

| Model | Parameters | Architecture | Context Window | Type |
|-------|:---------:|:----------:|:-------------:|:----:|
| LLaMA 3.1 8B Instruct | 8B | GQA | 131K | Standard |
| LLaMA 3.2 3B Instruct | 3B | GQA | 131K | Standard |
| LLaMA 3.2 1B Instruct | 1B | GQA | 131K | Standard |
| Mistral 7B Instruct v0.3 | 7B | SWA | 32K | Standard |
| Mistral Nemo 12B | 12B | SWA | 131K | Standard |
| Mistral Small 24B | 24B | SWA | 32K | Standard |
| Qwen 2.5 3B Instruct | 3B | GQA | 32K | Standard |
| Qwen 2.5 7B Instruct | 7B | GQA | 32K | Standard |
| Qwen 2.5 14B Instruct | 14B | GQA | 32K | Standard |
| Qwen 2.5 32B Instruct | 32B | GQA | 32K | Standard |
| Gemma 2 9B IT | 9B | GQA | 8K | Standard |
| Phi 3.5 Mini | 3.8B | GQA | 128K | Standard |
| **DeepSeek-R1-Distill-Llama-8B** | **8B** | **GQA** | **131K** | **🧠 System-2** |

> Model #13 is a **reasoning model** with latent Chain-of-Thought (hidden `<think>` blocks). This tests whether System-2 reasoning can prevent instruction drift — addressing a key peer review critique.

---

## 🏗️ Architecture

### Dual-Track Evaluation Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    POLY-DRIFTBENCH                              │
├──────────────────────────┬──────────────────────────────────────┤
│  Track 1: Static (CAT-D) │  Track 2: Dynamic (DeepSeek API)   │
│  Pre-generated user msgs │  Live user simulation on-the-fly   │
│  100% reproducible       │  Maximum ecological validity       │
│  Supports Gold-Context   │  Eliminates trajectory mismatch    │
├──────────────────────────┴──────────────────────────────────────┤
│  Exp 15: If DOP_static ≈ DOP_dynamic → Drift is REAL          │
└─────────────────────────────────────────────────────────────────┘
```

### Multi-Agent Data Generation Pipeline (Track 1)

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│     Scenario     │────▶│       User       │◀───▶│    Assistant     │
│    Architect     │     │    Simulator     │     │    Simulator     │
│  (plans arcs)    │     │  (CAT-D design)  │     │ (DDM L1-L5)     │
└──────────────────┘     └────────┬─────────┘     └────────┬─────────┘
                                 │                         │
                    ┌────────────▼─────────────────────────▼──┐
                    │     Rule-Based Validator (L1-L5)         │
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

Phase 4: Mechanistic & Diagnostic
┌────────────────────────────────────┐
│  Exp 16: Thought-Action Dissonance │
│  (reasoning models, <think> parse) │
│  Exp 17: Diagnostic Probe          │
│  (rule recall at DOP)              │
│  Exp 18: Information Shockwave     │
│  (RAG stress test)                 │
└────────────────────────────────────┘
```

### 3-Way Drift Decomposition

```
                          Instruction Drift
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
        RLHF Override    Pure Forgetting   Cascade Damage
        (L1/L3 fail     (L5 fails first;  (Free > Gold DOP;
         due to safety   counter requires  autoregressive
         training)       active memory)    error compound)
              │                │                │
              ▼                ▼                ▼
         Exp 12:          Exp 17:          Exp 14:
         Level Ordering   Diagnostic       Gold-Context
                          Probe            Scaffolding
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

# Regenerate English-only (fast, no translation)
python scripts/regenerate_en.py --output data/full_test --short 5 --medium 5 --long 5

# Quick test
python -m src.cli produce --num 2 --output data/test
```

---

## 🛡️ Quality-Hardened Generation Pipeline (v8)

The data generation pipeline was iteratively hardened through 3 rounds of quality auditing to eliminate 5 critical flaws found in the initial generations.

### The 5 Flaws (and How They Were Solved)

| # | Flaw | Root Cause | Fix | Result |
|---|------|-----------|-----|--------|
| 1 | **Lexical Overfitting** | LLM compresses to a single intro phrase | `frequency_penalty=0.4` + `temperature=0.85` at API level | 17x → 0x |
| 2 | **Citation Hallucination** | Cross-domain citations (financial source for cooking) | Mixed-topic citation rules + domain-aware `force_fix_ddm` | 5 → 0 mismatches |
| 3 | **Groundhog Day Loop** | User repeats same complaint verbatim | Resolved-problem tracker in batch prompt | Eliminated |
| 4 | **Meta-Language** | Robotic pivots ("Switching topics", "Plot twist") | Hard ban in prompts + `QUAL_META_LANGUAGE` validator | 13 → 0 |
| 5 | **Length Uniformity** | All user messages same length | Length variance mandate (20% ultra-short / 20% long / 60% normal) | CV: 0.14 → 0.33 |

### 16 Rule-Based Validators

| Category | Check | Type | What It Catches |
|----------|-------|------|----------------|
| **DDM** | `DDM_L1` – `DDM_L5` | Error | Missing tags, bullets, forbidden words, citations, turn counter |
| **Structural** | `STRUCT_ROLE`, `STRUCT_EMPTY`, `STRUCT_LENGTH` | Error/Warning | Role alternation, empty messages, length bounds |
| **Quality** | `QUAL_REPETITION` | Warning | Near-duplicate responses (trigram Jaccard) |
| | `QUAL_DIVERSITY` | Warning | Low lexical diversity (TTR) |
| | `QUAL_ASSISTANT_VARIETY` | Warning | Assistant reusing same opener |
| | `QUAL_USER_LENGTH_VARIANCE` | Warning | User messages too uniform (CV) |
| | `QUAL_META_LANGUAGE` | Warning | Robotic pivot phrases |
| | `QUAL_INTRO_REPETITION` | Warning | Same intro sentence reused 3+ times |
| | `QUAL_PHRASE_SPAM` | Warning | Specific banned phrases in openings |
| | `QUAL_CITATION_DOMAIN` | Warning | Citation-topic domain mismatch |
| | `QUAL_USER_VERBATIM` | Warning | User repeating complaints verbatim |
| **Coherence** | `COHER_PHANTOM_REF` | Warning | User references non-existent advice |
| | `QUAL_TOPIC_REPEAT` | Warning | Assistant repeating advice 10+ turns apart |
| **CAT-D** | `CATD_ECHO` | Warning | User echoing assistant-specific terminology |

### API-Level Diversity Controls

```python
# AssistantSimulator generates with:
temperature = 0.85          # Higher than default for structural diversity
frequency_penalty = 0.4     # Penalizes token reuse at the logit level

# 10 response styles in a shuffled rotation queue (no repeats until all used)
# Anti-repetition context: previous 2 styles injected as "AVOID" patterns
# Domain-aware force_fix_ddm: food→USDA, finance→CFPB, sleep→NIH, etc.
```

### Architecture Improvements

| Feature | Before | After |
|---------|--------|-------|
| **Batch sizing** | Fixed 15 | Dynamic: short=all-at-once, medium=20, long=15 |
| **Style rotation** | 5 styles, `random.choice()` | 10 styles, shuffled deque + anti-repeat |
| **Force-fix sources** | Generic "Standard professional guidelines" | Domain-aware (USDA/EPA/CFPB/NIH/APA) |
| **Safety net** | Only fix DDM errors | Fix ALL messages (L5 + phrase sanitization) |
| **Problem tracker** | None | Resolved-problems injected into next batch |

---

## 📁 Project Structure

```
Poly-DriftBench/
├── configs/
│   └── default.yaml                    # Models, languages, experiment config
├── docs/
│   └── paper_comparison/               # 📄 Related work analysis
│       ├── comparison_table.md         # Feature grid vs 8 related papers
│       └── related_work_analysis.md    # Detailed positioning + rebuttal guide
├── src/
│   ├── cli.py                          # CLI entry point
│   ├── data_gen/
│   │   ├── agents.py                   # 9 agents + CAT-D user sim + DDM force-fix
│   │   │                                #   freq_penalty=0.4, temp=0.85, style rotation
│   │   ├── pipeline.py                 # DataFactory orchestrator + safety net
│   │   ├── validators.py              # 16 rule-based validators (DDM + quality + CAT-D)
│   │   └── seed_generator.py          # 10 domain templates (5E/3M/2H)
│   ├── evaluation/
│   │   └── ddm.py                      # DDM v2 scoring (L1–L5, AUC, τ½, sDOP, CI95)
│   ├── experiments/
│   │   ├── runner.py                   # Master 15-experiment orchestrator
│   │   ├── inference.py                # GPU inference + dynamic + placement modes
│   │   ├── dynamic_user.py             # 🆕 DeepSeek user simulator (Track 2)
│   │   ├── exp6_reinjection.py         # Exp 6: System prompt re-injection
│   │   ├── exp7_context_budget.py      # Exp 7: Context budget analysis
│   │   ├── exp8_perplexity.py          # Exp 8: Perplexity at drift onset
│   │   ├── exp9_drift_velocity.py      # Exp 9: Drift velocity analysis
│   │   ├── exp10_cross_model.py        # Exp 10: Cross-model consistency
│   │   ├── exp11_tier_effect.py        # Exp 11: Tier effect analysis
│   │   ├── exp12_level_ordering.py     # Exp 12: Per-level failure ordering
│   │   ├── exp13_token_position.py     # Exp 13: Token position analysis
│   │   ├── exp14_gold_context.py       # Exp 14: Gold-Context scaffolding
│   │   └── exp15_static_vs_dynamic.py  # 🆕 Exp 15: Static vs Dynamic delta
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
│   ├── regenerate_en.py                # English data regeneration (v7)
│   └── run_production.py              # Data generation pipeline
├── annotation_ui/
│   ├── app.py                          # Flask annotation server
│   ├── start.sh                        # Server + Cloudflare tunnel
│   └── templates/                      # Login, guidelines, annotation pages
├── data/
│   └── production/
│       ├── short/parallel/{en,it,es,fr,de}/    # 25 × 5 short conversations
│       ├── medium/parallel/{en,it,es,fr,de}/   # 50 × 5 medium conversations
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
│   ├── level_ordering/                 # L1–L5 failure cascade
│   ├── token_position/                # System prompt ratio analysis
│   ├── gold_context/                   # 🆕 Forgetting vs cascade decomposition
│   └── exp15_static_vs_dynamic/        # 🆕 Static vs dynamic drift delta
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
- DDM Compliance — L1/L2/L3/L4/L5 (binary each)
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
| Kaplan-Meier | Exp 2, 14 | Survival analysis for instruction rule "death" |
| Paired t-test | Exp 15 | Static vs dynamic DOP comparison |
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
