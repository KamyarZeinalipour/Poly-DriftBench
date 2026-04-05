# 🔬 Poly-DriftBench: Multilingual Context Drift Benchmark

<p align="center">
  <b>A Multi-Agent Data Factory for Measuring Instruction-Following Degradation in Long-Context Multilingual LLMs</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Languages-EN%20%7C%20IT%20%7C%20ES%20%7C%20FR%20%7C%20DE-blue" alt="Languages">
  <img src="https://img.shields.io/badge/Conversations-100-green" alt="Conversations">
  <img src="https://img.shields.io/badge/Pipeline-v5--parallel--multiagent-purple" alt="Pipeline">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>

---

## 📖 Overview

**Poly-DriftBench** is a research framework for studying how Large Language Models (LLMs) degrade in instruction-following ability as conversations grow longer — and how this degradation accelerates in non-English languages due to **token fertility** (more tokens per word).

### The Token Squeeze Hypothesis

> *Languages with higher token fertility (e.g., Italian, German) consume more of the model's context window for the same semantic content, causing earlier onset of instruction-following drift compared to English.*

This repository provides:
1. **A multi-agent synthetic data factory** that generates 100 high-quality, DDM-constrained conversations across 5 languages
2. **A Drift Decay Model (DDM)** evaluation framework with 4 measurable constraint levels
3. **Rule-based + LLM-based quality assurance** pipelines
4. **A human annotation UI** for validating synthetic data quality
5. **Experiment runners** for token fertility profiling, drift measurement, and mechanistic analysis

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

### The DDM (Drift Decay Model) — 4 Constraint Levels

| Level | Constraint | What It Measures | Detection |
|-------|-----------|-----------------|-----------|
| **L1** | `[SYS_ACK: ACTIVE]` tag in every response | Basic instruction retention | Binary (present/absent) |
| **L2** | Numbered bullet points (1. 2. 3.) | Format compliance | Regex count |
| **L3** | Actionable, specific advice | Content quality | Semantic check |
| **L4** | `[Source: ...]` citations | Detail retention | Regex match |

The `[SYS_ACK: ACTIVE]` tag is a **canary token** — a meaningless format marker that the model must maintain. When models start dropping it, that's the measurable signal of context drift.

---

## 📊 Dataset: 3-Tier Length Strategy

| Tier | Conversations | Turns | ~Context Length | Purpose |
|------|:---:|:---:|:---:|---------|
| **Short** | 25 | 10-15 | ~5K-8K tokens | Control — no drift expected |
| **Medium** | 50 | 30-50 | ~12K-20K tokens | Drift onset zone |
| **Long** | 25 | 80-120 | ~30K-50K tokens | Deep drift — the money shot |

**Total: 100 conversations × 5 languages = 500 parallel corpora**

### Domains
- IT Troubleshooting
- Legal Document Review
- Customer Support
- Travel Planning
- Medical Consultation
- Financial Advisory
- Academic Tutoring
- Recipe Instruction
- Real Estate

---

## 🚀 Quick Start

### 1. Setup

```bash
# Clone
git clone https://github.com/KamyarZeinalipour/Poly-DriftBench.git
cd Poly-DriftBench

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your DEEPSEEK_API_KEY
```

### 2. Generate Data

```bash
# Quick test (2 conversations, 5 turns each)
python -m src.cli produce --num 2 --output data/test

# Full production (100 conversations, 3-tier strategy)
python scripts/run_production.py --parallel 3 --output data/production

# English only (no translation)
python -m src.cli produce --num 10 --no-translate --output data/en_only
```

### 3. Run Experiments

```bash
# Token fertility profiling
python -m src.cli fertility --data-dir data/production

# Drift measurement
python -m src.cli drift --data-dir data/production

# Full pipeline
python -m src.cli run-all --data-dir data/production
```

### 4. Human Annotation

```bash
# Start the annotation UI
cd annotation_ui
pip install flask gunicorn
bash start.sh

# Access at the Cloudflare URL shown in the terminal
```

---

## 📁 Project Structure

```
Poly-DriftBench/
├── configs/
│   └── default.yaml              # Experiment configuration
├── src/
│   ├── cli.py                    # CLI entry point
│   ├── data_gen/
│   │   ├── agents.py             # 9 specialized agents + PipelineStats
│   │   ├── pipeline.py           # DataFactory orchestrator
│   │   ├── validators.py         # Rule-based DDM + translation validators
│   │   └── seed_generator.py     # Domain templates + seed generation
│   ├── evaluation/
│   │   └── ddm.py                # DDM scoring engine
│   ├── experiments/
│   │   └── runner.py             # Experiment runner (fertility, drift, SPAR)
│   ├── tokenizer/
│   │   └── fertility.py          # Token fertility computation
│   ├── expansion/
│   │   └── strategies.py         # Paraphrastic expansion (BTE, CPI, CRI)
│   ├── attention/
│   │   └── spar.py               # SPAR attention analysis
│   └── visualization/
│       └── plots.py              # Result visualization
├── scripts/
│   └── run_production.py         # 3-tier production runner
├── annotation_ui/
│   ├── app.py                    # Flask annotation server
│   ├── start.sh                  # Server + Cloudflare tunnel launcher
│   └── templates/                # Login, guidelines, annotation UI
├── data/
│   └── production/
│       ├── short/                # 25 short conversations (10-15 turns)
│       ├── medium/               # 50 medium conversations (30-50 turns)
│       ├── long/                 # 25 long conversations (80-120 turns)
│       └── production_summary.json
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🔧 Pipeline Details

### 9 Specialized Agents

| Agent | Role | Calls/Conv |
|-------|------|:---:|
| **ScenarioArchitect** | Plans conversation arc, persona, emotional trajectory | 1 |
| **UserSimulator** | Generates realistic user messages with personality, pushback | N |
| **AssistantSimulator** | Generates DDM-compliant responses | N + rewrites |
| **QualityAuditor** | Scores quality, requests rewrites | 1-3 rounds |
| **ConversationValidator** | Rule-based DDM constraint checking (deterministic) | per round |
| **TranslatorAgent** | Initial translation to target language | N × 4 langs |
| **TranslationReviewerAgent** | Format preservation + naturalness review | N × 4 langs |
| **BackTranslatorAgent** | Semantic fidelity verification via back-translation | N × 4 langs |
| **TranslationValidator** | Rule-based format marker preservation check | per language |

### 3-Level Parallelism

```
Level 1: Multiple conversations generate simultaneously (--parallel 3)
Level 2: All 4 target languages translate in parallel per conversation
Level 3: Up to 10 messages translate in parallel per language
```

### Comprehensive Metadata Tracking

Every generated conversation includes paper-ready metadata:

```json
{
  "metadata": {
    "pipeline_stats": {
      "api_calls_total": 937,
      "tokens": { "prompt": 650000, "completion": 210000, "total": 860000 },
      "agent_activations": {
        "ScenarioArchitect": 1,
        "UserSimulator": 40,
        "AssistantSimulator": 120,
        "TranslatorAgent": 320,
        "BackTranslatorAgent": 320
      },
      "generation": {
        "ddm_violations_found": 3,
        "ddm_violations_auto_fixed": 3,
        "revision_rounds": 2
      },
      "translation": {
        "messages_translated": 320,
        "format_issues_force_fixed": 0,
        "low_fidelity_retries": 0
      },
      "phase_timings_seconds": {
        "planning": 5.2,
        "generation": 45.1,
        "audit_and_revise": 30.5,
        "translation": 22.3
      }
    }
  }
}
```

---

## 📈 Production Statistics

| Metric | Value |
|--------|------:|
| Total Conversations | 100 |
| Total Languages | 5 (EN, IT, ES, FR, DE) |
| Total API Calls | 114,137 |
| Total Tokens | 115M |
| DDM Compliance | 100% |
| Quality Score (mean) | 7.74/10 |
| Approved Rate | 100% |
| Production Time | 6.6 hours |
| Estimated Cost | ~$80 (DeepSeek-V3) |

---

## 📝 Human Evaluation

A web-based annotation UI is included for human validation of the generated data.

**Features:**
- Name-based login (no password)
- Per-annotator JSON files for persistence
- Resume from last annotated conversation
- Navigate back/forward to change annotations
- Comprehensive annotation guidelines
- 5 rating dimensions (1-5 Likert scale):
  - Naturalness, User Realism, Coherence, DDM Compliance, Overall Quality

**Deployment:**
```bash
cd annotation_ui && bash start.sh
# Publicly accessible via Cloudflare tunnel
```

---

## 🧪 Planned Experiments

| # | Experiment | Description |
|---|-----------|-------------|
| 1 | **Token Fertility Profiling** | Measure tokens/word ratio across languages using LLaMA/Mistral/Qwen tokenizers |
| 2 | **Baseline Drift Measurement** | Feed conversations to models, track DDM score decay per language |
| 3 | **Paraphrastic Expansion** | Expand English to match non-English token counts, isolate fertility effect |
| 4 | **Activation Patching** | Identify critical attention heads for instruction retention |
| 5 | **SPAR Analysis** | Sparse probing of attention patterns at drift onset points |

---

## 📚 Citation

```bibtex
@article{zeinalipour2026polydriftbench,
  title={Poly-DriftBench: A Multilingual Benchmark for Measuring 
         Instruction-Following Drift in Long-Context LLMs},
  author={Zeinalipour, Kamyar},
  year={2026}
}
```

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- **DeepSeek** — V3 model used for data generation and translation
- Built with the multi-agent orchestration paradigm for synthetic data quality assurance
