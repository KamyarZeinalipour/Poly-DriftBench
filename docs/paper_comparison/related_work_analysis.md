# Poly-DriftBench: Competitive Analysis & Our Contributions

> This document maps the 8 closest related works, with exact paper details, and states **precisely what we do that they don't**.

---

## The 8 Closest Related Works (Ranked by Similarity)

### 🥇 #1 — EvolIF (Closest Competitor)

| | |
|---|---|
| **Title** | *One Battle After Another: Probing LLMs' Limits on Multi-Turn Instruction Following* |
| **Authors** | Qi Jia, Kaiwei Zhang, Xiujie Song, Ye Shen, Shibo Wang, Dun Pei, Xiangyang Zhu, Guangtao Zhai |
| **Venue** | arXiv 2025 |
| **ArXiv** | Not yet assigned (2025 preprint) |

**What they do:**
- Multi-turn IF evaluation with evolving constraints
- Three-layer tracking mechanism (constraints, instructions, topics)
- Process-centric metrics (patience, endurance, robustness)
- ~20 turns of dynamic conversation

**What they DON'T do (our contribution):**

| Gap | Our Solution |
|---|---|
| Constraints **evolve** mid-conversation → can't isolate pure forgetting | We keep constraints **static** (same L1-L5 rules for 120 turns) → pure forgetting measurement |
| No drift **decomposition** into causes | Our Gold-Context mode decomposes `Drift = Intrinsic Forgetting + Cascade Damage` |
| English only | We test 5 languages + correlate drift with token fertility |
| No benchmark self-validation | Our Dual-Track (static vs dynamic) proves drift isn't a benchmark artifact |
| ~20 turns max | We go to **120 turns** — 6× longer |

> **One-liner:** *"EvolIF measures model patience with evolving rules. We measure pure memory decay with fixed rules over 6× longer conversations, and decompose WHY drift happens."*

---

### 🥈 #2 — MultiChallenge

| | |
|---|---|
| **Title** | *MultiChallenge: A Realistic Multi-Turn Conversation Evaluation Benchmark* |
| **Authors** | Ved Sirdeshmukh, Kaustubh Deshpande, Johannes Mols, Lifeng Jin, et al. |
| **Venue** | Findings of ACL 2025 |
| **ArXiv** | 2501.17399 |

**What they do:**
- 4 challenge categories: Instruction Retention, Inference Memory, Versioned Editing, Self-Coherence
- LLM-as-a-judge with instance-level rubrics
- Found even Claude 3.5 Sonnet < 50% accuracy

**What they DON'T do (our contribution):**

| Gap | Our Solution |
|---|---|
| Only 5-10 turns | We test **10-120 turns** with continuous measurement |
| Binary pass/fail per challenge | Our DDM gives **continuous 0-1 score per turn** with 5-level decomposition |
| No temporal modeling | We compute DOP (Drift Onset Point), AUC, T½ (Instruction Half-Life) |
| No causal analysis | Gold-Context decomposition (forgetting vs cascade) |
| English only | 5 languages |

> **One-liner:** *"MultiChallenge asks 'does the model fail?' in short conversations. We ask 'when, how fast, and why does it fail?' over 120 turns."*

---

### 🥉 #3 — LIFBench

| | |
|---|---|
| **Title** | *LIFBench: Evaluating the Instruction Following Performance and Stability of LLMs in Long-Context Scenarios* |
| **Authors** | Xiaodong Wu, Minhao Wang, Yichen Liu, Xiaoming Shi, He Yan, Xiangju Lu, Junmin Zhu, Wei Zhang |
| **Venue** | ACL 2024 |
| **ArXiv** | 2411.07037 |

**What they do:**
- 11 tasks across 3 long-context scenarios
- 2,766 instructions with automated rubric-based scoring (LIFEval)
- Tests stability across different context LENGTHS

**What they DON'T do (our contribution):**

| Gap | Our Solution |
|---|---|
| Each test is **independent** — no multi-turn conversation | We test **within** a single growing conversation |
| "Stability" = consistency across variations | Our "stability" = continuous decay curve over time |
| No turn-by-turn drift measurement | DDM score per turn → drift trajectory |
| No causal decomposition | Gold-Context mode |
| English only | 5 languages |

> **One-liner:** *"LIFBench measures IF at different context lengths. We measure IF decay within a single conversation as context grows naturally — and decompose why it decays."*

---

### #4 — XIFBench

| | |
|---|---|
| **Title** | *XIFBench: Evaluating LLMs on Multilingual Instruction Following* |
| **Authors** | Zhenyu Li, Kehai Chen, Yunfei Long, Xuefeng Bai, et al. |
| **Venue** | NeurIPS 2025 (Datasets & Benchmarks) |
| **ArXiv** | 2024 |

**What they do:**
- 558 instructions × 6 languages (EN, ZH, RU, AR, HI, SW)
- 5 constraint categories (Content, Style, Situation, Format, Numerical)
- Cultural accessibility annotation
- Requirement-based evaluation with semantic anchors

**What they DON'T do (our contribution):**

| Gap | Our Solution |
|---|---|
| **Single-prompt** evaluation | We do **multi-turn** (10-120 turns) |
| Measures cross-lingual IF quality | We measure cross-lingual **drift rate** over turns |
| No temporal analysis | DOP, AUC, T½ per language |
| No link to tokenization | We correlate drift with **Token Fertility Ratio** — more tokens = faster drift |

> **One-liner:** *"XIFBench asks 'how well do models follow instructions in Hindi?' We ask 'how much faster does Hindi drift than English, and is it because Hindi uses more tokens?'"*

---

### #5 — IFEval (The Foundation)

| | |
|---|---|
| **Title** | *Instruction-Following Evaluation for Large Language Models* |
| **Authors** | Jeffrey Zhou et al. |
| **Venue** | NeurIPS 2023 |

**Our position:** We build on IFEval's principle of **objective, verifiable constraints** (L1-L5 are all rule-based, no LLM judge needed). But IFEval is single-prompt. We extend it temporally.

---

### #6 — Lost in the Middle

| | |
|---|---|
| **Title** | *Lost in the Middle: How Language Models Use Long Contexts* |
| **Authors** | Nelson F. Liu, Kevin Lin, John Hewitt, et al. |
| **Venue** | TACL 2024 |

**Our position:** Liu et al. show models lose attention to the middle. We show they lose attention to the TOP (position 0 — the system prompt) over time. Our Prompt Placement experiment (top vs appended vs both) directly extends their positional analysis to multi-turn conversations.

---

### #7 — RULER

| | |
|---|---|
| **Title** | *RULER: What's the Real Context Size of Your Long-Context Language Models?* |
| **Authors** | Cheng-Yu Hsieh et al. (NVIDIA) |
| **Venue** | COLM 2024 |

**Our position:** RULER tests "can you handle this input?" We test "can you maintain output rules over time?" Different axis entirely — input comprehension vs output compliance.

---

### #8 — Needle in a Haystack

| | |
|---|---|
| **Title** | *LLMTest_NeedleInAHaystack* |
| **Authors** | Greg Kamradt |
| **Year** | 2023 |

**Our position:** NIAH tests **passive retrieval**. We test **active generative adherence**. A model can perfectly retrieve a "needle" while completely failing to maintain formatting rules 50 turns later.

---

## Summary: What We Uniquely Contribute

```
┌──────────────────────────────────────────────────────────────────────┐
│                 NOBODY ELSE DOES THESE:                             │
│                                                                     │
│  1. Gold-Context Decomposition                                      │
│     Drift = Pure Forgetting + Cascade Damage                        │
│     → Completely Novel. Zero prior work.                            │
│                                                                     │
│  2. Dual-Track Validation (Static vs Dynamic)                       │
│     If DOP_static ≈ DOP_dynamic → drift is real                    │
│     → Completely Novel. No benchmark validates itself this way.     │
│                                                                     │
│  3. 120-Turn Continuous Drift Curves                                │
│     With formal metrics (DOP, AUC, T½)                             │
│     → Closest is EvolIF (~20 turns) — we go 6× further             │
│                                                                     │
│  4. Cross-Lingual Drift × Token Fertility Correlation               │
│     Does Japanese drift faster because it costs more tokens?        │
│     → XIFBench is single-prompt. We test temporal drift.            │
│                                                                     │
│  5. L5 Dynamic State-Tracking                                       │
│     Model must compute [Turn: N] — separates memory from mimicry   │
│     → No other benchmark has a dynamic computational constraint     │
│                                                                     │
│  6. Kaplan-Meier Survival Analysis for IF Rules                     │
│     Borrowing from epidemiology for LLM failure analysis            │
│     → Novel framing — never applied to instruction following       │
│                                                                     │
│  7. Difficulty × Drift Interaction                                  │
│     State-tracking complexity vs format compliance                  │
│     → Does harder content cannibalize attention on rules?           │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Ready-to-Use "Related Work" Paragraph

> **Instruction-following evaluation** has evolved from single-prompt assessment (IFEval; Zhou et al., 2023) to long-context stability testing (LIFBench; Wu et al., 2024) and multi-turn conversation analysis (MultiChallenge; Sirdeshmukh et al., 2025; EvolIF; Jia et al., 2025). Cross-lingual instruction following has been explored by XIFBench (Li et al., 2025) across 6 languages. However, existing work suffers from three critical limitations: (1) **short temporal horizons** — MultiChallenge evaluates 5-10 turns, EvolIF ~20, while real-world applications involve 50-100+ turns; (2) **no causal decomposition** — no prior work distinguishes between intrinsic attention decay (the model forgets the prompt) and autoregressive degeneration (one mistake cascades into more); and (3) **no cross-lingual drift analysis** — XIFBench measures single-point compliance per language but not how compliance *decays differently* across languages over time. Poly-DriftBench addresses all three gaps with a 120-turn, 5-language, 5-level constraint benchmark featuring Oracle Context Scaffolding to mathematically decompose drift causality, and a novel Dual-Track validation framework that uses a live agentic simulator to prove our measurements reflect genuine model limitations rather than benchmark artifacts.

---

## Reviewer Rebuttal Cheat Sheet

| Reviewer says... | Our response |
|---|---|
| *"Just another IF benchmark"* | We don't just measure IF — we decompose its *temporal decay* and *causal structure*. Gold-Context mode is completely novel. |
| *"How is this different from EvolIF?"* | EvolIF evolves constraints → measures adaptation. We fix constraints → measures pure forgetting. Different phenomena. Plus: 6× longer, multilingual, causal decomposition. |
| *"How is this different from XIFBench?"* | XIFBench measures IF quality per language at a single point. We measure the *rate of IF decay* per language over 120 turns and correlate it with token fertility. |
| *"Why not just use IFEval?"* | IFEval is single-prompt. Real-world chat applications involve 50-100 turn conversations. We extend IFEval's verifiable-constraint philosophy temporally. |
| *"Is the drift you measure real or a benchmark artifact?"* | Experiment 15 (Dual-Track): we compare static (pre-written) vs dynamic (DeepSeek-simulated) user prompts. If DOP is similar → drift is real. |
| *"Why 5 constraint levels?"* | L1-L4 are static pattern-matching. L5 (turn counter) requires dynamic computation. This separates memory from mimicry — a novel diagnostic axis. |
| *"Why Kaplan-Meier?"* | Formal survival analysis treats "rule death" like patient events — giving us statistically rigorous Instruction Half-Life (T½) with confidence intervals. |
