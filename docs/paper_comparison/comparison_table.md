# Paper-by-Paper Comparison: Poly-DriftBench vs Related Work

## Master Comparison Table

| Feature | IFEval | LIFBench | RULER | NIAH | Lost in Middle | MultiChallenge | EvolIF | XIFBench | **Poly-DriftBench (Ours)** |
|---|---|---|---|---|---|---|---|---|---|
| **Paper** | Zhou et al. | Wu et al. | Hsieh et al. | Kamradt | Liu et al. | Sirdeshmukh et al. | Jia et al. | Li et al. | **Ours** |
| **Venue** | NeurIPS '23 | ACL '24 | COLM '24 | — | TACL '24 | ACL '25 | arXiv '25 | NeurIPS '25 | **Target: EMNLP '26** |
| | | | | | | | | | |
| **Multi-Turn** | ❌ (1 prompt) | ❌ (1 prompt) | ❌ (1 prompt) | ❌ (1 prompt) | ❌ (1 prompt) | ✅ (5-10 turns) | ✅ (~20 turns) | ❌ (1 prompt) | **✅ (10-120 turns)** |
| **Max Turns Tested** | 1 | 1 | 1 | 1 | 1 | ~10 | ~20 | 1 | **120** |
| **Continuous Drift Curve** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | Partial | ❌ | **✅ (per-turn DDM)** |
| **Drift Onset Point (DOP)** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **✅** |
| **Instruction Half-Life (T½)** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **✅** |
| **AUC Drift Score** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **✅** |
| | | | | | | | | | |
| **Multilingual** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (6 langs) | **✅ (5 langs)** |
| **Cross-lingual Drift Rate** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **✅** |
| **Token Fertility × Drift** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **✅** |
| | | | | | | | | | |
| **Constraint Levels** | 25 types | 11 tasks | 13 tasks | 1 | — | 4 categories | 12 groups | 5 categories | **5 levels (L1-L5)** |
| **Dynamic Constraint (State)** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **✅ (L5 Turn Counter)** |
| **Verifiable (No LLM Judge)** | ✅ | ✅ (LIFEval) | ✅ | ✅ | ✅ | ❌ (LLM judge) | ❌ (LLM judge) | Partial | **✅ (all rule-based)** |
| | | | | | | | | | |
| **Drift Decomposition** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **✅ (Gold-Context)** |
| **Forgetting vs Cascade** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **✅** |
| **Dual-Track Validation** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **✅ (Static + Dynamic)** |
| **Survival Analysis** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **✅ (Kaplan-Meier)** |
| **Prompt Placement Ablation** | ❌ | ❌ | ❌ | ❌ | ✅ (positional) | ❌ | ❌ | ❌ | **✅ (top/appended/both)** |

---

## Per-Paper Differential (What Exactly Changes)

### vs. IFEval (Zhou et al., NeurIPS 2023)

| Dimension | IFEval | Poly-DriftBench | Δ Change |
|---|---|---|---|
| Evaluation scope | Single prompt, 500 prompts | Multi-turn conversation, 10-120 turns | **Temporal extension** |
| Measurement | Binary pass/fail per prompt | Continuous 0-1 DDM score per turn | **Granularity upgrade** |
| Constraint types | 25 types (length, keyword, format) | 5 hierarchical levels (L1: format → L5: dynamic state) | **Hierarchical taxonomy** |
| Output | Prompt-level / instruction-level accuracy | Drift curve, DOP, AUC, T½, per-level decay | **Temporal analysis** |
| Languages | English only | EN, IT, ES, FR, DE | **Multilingual** |
| Causal analysis | None | Gold-Context decomposition | **Novel** |

---

### vs. LIFBench (Wu et al., ACL 2024)

| Dimension | LIFBench | Poly-DriftBench | Δ Change |
|---|---|---|---|
| Context growth | Artificial padding to increase length | Natural context growth over 120 turns | **Ecological validity** |
| Stability metric | Consistency across input variations | Temporal decay curve within one conversation | **Within-conversation** |
| Each test | Independent (reset between tests) | Cumulative (context grows, errors compound) | **Cascade modeling** |
| Evaluation | Rubric-based (LIFEval) | Rule-based DDM (5 levels, automated) | **Simpler, deterministic** |
| Causal analysis | None | Forgetting vs Cascade decomposition | **Novel** |
| Languages | English only | 5 languages | **Multilingual** |

---

### vs. RULER (Hsieh et al., COLM 2024)

| Dimension | RULER | Poly-DriftBench | Δ Change |
|---|---|---|---|
| Tests | Input comprehension (retrieval, tracing) | Output compliance (format adherence) | **Different axis** |
| Task type | Find info in long input | Generate compliant output over time | **Generative vs retrieval** |
| Multi-turn | No (synthetic single-prompt) | Yes (120-turn conversations) | **Multi-turn** |
| Languages | English only | 5 languages | **Multilingual** |

---

### vs. Lost in the Middle (Liu et al., TACL 2024)

| Dimension | Lost in the Middle | Poly-DriftBench | Δ Change |
|---|---|---|---|
| What decays | Retrieval of info at different positions | Adherence to system prompt at position 0 | **System prompt decay** |
| Mechanism | U-shaped positional bias | Monotonic temporal decay | **Temporal vs positional** |
| Test format | Single prompt | 120-turn conversation | **Multi-turn** |
| Prompt placement | Fixed | Ablation: top / appended / both | **Positional experiment** |

---

### vs. MultiChallenge (Sirdeshmukh et al., ACL 2025)

| Dimension | MultiChallenge | Poly-DriftBench | Δ Change |
|---|---|---|---|
| Turns | 5-10 | 10-120 | **12× more turns** |
| Scoring | LLM-as-judge (subjective) | Rule-based DDM (deterministic) | **Reproducibility** |
| Temporal analysis | None | DOP, AUC, T½, drift velocity | **Full temporal suite** |
| Challenges | 4 categories (retention, memory, editing, coherence) | 5 orthogonal constraint levels (L1-L5) | **Hierarchical failure analysis** |
| Causal | None | Gold-Context decomposition | **Novel** |
| Languages | English only | 5 languages | **Multilingual** |

---

### vs. EvolIF (Jia et al., 2025) — ⚠️ CLOSEST COMPETITOR

| Dimension | EvolIF | Poly-DriftBench | Δ Change |
|---|---|---|---|
| Constraints | **Evolve** mid-conversation | **Static** throughout | **Isolates pure forgetting** |
| What's measured | Adaptation to changing rules | Memory of fixed rules | **Different phenomenon** |
| Max turns | ~20 | 120 | **6× longer** |
| Scoring | LLM-based (patience, endurance) | Rule-based DDM (L1-L5) | **Deterministic, no judge** |
| Causal decomposition | None | Gold-Context (forgetting vs cascade) | **Novel** |
| Self-validation | None | Dual-Track (static vs dynamic) | **Novel** |
| Languages | English only | 5 languages | **Multilingual** |
| Token analysis | None | Token fertility × drift correlation | **Novel** |
| Difficulty control | Topic-based | State-tracking complexity | **Orthogonal to content** |

---

### vs. XIFBench (Li et al., NeurIPS 2025)

| Dimension | XIFBench | Poly-DriftBench | Δ Change |
|---|---|---|---|
| Languages | 6 (EN, ZH, RU, AR, HI, SW) | 5 (EN, IT, ES, FR, DE) | Different language set |
| Multi-turn | ❌ Single prompt | ✅ 10-120 turns | **Temporal extension** |
| What's measured | IF quality per language | IF **decay rate** per language | **Temporal × cross-lingual** |
| Tokenization link | None | Token fertility → drift rate | **Novel hypothesis** |
| Causal | None | Gold-Context decomposition | **Novel** |

---

## Visual: Feature Coverage Heat Map

```
                    IFEval  LIFBen  RULER   NIAH   LostMid  MultiC  EvolIF  XIFBen  OURS
Multi-turn            ·       ·       ·       ·       ·      ██      ███      ·     █████
120+ turns            ·       ·       ·       ·       ·       ·       █       ·     █████
Drift curve           ·       ·       ·       ·       ·       ·       █       ·     █████
DOP/AUC/T½            ·       ·       ·       ·       ·       ·       ·       ·     █████
Multilingual          ·       ·       ·       ·       ·       ·       ·      ████   █████
Drift × language      ·       ·       ·       ·       ·       ·       ·       ·     █████
Token fertility       ·       ·       ·       ·       ·       ·       ·       ·     █████
Dynamic constraint    ·       ·       ·       ·       ·       ·       ·       ·     █████
Verifiable (no LLM)  ████    ████    ████    ████    ████     ·       ·       █     █████
Gold-Context          ·       ·       ·       ·       ·       ·       ·       ·     █████
Dual-Track            ·       ·       ·       ·       ·       ·       ·       ·     █████
Survival analysis     ·       ·       ·       ·       ·       ·       ·       ·     █████
Prompt placement      ·       ·       ·       ·      ████     ·       ·       ·     █████

Legend: · = not supported  █ = partial  ████ = supported  █████ = full support
```

---

## Bottom Line: 6 Features ONLY We Have

| # | Feature | What It Proves |
|---|---|---|
| 1 | **Gold-Context Decomposition** | Separates "model forgets" from "one mistake causes more mistakes" |
| 2 | **Dual-Track Validation** | Proves our benchmark measures real drift, not benchmark artifacts |
| 3 | **120-Turn Drift Curves** | 6× longer than closest competitor (EvolIF ~20 turns) |
| 4 | **L5 Dynamic State Constraint** | Separates memory (pattern matching) from computation (counting) |
| 5 | **Cross-Lingual Drift × Token Fertility** | First empirical link between language "cost" and instruction decay |
| 6 | **Kaplan-Meier Survival Curves** | Medical-grade statistical framework applied to IF for first time |
