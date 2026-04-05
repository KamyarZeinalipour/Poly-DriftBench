
# THESIS / PROJECT PROPOSAL
**The Token Squeeze Hypothesis: A Mechanistic Evaluation of Cross-Lingual Attention Decay and Instruction Drift in Large Language Models**

### 1. Research Overview
Large Language Models (LLMs) are increasingly deployed as long-horizon agents that must maintain specific instructions, behavioral constraints, and system personas over extended interactions. However, a critical vulnerability exists: as the conversation lengthens, the transformer's attention mechanism naturally decays, causing the model to "forget" its initial instructions—a phenomenon termed *Instruction Drift*.

While the industry recognizes this drift, its cross-lingual dimensions remain largely unexplored. Empirical observations suggest that non-English languages suffer from Instruction Drift significantly faster than English. Previous theories have attributed this to cultural bias or a lack of non-English training data.

This research proposes a radically different, mechanistic explanation: **The Token Squeeze Hypothesis**. Because modern tokenizers are highly Anglocentric, expressing the same semantic content in non-English languages inherently requires more tokens (High Token Fertility). We hypothesize that non-English models forget their instructions faster not due to inferior language capabilities, but purely because the inefficient tokenizer artificially inflates the sequence length, mathematically pushing the system prompt out of the model's active attention window much faster.

This project introduces a fully automated, deterministic benchmarking framework across **five languages** (English, Italian, Spanish, French, German) to isolate, prove, and mitigate this architectural bottleneck through both behavioral measurement and direct mechanistic interpretability evidence.

### 2. Core Research Questions
This research addresses five primary, highly measurable research questions:

1. **RQ1 (The Fertility Gradient):** What is the precise token fertility ratio across English, Italian, Spanish, French, and German for standard goal-oriented dialogues, and how does this form a measurable gradient that correlates with instruction drift onset?
2. **RQ2 (The Core Proof):** Can the discrepancy in drift velocity across all five languages be mathematically isolated to raw sequence length? (If we artificially inflate English conversations to match each target language's token count using linguistically valid expansion, does English drift at equivalent rates?)
3. **RQ3 (Architectural Variance):** How do different foundational architectures (e.g., Llama-3's Grouped Query Attention vs. Mistral's Sliding Window Attention vs. Qwen-2's architecture) resist or succumb to the Token Squeeze?
4. **RQ4 (Mechanistic Evidence):** Can we directly observe attention weight decay on system prompt tokens as a function of sequence length, and does this decay rate correlate with token fertility ratio across languages?
5. **RQ5 (Mitigation Efficiency):** What is the optimal compute-to-fidelity ratio for mitigating this drift using algorithmic "Dynamic Prompt Injection" without overwhelming the context window?

### 3. What This Research Builds
To eliminate the subjective biases of "LLM-as-a-Judge" and the high costs of human evaluation, this project builds a 100% automated, deterministic evaluation pipeline.

#### 3.1 Poly-DriftBench — The Multilingual Parallel Dataset
Unlike traditional benchmarks that use complex, subjective personas (e.g., "Act like an angry chef"), Poly-DriftBench focuses on strict, goal-oriented system constraints that can be verified algorithmically.
* **The Corpus:** 100 long-horizon, multi-turn synthetic conversations (30–50 turns each) detailing goal-oriented tasks (e.g., IT troubleshooting, legal document review, customer support, travel planning).
* **The Parallel Translations:** The dataset exists as a semantically identical parallel corpus in **five languages**: English (EN), Italian (IT), Spanish (ES), French (FR), and German (DE). Professional-grade translations are produced via DeepL API with human spot-checking.
* **Language Selection Rationale:**
  - **English** — the tokenizer baseline (lowest fertility).
  - **Italian, Spanish, French** — three Romance languages with varying degrees of morphological richness and expected fertility inflation, enabling intra-family comparison.
  - **German** — a Germanic language with compound-word formation, offering a structurally different tokenization profile (long compound words may tokenize differently than Romance inflections).

#### 3.2 The Deterministic Drift Metric (DDM) — Multi-Level Constraint System
Instead of asking an LLM judge "Does this sound like the persona?", we measure instruction adherence through a **Multi-Level Constraint Protocol** embedded in the Turn 1 System Prompt:

| Level | Constraint | Measurement | Sensitivity |
| :--- | :--- | :--- | :--- |
| **L1** (Format) | Append the exact string `[SYS_ACK: ACTIVE]` to every response | Binary regex match | High (first to fail) |
| **L2** (Structure) | Always respond using numbered bullet points | Structural format classifier | Medium |
| **L3** (Lexical) | Never use the word "however" in any response | Lexical absence check | Medium |
| **L4** (Semantic) | Always cite a specific source before making any factual claim | Rule-based citation parser | Low (last to fail) |

Tracking decay at each level separately produces a **Decay Gradient** — revealing which types of constraints erode first and how this pattern varies across languages. The composite DDM score at turn $t$ is:

$$DDM(t) = \frac{1}{4}\sum_{i=1}^{4} \mathbb{1}[\text{Level}_i \text{ satisfied at turn } t]$$

The **Drift Onset Point (DOP)** is defined as the first turn where $DDM(t) < 1.0$, and the **Total Collapse Point (TCP)** is where $DDM(t) = 0$.

#### 3.3 The Paraphrastic Expansion Control — The Methodological Contribution
To isolate token count from linguistic capability, we introduce the **Paraphrastic Expansion Protocol**. This is a linguistically valid alternative to naive padding that ensures the inflated English text remains natural and semantically coherent.

**Why not `<pad>` tokens?** Models are trained to attend away from padding tokens. Inserting `<pad>` into English text does not create a fair comparison because the model has learned to ignore these tokens — the attention mechanism would not treat them as real context. The decay behavior under `<pad>` inflation would not match natural language inflation, invalidating the causal claim.

**Our approach — Three complementary expansion strategies:**

1. **Back-Translation Expansion (BTE):** Each English user message is translated to an intermediate language (e.g., Japanese → back to English). Back-translation naturally produces verbose, slightly rephrased English that inflates token count while preserving semantics.

2. **Controlled Paraphrastic Inflation (CPI):** Using a paraphrasing model (e.g., PEGASUS or GPT-4o with strict prompting), each English message is rewritten to be more verbose — adding clarifying clauses, hedging language, and elaboration — until its token count matches the target language equivalent. Semantic equivalence is verified via BERTScore ≥ 0.92.

3. **Contextual Repetition Injection (CRI):** Discourse-level filler is injected following natural conversational patterns (e.g., "As I mentioned earlier...", "To reiterate the key point...") to inflate turn length in a way that mimics natural conversational redundancy.

For each target language $L$, we create an expanded English dataset $EN_L$ where the token count at each turn matches language $L$. If $EN_{IT}$ (English inflated to Italian's token count) shows drift curves overlapping Italian's curve, **the Token Squeeze Hypothesis is confirmed**.

The use of three strategies also enables an **ablation study**: if all three produce similar drift curves despite different token compositions, this strengthens the claim that raw sequence length — not token content — drives the decay.

#### 3.4 Mechanistic Interpretability Module — Direct Attention Evidence
Beyond behavioral measurement (DDM), we provide **direct mechanistic evidence** by probing the transformer's internal representations:

**3.4.1 Attention Weight Extraction**
For each model, at each conversational turn $t$, we extract the attention weight matrices from all layers and heads. We compute the **System Prompt Attention Ratio (SPAR)**:

$$SPAR(t) = \frac{1}{L \cdot H} \sum_{l=1}^{L}\sum_{h=1}^{H} \frac{\sum_{i \in \text{sys}} \alpha^{(l,h)}_{last,i}}{\sum_{j} \alpha^{(l,h)}_{last,j}}$$

Where $\alpha^{(l,h)}_{last,i}$ is the attention weight from the last generated token to system prompt token $i$, across layers $l$ and heads $h$. SPAR measures **how much attention the model's final prediction still pays to the system prompt** at each turn.

**3.4.2 Layer-wise Decay Heatmaps**
We produce per-layer attention heatmaps showing system prompt attention across turns, revealing:
- Which layers "forget" first (early vs. late layers)
- Whether certain attention heads specialize in maintaining instruction context
- How decay patterns differ across languages at the same semantic turn

**3.4.3 Causal Intervention (Activation Patching)**
To confirm causality, we perform targeted activation patching:
- At a turn where Italian has drifted but English hasn't, we patch the Italian model's attention activations on the system prompt with the English model's activations
- If this restores Italian's instruction adherence, it proves that reduced attention to the system prompt (caused by Token Squeeze) is the direct mechanistic cause of the drift

#### 3.5 The Dynamic Injection Mitigation Script
A lightweight algorithmic mitigation strategy that tracks the token accumulation in real-time and dynamically re-injects the Turn 1 System Prompt into the context window *just before* the predicted attention decay threshold is reached, optimizing for both API/compute costs and instruction fidelity.

### 4. Experiments — What This Research Measures

* **Experiment 1 — Token Fertility Profiling (The Gradient):**
  Process the parallel Poly-DriftBench datasets through the tokenizers of 3+ major models (Llama-3-8B, Mistral-7B, Qwen-2-7B). Compute the Token Fertility Ratio (TFR) for each language pair:
  $$TFR(L) = \frac{\text{tokens}(L)}{\text{tokens}(EN)}$$
  Expected result: $TFR(EN) = 1.0 < TFR(DE) < TFR(FR) < TFR(ES) < TFR(IT)$ (exact ordering to be empirically determined). This establishes the **fertility gradient**.

* **Experiment 2 — Baseline Drift Measurement (5 Languages):**
  Run the standard EN, IT, ES, FR, DE datasets through all models. Plot the "Decay Curves" (Turn Number vs. DDM Score) for each language. Hypothesis: the drift onset ordering mirrors the fertility ordering.

* **Experiment 3 — The Token Squeeze Proof (Paraphrastic Control):**
  Run the three expanded English datasets ($EN_{IT}^{BTE}$, $EN_{IT}^{CPI}$, $EN_{IT}^{CRI}$) through the models. If all three expanded English decay curves cluster with the Italian decay curve (and away from baseline English), this conclusively proves RQ2: **Instruction Drift across languages is primarily a function of Anglocentric tokenization inefficiency, not language-specific model degradation.**

* **Experiment 4 — Regression Analysis (Fertility → Drift):**
  With 5 languages × 3 models = 15 data points, fit a regression model:
  $$DOP = \beta_0 + \beta_1 \cdot TFR + \epsilon$$
  A high $R^2$ and significant $\beta_1$ provides the statistical backbone for the Token Squeeze Hypothesis.

* **Experiment 5 — Mechanistic Validation (Attention Probing):**
  Extract SPAR scores across all turns for all languages. Verify that:
  (a) SPAR decays monotonically with turn count,
  (b) SPAR decays faster for high-fertility languages,
  (c) SPAR at the DDM failure point is approximately equal across languages (i.e., all languages fail at the same attention threshold, just reaching it at different speeds).

* **Experiment 6 — Mitigation Cost-Benefit Analysis:**
  Apply the Dynamic Injection script to the Italian and Spanish datasets. Measure the exact compute overhead (additional tokens injected) required to maintain $DDM \geq 0.75$ over 50 turns.

### 5. Project / Thesis Structure

| Chapter/Phase | Title | Content |
| :--- | :--- | :--- |
| **1** | **Introduction** | Problem motivation, the LLM "Language Tax", Research Questions. |
| **2** | **Related Work** | Transformer attention decay, context window limits, tokenization bias (Petrov et al., 2024), multilingual NLP gaps, mechanistic interpretability (Olsson et al., 2022). |
| **3** | **Poly-DriftBench & Methodology** | Dataset creation across 5 languages, Token Fertility calculations, Multi-Level DDM definition. |
| **4** | **The Paraphrastic Control Framework** | Three expansion strategies (BTE, CPI, CRI), semantic equivalence verification, experimental design to isolate sequence length. |
| **5** | **Results: The Anatomy of Decay** | Decay curves across 5 languages, fertility gradient analysis, regression results. |
| **6** | **Mechanistic Evidence** | SPAR analysis, layer-wise heatmaps, activation patching results. |
| **7** | **Mitigation: Dynamic Injection** | Prompt-refreshing algorithm, compute overhead vs. fidelity tradeoffs. |
| **8** | **Discussion & Future Work** | Implications for EU AI Act, API pricing for non-English users, tokenizer design recommendations, extension to low-resource languages. |
| **9** | **Conclusion** | Final summary and open-source release of code, datasets, and evaluation framework. |

### 6. Technology Stack

| Component | Tools / Libraries |
| :--- | :--- |
| **Language** | Python 3.11+ |
| **Model Hosting/Inference** | HuggingFace `transformers`, `vLLM` (for fast batch inference), or local GPU (Ollama) |
| **Target Models (Open Weights)** | Llama-3-8B, Mistral-7B-v0.3, Qwen-2-7B |
| **Attention Extraction** | HuggingFace `transformers` (with `output_attentions=True`), custom hooks for KV-cache inspection |
| **Mechanistic Interpretability** | `TransformerLens` (Neel Nanda), `baukit` for activation patching |
| **Data Generation & Translation** | DeepL API (for ground-truth parallel translations), GPT-4o for English seed generation |
| **Paraphrastic Expansion** | Back-translation via MarianMT/OPUS-MT, GPT-4o for controlled inflation, PEGASUS for paraphrasing |
| **Semantic Verification** | `BERTScore`, `sentence-transformers` (cosine similarity verification) |
| **Evaluation Framework** | Custom Python scripts (regex, structural parsers, lexical checkers) |
| **Statistical Analysis** | `scipy.stats`, `statsmodels` (regression, KS-tests, effect sizes) |
| **Data Visualization** | `matplotlib`, `seaborn` (decay curves), `plotly` (interactive attention heatmaps) |

### 7. Expected Contributions
1. **Empirical Proof of the "Language Tax":** The first mechanistic proof — combining behavioral evidence with direct attention probing — that non-English languages suffer faster context decay purely due to tokenization architecture, across a five-language gradient (EN, IT, ES, FR, DE).
2. **Poly-DriftBench:** An open-source, multilingual, mathematically verifiable dataset for testing long-context instruction adherence without relying on subjective LLM judges.
3. **The Paraphrastic Expansion Methodology:** A linguistically valid control framework (BTE + CPI + CRI) that future researchers can use to isolate sequence length variables in any cross-lingual study — replacing naive padding approaches.
4. **SPAR Metric & Attention Decay Maps:** A reusable interpretability toolkit for measuring how transformer attention to instructions evolves over conversation length.
5. **Compute-Optimized Mitigation:** A quantifiable analysis of how to solve this drift programmatically, providing immediate value to enterprise engineers building non-English RAG or agentic systems.
6. **Fertility-Drift Regression Model:** A predictive formula allowing practitioners to estimate drift onset for *any* language given its fertility ratio, without running full experiments.