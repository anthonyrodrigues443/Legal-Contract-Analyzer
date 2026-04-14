# Legal Contract Analyzer

Automatic detection of unfair and risky clauses in legal contracts using NLP and transformer models, benchmarked against the UNFAIR-ToS dataset from LexGLUE.

## Problem

Terms of Service agreements contain clauses that may be unfair to consumers under EU Directive 93/13/EEC. Manually reviewing these is time-consuming — this project builds a classifier that flags potentially unfair clauses across 8 category types.

## Dataset

**UNFAIR-ToS** from the [LexGLUE benchmark](https://huggingface.co/datasets/coastalcph/lex_glue) — sentences from 50 Terms of Service documents annotated for 8 types of potentially unfair clauses.

| Split | Samples | Unfair | Fair |
|-------|---------|--------|------|
| Train | 5,532 | 630 (11.4%) | 4,902 (88.6%) |
| Val | 2,275 | — | — |
| Test | 1,607 | — | — |

**Primary metric:** Micro-F1 (multi-label classification)

## Current Status

- **Phase completed:** Phase 2 — Multi-Model Experiment (2026-04-14)
- **Best model (overall):** TF-IDF(20K) + LogReg — Macro-F1: **0.6146** (CUAD)
- **Best model (high-risk clauses):** XGBoost + TF-IDF(20K) — High-Risk Macro-F1: **0.576**
- **Gap to published RoBERTa-large:** -0.035 macro-F1

## Key Findings

1. **TF-IDF dominates fine-tuned transformers by +0.225 macro-F1 on CUAD.** Fine-tuned BERT (0.350) is the worst model tested — 512-token truncation sees only 5% of each contract (7,861-word average).
2. **The bottleneck is document coverage, not model sophistication.** TF-IDF processes 100% of each contract; BERT truncates at 512 tokens. The published RoBERTa result (0.650) uses sliding windows — truncation explains the gap entirely.
3. **XGBoost beats LogReg on high-risk clauses by +0.059 macro-F1** even though LogReg wins overall (0.615 vs 0.605). XGBoost wins Liquidated Damages by +0.214 — overall F1 is the wrong metric for legal review.
4. **Vocabulary Goldilocks zone at 20K bigrams.** 100K features underperforms 5K features (0.565 vs 0.594) — past 20K, each new dimension is a coefficient to estimate from N=408 imbalanced contracts.
5. **Domain features HURT TF-IDF by -0.118 micro-F1 on UNFAIR-ToS.** Binary regex flags are redundant noise when the base model already captures the same phrases via bigrams with greater precision.

## Models Compared

| Model | Micro-F1 | Macro-F1 | Precision | Recall |
|-------|----------|----------|-----------|--------|
| TF-IDF + LogReg | **0.6555** | **0.6729** | 0.5358 | 0.8441 |
| TF-IDF + Domain + LogReg | 0.5379 | 0.5166 | 0.4049 | 0.8011 |
| Rule-Based | 0.2558 | 0.3091 | 0.1684 | 0.5323 |
| Domain Only + LogReg | 0.2146 | 0.2302 | 0.1237 | 0.8118 |
| Majority Class | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| *BERT-base (published)* | *0.8300* | — | — | — |
| *Legal-BERT (published)* | *0.8500* | — | — | — |

**Total experiments:** 20

## Iteration Summary

### Phase 1: Domain Research + Dataset + EDA + Baseline — 2026-04-13

<table>
<tr>
<td valign="top" width="38%">

**EDA Run 1:** Explored UNFAIR-ToS dataset (9,414 samples, 8 label types, 7.8:1 class imbalance). Tested 5 baselines from majority-class through TF-IDF+LogReg. Best result: Micro-F1 = 0.6555 with 10K TF-IDF features and balanced class weights.<br><br>
**Key Contrast:** Adding 27 hand-crafted domain regex features to TF-IDF dropped performance to 0.5379 (delta: -0.118) — the same patterns captured more precisely by bigrams become noisy binary signals when added explicitly.

</td>
<td align="center" width="24%">

<img src="results/phase1_model_comparison.png" width="220">

</td>
<td valign="top" width="38%">

**Combined Insight:** Statistical features (TF-IDF bigrams) subsume crude domain signals. When a model already learns "sole discretion" and "reserve the right" as high-weight bigrams, adding binary flags for the same patterns introduces redundant noisy features — the classifier penalizes them rather than benefitting.<br><br>
**Surprise:** Domain features HURT by -0.118 micro-F1. The conventional wisdom that "domain knowledge always helps" breaks down when the base model already captures the same signals with greater precision.<br><br>
**Research:** Chalkidis et al. (2022, LexGLUE/ACL) — BERT-base achieves 0.83 micro-F1, Legal-BERT 0.85; Lippi et al. (2019, CLAUDETTE) — pioneered rule-based detection, establishing the ceiling of regex-based approaches at ~0.25 F1.<br><br>
**Best Model So Far:** TF-IDF + LogReg — Micro-F1: 0.6555

</td>
</tr>
</table>

### Phase 2: Multi-Model Experiment — 2026-04-14

<table>
<tr>
<td valign="top" width="38%">

**Model Run 1:** Tested 6 models across classical ML, dense embeddings, and fine-tuned transformers on CUAD. Best: TF-IDF+LightGBM at Macro-F1 = 0.575. Fine-tuned BERT-base scored 0.350 — the worst of all 7 models — because 512-token truncation sees only 5% of each 7,861-word contract.<br><br>
**Model Run 2:** Vocabulary ablation (5K–100K features) revealed a Goldilocks zone at 20K bigrams (Macro-F1 = 0.603); 100K features HURTS to 0.565. TF-IDF(20K)+LR achieves 0.6146 overall, but XGBoost(20K) wins on high-risk clause detection — Macro-F1 0.576 vs LR's 0.517, including +0.214 on Liquidated Damages.

</td>
<td align="center" width="24%">

<img src="results/phase2_mark_model_comparison.png" width="220">

</td>
<td valign="top" width="38%">

**Combined Insight:** Classical ML beats transformers by +0.225 macro-F1 because BERT truncation at 512 tokens is fatal on long contracts. The "best overall model" (LogReg, 0.615) and the "best legal model" (XGBoost, high-risk Macro-F1 0.576) diverge — overall F1 is the wrong optimization target for legal clause review.<br><br>
**Surprise:** Fine-tuning HURTS: frozen Legal-BERT CLS + LogReg (0.514) beats fine-tuned Legal-BERT (0.410). With 408 training contracts, fine-tuning destroys pre-trained representations faster than it learns task-specific patterns. Also: 100K vocabulary features underperform 5K (0.565 vs 0.594).<br><br>
**Research:** Chalkidis et al. (2020, LEGAL-BERT, EMNLP) — domain pre-training adds +5–10%, confirmed by +0.060 gap, but cannot compensate for 5% document coverage; Hendrycks et al. (2021, CUAD) — published RoBERTa uses sliding windows, explaining the ~0.650 vs our 0.575 gap.<br><br>
**Best Model So Far:** TF-IDF(20K) + LogReg — Macro-F1: 0.6146 (CUAD) / XGBoost for high-risk clauses

</td>
</tr>
</table>

## Project Structure

```
Legal-Contract-Analyzer/
├── data/               # Dataset files
├── notebooks/          # Jupyter notebooks (phase1_eda_baseline.ipynb)
├── src/                # Source code
├── models/             # Saved model artifacts
├── results/            # Metrics, plots, experiment logs
├── reports/            # Detailed phase reports
├── tests/              # Unit and integration tests
└── config/             # Configuration files
```

## References

1. Chalkidis et al. (2022) — [LexGLUE: A Benchmark Dataset for Legal Language Understanding](https://aclanthology.org/2022.acl-long.297/) — ACL 2022
2. Lippi et al. (2019) — [CLAUDETTE: automated detector of potentially unfair clauses](https://doi.org/10.1007/s10506-019-09243-2)
3. EU Directive 93/13/EEC — Unfair Terms in Consumer Contracts
