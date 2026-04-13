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

- **Phase completed:** Phase 1 — Domain Research + Dataset + EDA + Baseline (2026-04-13)
- **Best model:** TF-IDF + Logistic Regression — Micro-F1: **0.6555**
- **Gap to published BERT-base:** -0.175 micro-F1

## Key Findings

1. **TF-IDF + LogReg achieves 0.656 micro-F1** — strong non-transformer baseline, 0.175 below published BERT-base (0.83).
2. **Domain features HURT when combined with TF-IDF** (delta: -0.118). 27 binary regex features add noise because TF-IDF bigrams already capture the same legal phrases more precisely.
3. **Severe class imbalance** (88.6% fair) is the primary challenge — all models struggle with precision.
4. **Domain features alone have high recall (81%) but terrible precision (12%)** — wide net, too many false alarms.
5. **Distinguishing actor matters:** "the company may terminate" (unfair) vs "you may terminate" (fair) requires syntactic understanding that TF-IDF lacks — the gap to transformers.

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

**Total experiments:** 5

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
