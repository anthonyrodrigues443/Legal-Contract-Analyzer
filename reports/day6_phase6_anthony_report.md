# Phase 6: Explainability & Model Understanding — Legal Contract Analyzer
**Date:** 2026-04-19
**Session:** 6 of 7
**Researcher:** Anthony Rodrigues

## Objective
Understand WHY the Phase 5 champion (LGBM-only + 40K word 1-3gram + class-prior thresholds) works — and where it fails. Specifically: how many features does it actually use, do trigrams matter, what drives TP vs FP predictions, and do the top features align with legal domain knowledge?

## Research & References
1. Lundberg & Lee (NeurIPS 2017) — TreeSHAP provides model-faithful feature attribution for tree ensembles, unlike split-gain which measures split frequency.
2. Fan & Lin (2007) — Class-prior thresholds as F1-optimal plug-in rules for calibrated classifiers (used in Phase 5).
3. CUAD v1 Annotation Guidelines (Hendrycks et al. 2021) — defines what each clause category means legally, used for domain validation.

How research influenced today's experiments: TreeSHAP was chosen over split-gain because Mark's Phase 6 already used split-gain/LR coefficients — SHAP provides direction and magnitude per prediction, enabling TP vs FP decomposition.

## Dataset
| Metric | Value |
|--------|-------|
| Total contracts | 510 |
| Train | 408 |
| Test | 102 |
| Features | 40,000 (word 1-3gram TF-IDF) |
| Valid clauses | 28 |
| HIGH-RISK clauses | 5 (Uncapped Liability, IP Ownership, Change of Control, Non-Compete, Liquidated Damages) |
| Primary metric | Macro-F1 (CUAD benchmark standard) |

## Experiments

### Experiment 6.1: Feature Utilization Audit
**Hypothesis:** With max_depth=4 and 50 trees, each LGBM uses <5% of 40K features. Most vocabulary is dead weight.
**Result:**
| Metric | Value |
|--------|-------|
| Mean features used per clause | 277 / 40,000 (0.7%) |
| Min used | 219 (Warranty Duration) |
| Max used | 361 (Most Favored Nation) |
| Union across ALL clauses | 4,886 / 40,000 (12.2%) |
| NEVER used by any clause | 35,114 (87.8%) |

**Interpretation:** The model is naturally sparse. Each clause detector uses <1% of available features, and 88% of the vocabulary is never split on by any model. This means the Phase 4 trigram expansion from 20K→40K helped not by providing 2x more features, but by ensuring the ~277 USEFUL features per clause were available in the vocabulary. A 5K vocabulary could theoretically work — but you'd need to know WHICH 5K in advance.

### Experiment 6.2: N-gram Size Decomposition
**Hypothesis:** If trigrams carry <5% of importance, Phase 4's expansion was really about getting more bigrams.
**Result:**
| N-gram size | Feature count | Importance share |
|------------|--------------|-----------------|
| Unigrams | 5,678 (14.2%) | 39.0% |
| Bigrams | 19,495 (48.7%) | 42.1% |
| Trigrams | 14,827 (37.1%) | 18.9% |

Per HIGH-RISK clause:
| Clause | Unigram | Bigram | Trigram |
|--------|---------|--------|---------|
| Uncapped Liability | 34.5% | 41.6% | 23.9% |
| Change Of Control | 31.7% | 44.8% | 23.5% |
| Liquidated Damages | 35.2% | 45.9% | 18.9% |
| Non-Compete | 40.4% | 43.7% | 15.9% |
| IP Ownership Assignment | 39.2% | 43.6% | 17.1% |

**Interpretation:** Trigrams carry 18.9% of importance — meaningful and justified. The top trigram features are legally specific phrases: "change of control" (SHAP=0.52), "title and interest" (0.38), "written consent of" (0.15). These are exact legal terms that unigrams and bigrams can't capture unambiguously. Bigrams dominate (42.1%) because most legal phrases are 2-word collocations.

### Experiment 6.3: TreeSHAP for HIGH-RISK Clauses
**Method:** SHAP TreeExplainer on 5 HR clause models, 102 test contracts.
**Result — Top SHAP features per clause:**

| Clause | #1 Feature | |SHAP| | #2 Feature | |SHAP| |
|--------|-----------|-------|-----------|-------|
| Uncapped Liability | consequential | 1.184 | under section | 0.269 |
| IP Ownership Assignment | title and interest | 0.379 | intellectual property | 0.321 |
| Change Of Control | change of control | 0.522 | change of | 0.359 |
| Liquidated Damages | liquidated damages | 0.488 | to any person | 0.457 |
| Non-Compete | promote | 0.270 | directly | 0.239 |

**Interpretation:** The model anchors on legally relevant terms — "consequential" for liability, "intellectual property" + "title and interest" for IP assignment, "change of control" for CoC. But Non-Compete is concerning: "promote" and "directly" are proxy features, not legal terms. This explains Non-Compete's lower F1 (0.531).

### Experiment 6.4: TP vs FP Feature Analysis
**Hypothesis:** False positives are driven by different features than true positives.
**Result:**

| Clause | TP | FP | FN | F1 | Key TP→FP discriminator |
|--------|---:|---:|---:|-----|------------------------|
| Uncapped Liability | 19 | 17 | 5 | 0.633 | "consequential" triggers both TP AND FP equally |
| IP Ownership Assignment | 16 | 9 | 7 | 0.667 | "hereby assigns" separates TP from FP (gap=+0.68) |
| Change Of Control | 17 | 19 | 5 | 0.586 | "change of control" — 4x stronger in TP vs FP |
| Non-Compete | 17 | 22 | 8 | 0.531 | No single feature separates TP/FP; proxy features dominate |
| Liquidated Damages | 7 | 6 | 7 | 0.519 | "liquidated damages" is a near-perfect TP discriminator (+2.91 gap) |

**Key finding — Uncapped Liability paradox:** The #1 SHAP feature ("consequential") has HIGHER activation in false positives (SHAP +1.50) than true positives (SHAP +1.36). The word "consequential" appears in BOTH uncapped liability clauses AND standard liability exclusion clauses ("exclusive of consequential damages"). The model can't distinguish "we're liable for consequential damages" from "we exclude consequential damages" using unigrams. This is the main precision bottleneck for HR clauses.

### Experiment 6.5: Cross-Clause Feature Sharing
**Result:**
| Metric | Value |
|--------|-------|
| Mean pairwise Jaccard | 0.025 |
| Most similar pair | Governing Law ↔ Anti-Assignment (J=0.072) |
| Features used by 10+ clauses | 2 |
| Features used by exactly 1 clause | 3,292 |

**Interpretation:** Clause detectors are near-independent (Jaccard=0.025). 3,292 features are used by exactly one clause — the model learns clause-specific vocabularies. The only "universal" features are generic terms like "10" and "cause" (used by 13 and 10 clauses respectively). This validates the per-clause LGBM architecture over a shared multi-label model.

### Experiment 6.6: SHAP vs Split-Gain Agreement
| Clause | Spearman ρ | Top-10 overlap |
|--------|-----------|---------------|
| Uncapped Liability | 0.512 | 6/10 |
| IP Ownership Assignment | 0.559 | 5/10 |
| Change Of Control | 0.573 | 8/10 |
| Non-Compete | 0.604 | 7/10 |
| Liquidated Damages | 0.490 | 8/10 |

**Interpretation:** Moderate agreement (ρ~0.5-0.6). Split-gain and SHAP agree on the MOST important features but disagree on the tail. Mark's LR coefficient analysis (Phase 6) and split-gain importance give directionally correct signals but would miss the TP/FP decomposition that SHAP enables.

### Experiment 6.7: Domain Validation
| Clause | Matched/Expected | Coverage | Verdict |
|--------|-----------------|---------|---------|
| Uncapped Liability | 2/9 | 22% | PARTIAL |
| IP Ownership Assignment | 3/7 | 43% | PARTIAL |
| Change Of Control | 3/7 | 43% | PARTIAL |
| Non-Compete | 3/7 | 43% | PARTIAL |
| Liquidated Damages | 1/6 | 17% | PROXY |

**Interpretation:** PARTIAL validation across most clauses — the model finds 2-3 of the expected legal terms in its top-20 SHAP features, but also relies on proxy features that co-occur with legal language. Liquidated Damages is the most proxy-dependent: only "liquidated damages" itself matches; other top features are contract-type indicators ("contractor", "security") rather than legal terms.

## Key Findings

1. **88% of the 40K vocabulary is dead weight.** Each clause uses 277 features on average. The model is naturally sparse — the large vocabulary ensures useful features are available, but a vocabulary audit could reduce inference cost by 8x without hurting accuracy.

2. **Trigrams carry 18.9% of importance — justified.** "Change of control", "title and interest", "written consent of" are exact legal phrases that shorter n-grams split ambiguously. The Phase 4 trigram expansion wasn't just about capacity — it was about capturing precise legal collocations.

3. **"Consequential" is the best and worst feature.** It's the #1 predictor for Uncapped Liability (|SHAP|=1.18) but triggers EQUALLY in true and false positives because it appears in both uncapped liability AND standard exclusion clauses. Improving precision requires understanding negation context ("liable for" vs "excluding").

4. **Clause detectors are near-independent (Jaccard=0.025).** Each clause has its own ~277-feature vocabulary with almost no overlap. This validates per-clause modeling over shared architectures and explains why multi-label approaches didn't help in Phase 2.

5. **Non-Compete relies on proxy features, not legal terms.** Top SHAP features are "promote", "directly", "maintain the" — general business language, not legal non-compete terminology. This explains its worst-in-class HR F1 (0.531) and suggests domain-specific feature engineering could help.

## Error Analysis
- **Uncapped Liability FP pattern:** 17 false positives, driven by "consequential" in standard liability exclusion clauses. Fix requires negation-aware features or clause-level context.
- **Change Of Control FP pattern:** 19 false positives — "change of" bigram fires on generic change language. The trigram "change of control" is 4x stronger in TP than FP, but the bigram "change of" is nearly equal in both.
- **Non-Compete has the worst TP/FP separation:** No single feature cleanly separates true from false positives. The model uses indirect proxies rather than legal terms.

## Next Steps
- Phase 7: pytest suite, README consolidation, fair LLM re-comparison with sliding-window (full-document context)
- Vocabulary pruning experiment: can we reduce from 40K to 5K without losing F1?
- Negation-aware features for "consequential" disambiguation (biggest precision lever)

## References Used Today
- [1] Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions" (NeurIPS 2017) — TreeSHAP methodology
- [2] Fan & Lin, "A Study on Threshold Selection for Multi-label Classification" (2007) — Class-prior thresholds
- [3] Hendrycks et al., "CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review" (NeurIPS 2021) — Domain validation benchmarks

## Code Changes
- notebooks/phase6_anthony_explainability.ipynb (12 code cells, 7 markdown cells, fully executed)
- results/phase6_anthony_explainability.json (all metrics and SHAP results)
- results/phase6_anthony_ngram_decomposition.png
- results/phase6_anthony_shap_hr_clauses.png
- results/phase6_anthony_cross_clause_heatmap.png
- reports/day6_phase6_anthony_report.md (this file)
