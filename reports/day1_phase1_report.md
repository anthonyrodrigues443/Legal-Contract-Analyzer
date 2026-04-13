# Phase 1: Domain Research + Dataset + EDA + Baseline — Legal Contract Analyzer
**Date:** 2026-04-13
**Session:** 1 of 7
**Researcher:** Anthony Rodrigues

## Objective
Can we automatically detect unfair/risky clauses in legal contracts? Establish baselines and understand the dataset before applying transformer models.

## Research & References
1. Chalkidis et al. (2022) — LexGLUE: A Benchmark Dataset for Legal Language Understanding (ACL 2022). Established the UNFAIR-ToS benchmark. BERT-base achieves ~0.83 micro-F1, Legal-BERT ~0.85.
2. Lippi et al. (2019) — CLAUDETTE: Automated detector of potentially unfair clauses in online ToS. Pioneered rule-based + ML approach to unfair clause detection.
3. EU Directive 93/13/EEC — Legal framework defining unfair contract terms. The 8 label categories in UNFAIR-ToS directly map to EU consumer protection law.

How research influenced experiments: The LexGLUE benchmark set micro-F1 as our primary metric and established BERT/Legal-BERT as targets. CLAUDETTE inspired our rule-based baseline using legal-domain regex patterns.

## Dataset
| Metric | Value |
|--------|-------|
| Total samples | 9,414 |
| Train / Val / Test | 5,532 / 2,275 / 1,607 |
| Unfair clauses (train) | 630 (11.4%) |
| Fair clauses (train) | 4,902 (88.6%) |
| Imbalance ratio | 7.8:1 (fair:unfair) |
| Unique unfairness types | 8 |
| Multi-label samples | ~2.5% |
| Avg word count | ~34 words |

**Dataset:** UNFAIR-ToS from LexGLUE benchmark (coastalcph/lex_glue on HuggingFace). Sentences from 50 Terms of Service documents, annotated for 8 types of potentially unfair clauses under EU consumer protection law.

## Experiments

### Experiment 1.1: Majority Class Baseline
**Hypothesis:** Predict all sentences as "fair" — lower bound.
**Result:** Micro-F1 = 0.0000. Expected for a multi-label task where predicting no labels gives zero F1.

### Experiment 1.2: Rule-Based Classifier
**Hypothesis:** Domain regex patterns (e.g., "sole discretion", "reserve the right", "at any time") can detect unfair clauses without ML.
**Method:** 27 hand-crafted regex features mapped to 8 clause types via if/then rules.
**Result:** Micro-F1 = 0.2558, Precision = 0.1684, Recall = 0.5323
**Interpretation:** Rules catch ~53% of unfair clauses but with very low precision (83% false alarm rate). Legal language is too varied for rigid patterns.

### Experiment 1.3: TF-IDF + Logistic Regression (OneVsRest)
**Hypothesis:** Standard NLP baseline — 10K TF-IDF features with balanced class weights.
**Method:** TfidfVectorizer(max_features=10000, ngram_range=(1,2), sublinear_tf=True) + LogReg(class_weight='balanced')
**Result:** Micro-F1 = 0.6555, Macro-F1 = 0.6729, Precision = 0.5358, Recall = 0.8441
**Interpretation:** Strong baseline. High recall (84%) but mediocre precision (54%). Gap to published BERT-base: -0.175.

### Experiment 1.4: TF-IDF + Domain Features + LogReg
**Hypothesis:** Adding 27 domain-specific legal features to TF-IDF will improve performance.
**Method:** Concatenated 27 legal regex features to 10K TF-IDF features.
**Result:** Micro-F1 = 0.5379 (delta: -0.1176 vs TF-IDF only)
**Interpretation:** COUNTERINTUITIVE. Domain features HURT performance by -0.118 micro-F1. The noisy binary features (low precision regex) likely introduce false signals that confuse the classifier. TF-IDF bigrams already capture "sole discretion" and "reserve the right" more granularly.

### Experiment 1.5: Domain Features Only + LogReg
**Hypothesis:** How much can 27 domain features alone achieve?
**Method:** Only the 27 hand-crafted legal features as input to LogReg.
**Result:** Micro-F1 = 0.2146, Recall = 0.8118, Precision = 0.1237
**Interpretation:** High recall (81%) means the features detect most unfair clauses, but precision is terrible (12%). The features are necessary but far from sufficient.

## Head-to-Head Comparison
| Rank | Model | Micro-F1 | Macro-F1 | Precision | Recall | Features |
|------|-------|----------|----------|-----------|--------|----------|
| 1 | TF-IDF + LogReg | **0.6555** | **0.6729** | 0.5358 | **0.8441** | 10,000 |
| 2 | TF-IDF + Domain + LogReg | 0.5379 | 0.5166 | 0.4049 | 0.8011 | 10,027 |
| 3 | Rule-Based | 0.2558 | 0.3091 | 0.1684 | 0.5323 | 27 |
| 4 | Domain Only + LogReg | 0.2146 | 0.2302 | 0.1237 | 0.8118 | 27 |
| 5 | Majority Class | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 |

**Published benchmarks:** BERT-base ~0.83, Legal-BERT ~0.85 micro-F1

## Key Findings
1. **TF-IDF + LogReg achieves 0.656 micro-F1** — solid baseline, 0.175 below published BERT-base.
2. **Domain features HURT when combined with TF-IDF** (delta: -0.118). The 27 binary regex features add noise because they're low-precision indicators. TF-IDF bigrams already capture the same legal phrases more precisely. This is the key insight: crude domain knowledge can be COUNTERPRODUCTIVE when mixed with good statistical features.
3. **Severe class imbalance** (88.6% fair) is the primary challenge. All models struggle with precision.
4. **Domain features have high recall (81%) but terrible precision (12%)** — they cast too wide a net.
5. **The gap to BERT is ~0.175** — transformer contextual understanding is needed to distinguish genuinely unfair clauses from similar-sounding fair ones.

## Error Analysis
- **Most missed clause types:** Content removal and Arbitration have lowest recall across all baselines.
- **False positives dominate:** All models flag too many fair clauses as unfair. The word patterns overlap significantly.
- **Multi-label confusion:** Models struggle to assign multiple labels to the same sentence.
- **Key difficulty:** Distinguishing "the company may terminate" (unfair) from "you may terminate" (fair) requires understanding WHO can do WHAT — syntax-level understanding that TF-IDF lacks.

## Next Steps
- Phase 2 should test BERT-base, Legal-BERT, RoBERTa, DeBERTa to close the 0.175 gap.
- Key question: Will domain-adapted models (Legal-BERT) significantly outperform general BERT?
- Try class-weighted loss vs SMOTE vs threshold tuning for the imbalance problem.
- Investigate if fine-tuned models can solve the "who can do what" syntactic challenge.

## References Used Today
- [1] Chalkidis et al. (2022) — LexGLUE: A Benchmark Dataset for Legal Language Understanding — https://aclanthology.org/2022.acl-long.297/
- [2] Lippi et al. (2019) — CLAUDETTE: automated detector of potentially unfair clauses — https://doi.org/10.1007/s10506-019-09243-2
- [3] EU Directive 93/13/EEC — Unfair Terms in Consumer Contracts

## Code Changes
- Created project structure: src/, data/, models/, results/, notebooks/, reports/, tests/, config/
- Created notebooks/phase1_eda_baseline.ipynb (executed, 31 cells, 5 plots)
- Created results/metrics.json, 5 analysis plots
- Created requirements.txt, .gitignore
