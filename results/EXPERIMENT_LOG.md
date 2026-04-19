# Experiment Log — Legal Contract Analyzer
## All phases · Both researchers · CUAD dataset (510 contracts, 39 clause types)

Primary metric: **Macro-F1** (treats every clause equally; standard CUAD leaderboard metric).
Published RoBERTa-large baseline: **~0.650** (Hendrycks et al., NeurIPS 2021).

---

## Phase 1: Baselines (2026-04-13)

### Anthony — Rule-based + TF-IDF baselines

| # | Model | Macro-F1 | Micro-F1 | Precision | Recall | Features | Notes |
|---|-------|----------|----------|-----------|--------|----------|-------|
| 1.1 | Majority Class | 0.000 | 0.000 | 0.000 | 0.000 | 0 | Floor |
| 1.2 | Rule-Based (27 regex) | 0.309 | 0.256 | 0.168 | 0.532 | 27 | High FP rate |
| 1.3 | **TF-IDF + LogReg** | **0.673** | **0.656** | 0.536 | 0.844 | 10,000 | **Phase 1 champion** |
| 1.4 | TF-IDF + Domain + LogReg | 0.517 | 0.538 | 0.405 | 0.801 | 10,027 | Domain features HURT (-0.118) |
| 1.5 | Domain Only + LogReg | 0.230 | 0.215 | 0.124 | 0.812 | 27 | High recall, terrible precision |

**Published benchmarks:** BERT-base ~0.83 micro-F1, Legal-BERT ~0.85 micro-F1

**Key insight:** Domain features (binary regex) hurt when combined with TF-IDF. TF-IDF bigrams capture legal phrases more precisely than hand-coded rules.

### Mark — Keyword rules + compact ML baseline

| # | Model | Macro-F1 | Macro-AUC | HR-F1 | Notes |
|---|-------|----------|-----------|-------|-------|
| 1.1 | Majority Class | 0.222 | — | — | Floor |
| 1.2 | Keyword Rules (CTRL+F) | 0.491 | — | 0.440 | Industry "standard" |
| 1.3 | TF-IDF + LogReg (C=0.1) | 0.616 | 0.851 | — | Too regularized |
| 1.4 | **TF-IDF + LogReg (C=1.0)** | **0.642** | **0.851** | — | **Within 0.008 of RoBERTa!** |

**Key insight:** TF-IDF+LR (0.642) is only 0.008 below published RoBERTa-large — simple bag-of-words almost matches a 340M-parameter transformer.

---

## Phase 2: Multi-Model Experiment (2026-04-14)

### Anthony — Transformer vs classical sweep (7 models)

| Rank | Model | Macro-F1 | Micro-F1 | Precision | Recall | AUC | Time (s) |
|------|-------|----------|----------|-----------|--------|-----|----------|
| 1 | **TF-IDF + LightGBM** | **0.575** | **0.772** | 0.696 | 0.525 | 0.906 | 202 |
| 2 | TF-IDF + LogReg (baseline) | 0.567 | 0.712 | 0.598 | 0.572 | 0.869 | 0.3 |
| 3 | TF-IDF + SVM | 0.532 | 0.718 | 0.647 | 0.488 | 0.872 | 2.8 |
| 4 | Legal-BERT CLS + LogReg | 0.514 | 0.653 | 0.494 | 0.548 | 0.811 | 502 |
| 5 | SBERT + LogReg | 0.472 | 0.580 | 0.409 | 0.608 | 0.786 | 45 |
| 6 | Legal-BERT (fine-tuned) | 0.410 | 0.483 | 0.369 | 0.553 | 0.695 | 300 |
| 7 | BERT-base (fine-tuned) | 0.350 | 0.433 | 0.311 | 0.521 | 0.637 | 406 |

**Key insight:** Fine-tuned BERT/Legal-BERT FAIL because 512-token truncation covers only ~5% of an average 8,641-word contract. TF-IDF reads 100% of the document.

### Mark — Vocabulary sweep + XGBoost vs LR vs NB (10 runs)

| # | Approach | Macro-F1 | HR-F1 | Δ vs Phase 1 | Verdict |
|---|----------|----------|-------|--------------|---------|
| 2.1a | TF-IDF+LR (5K) | 0.594 | — | -0.048 | Too small vocab |
| 2.1b | TF-IDF+LR (10K) | 0.599 | — | -0.043 | Better |
| 2.1c | **TF-IDF+LR (20K)** | **0.603** | — | **Goldilocks** | Best compact |
| 2.1d | TF-IDF+LR (50K) | 0.584 | — | -0.058 | Past optimum |
| 2.1e | TF-IDF+LR (100K) | 0.565 | — | -0.077 | Too noisy |
| 2.2a | Complement NaiveBayes | 0.468 | — | -0.174 | Fails — independence assumption |
| 2.2b | Multinomial NB | 0.549 | — | -0.093 | Better than CNB |
| 2.3 | **XGBoost+TF-IDF(20K)** | 0.605 | **0.576** | -0.037 | **HR champion** |
| 2.4 | LSA-256+RandomForest | 0.531 | 0.443 | -0.112 | LSA kills rare legal bigrams |
| 2.5 | TF-IDF(20K)+LR corrected | 0.615 | 0.517 | -0.027 | Overall winner |

**COUNTERINTUITIVE:** 100K features drops F1 by -0.037 vs 20K. Vocabulary Goldilocks zone confirmed.
**Key insight:** XGBoost wins on HIGH-RISK clauses (HR-F1=0.576) even though LR wins overall macro-F1. The "best" overall model is NOT the legally correct model.

---

## Phase 3: Feature Engineering Deep Dive (2026-04-15)

### Anthony — Iterative feature ablation on LightGBM

| # | Approach | Macro-F1 | Δ | Notes |
|---|----------|----------|---|-------|
| 3.1 | TF-IDF + LGBM (10K) | 0.575 | baseline | Phase 2 replay |
| 3.2 | TF-IDF + LGBM (20K) | 0.589 | +0.014 | Vocabulary expansion |
| 3.3 | TF-IDF + LGBM + contract length | 0.591 | +0.016 | Small structural lift |
| 3.4 | Sentence embeddings LGBM | 0.523 | -0.052 | Worse — embeddings lose rare terms |
| 3.5 | **TF-IDF + LGBM (25K, tuned)** | **0.601** | **+0.026** | **Anthony Phase 3 best** |

### Mark — 7 feature engineering strategies

| # | Approach | Macro-F1 | HR-F1 | Δ vs P2 | Verdict |
|---|----------|----------|-------|---------|---------|
| 3.1a | Domain-Only LR (51 features) | 0.492 | 0.469 | -0.122 | Far below TF-IDF |
| 3.1b | TF-IDF(20K)+Domain → LR | 0.516 | 0.472 | -0.099 | Domain DESTROYS LR |
| 3.1c | TF-IDF(20K)+Domain → XGB | 0.610 | 0.495 | -0.004 | Domain neutral on XGB |
| 3.2 | Chi2 per-clause, best K=1000 | 0.612 | 0.451 | -0.002 | Near-global, not better |
| 3.3 | **Sliding Window MaxPool** | **0.615** | **0.510** | **+0.001** | **Helps Non-Compete +0.090** |
| 3.4a | Char 4-6gram TF-IDF | 0.597 | 0.474 | -0.018 | Char alone is weak |
| 3.4b | **Word+Char combined** | **0.619** | 0.485 | **+0.004** | **Phase 3 macro-F1 champion** |

**COUNTERINTUITIVE:** Domain features DESTROY LR (-0.099) but are neutral for XGBoost (-0.004). Mixed-scale features break LR's uniform regularization.
**Key insight:** The bottleneck is the MODEL architecture, not the features. 7 engineering strategies improved macro-F1 by at most +0.004.

---

## Phase 4: Hyperparameter Tuning + Error Analysis (2026-04-16)

### Anthony — Optuna tuning on LGBM

| # | Approach | Macro-F1 | Δ | Notes |
|---|----------|----------|---|-------|
| 4.1 | LGBM default (25K) | 0.601 | baseline | Phase 3 champion |
| 4.2 | LGBM Optuna (50 trials) | 0.614 | +0.013 | Best: n_est=150, depth=6, lr=0.05 |
| 4.3 | LGBM Optuna + class weights | 0.629 | +0.028 | Class weighting adds +0.015 |
| 4.4 | **LGBM per-clause thresholds** | **0.641** | **+0.040** | **Anthony Phase 4 best** |

### Mark — LightGBM default vs Optuna + error analysis

| # | Approach | Macro-F1 | HR-F1 | Δ | Verdict |
|---|----------|----------|-------|---|---------|
| 4.1 | LR C sweep (40K Word+Char) | 0.623 | 0.510 | +0.004 | Near-optimal at C=2.0 |
| 4.2a | **LightGBM default (20K)** | **0.666** | 0.499 | **+0.060** | **BEATS RoBERTa-large!** |
| 4.2b | LightGBM Optuna tuned | 0.656 | 0.464 | -0.010 vs default | Tuning HURT |
| 4.3a | **LR + Youden threshold** | 0.659 | **0.502** | **+0.037** | **Best Phase 4 HR** |
| 4.3b | LR + Legal-Review (recall≥0.80) | 0.657 | 0.484 | +0.035 | All HR clauses recall ≥0.80 |

**COUNTERINTUITIVE:** Optuna tuning HURT LightGBM by -0.010. Default was already near-optimal for the 20K feature space.
**Key finding:** LightGBM default BEATS published RoBERTa-large (0.666 vs 0.650). Full-document TF-IDF wins over 512-token transformer truncation.
**Error analysis:** Corr(training_size, F1) = 0.742. Data scarcity explains 55% of F1 variance. 9/102 test contracts miss at least one HIGH-RISK clause.

---

## Phase 5: Advanced Techniques + Ablation + LLM Comparison (2026-04-17)

### Anthony — Ensemble experiments

| # | Approach | Macro-F1 | HR-F1 | Δ | Verdict |
|---|----------|----------|-------|---|---------|
| 5.1 | Stacking (LGBM + LR meta-learner) | 0.658 | 0.521 | baseline | Weak meta-learner signal |
| 5.2 | **Blend α=0.5 (LGBM + LR)** | **0.678** | **0.545** | **+0.020** | **Best ensemble** |
| 5.3 | Blend α=0.7 (LGBM-heavy) | 0.671 | 0.538 | +0.013 | LGBM-heavy hurts rare clauses |
| 5.4 | Blend + SHAP threshold calibration | 0.679 | 0.541 | +0.021 | Marginal SHAP lift |

### Mark — Ablation study + LLM comparison

| # | Approach | Macro-F1 | HR-F1 | Δ | Verdict |
|---|----------|----------|-------|---|---------|
| 5.1a | Champion (all components) | 0.640 | 0.499 | baseline | Reconstructed Phase 4 |
| 5.1b | Remove class reweighting | 0.560 | 0.470 | **-0.080** | **BIGGEST drop** |
| 5.1c | Swap word→char(4-6) n-grams | 0.568 | 0.449 | -0.072 | Legal bigrams not capturable |
| 5.1d | Reduce 20K→5K features | 0.574 | 0.471 | -0.065 | Confirms Goldilocks finding |
| 5.1e | Remove bigrams (1-gram only) | 0.598 | 0.453 | -0.041 | "change of control" is a bigram |
| 5.1f | Reduce tree depth 4→2 | 0.637 | **0.523** | -0.003 | **COUNTERINTUITIVE: depth barely matters** |
| 5.2 | **Best blend (LGBM 50%+LR 50%)** | **0.691** | **0.582** | **+0.025** | **New all-time best** |
| 5.3a | LightGBM (full doc) N=30 subset | — | 0.499 | — | Full-document advantage |
| 5.3b | Claude Sonnet zero-shot (400-word) | — | 0.162 | -0.337 | 4.6% contract coverage |
| 5.3c | Claude Sonnet few-shot (400-word) | — | 0.121 | -0.377 | Few-shot WORSE than zero-shot |

**Ablation verdict:** Class reweighting is the single most critical component (-0.080 without it). More impactful than all feature engineering combined.
**LLM finding:** LightGBM beats Claude Sonnet 3× on high-risk clauses and runs 5,547× faster. Root cause: TF-IDF reads 100% of contract; Claude read 4.6%.

---

## Phase 6: Production Pipeline + Explainability (2026-04-18)

### Anthony — Full production pipeline with per-clause CV thresholds

| # | Approach | Score | Notes |
|---|----------|-------|-------|
| 6.1 | Production pipeline (train+serialize) | macro-F1=0.713 / HR-F1=0.545 | 3-fold CV thresholds |
| 6.2 | SHAP per-clause analysis | — | Top features domain-validated |
| 6.3 | Streamlit UI | — | Contract upload + clause highlights |

### Mark — Production blend + explainability deep dive

| # | Approach | Score | Δ | Verdict |
|---|----------|-------|---|---------|
| 6.1 | Production pipeline (train+serialize) | macro-F1=**0.7163** / HR-F1=0.582 | +0.025 vs P5 | New all-time best |
| 6.2 | LGBM feature importance (4 HR clauses) | — | — | Domain-aligned 3/4 clauses |
| 6.3 | LR coefficient analysis | `liquidated damages` coef=0.826 | — | Highest discriminative bigram |
| 6.4 | LGBM vs LR feature overlap | 3–21% per clause | — | Explains why blend wins |
| 6.5 | Feature position analysis | Median 28–40% into contract | — | Confirms Claude truncation root cause |
| 6.6 | Domain validation | Proxy features for Uncapped Liability | — | Model learned absence patterns |

**COUNTERINTUITIVE finding:** Uncapped Liability detected at F1=0.667 using ZERO direct legal keywords ("unlimited", "uncapped"). Model learned proxy contextual signals — because uncapped liability is drafted by OMISSION, not explicit language.

---

## Phase 7: Testing + README + Polish (2026-04-19)

| Deliverable | Status | Notes |
|-------------|--------|-------|
| tests/test_data_pipeline.py | ✅ Done | 18 tests — taxonomy, preprocessing, parquet sanity |
| tests/test_model.py | ✅ Done | 18 tests — bundle structure, vectorizer, predictions |
| tests/test_inference.py | ✅ Done | 28 tests — E2E pipeline, latency, edge cases |
| README.md | ✅ Done | Comprehensive with all results tables |
| reports/final_report.md | ✅ Done | Consolidated findings |
| All 64 pytest tests pass | ✅ 64/64 | 107s runtime |

---

## All-Time Leaderboard (both researchers, all phases)

| Rank | Model | Macro-F1 | HR-F1 | Phase | Author |
|------|-------|----------|-------|-------|--------|
| 🥇 1 | **Production LGBM+LR Blend (Youden)** | **0.7163** | 0.582 | P6 | Mark |
| 🥈 2 | Anthony Production Blend (CV thresh) | 0.713 | 0.545 | P6 | Anthony |
| 🥉 3 | LGBM+LR Blend α=0.5 | 0.691 | 0.582 | P5 | Mark |
| 4 | Anthony Blend + SHAP calibration | 0.679 | 0.541 | P5 | Anthony |
| 5 | LightGBM default (20K TF-IDF) | 0.666 | 0.499 | P4 | Mark |
| 6 | Anthony LGBM per-clause thresholds | 0.641 | — | P4 | Anthony |
| 7 | LR + Youden threshold | 0.659 | 0.502 | P4 | Mark |
| 8 | Word(20K)+Char(20K) → LR | 0.619 | 0.485 | P3 | Mark |
| 9 | TF-IDF(20K) + LR (corrected) | 0.615 | 0.517 | P2 | Mark |
| 10 | XGBoost + TF-IDF(20K) | 0.605 | **0.576** | P2 | Mark |
| — | **Published RoBERTa-large** | **0.650** | — | ref | Hendrycks+ |
| — | Claude Sonnet zero-shot | — | 0.162 | P5 | Mark |
| — | Claude Sonnet few-shot | — | 0.121 | P5 | Mark |

**Bottom line:** Custom LGBM+LR blend beats published RoBERTa-large by **+0.066 macro-F1**, runs at ~12ms/contract vs ~3s for a frontier model, and costs $0/query vs ~$0.02/query.

---

## Key Findings Across All Phases

1. **Full-document access beats model capacity** — TF-IDF on 100% of text outperforms BERT on 5%. The decisive architectural choice is document coverage, not parameter count.
2. **Vocabulary has a Goldilocks zone at 20K bigrams** — 100K features drops F1 by -0.037 on N=408 training contracts. More features ≠ better.
3. **Class reweighting is the single most critical component** — Ablation shows -0.080 macro-F1 without it. More impactful than all feature engineering combined.
4. **The "best" overall model is NOT the legally correct model** — XGBoost has lower macro-F1 than LR but higher HR-F1. Always evaluate by clause risk level.
5. **LightGBM beats Claude Sonnet 3× on high-risk clauses, runs 5,547× faster** — Domain-specific small models still win when architecture fits the task.
6. **Uncapped Liability is detected via proxy features, not explicit keywords** — The model learned that uncapped liability is drafted by OMISSION. Standard keyword search fails here.
7. **Data scarcity explains 55% of F1 variance (r=0.742)** — Hard clauses average 33 training positives; easy clauses average 205. The model ceiling IS the data ceiling.
