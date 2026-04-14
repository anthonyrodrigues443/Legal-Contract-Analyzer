# Experiment Log — Legal Contract Analyzer

## Phase 1: Baselines (2026-04-13)

| # | Model | Micro-F1 | Macro-F1 | Precision | Recall | Features | Notes |
|---|-------|----------|----------|-----------|--------|----------|-------|
| 1.1 | Majority Class | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | Floor |
| 1.2 | Rule-Based (27 regex) | 0.2558 | 0.3091 | 0.1684 | 0.5323 | 27 | High FP rate |
| 1.3 | TF-IDF + LogReg | **0.6555** | **0.6729** | 0.5358 | 0.8441 | 10,000 | **Phase 1 champion** |
| 1.4 | TF-IDF + Domain + LogReg | 0.5379 | 0.5166 | 0.4049 | 0.8011 | 10,027 | Domain features HURT (-0.118) |
| 1.5 | Domain Only + LogReg | 0.2146 | 0.2302 | 0.1237 | 0.8118 | 27 | High recall, terrible precision |

**Published benchmarks:** BERT-base ~0.83, Legal-BERT ~0.85 micro-F1

**Key insight:** Crude domain features (binary regex) hurt when combined with TF-IDF. The features are too noisy — TF-IDF bigrams capture the same legal phrases more precisely.


## Phase 2: Multi-Model Comparison (2026-04-14)

| Rank | Model | Macro-F1 | Micro-F1 | Precision | Recall | AUC | Time (s) |
|------|-------|----------|----------|-----------|--------|-----|----------|
| 1 | **TF-IDF + LightGBM** | 0.5750 | 0.7719 | 0.6958 | 0.5249 | 0.9059 | 202.2 |
| 2 | TF-IDF + LogReg (baseline) | 0.5669 | 0.7118 | 0.5979 | 0.5717 | 0.8693 | 0.3 |
| 3 | TF-IDF + SVM | 0.5316 | 0.7181 | 0.6472 | 0.4878 | 0.8721 | 2.8 |
| 4 | Legal-BERT CLS + LogReg | 0.5144 | 0.6525 | 0.4937 | 0.5480 | 0.8111 | 502.2 |
| 5 | SBERT + LogReg | 0.4721 | 0.5795 | 0.4091 | 0.6078 | 0.7855 | 45.4 |
| 6 | Legal-BERT (fine-tuned) | 0.4098 | 0.4828 | 0.3693 | 0.5529 | 0.6946 | 299.5 |
| 7 | BERT-base (fine-tuned) | 0.3501 | 0.4334 | 0.3113 | 0.5206 | 0.6373 | 406.3 |

**Champion:** TF-IDF + LightGBM (Macro-F1=0.5750)
**Key insight:** Domain pre-training effect: Legal-BERT vs BERT-base = +0.0597
