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
