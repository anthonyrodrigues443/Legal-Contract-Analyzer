# Phase 3: Feature Engineering Deep Dive — Legal Contract Analyzer
**Date:** 2026-04-15
**Session:** 3 of 7
**Researcher:** Mark Rodrigues

## Objective
Answer: **Is the bottleneck the MODEL or the FEATURES?**
Phase 2 found TF-IDF(20K)+LR gets macro-F1=0.6146 — only 0.035 below published RoBERTa-large (~0.650). Can feature engineering close this gap, or is classical ML fundamentally limited?

## Building on Anthony's Work
**Anthony found:** BERT fails on CUAD because 512-token truncation covers only ~5% of the average contract (7,643 words). Fine-tuned BERT-base got 0.350 macro-F1 — the worst result of Phase 2. Conclusion: transformers are architecturally broken on long legal contracts.

**My approach:** If transformers fail on long docs, can long-doc-aware feature engineering fix classical ML? I tested 4 strategies:
1. Enhanced domain features (51 hand-crafted legal features)
2. Per-clause chi-squared feature selection
3. Sliding window max-pooled TF-IDF (directly inspired by Anthony's truncation finding)
4. Character n-gram representations

**Combined insight:** Anthony's BERT finding AND my feature ceiling finding together tell the same story: the bottleneck is NEITHER features NOR fine-tuning — it is the absence of a long-document architecture that can attend to clause-specific passages. Phase 5 should test sliding-window BERT (Legal-BERT with stride) or longformer-style approaches.

## Research & References
1. **Hendrycks et al. (2021) — "CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review"** (NeurIPS). The CUAD paper shows each clause type has distinct vocabulary patterns. This motivated per-clause chi-squared selection (Exp 3.2). Their reported RoBERTa-large macro-F1 is ~0.647-0.650.
2. **Bojanowski et al. (2017) — "Enriching Word Vectors with Subword Information"** (fastText, TACL). Character n-gram representations improve morphologically-rich text. Legal language has distinctive suffixes (-ification, -ability, -ility) and Latin roots that word-level models may tokenize inconsistently.
3. **Karpukhin et al. (2020) — "Dense Passage Retrieval for Open-Domain QA"** (DPR, EMNLP). The sliding window approach in Exp 3.3 is inspired by passage-level retrieval: find the highest-signal window per contract rather than computing one TF-IDF vector over 7,643 words.

How research influenced today's experiments: Hendrycks et al. motivated Exp 3.2 (clause-specific features). DPR motivated Exp 3.3. The domain feature taxonomy (Exp 3.1) was grounded in CUAD's clause annotation guidelines and standard corporate due diligence checklists.

## Dataset
| Metric | Value |
|--------|-------|
| Total samples | 510 |
| Train samples | 408 |
| Test samples | 102 |
| Valid clause types | 39 |
| Avg words per contract | 7,643 |
| Text length range | 645 – 300,768 chars |
| Class distribution | 2–97% per clause (highly imbalanced) |
| HIGH-RISK measurable | Uncapped Liability, Change of Control, Non-Compete, Liquidated Damages |

## Experiments

### Experiment 3.1a: Domain-Only LR (51 features)
**Hypothesis:** 51 hand-crafted legal features (negation patterns, monetary amounts, party asymmetry, high-risk triggers) should carry enough signal to compete with TF-IDF on clause detection.

**Method:** Extracted 51 features per contract covering structural patterns, obligation strength (shall/must vs. may), negation context near key terms, monetary amounts ($X per day = Liquidated Damages signal), party asymmetry ratios, and clause-specific boolean triggers. Trained OvR LR with class_weight='balanced'.

**Result:**
| Metric | Value | Delta vs Phase 2 LR |
|--------|-------|---------------------|
| Macro-F1 | 0.4923 | -0.1223 |
| HR-F1 | 0.469 | -0.048 |
| Train time | 0.8s | — |

**Interpretation:** 51 features get 0.492 vs TF-IDF(20K)'s 0.615. Even 51 well-designed features are far outperformed by 20,000 learned vocabulary features. Legal signal is too distributed and contextual for hand-crafted patterns to capture alone. The features encode presence/absence but not the legal *context* that makes them significant.

---

### Experiment 3.1b: TF-IDF(20K) + Domain Features → LR
**Hypothesis:** Adding domain features should complement TF-IDF — they capture patterns (negation context, monetary amounts) that TF-IDF misses.

**Result:**
| Metric | Value | Delta vs Phase 2 LR |
|--------|-------|---------------------|
| Macro-F1 | 0.5161 | -0.0985 |
| HR-F1 | 0.472 | -0.045 |

**Interpretation:** SHOCKING — adding domain features to TF-IDF DESTROYS performance by -0.098 macro-F1. Why? In a 20,051-dimensional space (20K TF-IDF + 51 domain), LR's L2 regularization distributes penalty across ALL features uniformly. The 51 domain features have different scales and correlations with TF-IDF features, causing regularization interference. This is the "curse of feature heterogeneity" — mixing sparse boolean features with dense TF-IDF scores confuses LR's optimization.

---

### Experiment 3.1c: TF-IDF(20K) + Domain Features → XGBoost
**Hypothesis:** XGBoost can learn to ignore irrelevant features via tree structure; the negative interference that hurts LR should not hurt XGBoost.

**Result:**
| Metric | Value | Delta vs Phase 2 XGB |
|--------|-------|----------------------|
| Macro-F1 | 0.6102 | +0.0050 |
| HR-F1 | 0.495 | -0.081 |

**Interpretation:** Confirmed — XGBoost is near-neutral with domain features added (+0.005 overall). Tree-based models naturally handle mixed-scale, heterogeneous features via feature selection at each split. BUT XGBoost's HR-F1 drops from 0.576 to 0.495 (-0.081). Adding domain features may cause XGBoost to focus on "easier" common clauses and neglect the rare high-risk ones where domain features provide misleading signal.

---

### Experiment 3.2: Per-Clause Chi-Squared Feature Selection (K = 100, 300, 500, 1000)
**Hypothesis:** Each clause type has distinct vocabulary. The word "liquidated" is highly discriminative for Liquidated Damages but noise for Non-Compete. Selecting top-K features per clause should outperform the global 20K vocabulary.

**Method:** For each of 39 clauses, compute chi2(TF-IDF feature, label) and select top-K features. Train clause-specific LR on only those features.

**Result:**
| K | Macro-F1 | HR-F1 | Delta vs Phase 2 LR |
|---|----------|-------|---------------------|
| 100 | 0.5824 | 0.370 | -0.0322 |
| 300 | 0.5964 | 0.367 | -0.0182 |
| 500 | 0.5982 | 0.443 | -0.0164 |
| 1000 | **0.6124** | 0.451 | **-0.0022** |

**Per-clause at best K=1000:**
| High-Risk Clause | Chi2-F1 |
|-----------------|---------|
| Uncapped Liability | 0.571 |
| Change Of Control | 0.471 |
| Non-Compete | 0.513 |
| Liquidated Damages | 0.250 |

**Interpretation:** Per-clause selection does NOT beat the global vocabulary — the best K=1000 still underperforms global TF-IDF(20K) by -0.0022. This is counterintuitive: the legal literature says clause vocabularies are distinct, but the model needs global context features (e.g., "the contract mentions indemnification AND warranty" together) that per-clause selection removes. Also: chi2 feature selection is noisy on small N=408 training samples — many high-signal features for rare clauses don't reach chi2 significance thresholds.

---

### Experiment 3.3: Sliding Window Max-Pooled TF-IDF
**Hypothesis:** Directly addressing Anthony's finding — BERT's 512-token truncation = 5% coverage. TF-IDF on a 7,643-word document dilutes rare clause terms. Max-pooling over 400-word windows captures the highest-signal passage.

**Method:** Each contract split into overlapping 400-word windows (stride=200). TF-IDF computed per window. Element-wise max across all windows. Compare vs standard TF-IDF on same 10K vocabulary.

**Result:**
| Model | Macro-F1 | HR-F1 | Delta |
|-------|----------|-------|-------|
| Standard TF-IDF(10K)+LR | 0.6183 | 0.474 | — |
| **Sliding Window MaxPool(10K)+LR** | **0.6151** | **0.510** | -0.003 overall |

**Per-clause on HIGH-RISK clauses:**
| Clause | Sliding Window | Standard TF-IDF | Delta |
|--------|---------------|-----------------|-------|
| Non-Compete | **0.513** | 0.423 | **+0.090** |
| Liquidated Damages | **0.364** | 0.316 | **+0.048** |
| Change Of Control | **0.549** | 0.515 | +0.034 |
| Uncapped Liability | 0.615 | **0.643** | -0.027 |

**Interpretation:** MIXED BUT INTERESTING — aggregate macro-F1 barely changes (-0.003), but sliding window dramatically improves Non-Compete (+0.090) and Liquidated Damages (+0.048). These are clauses that appear in specific sections of a contract (Non-Compete typically in a dedicated section, not distributed throughout). Max-pooling finds the high-signal section. Uncapped Liability *worsens* (-0.027) because it appears distributed across the contract — standard full-doc TF-IDF captures all instances, while max-pooling may fix on one misleading window.

**Verdict:** Anthony's long-document insight IS partially validated at the clause level, but the effect is heterogeneous — some clauses need global context, others need local passage focus.

---

### Experiment 3.4a: Character 4-6 gram TF-IDF
**Hypothesis:** Legal morphology is distinctive — suffixes like -ification, -ility, -ation, and Latin roots distinguish legal vocabulary. Character n-grams (4-6 chars) capture subword patterns that word bigrams miss.

**Result:**
| Metric | Value | Delta vs Phase 2 LR |
|--------|-------|---------------------|
| Macro-F1 | 0.5971 | -0.0175 |
| HR-F1 | 0.474 | -0.043 |

**Interpretation:** Character n-grams alone underperform word bigrams by -0.018 macro-F1. Legal signal is primarily at the word/phrase level (specific terms matter), not at the character level.

---

### Experiment 3.4b: Word Bigrams + Character 4-6 grams Combined
**Hypothesis:** Combining word bigrams (phrase-level signal) + character n-grams (morphological signal) should be additive.

**Result:**
| Metric | Value | Delta vs Phase 2 LR |
|--------|-------|---------------------|
| Macro-F1 | **0.6187** | **+0.0041** |
| HR-F1 | 0.485 | -0.032 |
| Features | 40,000 | 2× baseline |

**Interpretation:** Word+Char combined is the Phase 3 Macro-F1 winner (+0.004 vs Phase 2). The marginal gain is real but small — character morphology adds subword-level disambiguation (e.g., "liability" vs "liabilities" vs "liable" → different char n-grams) but TF-IDF already has good coverage of these as separate word tokens. The gain comes at 2× feature cost.

## Head-to-Head Comparison

| Rank | Model | Macro-F1 | HR-F1 | Delta P2 | Notes |
|------|-------|----------|-------|----------|-------|
| 1 | **Word(20K)+Char(20K) → LR** | **0.6187** | 0.485 | **+0.0041** | Phase 3 Macro-F1 champion |
| 2 | Standard TF-IDF(10K)+LR | 0.6183 | 0.474 | +0.0037 | Near-equivalent to 20K |
| 3 | **SlidingWindow-MaxPool(10K)+LR** | 0.6151 | **0.510** | +0.0005 | **Phase 3 HR-F1 winner** |
| 4 | Phase2 TF-IDF(20K)+LR [P2] | 0.6146 | 0.517 | baseline | Phase 2 champion |
| 5 | Chi2-SelectK(k=1000)+LR | 0.6124 | 0.451 | -0.0022 | Near-global |
| 6 | TF-IDF(20K)+Domain → XGBoost | 0.6102 | 0.495 | -0.0044 | Domain adds noise |
| 7 | Phase2 XGBoost+TF-IDF [P2] | 0.6052 | 0.576 | — | P2 HR champion |
| 8 | Char N-gram(4-6, 20K)+LR | 0.5971 | 0.474 | -0.0175 | Subword alone is weak |
| 9 | TF-IDF(20K)+Domain → LR | 0.5161 | 0.472 | **-0.0985** | **Domain DESTROYS LR** |
| 10 | Domain-Only LR (51 features) | 0.4923 | 0.469 | -0.1223 | Cannot compete with TF-IDF |

## Key Findings

1. **THE BOTTLENECK IS THE MODEL, NOT THE FEATURES.** We tried 10 feature variants and the best improvement was +0.004 macro-F1. The gap from 0.615 to 0.650 (published RoBERTa) and 0.780 (human) cannot be closed with classical features.

2. **DOMAIN FEATURES DESTROY LR BUT ARE NEUTRAL FOR XGBOOST.** Adding 51 hand-crafted legal features to TF-IDF+LR loses -0.098 macro-F1. The same features with XGBoost: -0.004. This is NOT a domain knowledge problem — it is a regularization/optimization problem with mixed feature types in linear models.

3. **SLIDING WINDOW HELPS SPECIFIC HIGH-RISK CLAUSES.** Non-Compete: +0.090 F1, Liquidated Damages: +0.048 F1, Change of Control: +0.034. But Uncapped Liability: -0.027. Each clause type needs a *different* context window — there is no universal optimal window size.

4. **10K VOCABULARY MATCHES 20K.** Standard TF-IDF(10K)+LR gets 0.6183 vs TF-IDF(20K)+LR's 0.6146. Contradicts Phase 2's ablation which found 20K > 10K. The difference is now within noise — Phase 2's ablation may have been run without `class_weight='balanced'`.

5. **CHI-SQUARED SELECTION DOESN'T HELP.** Even with clause-specific feature selection, we can't beat the global vocabulary. With only N=408 training contracts, chi2 feature selection is noisy — rare clause examples don't have enough signal to find truly discriminative features reliably.

## Frontier Model Comparison
Not applicable for Phase 3 (feature engineering phase). Scheduled for Phase 5.

## Error Analysis
- Liquidated Damages remains the hardest clause at 0.250-0.364 across all approaches. The term "liquidated damages" appears explicitly in most contracts, but the clause itself is rare (only a few positive examples in test). The model is overfitting the keyword without learning context.
- Non-Compete improves dramatically with sliding window (0.423 → 0.513) suggesting these clauses appear in concentrated sections. Future work: use section-specific features.
- XGBoost + Domain Features loses HR-F1 (-0.081) despite gaining macro-F1. The domain features appear to help easy clauses (common, frequent) at the cost of hard ones (rare, high-risk).

## Next Steps
- **Phase 4 direction:** Threshold optimization per clause (not just 0.5 decision boundary) — high-risk clauses should use a lower threshold (higher recall). Also: error analysis on which specific contracts fail systematically (long vs. short, industry type, contract vintage).
- **Key Phase 4 hypothesis:** "Optimizing the decision threshold per clause type (calibration) improves HR-F1 more than any feature engineering did." Uncapped Liability and Non-Compete may benefit from different thresholds.
- **Broader implication:** The Phase 3 verdict ("bottleneck = model") means Phase 5's LLM comparison will be particularly interesting — if GPT-5.4/Opus 4.6 can close the gap from 0.62 to ~0.78 (human level), that tells us whether architectural capacity or legal training data is the limiting factor.

## References Used Today
- [1] Hendrycks, D. et al. (2021). "CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review." NeurIPS Datasets Track. https://arxiv.org/abs/2103.06268
- [2] Bojanowski, P. et al. (2017). "Enriching Word Vectors with Subword Information." Transactions of the ACL. https://arxiv.org/abs/1607.04606
- [3] Karpukhin, V. et al. (2020). "Dense Passage Retrieval for Open-Domain Question Answering." EMNLP. https://arxiv.org/abs/2004.04906

## Code Changes
- `notebooks/phase3_mark_feature_engineering.py` — Phase 3 experimental script (all 4 experiments)
- `results/phase3_mark_feature_engineering.png` — 5-panel visualization
- `results/phase3_mark_metrics.json` — All experiment metrics
- `reports/day3_phase3_mark_report.md` — This report
