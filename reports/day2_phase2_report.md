# Phase 2: Multi-Model Experiment — Legal Contract Analyzer
**Date:** 2026-04-14
**Session:** 2 of 7
**Researcher:** Anthony Rodrigues

## Objective
Can transformer models (BERT, Legal-BERT) close the gap between classical TF-IDF baselines (0.567 macro-F1) and published SOTA (~0.650 macro-F1) on CUAD multi-label clause detection?

## Research & References
1. **Chalkidis et al. (2020)** — "LEGAL-BERT: The Muppets straight out of Law School" — Domain pre-training on 12GB legal corpora improves legal NLP tasks by +5-10% over BERT-base. We test whether this holds on CUAD clause detection.
2. **He et al. (2021)** — DeBERTa: Disentangled attention mechanism improves position-dependent token understanding. Relevant because legal clause structure depends heavily on position within contract sections.
3. **Hendrycks et al. (2021)** — CUAD paper: RoBERTa-large achieves ~0.65 macro-F1 with standard fine-tuning, but uses sophisticated sliding window approach for long documents. Our initial test uses 512 truncation to isolate the effect of model choice vs document handling.

How research influenced experiments: The Chalkidis finding that domain pre-training helps guided our comparison of Legal-BERT vs BERT-base. The CUAD paper's sliding window approach explains why published numbers are higher than our truncated results — the gap is document coverage, not model architecture.

## Dataset
| Metric | Value |
|--------|-------|
| Dataset | CUAD v1 (official train/test split) |
| Total contracts | 510 (408 train / 102 test) |
| Clause categories | 36 (risk-relevant) |
| Primary metric | Macro-F1 |
| Mean contract length | ~7,861 words |
| Avg labels per contract | ~12 |

## Experiments

### Experiment 2.1: TF-IDF + SVM (Linear)
**Hypothesis:** Max-margin SVM should outperform LogReg on rare clauses with tight decision boundaries.
**Method:** LinearSVC with CalibratedClassifierCV, class_weight='balanced', OneVsRest, 50K TF-IDF features.
**Result:** Macro-F1 = 0.5316, Macro-AUC = 0.8721
**Interpretation:** SVM underperforms LogReg by -0.035 macro-F1. The max-margin objective is too conservative for rare classes — it trades recall for precision, but with imbalanced data we need recall more. LogReg's probabilistic output + balanced class weights handles the imbalance better.

### Experiment 2.2: TF-IDF + LightGBM
**Hypothesis:** Gradient boosting captures non-linear feature interactions that linear models miss (e.g., "indemnification" AND "uncapped" together signals different risk).
**Method:** LightGBM per clause, scale_pos_weight for imbalance, 200 estimators, max_depth=6.
**Result:** Macro-F1 = 0.5750, Macro-AUC = 0.9059
**Interpretation:** **Phase 2 champion.** LightGBM beats LogReg by +0.008 macro-F1 and achieves the highest AUC (0.906). The tree-based model captures keyword co-occurrence patterns that linear models cannot. Notably high precision (0.696) suggests it learns tight, specific rules per clause type.

### Experiment 2.3: Sentence-BERT (all-MiniLM-L6-v2) + LogReg
**Hypothesis:** Dense 384d embeddings capture semantic similarity across different phrasings of the same legal concept.
**Method:** Mean-pooled 400-word chunk embeddings from all-MiniLM-L6-v2, then LogReg classifier.
**Result:** Macro-F1 = 0.4721, Macro-AUC = 0.7855
**Interpretation:** Dense embeddings lose the precise legal vocabulary that TF-IDF captures. "Governing law" is a near-perfect TF-IDF feature (0.971 F1) but gets diluted in a 384d embedding space averaged across 20+ chunks. General-purpose sentence embeddings are too lossy for legal clause detection.

### Experiment 2.4: Legal-BERT CLS Embeddings + LogReg
**Hypothesis:** Legal-BERT's domain-specific 768d representations should outperform general Sentence-BERT.
**Method:** CLS token extraction from Legal-BERT (nlpaueb/legal-bert-base-uncased), mean-pooled across 512-token chunks.
**Result:** Macro-F1 = 0.5144, Macro-AUC = 0.8111
**Interpretation:** Legal domain pre-training adds +0.042 over Sentence-BERT. But still below TF-IDF (+0.053 gap). The CLS token captures legal language patterns better than general embeddings, but averaging across chunks loses the precise location information that TF-IDF preserves.

### Experiment 2.5: BERT-base Fine-tuned (Multi-label)
**Hypothesis:** Fine-tuning adapts BERT's representations to the specific clause detection task, unlike frozen embeddings.
**Method:** bert-base-uncased, 36-output classification head, BCEWithLogitsLoss with pos_weight, 512 truncation, 5 epochs, lr=2e-5.
**Result:** Macro-F1 = 0.3501, Macro-AUC = 0.6373
**Interpretation:** **Fine-tuned BERT is the WORST model tested.** With only 408 training contracts and 512-token truncation losing ~94% of each contract, fine-tuning overfits to superficial patterns in the truncated preamble while ignoring the actual clause content later in the document.

### Experiment 2.6: Legal-BERT Fine-tuned (Multi-label)
**Hypothesis:** Domain pre-training should help even with limited fine-tuning data.
**Method:** nlpaueb/legal-bert-base-uncased, same architecture as Exp 2.5.
**Result:** Macro-F1 = 0.4098, Macro-AUC = 0.6946
**Interpretation:** Domain pre-training adds +0.060 over BERT-base (consistent with Chalkidis et al.'s +5-10% finding). But Legal-BERT fine-tuned (0.410) is STILL worse than frozen Legal-BERT CLS + LogReg (0.514). Fine-tuning with only 408 examples destroys the pre-trained representations faster than it learns task-specific patterns.

## Head-to-Head Comparison
| Rank | Model | Macro-F1 | Micro-F1 | Precision | Recall | AUC | Time (s) |
|------|-------|----------|----------|-----------|--------|-----|----------|
| 1 | **TF-IDF + LightGBM** | **0.5750** | 0.7719 | 0.6958 | 0.5249 | 0.9059 | 202.2 |
| 2 | TF-IDF + LogReg | 0.5669 | 0.7118 | 0.5979 | 0.5717 | 0.8693 | 0.3 |
| 3 | TF-IDF + SVM | 0.5316 | 0.7181 | 0.6472 | 0.4878 | 0.8721 | 2.8 |
| 4 | Legal-BERT CLS + LogReg | 0.5144 | 0.6525 | 0.4937 | 0.5480 | 0.8111 | 502.2 |
| 5 | SBERT + LogReg | 0.4721 | 0.5795 | 0.4091 | 0.6078 | 0.7855 | 45.4 |
| 6 | Legal-BERT (fine-tuned) | 0.4098 | 0.4828 | 0.3693 | 0.5529 | 0.6946 | 299.5 |
| 7 | BERT-base (fine-tuned) | 0.3501 | 0.4334 | 0.3113 | 0.5206 | 0.6373 | 406.3 |

Published: RoBERTa-large ~0.650, Human ~0.780

## Key Findings

1. **TF-IDF dominates transformers on long-document legal clause detection.** All three TF-IDF models outperform all four transformer-based models. The gap: +0.225 (best classical vs best fine-tuned transformer). This contradicts the assumption that transformers are always better for NLP tasks.

2. **The bottleneck is document coverage, not model sophistication.** Contracts average 7,861 words (~10K tokens). BERT sees only the first 512 tokens (~5%). TF-IDF processes 100% of the document. The published RoBERTa result (0.650) uses sliding window — our Phase 3 must implement this.

3. **Fine-tuning HURTS with small N + long documents.** Fine-tuned Legal-BERT (0.410) is worse than frozen Legal-BERT CLS + LogReg (0.514). With only 408 training contracts, fine-tuning destroys pre-trained representations faster than it learns new ones. Frozen features + simple classifier wins.

4. **Domain pre-training consistently helps, but not enough.** Legal-BERT beats BERT-base by +0.060 (fine-tuned) and +0.042 (frozen embeddings). Chalkidis et al.'s finding holds. But domain knowledge cannot compensate for seeing only 5% of each document.

5. **LightGBM captures clause co-occurrence patterns.** The tree model's +0.008 over LogReg comes from learning that certain clause types co-occur (e.g., Indemnification + Cap on Liability). Linear models treat each clause independently; trees can split on feature combinations.

## Per-Clause Analysis
- Classical ML wins 22/26 clauses. Transformers win only 4.
- Transformer advantage is concentrated on RARE clauses with <5% prevalence: Unlimited License (+0.40), Third Party Beneficiary (+0.23), Volume Restriction (+0.08), Change of Control (+0.07).
- Classical advantage is largest on WELL-DEFINED clauses: No-Solicit (-0.50), Governing Law (-0.45), Insurance (-0.47). These have precise legal vocabulary that TF-IDF captures perfectly.

## Error Analysis
- **Why transformers fail:** 512-token truncation means the model only sees the contract preamble (parties, dates, definitions). The actual risk clauses appear in Sections 5-15, well beyond the cutoff.
- **Where transformers help:** On rare clauses (Unlimited License, Third Party Beneficiary), the transformer's pre-trained legal language understanding provides some signal even from truncated text, while TF-IDF has too few positive examples to learn from.
- **LogReg vs Mark's result:** My reproduction gets 0.567 vs Mark's 0.642. The difference: Mark used random 80/20 split, I used the official CUAD train/test split. The official split may be harder due to contract-level stratification.

## Next Steps
Phase 3 should investigate:
1. **Sliding window for transformers** — Process full contract in overlapping 512-token windows, aggregate predictions. This is how the published RoBERTa result (0.650) works.
2. **Hierarchical transformer** — Encode chunks, then use a second-level aggregator to combine chunk representations.
3. **Feature engineering on TF-IDF champion** — Add n-gram features (trigrams), section-aware features, contract structure features.
4. **Hybrid approach** — TF-IDF for common clauses + transformer for rare clauses. Each paradigm has complementary strengths.

## References Used Today
- [1] Chalkidis, I., et al. (2020). LEGAL-BERT: The Muppets straight out of Law School. EMNLP 2020 Findings. https://arxiv.org/abs/2010.02559
- [2] He, P., et al. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention. ICLR 2021. https://arxiv.org/abs/2006.03654
- [3] Hendrycks, D., et al. (2021). CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review. arXiv:2103.06268. https://arxiv.org/abs/2103.06268
- [4] Chalkidis, I., et al. (2022). LexGLUE: A Benchmark for Legal NLP. ACL 2022. https://arxiv.org/abs/2110.00976

## Code Changes
- `notebooks/phase2_model_comparison.ipynb` — Full Phase 2 research notebook (36 cells, executed, 3 plots)
- `data/raw/CUADv1.json`, `data/raw/test.json`, `data/raw/train_separate_questions.json` — CUAD QA data from GitHub
- `results/phase2_metrics.json` — All experiment metrics with per-clause breakdown
- `results/phase2_model_comparison.png` — Bar chart + speed/accuracy tradeoff
- `results/phase2_per_clause_delta.png` — Per-clause transformer vs classical comparison
- `results/phase2_training_curves.png` — Transformer training convergence
- `results/EXPERIMENT_LOG.md` — Updated with Phase 2 results
