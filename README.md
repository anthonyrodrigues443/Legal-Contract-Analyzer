# Legal Contract Analyzer

**Multi-label clause detection on real commercial contracts.** A per-clause LightGBM trained on 40K word 1-3gram TF-IDF features matches published RoBERTa-large SOTA (macro-F1 = 0.647 vs 0.650) while running in ~9 ms per contract at $0/query — 300× faster than a frontier LLM making zero-shot calls, at 3.6× the HIGH-RISK clause F1.

> **Headline finding:** We reached SOTA parity by *deleting* threshold-learning entirely. Our champion uses `threshold = training positive rate` per clause — no CV, no validation set, no test-set contact. Plug-in F1-optimal rules from statistical theory beat every CV-tuned variant we measured.

> **Second finding:** 88% of the 40,000-feature TF-IDF vocabulary is *never used* by any LightGBM. Each clause detector picks ~277 features from 40K. The "#1 predictor" for Uncapped Liability — the word *"consequential"* — fires **equally** in true and false positives, because bag-of-words can't distinguish *"liable for consequential damages"* from *"excluding consequential damages."*

---

## Dataset

**[CUAD v1](https://www.atticusprojectai.org/cuad)** — Contract Understanding Atticus Dataset (Hendrycks et al., NeurIPS 2021)

| Metric | Value |
|--------|-------|
| Total contracts | 510 real SEC commercial contracts |
| Clause types annotated | 41 (28 valid after test-positive filter) |
| Average contract length | 8,641 words |
| Train / Test split | 408 / 102 (seed=42, deterministic shuffle) |
| Class balance | Highly imbalanced; rarest valid clause has 3 test positives |
| Annotators | Lawyers (Atticus Project) |

**Primary metric:** Macro-F1 (treats every clause equally, matching CUAD benchmark convention).
**Secondary metric:** HR-F1 (macro-F1 on 5 HIGH-RISK clauses: Uncapped Liability, IP Ownership, Change of Control, Non-Compete, Liquidated Damages).

---

## Final Results

### Champion model — Phase 5 rework (LGBM + 40K trigrams + class-prior thresholds)

| Metric | Value | Reference |
|--------|------:|-----------|
| **Macro-F1** | **0.6471** | RoBERTa-large CUAD SOTA ~0.65 (within ~0.01 noise band) |
| **HR-F1** | **0.5872** | Claude Sonnet zero-shot: 0.162 (3.6× better) |
| **Macro-AUC** | **0.8690** | — |
| **Macro-Recall** | 0.7051 | — |
| **Macro-Precision** | 0.6081 | — |
| **Inference** | 9 ms/contract | Claude ~3,000 ms (300× faster) |
| **Cost/1K contracts** | ~$0 | Claude ~$15–20 |

### Head-to-head vs frontier LLMs (HR clause detection, fair comparison)

| Model | HR-F1 | Latency | Cost/1K | Contract coverage |
|-------|------:|--------:|--------:|------------------:|
| **LightGBM (ours, full doc)** | **0.587** | **9 ms** | **$0** | 100% |
| Claude Sonnet zero-shot (400-word excerpt) | 0.162 | 3,200 ms | $15 | 4.6% |
| Claude Sonnet few-shot (400-word excerpt) | 0.121 | 4,100 ms | $20 | 4.6% |

The comparison is constrained by Claude's prompt cost: a full-document pass across 102 test contracts at ~$0.02/call would run ~$25 just for the baseline. The LLM numbers are consistent across sampled shards.

---

## Key Findings

1. **Class-prior thresholds beat CV-tuned thresholds.** SCut (per-fold F1 search) is structurally broken on rare labels at n=408 — per-fold validation F1 on ~5 positives is noise-dominated (Fan & Lin 2007). The plug-in rule `threshold = train_positive_rate` matches SOTA on macro-F1 and wins HR-F1 by +0.029 — zero hyperparameters.

2. **The LR blend is NOT universally additive.** It helps on 20K word 1-2gram (+0.01) but HURTS on 40K word 1-3gram by -0.038. Phase 6 reworked the pipeline to ship LGBM-alone.

3. **Mark's Phase 5/6 macro-F1 of 0.69–0.72 was inflated by a test-set leak** — Youden thresholds were fit on `y_test`. The honest, deployable number is 0.647. We shipped the correction.

4. **88% of the 40K vocabulary is dead weight.** Each clause uses 277 features on average; union across all clauses = 4,886 (12.2%). A 5K vocab could match performance — but you'd need to know which 5K a priori.

5. **Trigrams carry 18.9% of importance — justified.** "Change of control", "title and interest", "written consent of" — exact legal phrases that shorter n-grams split ambiguously.

6. **Clause detectors are near-independent (Jaccard = 0.025).** 3,292 features are used by exactly one clause. This validates per-clause architecture over shared multi-label models.

7. **"Consequential" is the best AND worst feature.** #1 SHAP contributor for Uncapped Liability (|SHAP|=1.18) but triggers equally in true and false positives because bag-of-words can't understand negation ("liable for" vs "excluding consequential"). Main precision bottleneck.

8. **Full-document access beats frontier-model zero-shot** — at a 300× speed advantage. The model reads 100% of a 9K-word contract in 9 ms. Claude with 400 words of excerpt reads 4.6% in 3 seconds.

---

## Architecture

```
Contract text (full document, avg 8,641 words)
             │
             ▼
    TF-IDF Vectorizer
    ────────────────────────
    40K word 1-3gram
    sublinear_tf, min_df=2
    Phase 4/5 ablation winner
             │
             ▼
    Per-clause LightGBM (28 models)
    ────────────────────────────────
    n_estimators=50, depth=4, lr=0.15
    subsample=0.8, colsample=0.4
    scale_pos_weight per-clause
             │
             ▼
    Class-prior thresholds
    ────────────────────────
    threshold[c] = train_positive_rate[c]
    No CV, no val split, no test contact.
    F1-optimal plug-in rule (Lipton & Elkan 2014).
             │
             ▼
    28 binary clause predictions
    + overall risk: HIGH / MEDIUM / LOW
```

---

## Setup

```bash
git clone https://github.com/anthonyrodrigues443/Legal-Contract-Analyzer.git
cd Legal-Contract-Analyzer
pip install -r requirements.txt

# Train the production model (~1 minute on CPU)
python -m src.train
# Expected: macro-F1≈0.647  HR-F1≈0.587  macro-AUC≈0.869

# Run inference
python -m src.predict --demo
python -m src.predict --file path/to/contract.txt
python -m src.predict --text "contract text..."

# Streamlit UI
streamlit run app.py
```

---

## Usage

```python
from pathlib import Path
from src.predict import load_model, predict

bundle = load_model()  # reads models/ directory
result = predict("Full contract text goes here...", bundle)

print(result["overall_risk"])          # "HIGH" / "MEDIUM" / "LOW"
print(result["high_risk_detected"])    # ["Non-Compete", "Liquidated Damages"]
print(result["detected_count"])        # 7
print(result["latency_ms"])            # ~9

for clause, info in result["clauses"].items():
    if info["detected"]:
        print(f"{clause}: prob={info['probability']:.3f} "
              f"thr={info['threshold']:.3f} risk={info['risk_level']}")
```

---

## Clause Risk Taxonomy

| Risk Level | Clauses |
|------------|---------|
| **HIGH** | Uncapped Liability, IP Ownership Assignment, Change Of Control, Non-Compete, Liquidated Damages, Joint IP Ownership |
| **MEDIUM** | Indemnification, Cap On Liability, Termination For Convenience, Exclusivity, No-Solicit Of Employees, No-Solicit Of Customers, Revenue/Profit Sharing, Most Favored Nation, Covenant Not To Sue |
| **STANDARD** | All remaining 25 clause types |

Per-clause F1 (test set, Phase 5 champion):

| Clause | F1 | Clause | F1 |
|--------|---:|--------|---:|
| Governing Law | 0.978 | Uncapped Liability | 0.633 |
| License Grant | 0.932 | ROFR/ROFO/ROFN | 0.629 |
| Anti-Assignment | 0.930 | Warranty Duration | 0.615 |
| Cap On Liability | 0.912 | Revenue/Profit Sharing | 0.615 |
| Audit Rights | 0.900 | Non-Transferable License | 0.600 |
| Insurance | 0.889 | Change Of Control | 0.586 |
| No-Solicit Of Employees | 0.833 | Termination For Convenience | 0.584 |
| Renewal Term | 0.769 | Unlimited/All-You-Can-Eat | 0.571 |
| Covenant Not To Sue | 0.756 | Third Party Beneficiary | 0.556 |
| Minimum Commitment | 0.734 | Non-Compete | 0.531 |
| IP Ownership Assignment | 0.667 | Liquidated Damages | 0.519 |
| Post-Termination Services | 0.667 | Notice Period… | 0.512 |
| Volume Restriction | 0.578 | Non-Disparagement | 0.286 |
| Most Favored Nation | 0.211 | No-Solicit Of Customers | 0.125 |

---

## Tests

```bash
python -m pytest tests/ -v
# 59 passed, 7 skipped (parquet-only) in ~40s
```

| Test file | Tests | Covers |
|-----------|------:|--------|
| `tests/test_data_pipeline.py` | 18 | CUAD taxonomy, question→category mapping, parquet sanity |
| `tests/test_model.py` | 18 | Bundle structure, vectorizer 40K trigram config, training manifest |
| `tests/test_inference.py` | 23 | E2E pipeline, latency < 3s, edge cases, semantic correctness |

Training-manifest tests enforce the Phase 5 champion invariants:
- macro-F1 ≥ 0.63 (SOTA parity band)
- macro-AUC ≥ 0.85 (ranking quality)
- HR-F1 ≥ 0.50 (beats LLM baseline 3×)
- vocabulary is 30K–40K with ngram_range = (1, 3)

---

## Research Journey

### Phase 1 — Domain Research + Baselines (Apr 13)
TF-IDF + LR reaches macro-F1 = 0.642 — only **0.008 below** published RoBERTa-large. Keyword rules get F1 = 0.440 on Uncapped Liability because "unlimited" appears in both present-and-absent contexts.

### Phase 2 — Multi-model experiment (Apr 14)
Tested 6 paradigms. **Fine-tuned BERT is the worst model** (macro-F1 = 0.350) — 512-token truncation loses 95% of document context. TF-IDF + LGBM wins at macro-F1 = 0.615 over fine-tuned transformers because the document is fully visible.

### Phase 3 — Feature engineering (Apr 15)
Hand-crafted legal-syntax features HURT macro-F1 by -0.007; hybrid (global + positional + syntactic) is the **worst** combination at -0.008. "More features hurt" confirmed at n=408. Positional TF-IDF (4 quartiles) was the Phase 3 winner at +0.021 macro-F1.

### Phase 4 — Feature ablation + router (Apr 18 catch-up)
Word 1-3gram @ 40K beats positional TF-IDF by +0.025 — Phase 3's "positional wins" story was mostly capacity, not structure. Oracle per-clause model selection reaches 0.676 (above SOTA), but a learned router only agrees with oracle 46% at n=408. **Scale-bound ceiling.**

### Phase 5 — Threshold ablation (Apr 18)
The 0.058 oracle-vs-CV gap identified in Phase 4 → close via threshold work. 5-fold SCut HURTS by -0.011 (more folds don't help on ~5 rare-class positives). Literature pivot: Fan & Lin 2007 → PCut / class-prior rules. **Class-prior threshold wins at macro-F1 = 0.6471 and HR-F1 = 0.5872**, beating every CV-tuned variant on HR-F1 by +0.029.

### Phase 6 — Explainability (Apr 19)
TreeSHAP on the champion: 88% of features unused, Jaccard = 0.025 across clauses (near-independent), "consequential" is the best AND worst feature (|SHAP| = 1.18, fires equally in TP and FP). Trigrams carry 18.9% of importance — justified. Domain validation: 3/5 HR clauses show PARTIAL alignment with legal terminology; Non-Compete uses proxy features not legal terms.

### Phase 7 — Testing + README polish (Apr 19)
pytest suite rewritten for the Phase 5 champion (blend pipeline removed). 59 tests pass. Training-manifest tests enforce macro-F1 ≥ 0.63 and vocabulary = 40K trigrams as regression guardrails.

See [`results/EXPERIMENT_LOG.md`](results/EXPERIMENT_LOG.md) for the full cross-researcher experiment log.

---

## Project Structure

```
Legal-Contract-Analyzer/
├── README.md
├── requirements.txt
├── app.py                              # Streamlit UI
├── config/config.yaml
├── src/
│   ├── feature_engineering.py          # CUAD loader, clause taxonomy, vectorizer config
│   ├── data_pipeline.py                # HF dataset loader (lazy, legacy)
│   ├── train.py                        # Phase 5 champion training pipeline
│   ├── predict.py                      # Inference API (load_model + predict)
│   └── evaluate.py                     # Evaluation suite
├── models/
│   ├── vectorizer.joblib               # 40K word 1-3gram TF-IDF
│   ├── lgbm_models.joblib              # Per-clause LGBMClassifier list
│   ├── thresholds.json                 # Class-prior thresholds per clause
│   ├── valid_clauses.json              # Clause ordering
│   ├── training_manifest.json          # Metrics + config
│   └── model_card.md
├── data/
│   ├── raw/CUADv1.json                 # CUAD source (not checked in)
│   └── README.md
├── notebooks/
│   ├── phase1_eda_baseline.ipynb
│   ├── phase2_model_comparison.ipynb
│   ├── phase3_anthony_iterative.ipynb
│   ├── phase4_anthony_iterative.ipynb
│   ├── phase5_anthony_iterative.ipynb
│   └── phase6_anthony_explainability.ipynb
├── results/
│   ├── EXPERIMENT_LOG.md
│   ├── phase6_anthony_explainability.json
│   ├── phase6_anthony_ngram_decomposition.png
│   ├── phase6_anthony_shap_hr_clauses.png
│   └── phase6_anthony_cross_clause_heatmap.png
├── reports/
│   ├── day1_phase1_report.md
│   ├── ...
│   └── day7_phase7_anthony_report.md
└── tests/
    ├── test_data_pipeline.py           # 18 tests
    ├── test_model.py                   # 18 tests
    └── test_inference.py               # 23 tests
```

---

## Limitations

- **Training data:** 408 real SEC commercial contracts. Rare clauses (<10 test positives) carry F1 below 0.30 — not enough signal for confident detection.
- **Document type:** US commercial contracts only. Non-English, consumer ToS, and OCR'd documents are out of scope.
- **Negation blind spot:** The "consequential" paradox means bag-of-words cannot distinguish uncapped liability from liability exclusions. A negation-aware feature pipeline is needed for precision gains on Uncapped Liability.
- **LLM baseline is coverage-constrained:** Claude results are measured at 400-word excerpts (4.6% of document). With full sliding-window context, LLM HR-F1 would likely improve — but so would cost and latency by another order of magnitude.
- **Not legal advice:** Model provides first-pass triage. All flagged clauses require attorney review before signing.

---

## References

1. Hendrycks et al. (2021) — [CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review](https://arxiv.org/abs/2103.06268) — NeurIPS 2021
2. Chalkidis et al. (2022) — [LexGLUE: A Benchmark Dataset for Legal Language Understanding](https://aclanthology.org/2022.acl-long.297/) — ACL 2022
3. Chalkidis et al. (2020) — [LEGAL-BERT: The Muppets straight out of Law School](https://aclanthology.org/2020.findings-emnlp.261/) — EMNLP 2020
4. Ke et al. (2017) — [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html) — NeurIPS 2017
5. Lundberg & Lee (2017) — [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874) — NeurIPS 2017 (TreeSHAP)
6. Fan & Lin (2007) — [A Study on Threshold Selection for Multi-label Classification](https://www.csie.ntu.edu.tw/~cjlin/papers/threshold.pdf) — class-prior thresholds
7. Lipton & Elkan (2014) — [Thresholding Classifiers to Maximize F1 Score](https://arxiv.org/abs/1402.1892) — plug-in rule theory
8. Mitchell et al. (2018) — [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) — FAT* 2019

---

*Built by Anthony Rodrigues & Mark Rodrigues as part of the YC Portfolio Projects series.*
*Dataset: CUAD (CC BY 4.0). Code: MIT license.*
