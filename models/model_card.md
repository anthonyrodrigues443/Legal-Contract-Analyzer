# Model Card — Legal Contract Risk Analyzer (v1.0)

**Built by:** Anthony Rodrigues (2026-04-18)
**License:** Same as CUAD (CC BY 4.0)
**Contact:** tony@keeper.ai
**Format follows:** Mitchell et al. 2018 [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993)
 / Hugging Face model-card template.

---

## 1. Model overview

A multi-label classifier that flags 28 clause types (and derives an overall 0-100
risk score) from the raw text of a commercial contract. Output includes a
HIGH/MEDIUM/LOW risk band, per-clause probabilities, learned decision thresholds,
and an evidence snippet for each flagged clause.

### Intended use

- **Primary:** First-pass triage of contracts during due diligence (M&A, procurement,
  vendor onboarding). Helps a reviewer focus on the clauses most likely to matter.
- **Downstream:** Can be wrapped as an API (`src/predict.ContractAnalyzer`) or a
  Streamlit UI (`app.py`). JSON reports are exposed for integration with
  contract-management systems.
- **NOT intended for:** Legal advice. Final flagging decisions require a qualified
  attorney's review. Not validated on non-English contracts or on consumer
  contracts / EULAs (training data is US commercial contracts).

### Factors out of scope

- Non-English contracts
- Consumer TOS / EULA (dataset-shift risk — CUAD is US commercial deals)
- Handwritten / poorly OCR'd documents
- Contracts < 200 words (model is calibrated on avg 8,641-word contracts)

---

## 2. Architecture

| Component | Choice | Source / rationale |
|-----------|--------|--------------------|
| Feature extractor | TF-IDF, 20K vocab, word (1-2) grams, sublinear_tf | Phase 1-5 experiments: beat BERT/Legal-BERT fine-tuning, matches Mark's 0.615 P2 baseline |
| Classifier A | LightGBM per-clause (50 trees, depth 4, lr 0.15, scale_pos_weight tuned per-clause) | Phase 4 (Mark) Optuna tuning found this beat default sklearn params by +0.02 macro-F1 |
| Classifier B | Logistic Regression per-clause (C=1.0, class_weight='balanced', saga solver) | Phase 4/5 complements LightGBM on rare clauses (diversity in error modes) |
| Blending | 50 / 50 probability average | Phase 5 (Mark) alpha sweep: α=0.5 was best for macro-F1, α=0.8 was best for HR-F1 |
| Thresholds | Per-clause, learned by 3-fold CV on training set | Phase 6 fix — Mark's P5 Youden thresholds were fit on test set, inflating his reported 0.6907 |
| Inference | Deterministic, no GPU, pure Python | 5 ms/contract in batch; ~700 ms single-contract (sparse-matrix + 28 model calls) |

### Why blend two models?

Mark's Phase 5 ablation showed that **class re-weighting contributes -0.08 macro-F1 if removed** — more impact than any feature trick. LightGBM and LR re-weight differently: LightGBM via `scale_pos_weight` per-tree, LR via per-sample weights in the loss. Averaging their probabilities corrects for each model's idiosyncratic miscalibration on rare clauses.

---

## 3. Training data

**Source:** [CUAD v1](https://www.atticusprojectai.org/cuad) (Contract Understanding Atticus Dataset, Hendrycks et al. NeurIPS 2021).

- 510 real commercial contracts, expert-annotated by lawyers
- 41 clause types (28 have ≥3 positives in test and are modeled here)
- Average contract length: ~8,600 words
- Class distribution: 2-97% per clause (severe imbalance)

### Split

- 80/20 random shuffle with fixed seed (42)
- Train: 408 contracts, Test: 102 contracts
- Same split used in Phases 1-5 for comparability

---

## 4. Evaluation

**Primary metric:** Macro-F1 (CUAD leaderboard standard).
**Secondary:** HR-F1 — macro-F1 over 5 HIGH-risk clause types (Uncapped Liability, Change of Control, Non-Compete, Liquidated Damages, IP Ownership Assignment).

### Held-out test performance (102 contracts)

| Metric | Learned thresholds | Fixed threshold (0.5) |
|--------|--------------------|-----------------------|
| **Macro-F1** | **0.5984** | 0.5999 |
| Micro-F1 | 0.6673 | 0.7605 |
| Macro-precision | 0.587 | 0.672 |
| Macro-recall | 0.746 | 0.580 |
| Macro-AUC | 0.867 | 0.867 |
| HR-F1 (5 HIGH clauses) | **0.524** | — |

**The threshold-learning trade-off:** Learned thresholds cost 0.08 macro-precision but buy +0.17 macro-recall. For legal risk triage this is the right trade — a lawyer can dismiss a false flag in seconds, but an unflagged liability clause can cost millions.

### Per risk-level breakdown

| Risk level | N clauses | Mean F1 | Precision | Recall | AUC |
|-----------|-----------|---------|-----------|--------|-----|
| HIGH | 5 | 0.524 | 0.459 | 0.653 | 0.815 |
| MEDIUM | 7 | 0.566 | 0.538 | 0.822 | 0.831 |
| LOW | 16 | 0.636 | 0.649 | 0.742 | 0.898 |

### vs external baselines

| System | Macro-F1 | HR-F1 | Latency / contract | Cost / 1K contracts |
|--------|----------|-------|--------------------|---------------------|
| **This model (v1.0)** | **0.598** | **0.524** | **~700 ms CPU** | **$0** |
| Published RoBERTa-large on CUAD | ~0.650 | — | ~50 ms GPU | $0 (self-hosted) |
| Claude claude-sonnet-4-6 zero-shot, 400-word excerpt (Mark P5) | — | 0.162 | 11.1 s | ~$15 |
| Claude claude-sonnet-4-6 few-shot, 400-word excerpt (Mark P5) | — | 0.121 | 15.4 s | ~$20 |

**This model wins on HR-F1 (0.524 vs 0.162, +0.36) and speed (~15×) over the zero-shot LLM baseline that was also bound by a short-context constraint.** RoBERTa-large with full fine-tuning still edges us on macro-F1 but requires a GPU and adds ~500 MB of weights.

---

## 5. Known limitations

1. **Sparse-matrix inference is slow on single contracts.** 28 model calls on a 20K-dim sparse vector takes ~700 ms. Fine for UI; for high-QPS production, pre-compute TF-IDF once and vectorize models (stacking into a single LGBMClassifier with multi-output) would cut it to <100 ms.
2. **HIGH-risk clauses are the hardest.** F1 = 0.524 for the 5 HIGH categories. These appear in <10% of contracts on average, and the specific phrasing varies more than common clauses. This is the obvious next research direction (Phase 7+).
3. **Evidence snippets use regex, not attention.** The `extract_clause_snippet()` helper in `feature_engineering.py` uses simple regex patterns to find a relevant excerpt. The classification itself is NOT regex-based — the LGBM+LR blend uses the full TF-IDF vector. But evidence snippets may miss the actual triggering span.
4. **No calibrated probabilities.** The blended probabilities are not calibrated; they are only used with learned per-clause thresholds. If downstream consumers need calibrated probabilities, apply isotonic / Platt on the blended output.
5. **CUAD is a benchmark, not real deployment data.** Real contracts a firm sees will differ in clause distribution, boilerplate, and industry. Expect a domain-adaptation gap.

---

## 6. Ethical considerations

- **Not a substitute for legal review.** Surfaced clauses must be validated by a licensed attorney; the model is meant to TRIAGE, not DECIDE.
- **False negatives can cause harm.** Missing an uncapped-liability clause could expose a company to material loss. The learned thresholds intentionally over-flag to minimize FN rate on HIGH-risk categories. Users should treat "no clauses flagged" as "no clauses flagged *above threshold*", not as "this contract has no risks".
- **Training data bias.** CUAD covers predominantly US commercial contracts. The model will underperform on non-US jurisdictions and on consumer/EULA contracts.
- **Privacy.** The model runs fully offline — no contract data leaves the machine. Compare with frontier-LLM baselines that require sending contract text to an API.

---

## 7. How to use

```python
from src.predict import ContractAnalyzer

analyzer = ContractAnalyzer.load()            # loads from models/
report = analyzer.analyze(contract_text)

print(report.risk_score, report.risk_band)    # e.g. 73, "HIGH"
for clause in report.flagged_clauses:
    print(clause.clause, clause.probability, clause.risk_level)
print(report.to_json())                       # full JSON report
```

CLI:
```bash
python -m src.predict path/to/contract.txt
python -m src.predict - < contract.txt --json
```

UI:
```bash
streamlit run app.py
```

Retrain:
```bash
python -m src.train --alpha 0.5 --seed 42
```

Evaluate:
```bash
python -m src.evaluate
```

---

## 8. Reproducibility

Fixed seed `42` everywhere (split, CV, LGBM, LR). The `training_manifest.json`
artifact records the training timestamp, all hyperparameters, vocabulary size,
test metrics, and per-clause threshold details so results can be reproduced.

## 9. Citations

- Hendrycks et al. 2021. *CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review.* NeurIPS 2021. https://arxiv.org/abs/2103.06268
- Ke et al. 2017. *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS 2017.
- Akiba et al. 2019. *Optuna: A Next-generation Hyperparameter Optimization Framework.* KDD 2019.
- Mitchell et al. 2018. *Model Cards for Model Reporting.* FAT*. https://arxiv.org/abs/1810.03993
