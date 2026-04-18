# Model Card — Legal Contract Risk Analyzer (v2.0, Phase 6 rework)

**Built by:** Anthony Rodrigues (2026-04-18)
**License:** Same as CUAD (CC BY 4.0)
**Contact:** tony@keeper.ai
**Format follows:** Mitchell et al. 2018 [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) / Hugging Face model-card template.

---

## 1. Model overview

A multi-label classifier that flags 28 clause types and derives an overall HIGH/MEDIUM/LOW risk band from the raw text of a commercial contract. Output includes per-clause probabilities, per-clause decision thresholds, and an overall risk classification used for due-diligence triage.

### Intended use

- **Primary:** First-pass triage of contracts during due diligence (M&A, procurement, vendor management) — surface clauses a human reviewer should look at first.
- **Secondary:** Screening of unsigned contracts by non-lawyers (startup founders, product managers) before escalation to counsel.
- **Not intended for:** Binding legal advice. Deciding whether to sign. Regulatory compliance determinations. Jurisdictions other than U.S. commercial law (training data is U.S.-focused CUAD).

---

## 2. Pipeline (v2.0, Phase 6 rework)

| Component | Choice | Why |
|-----------|--------|-----|
| Features | **TF-IDF word 1-3gram, 40,000 features, sublinear_tf=True** | Phase 4 ablation: word 1-3gram @ 40K beats word 1-2gram @ 20K by +0.045 macro-F1 |
| Model | **LightGBM, one-vs-rest per clause** (50 estimators, depth=4, lr=0.15, scale_pos_weight by class) | Phase 5: LR blend HURTS by −0.038 macro-F1 at 40K features (LR saga fails to converge on 5/28 clauses). LGBM alone is the champion. |
| Thresholds | **Class-prior per clause** (`threshold = training positive rate`) | Phase 5 counterintuitive finding: SCut CV threshold tuning overfits on rare labels at n=408 (Fan & Lin 2007). Plug-in rule wins on macro-F1 *and* beats CV by +0.029 HR-F1. |
| Train/test split | 408 / 102 contracts, deterministic permutation (seed=42) | Matches every prior phase for apples-to-apples comparison |

### Key architectural change from v1.0

v1.0 was a **LGBM+LR blend with CV-learned thresholds**. v2.0 drops both LR and CV threshold learning:
- **LR removed**: on 40K features (vs v1.0's 20K), LR saga fails to converge on 5/28 clauses and its predictions DRAG the ensemble DOWN by 0.038 F1. The historical "blend always lifts" assumption only holds at smaller vocabularies.
- **CV thresholds replaced by class priors**: `threshold = train_positive_rate` per clause. No CV, no hyperparameter, no test-set touch. Phase 5 showed this beats 3-fold and 5-fold SCut CV thresholds on both macro-F1 and HR-F1.

The v2.0 pipeline is **simpler** (no LR solver, no CV threshold search), **faster to train** (2 min vs 7 min end-to-end), **faster to infer** (~8ms vs ~12ms per contract), and **better** on the metric lawyers care about (HR-F1 0.5872 vs v1.0's 0.5244 on main).

---

## 3. Metrics (held-out test, 102 contracts, 28 valid clauses)

| Metric | v2.0 (this model) | v1.0 (prior main) | RoBERTa-large SOTA |
|--------|------------------:|------------------:|-------------------:|
| Macro-F1 | **0.6471** | 0.598 | ~0.65 |
| HR-F1 (5 high-risk clauses) | **0.5872** | 0.524 | — |
| Macro-AUC | **0.8690** | 0.867 | — |
| Micro-F1 | 0.6735 | 0.667 | — |
| Macro precision | 0.6081 | 0.587 | — |
| Macro recall | 0.7051 | 0.746 | — |
| Inference latency | ~8 ms / contract | ~443 ms | ~50 ms GPU |

**SOTA parity**: RoBERTa-large on CUAD varies ~0.01 across papers; 0.6471 is within that noise band. HR-F1 is our practical metric for legal due diligence (it's the score on the five clauses a lawyer MUST find); v2.0 beats v1.0 by +0.063 HR-F1.

---

## 4. Per-clause performance

See `results/phase6_evaluation.json` and `results/phase6_evaluation.png` for full per-clause F1/precision/recall breakdowns (regenerate by running `python -m src.evaluate`).

High-level shape:
- 12 frequent clauses (positive rate ≥30%): F1 ≈ 0.65–0.97 (e.g., Governing Law, License Grant, Renewal Term)
- 11 medium clauses (10-30%): F1 ≈ 0.30–0.72
- 5 rare clauses (<10%): F1 ≈ 0.05–0.50 (noise-limited due to small positive counts)

---

## 5. Frontier LLM comparison

| Model | HR-F1 | Latency | Cost / 1K contracts |
|-------|------:|--------:|--------------------:|
| **This model (v2.0)** | **0.5872** | ~8 ms | $0 |
| Claude Sonnet 4.6, zero-shot | 0.162 | 11,100 ms | $15 |
| Claude Sonnet 4.6, few-shot | 0.121 | 15,400 ms | $20 |

**3.6× higher HR-F1 than Claude Sonnet 4.6** zero-shot (0.5872 vs 0.162). At 100K contracts/year, runs cost $0 vs $1,500 for Claude.

---

## 6. Training data

- **CUAD v1** ([Hendrycks et al. 2021](https://arxiv.org/abs/2103.06268), CC BY 4.0): 510 commercial contracts from SEC EDGAR, expert-annotated for 41 clause types. Dropped 5 metadata clauses (Document Name, Parties, Agreement Date, Effective Date, Expiration Date) and 8 clauses with <3 positives in the test split → 28 valid clauses remain.
- No external data, no synthetic augmentation.

---

## 7. Limitations

- **U.S. commercial law bias**: CUAD is SEC-filed U.S. contracts. Do not apply to non-U.S. contracts or non-commercial domains (employment, consumer, medical) without re-validation.
- **Small training set (408 contracts)**: rare clauses (positive rate < 10%) have high F1 variance. A learned per-clause router would in principle reach 0.676 macro-F1 (Phase 4 oracle ceiling), but at this scale the routing signal doesn't transfer — Phase 4 confirmed.
- **Uppercase / extreme formatting**: demo contract showed the model can confuse "UNLIMITED LIABILITY" (should be Uncapped Liability) with Cap On Liability because both share "liability" token patterns. Known failure mode; addressing it requires either more training examples for Uncapped Liability or a rule-layer on the final prediction.
- **No span extraction**: the model predicts *whether* a clause is present, not *which span*. The UI uses a separate regex (`CLAUSE_HIGHLIGHT_PATTERNS` in `src/feature_engineering.py`) to surface evidence snippets.

---

## 8. Ethical considerations

- Model outputs are probabilistic; a "HIGH risk" flag is a triage signal, not a legal conclusion. Explicit disclaimer should appear in any downstream UI (as the Streamlit app does).
- No personally identifiable information in training data (CUAD contracts are SEC public filings).
- Misuse risk: a user could treat model output as binding legal advice. Mitigation is documentation (this card) and UI disclaimers.

---

## 9. How to reproduce

```bash
pip install -r requirements.txt
python -m src.train           # trains + writes models/*.joblib
python -m src.evaluate        # runs held-out evaluation, writes results/phase6_evaluation.{json,png}
python -m src.predict --demo  # inference on built-in demo contract
streamlit run app.py          # Streamlit UI
```

All training / evaluation / inference is CPU-only. No GPU required.

---

## 10. Version history

- **v2.0 (this card)** — Phase 6 rework: 40K word 1-3gram TF-IDF + LGBM-only + class-prior thresholds. Macro-F1 0.6471, HR-F1 0.5872.
- **v1.0** — LGBM+LR blend + CV Youden thresholds. Macro-F1 0.598, HR-F1 0.524. (Superseded.)
