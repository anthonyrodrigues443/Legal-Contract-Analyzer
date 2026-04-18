# Phase 6: Production Pipeline + Streamlit UI — Legal Contract Analyzer
**Date:** 2026-04-18
**Session:** 6 of 7
**Researcher:** Anthony Rodrigues

## Objective
Two concrete questions for Phase 6:

1. **Deployability.** Turn the Phase 1-5 research (best number: Mark's LGBM+LR blend at macro-F1=0.6907) into a clean, serializable production pipeline that loads in <1 s and predicts on a fresh contract in under a second.
2. **Honest thresholds.** Mark's Phase 5 Youden thresholds were fit directly on the held-out test set — a subtle data leak that inflates the reported score. Fit them instead with 3-fold CV on training data only, and report what the *deployable* macro-F1 actually is.

## Research & References
1. **Mitchell et al. 2018 — "Model Cards for Model Reporting"** ([arXiv:1810.03993](https://arxiv.org/abs/1810.03993)). Google's standard format for documenting model limitations, ethical considerations, and intended use. Our `models/model_card.md` follows this template.
2. **Bender & Friedman 2018 — "Data Statements for NLP"** — informed the "NOT intended for" section in the model card (non-English, consumer TOS, OCR'd docs).
3. **Streamlit deployment best practices (2024)** — cache the model with `@st.cache_resource` (once-per-session), cache data transformations with `@st.cache_data`, and avoid UI reruns triggering retraining. We follow both patterns in `app.py`.

How research influenced today's work: The model card format directly shapes the disclosure of the threshold-leak caveat — we document that the honest number is 0.598 macro-F1, not 0.691, and explain why. Streamlit caching patterns let the 700 ms single-contract inference feel instant in the UI without reloading the 5.4 MB vectorizer on every interaction.

## Dataset
| Metric | Value |
|--------|-------|
| Total contracts | 510 |
| Train / Test | 408 / 102 |
| Clauses modeled | 28 (clauses with ≥3 positives in test) |
| HIGH-risk clauses | 5 (Uncapped Liability, Change Of Control, Non-Compete, Liquidated Damages, IP Ownership Assignment) |
| MEDIUM-risk clauses | 7 |
| LOW-risk clauses | 16 |
| Avg contract length | 8,641 words |

## Experiments

### Experiment 6.1: Honest threshold learning (train-only CV)
**Hypothesis:** Mark's reported P5 macro-F1 of 0.6907 is inflated because his per-clause Youden thresholds were optimized on `y_test`. The deployable number — what we'd actually see on unseen data — should be lower.

**Method:** Implemented `learn_thresholds_cv()` in `src/train.py` that runs 3-fold CV on the training set only. For each fold, refit the TF-IDF vectorizer and both models on the non-held part, predict on the held part, and aggregate out-of-fold (OOF) probabilities. Then sweep per-clause thresholds (0.05 → 0.95 step 0.02) on the OOF probs, picking the threshold that maximizes per-clause F1. 25 of 28 clauses got non-default thresholds.

**Result:**
| Metric | Mark P5 (test-fit thresholds) | Phase 6 (CV-fit thresholds) | Δ |
|--------|------------------------------|-----------------------------|---|
| Macro-F1 | 0.6907 | 0.5984 | **-0.0923** |
| HR-F1 | 0.5818 | 0.5244 | -0.0574 |
| Macro-AUC | — | 0.867 | — |
| Macro-recall | — | 0.746 | — |
| Macro-precision | — | 0.587 | — |

**Interpretation:** Nearly a full 0.1 macro-F1 of the headline number was test-set optimization. The honest 0.598 is still comparable to RoBERTa-large (≈0.65) and still crushes Claude zero-shot (0.162 HR-F1). But the "we beat RoBERTa-large" claim from Phase 5 doesn't survive — we're ~0.05 F1 short of it. This is a Keeper-style "correction when wrong" moment worth calling out.

### Experiment 6.2: Fixed 0.5 vs learned-threshold trade-off
**Hypothesis:** Even when learned on CV, per-clause thresholds should still trade precision for recall. For legal triage this is the right direction.

**Method:** Evaluated the same trained models twice — once with fixed `t=0.5`, once with CV-learned per-clause thresholds — on the same test set.

**Result:**
| Metric | Fixed 0.5 | CV-learned | Δ |
|--------|-----------|-----------|---|
| Macro-F1 | 0.5999 | 0.5984 | -0.0015 (noise) |
| Micro-F1 | 0.7605 | 0.6673 | -0.0932 |
| Macro-precision | 0.672 | 0.587 | -0.085 |
| Macro-recall | 0.580 | 0.746 | **+0.166** |

**Interpretation:** The macro-F1 is essentially identical (-0.0015). What changed is the P/R operating point — learned thresholds flag 17 pp more positives (true and false). In a due-diligence workflow, a missed uncapped-liability clause can cost millions while a false flag costs a reviewer 20 seconds. **Threshold learning is the right move for this domain even though it looks like a wash on aggregate F1.** This is the type of finding Keeper's research would surface — the metric doesn't move but the business value does.

### Experiment 6.3: Latency under production conditions
**Hypothesis:** LightGBM + LR on sparse TF-IDF will be ≥1000× faster than the Claude zero-shot baseline (11.1 s from Mark P5). Single-contract latency will be dominated by the TF-IDF transform, not the 28 model calls.

**Method:** Timed the end-to-end `analyzer.analyze(text)` pipeline on 1 random test contract (median of 10 runs), and separately on a batch of all 102 test contracts.

**Result:**
| Condition | ms/contract | Speedup vs Claude |
|-----------|-------------|-------------------|
| Batch of 102 | 12.0 ms | 925× |
| Single contract (median) | 443 ms | 25× |
| Claude claude-sonnet-4-6 zero-shot (Mark P5) | 11,100 ms | 1× |
| Claude claude-sonnet-4-6 few-shot (Mark P5) | 15,400 ms | 0.7× |

**Interpretation:** Batch mode is effectively free (12 ms amortized). Single-contract is slower than expected (443 ms) because sparse-matrix indexing + 28 separate `predict_proba` calls have high per-call overhead. A future optimization: stack all 28 LightGBMs into a single multi-output booster, or replace sklearn-level OvR with a pre-transformed batch interface. Even at 443 ms the latency is invisible in a UI context.

### Experiment 6.4: Risk scoring heuristic
**Hypothesis:** A simple weighted sum of flagged clauses (10 pts per HIGH, 4 per MEDIUM, 1 per LOW, capped at 100) will produce reasonable risk bands on real contracts.

**Method:** Applied the scoring rule to all 102 test contracts and inspected the distribution against the actual clause counts.

**Result:** 102 contracts produced a risk-score distribution with median ~25 (LOW), 3rd quartile ~45 (MEDIUM), max ~92 (HIGH). The test sample with all 5 HIGH-risk clauses scored 73 (HIGH band). Contracts with a single NDA-style exclusivity clause scored <10 (LOW).

**Interpretation:** The rule is intentionally heuristic — a fractional-reviewer would tune weights per industry. We surface the score alongside the full per-clause breakdown so a reviewer can ignore the score and look at the underlying flags if they want.

## Production pipeline architecture

```
Raw contract text
      │
      ▼
┌─────────────────────┐
│ TfidfVectorizer     │  20K vocab, (1,2) grams, sublinear_tf
│ models/vectorizer   │  fit on 408 training contracts
└──────────┬──────────┘
           │ 20K-dim sparse vector
           ▼
    ┌──────┴──────┐
    │             │
┌───▼────┐   ┌────▼─────┐
│ 28x    │   │ 28x      │
│ LGBM   │   │ LogReg   │
│ (per   │   │ (per     │
│ clause)│   │ clause)  │
└────┬───┘   └────┬─────┘
     │            │
     │  P_lgbm    │ P_lr
     ▼            ▼
    α·P_lgbm + (1-α)·P_lr    α = 0.5
           │
           ▼
   Per-clause threshold
   (learned via 3-fold CV
    on training set)
           │
           ▼
  Binary flag per clause + prob + threshold
           │
           ▼
  Risk score (10·HIGH + 4·MED + 1·LOW, cap 100)
           │
           ▼
  ContractReport (JSON / Streamlit / CLI)
```

## Files delivered

| File | Purpose |
|------|---------|
| `src/feature_engineering.py` | Data loading from CUAD JSON, train/test split, vectorizer factory, clause taxonomy (HIGH/MED/LOW), evidence-snippet regex helpers |
| `src/train.py` | End-to-end training: CV threshold learning → final fit → test evaluation → artifact serialization |
| `src/predict.py` | `ContractAnalyzer` class with `.load()`, `.analyze()`, `.predict_proba()`, `.predict()`, plus CLI |
| `src/evaluate.py` | Evaluation suite: aggregate + per-clause + by-risk-level metrics, latency benchmark, plot generation |
| `app.py` | Streamlit UI — paste contract, get risk score + gauge + flagged clauses with evidence + coverage report |
| `models/vectorizer.joblib` | Fitted TF-IDF (20K features, 5.4 MB) |
| `models/lgbm_models.joblib` | 28 LightGBM classifiers (20 MB) |
| `models/lr_models.joblib` | 28 LogReg classifiers (4.5 MB) |
| `models/thresholds.json` | Per-clause CV-learned decision thresholds |
| `models/training_manifest.json` | Training timestamp, params, test metrics, per-clause threshold details |
| `models/model_card.md` | HF-style model card with limitations + ethical considerations |
| `scripts/capture_screenshot.py` | Playwright-based Streamlit screenshot automation (reproducible for future phases) |
| `results/phase6_evaluation.json` | Full evaluation metrics + per-clause breakdown |
| `results/phase6_evaluation.png` | 3-panel chart: per-clause F1 by risk, threshold comparison, latency vs Claude |
| `results/ui_screenshot.png` | Streamlit UI running on a real CUAD contract, showing HIGH-risk result |

## Key Findings

1. **Mark's Phase 5 macro-F1 of 0.6907 was inflated by 0.09 F1 due to fitting per-clause Youden thresholds on the test set.** The honest, deployable number with CV-learned thresholds is 0.598. Still competitive with RoBERTa-large (0.65), still crushes Claude zero-shot, but not *better than* RoBERTa anymore. This is the kind of correction I'd rather ship than hide.

2. **Threshold learning looks neutral on aggregate F1 (-0.0015) but shifts the precision/recall operating point by +17 pp recall.** For legal due-diligence the recall trade is worth ~infinite precision loss — a missed uncapped-liability clause costs millions, a false flag costs 20 seconds of a reviewer's time.

3. **Single-contract latency is 443 ms — slower than advertised "2 ms" but still 25× faster than Claude.** The bottleneck is per-call overhead across 28 separate `predict_proba` calls on sparse matrices, not the TF-IDF transform. A single multi-output LGBM would likely cut this under 100 ms. Future optimization target.

4. **Production thresholds don't center on 0.5 — 25 of 28 clauses have non-default thresholds.** The "Non-Compete" threshold landed at 0.29 (aggressive recall), "Governing Law" at 0.50 (default), "Minimum Commitment" at 0.67 (conservative). These numbers are the real per-clause operating points a deployed system needs — and they would NOT transfer from a different train/test split.

## Frontier Model Comparison

| Model | Macro-F1 | HR-F1 | Latency | Cost / 1K contracts | Deploys offline? |
|-------|----------|-------|---------|---------------------|------------------|
| **This model v1.0** | **0.598** | **0.524** | ~700 ms CPU / 12 ms batch | ~$0 | **Yes** |
| Mark P5 blend (test-fit thresholds) | 0.691 | 0.582 | — | — | — (inflated) |
| Published RoBERTa-large (CUAD) | ~0.65 | — | ~50 ms GPU | Self-hosted | Yes |
| Claude zero-shot, 400-word excerpt | — | 0.162 | 11.1 s | ~$15 | No |
| Claude few-shot, 400-word excerpt | — | 0.121 | 15.4 s | ~$20 | No |

## Error Analysis

- **HIGH-risk clauses are the hardest** (F1=0.524). Within HIGH: Non-Compete and Change Of Control detect well (F1 >0.6), but Liquidated Damages and Uncapped Liability remain ~0.3-0.4. These clauses often appear in a single paragraph buried in section 12+ of a 50-page contract — the full-document TF-IDF averages them out with surrounding vocabulary.
- **MEDIUM-risk clauses perform best on recall** (0.822) — the model catches almost every Indemnification / Cap on Liability / Termination-for-Convenience clause, at the cost of some precision.
- **LOW-risk clauses have the highest F1** (0.636) because they are mostly well-defined header clauses (Governing Law, Insurance, Audit Rights) with distinctive vocabulary.

## Next Steps (Phase 7)

1. **Polish + tests** — `pytest` covering data pipeline, predict API, evaluate smoke test; integration test that trains a toy model and round-trips a sample contract.
2. **Comprehensive README** — mermaid diagram, full experiment log linking to daily reports, LinkedIn-ready screenshot, setup instructions, reproducibility notes.
3. **Long-document SHAP-lite** — run a sliding-window version of the model to surface which paragraph most contributed to each clause prediction. This would upgrade evidence snippets from regex to actual model-signal.
4. **Fair LLM re-comparison** — Phase 5 Claude test used 400-word excerpts (Claude's handicap, not its ceiling). Retest with a sliding-window chunking strategy to get a fair frontier-model baseline.

## References Used Today
- [1] Mitchell et al. 2018. *Model Cards for Model Reporting.* FAT*. https://arxiv.org/abs/1810.03993
- [2] Hendrycks et al. 2021. *CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review.* NeurIPS. https://arxiv.org/abs/2103.06268
- [3] Streamlit — caching docs. https://docs.streamlit.io/library/advanced-features/caching (`@st.cache_resource` for models, `@st.cache_data` for datasets)
- [4] Phase 5 reports (Mark) — ensemble architecture and LLM baseline comparison. `reports/day5_phase5_mark_report.md`

## Code Changes
- `src/__init__.py` — already present
- `src/feature_engineering.py` — NEW (260 lines): CUAD loader, split, vectorizer factory, clause taxonomy, evidence helpers
- `src/train.py` — NEW (250 lines): CV threshold learning + final fit + eval + save
- `src/predict.py` — NEW (210 lines): `ContractAnalyzer` class + CLI
- `src/evaluate.py` — NEW (190 lines): eval suite + plot + JSON report
- `app.py` — NEW (270 lines): Streamlit UI with risk gauge + flagged clauses + evidence + coverage
- `models/model_card.md` — NEW: HF-style model card
- `scripts/capture_screenshot.py` — NEW: Playwright automation for UI screenshots
- `.claude/launch.json` — NEW (in parent dir): preview_start config (for future automation)
- `requirements.txt` — added streamlit, plotly, playwright
