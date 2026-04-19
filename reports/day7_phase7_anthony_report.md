# Phase 7: Testing + README + Polish — Legal Contract Analyzer
**Date:** 2026-04-19
**Session:** 7 of 7
**Researcher:** Anthony Rodrigues

## Objective
Close out the 7-day project sprint: rewrite the pytest suite against the current Phase 5 champion pipeline, produce a comprehensive README reflecting the honest metrics, and confirm the full test-and-README pair gives a future reader (or future Anthony) the real story — not the test-leak inflated one.

## Research & References
1. Mitchell et al. (2018) "Model Cards for Model Reporting" — Model card conventions; we use the template for `models/model_card.md`.
2. Production ML test pyramid (Ebert et al. 2016) — unit tests for data pipeline, integration tests for bundle, E2E tests for predict().
3. CUAD paper evaluation conventions — macro-F1 as primary, per-clause transparency as reporting standard.

How research influenced today's work: the test suite tests *invariants* (vocab size, ngram range, macro-F1 floor, vocabulary must contain "change of control" — a Phase 6 explainability finding) rather than happy-path smoke tests, so regression in any phase's conclusion would actually trigger a failing test.

## Work Completed

### 7.1 — pytest suite rewrite (tests/)
The existing tests were written against the OLD `blend_pipeline.joblib` artifact from Mark's Phase 5/6 pipeline. That pipeline was replaced in Phase 6 rework (PR #17) with:
- `vectorizer.joblib`
- `lgbm_models.joblib` (list, not dict)
- `thresholds.json`
- `valid_clauses.json`
- `training_manifest.json`

The old tests were asserting on the wrong file structure and the wrong metrics (macro-F1 ≥ 0.65 from Mark's test-leak threshold fit). I rewrote:

| File | Before | After | Notes |
|------|-------:|------:|-------|
| `tests/test_data_pipeline.py` | 18 (2 failing) | 18 | One-line fix: lazy import of `datasets` in `src/data_pipeline.py` |
| `tests/test_model.py` | 18 (all failing) | 18 | Now asserts 40K trigrams, class-prior thresholds, manifest invariants |
| `tests/test_inference.py` | 28 (all failing) | 23 | Trimmed redundant tests, rewrote for `src.predict.load_model`/`predict` |

**Result:** `59 passed, 7 skipped (parquet-only)` in 36s.

### 7.2 — Manifest-backed regression guardrails
New invariants in `tests/test_model.py::TestTrainingManifest`:
- `macro_f1 ≥ 0.63` (SOTA parity band, catches regressions below the Phase 5 champion's 0.647)
- `macro_auc ≥ 0.85` (ranking quality, independent of thresholds)
- `hr_f1 ≥ 0.50` (beats LLM baseline 3×, catches HR-clause regressions)
- `tfidf_max_features == 40_000`
- `tfidf_ngram_range == [1, 3]`
- `pipeline` string identifies as Phase 5/6 champion

And a Phase 6 explainability finding encoded as a structural test:
- `test_vocabulary_contains_legal_trigrams` — the 40K vocab must contain `"change of control"`, `"liquidated damages"`, `"intellectual property"`. These were among the SHAP top-5 features; if a future vectorizer rebuild accidentally drops them (e.g. min_df too aggressive), this test catches it.

### 7.3 — README rewrite
The existing README reported Mark's test-leak inflated macro-F1 of 0.716 as the "production" number. Rewrote top-to-bottom to reflect the honest Phase 5 champion:
- Headline: macro-F1 = 0.647, matching RoBERTa-large SOTA within noise band
- HR-F1 = 0.587, 3.6× Claude zero-shot
- Latency: ~9 ms single-contract inference
- Architecture diagram updated to LGBM-only (no LR blend, Phase 5 showed it HURTS at 40K)
- Phase 6 findings ("consequential paradox", 88% dead vocabulary, trigrams justified at 18.9%) included in "Key Findings"
- Phase-by-phase journey rewritten to reflect what actually happened across both researchers

### 7.4 — Minor: lazy import in data_pipeline.py
`src/data_pipeline.py` unconditionally imported `datasets` (HuggingFace) at module level, breaking test collection on environments without HF installed. Moved to lazy import inside `load_cuad_dataset()` — the only function that uses it. The Phase 5 pipeline loads CUAD from the local JSON via `src/feature_engineering.load_cuad_from_json()`, so HF isn't a runtime dependency.

## Key Findings
1. **All 4 test failures after initial rewrite were test-code bugs, not model bugs.** The Phase 5 champion pipeline is solid — the old test suite was pointing at a stale `blend_pipeline.joblib` that no longer exists.
2. **Tests caught the real-world regression risk.** If Phase 6 explainability said "change of control" was the #1 trigram for Change of Control clause, then the test `test_vocabulary_contains_legal_trigrams` encodes that expectation. Future pipeline changes that would break the Phase 6 story now fail a test.
3. **59 tests in 36s is well inside developer-loop budget.** No test needs network, GPU, or the full CUAD JSON — all exercise the saved artifacts + small synthetic contracts.

## What Didn't Work
- 3 out of 7 originally-skipped `TestProcessedData` tests remain skipped — they depend on `data/processed/cuad_classification.parquet` which isn't in the repo (excluded via .gitignore for size). The modeling path doesn't need it (the train.py uses the raw JSON directly), but the legacy parquet path still exists. Decided to leave these as `skipif(not path.exists())` rather than delete them; they're dormant, not broken.

## Final Test Inventory
- `tests/test_data_pipeline.py` — 18 passed (7 skipped on parquet path)
- `tests/test_model.py` — 18 passed
- `tests/test_inference.py` — 23 passed

**Total: 59 passed, 7 skipped.**

## Next Steps (project-complete)
- Vocabulary pruning experiment — verify that shipping 5K instead of 40K features doesn't hurt F1 (Phase 6 finding suggests it shouldn't)
- Negation-aware features — fix the "consequential" paradox identified in Phase 6
- Fair LLM re-comparison with full-document sliding-window context — measure whether frontier LLM HR-F1 at full coverage still loses to LightGBM

## References Used Today
- [1] Mitchell et al. (2018) "Model Cards for Model Reporting" — https://arxiv.org/abs/1810.03993
- [2] pytest docs — fixtures, skipif patterns, parameterize
- [3] CUAD leaderboard conventions — primary = macro-F1, secondary = per-clause F1

## Code Changes
- `tests/test_model.py` (rewritten for Phase 5 champion artifacts)
- `tests/test_inference.py` (rewritten for `src.predict` API)
- `src/data_pipeline.py` (lazy import of `datasets`)
- `README.md` (top-to-bottom rewrite with honest metrics)
- `reports/day7_phase7_anthony_report.md` (this file)
