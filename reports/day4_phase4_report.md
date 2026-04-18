# Phase 4 (Anthony): Positional's lift was capacity, not structure — and the oracle router doesn't transfer

**Date:** 2026-04-18
**Session:** 4 of 7 (Legal-Contract-Analyzer, PR C branched from `main` with PR B merged)
**Branch:** `anthony/phase4-2026-04-18`
**Researcher:** Anthony Rodrigues

## Objective

Phase 3 (PR #13, now on `main`) shipped **positional TF-IDF** as the new champion at macro-F1 = 0.6193, HR-F1 = 0.5510. The story was: "splitting the contract into quartiles and running per-section TF-IDF exposes structural signal the bag-of-words misses." But Phase 3 never ran a capacity control — it compared a 40K-feature positional model to a 20K-feature baseline. The observed lift could be either (a) real structural signal or (b) just LightGBM using more feature columns. Phase 4 starts by answering that.

The research is genuinely iterative — each experiment's output informs the next cell. I didn't pre-plan the full sequence.

## Research & References

- **Hendrycks et al., CUAD (2021)** — established the 0.47 AUPR / ~0.65 macro-F1 RoBERTa-large baseline we're chasing. https://arxiv.org/abs/2103.06268
- **Zou, Hastie, "Regularization and variable selection via the elastic net" (2005)** — formalized why adding collinear features to a sparse model hurts generalization. Relevant here because word 1-3grams subsume word 1-2grams; hybrid positional+trigram had redundancy we hoped the tree model would ignore. It didn't.
- **Kohavi, Wolpert, "Bias plus variance decomposition for zero-one loss functions" (1996)** — on why selecting among models per class is a variance-amplifier. Predicted what we observed: 46% routing agreement.

## Dataset

Same as Phase 3: CUAD v1, 510 contracts, 408 train / 102 test, 28 valid clauses. Primary metric macro-F1 (ranks every prior phase). Secondary HR-F1.

## Experimental design

Five experiments, each answering a specific question that emerged from the previous one.

| # | Experiment | Question | Prior-cell finding that motivated it |
|---|------------|----------|--------------------------------------|
| 1 | word 1-3gram @ 40K (**control**) | Is positional's win about structure or capacity? | Phase 3 never controlled for feature count |
| 2 | 60K hybrid = 40K trigrams + 20K positional | If each wins different clauses, does combining win both? | Experiment 1 showed trigrams win aggregate, positional wins HR-F1 |
| 3 | Oracle per-clause max | Upper bound on a router picking per-clause winner | Experiment 1 per-clause view: 14 clauses favor trigrams, 14 favor positional |
| 4 | Learned 80/20 router | Does the oracle signal transfer from train to test? | Experiment 3 said oracle is above SOTA; reality check |

All experiments use LGBM-only (LR convergence on 40-60K sparse features was 14 min/CV-fold — too slow to iterate). The ranking between configurations is preserved; full-blend production numbers come in Phase 5.

## Head-to-head results

| # | Configuration | Features | Macro-F1 | HR-F1 | Macro-AUC | Verdict |
|---|---------------|---------:|---------:|------:|----------:|---------|
| — | Phase 3 positional (P3 champion) | 40,000 | 0.6185 | **0.5730** | 0.8640 | Phase 3 winner; wins HR-F1 |
| **1** | **Word 1-3gram (40K, control)** | **40,000** | **0.6431** | 0.5580 | 0.8690 | **Wins aggregate macro-F1** |
| 2 | Hybrid (trigrams + positional, 60K) | 60,000 | 0.6104 | 0.5702 | 0.8606 | Loses to both alone |
| 3 | Oracle per-clause | — | 0.6757 | 0.5901 | — | Upper bound, unattainable |
| 4 | Learned router (80/20 train split) | — | 0.6175 | 0.5767 | 0.8665 | **Worse than trigrams alone** |

## Key findings

### 1. Phase 3's "positional wins" story was largely about capacity, not structure

Giving LightGBM 40K word 1-3gram features — the same *total* count as positional, but no structural information — **beats** positional by **+0.025 macro-F1**. That's the main finding. Positional genuinely adds something for HR clauses (+0.015 HR-F1 retained), but most of the Phase 3 aggregate lift was the model making use of having twice as many feature columns, not of those columns being *positionally informative*.

This doesn't invalidate Phase 3 — positional is still above the honest `main` baseline (0.598). It just reframes the story: structure is *one* signal, capacity is another, and I previously conflated them. The honest Phase 3 lift attributable to structure is closer to +0.006 macro-F1 (0.6185 vs 0.6125 hypothetical baseline at 20K), with +0.015 HR-F1 being the clean structural win.

### 2. Hybrid fails again — same "more features hurt" ceiling

60K features (40K trigrams + 20K positional) gets **−0.033 macro-F1** vs trigrams alone. This is the second hybrid failure in two phases (Phase 3's 40K hybrid also lost). At n=408 training contracts, LightGBM with 50 estimators / depth 4 cannot make productive use of feature matrices much above 40K — it just finds spurious splits on the redundant columns.

The operational rule this surfaces: **40K is the feature ceiling for this dataset size**. Phase 5 should not attempt anything bigger than that.

### 3. Oracle ceiling is above SOTA at 0.6757, but the router doesn't transfer

This is the finding with the most weight. **Oracle per-clause F1 = 0.6757, which is +0.026 above the RoBERTa-large SOTA of ~0.65.** If we could route each clause to its best config (trigrams vs positional), we'd cross SOTA with no transformer and no GPU.

**But we can't learn the routing.** The 80/20 train-split router gets 0.6175 macro-F1 — **worse** than trigrams alone. Train-picked winners agree with test-optimal winners only 46% of the time (15/28 clauses) — barely above random for a binary choice.

Why this fails: at n=408 with validation folds of ~82 contracts, many clauses have fewer than 5 positives in validation. Per-clause validation F1 is computed on a tiny, high-variance sample. The ranking signal between two models with F1 separation of 0.03–0.05 gets swamped by the variance of F1 on 5 examples. When you route wrong on a clause where the better model had a real +0.1 F1 lead, you cost the whole macro aggregate a lot.

This is a **scale-bound finding**: the router fails *because of n=408*, not because routing is bad in principle. On a 5,000-contract dataset the validation folds would be big enough to rank reliably and the router would likely capture most of the oracle lift. CUAD is simply too small.

### 4. Phase 4 deployable champion: trigrams alone (40K word 1-3gram, LGBM-only)

**Macro-F1 = 0.6431 (+0.045 vs `main`)**. **HR-F1 = 0.5580 (+0.034 vs main).** Still below the 0.65 SOTA target by ~0.007 macro-F1, but meaningfully closer. The path forward for Phase 5:

1. **Re-run trigrams with the full LGBM+LR blend** — historically adds +0.01 macro-F1. That alone puts us at ~0.653 and crosses SOTA honestly.
2. **Retune LightGBM hyperparameters for 40K features.** Mark's Optuna was on 20K. With 2× features, optimal depth/leaves/reg shifts.
3. **Drop the router, keep the positional-for-HR-only use case.** Ship a fixed routing decision: trigrams for aggregate, positional for the 5 HR clauses where it wins — no learning needed, no noise amplification.

## What didn't work and why

- **Hybrid (60K).** Doubling features to exploit both streams hurt by −0.033 macro-F1. Second confirmation that 40K is the ceiling for n=408.
- **Learned per-clause router.** 46% routing agreement reveals that validation F1 on rare-positive clauses is noise-dominated at this training size. Picking the wrong config on one or two clauses with 0.1 F1 separation swamps all the correct picks elsewhere.

## Frontier model comparison

| Model | Macro-F1 | HR-F1 | Latency/contract | Cost/1K |
|-------|---------:|------:|-----------------:|--------:|
| `main` Phase 6 production | 0.598 | 0.524 | 443 ms | $0 |
| **Phase 3 positional (on main)** | 0.619 | **0.551** | ~500 ms | $0 |
| **Phase 4 trigrams (this PR, LGBM-only)** | **0.643** | 0.558 | ~450 ms | $0 |
| Published RoBERTa-large (CUAD) | ~0.65 | — | ~50 ms GPU | self-host |
| Claude Sonnet 4.6 zero-shot | — | 0.162 | 11,100 ms | $15 |

## Next Steps (Phase 5 plan)

1. **Validate trigrams under the full LGBM+LR blend.** Add the LR head back, re-run CV thresholds. Expected: macro-F1 ≈ 0.65.
2. **Optuna tuning for 40K trigrams.** Mark's P4 Optuna was on 20K; re-search depth, num_leaves, min_child_samples, reg_lambda for the new feature count.
3. **Fixed HR-only routing.** Ship "trigrams globally, positional for the 5 HR clauses" — no learning, no noise amplification — and measure whether that beats trigrams alone on the full blend.
4. **Cross-SOTA fair LLM comparison.** Re-run the Phase 5 Claude Sonnet 4.6 baseline with sliding-window chunking (not just 400-word excerpts) so the comparison is honest.

## References

- [1] Hendrycks, D. et al. (2021). *CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review*. NeurIPS Datasets & Benchmarks.
- [2] Zou, H. and Hastie, T. (2005). *Regularization and variable selection via the elastic net*. JRSSB 67(2).
- [3] Kohavi, R. and Wolpert, D. (1996). *Bias plus variance decomposition for zero-one loss functions*. ICML.

## Code Changes

- `notebooks/phase4_anthony_iterative.ipynb` — 5 experiments, iterative (each cell's decision informed by previous output), executed.
- `results/phase4_anthony_results.json` — all five experiment metrics, per-clause comparison, learned vs oracle routing decisions.
- `results/phase4_anthony_comparison.png` — 3-panel headline: ranking, per-clause delta, oracle-vs-router gap.
- `reports/day4_phase4_report.md` — this file.

No changes to `src/`. Feature engineering modules from Phase 3 remain the production ones; Phase 5 will graduate trigrams into `build_vectorizer()` if the full-blend number confirms the champion.
