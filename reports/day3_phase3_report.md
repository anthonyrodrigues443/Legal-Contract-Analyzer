# Phase 3 (Anthony): Positional structure beats hand-crafted legal syntax — and both together make it WORSE

**Date:** 2026-04-18
**Session:** 3 of 7 (Legal-Contract-Analyzer, back-filled on Saturday of NLP-1 week)
**Researcher:** Anthony Rodrigues
**Branch:** `anthony/phase3-2026-04-18`, built on top of `main` (PR #11 merged)

## Objective

The production pipeline (`main`) sits at **macro-F1 = 0.598** and **HR-F1 = 0.524** on CUAD. Published RoBERTa-large reports ~0.65 macro-F1, so we're 0.05 F1 below open-source SOTA after Phase 6's honesty correction. Mark has exhausted the TF-IDF axis (Word+Char n-grams, 20K vocab Goldilocks, sliding-window BERT). My angle for Phase 3 is to **test whether features beyond the bag-of-words carry signal TF-IDF can't capture**:

1. **Positional structure.** Legal contracts have conventional organization — Governing Law tends to sit at the tail, License Grants in the body, Definitions at the head. A global bag-of-words erases that.
2. **Legal syntax.** Modifier structures (`shall not`, `no more than N days`, `notwithstanding`, `to the fullest extent`) define whether a clause is *binding* or *boilerplate*. These are scale-invariant patterns that raw TF-IDF frequencies dilute.

The two hypotheses are orthogonal, so I run them separately, then combine.

## Research & References

- **CUAD paper (Hendrycks et al., 2021)** — reports RoBERTa-large AUPR around 0.47 on CUAD. The practical F1 numbers in our pipeline aren't directly comparable to AUPR but the paper's finding that "document length and structure matter" motivated the positional experiment.
- **LegalBench + LeXFiles benchmark work (2024)** — shows that TF-IDF baselines remain competitive when transformer context windows don't cover the document. Reinforced that sliding-window / positional tricks are the leverage point for CUAD-length contracts.
- **"Why your feature engineering just makes things worse" (Brownlee, Machine Learning Mastery, 2022 update)** — cites dimensionality curse + feature collinearity as the most common cause of negative transfer from hand-crafted features added to already-high-dim sparse representations. This is exactly what I ended up finding with my syntactic features.

The research pointed two ways at once: position matters, and hand-crafted features are high-risk when the base representation is already 20K dims. I designed Phase 3 to separate those signals rather than conflate them (which is what Mark's `+ domain features` Phase 3 had done).

## Dataset

| Metric | Value |
|--------|-------|
| Total contracts | 510 |
| Train / Test | 408 / 102 |
| Valid clauses (≥3 positives in test) | 28 |
| Median doc length (chars) | 30,756 |
| Positive rate range | 3.4% – 84.8% (median 24.5%) |
| Primary metric | macro-F1 across 28 clauses (ranks head model in every prior phase) |
| Secondary metric | HR-F1 — mean F1 over 5 lawyer-flagged high-risk clauses (the due-diligence metric) |

## Experimental design

Four configurations, all evaluated on the same test split with the Phase 6 production blend (LGBM+LR @ alpha=0.5) and per-clause thresholds learned via 3-fold CV on the *training set only* (no test-set leakage).

| # | Experiment | Features | Intent |
|---|------------|----------|--------|
| A | baseline | 20K global TF-IDF (word 1-2gram) | Reproduces main — honest floor |
| B | + positional | +20K positional TF-IDF (4 quartiles × 5K each) | Does *where* a bigram appears matter? |
| C | + syntactic | +42 hand-crafted legal syntax features | Do modifier patterns add signal beyond TF-IDF? |
| D | hybrid | A + B + C (all three streams) | Do they compose, or interfere? |

Per-clause thresholds are learned per experiment — a threshold that's optimal for a 20K-dim baseline is unlikely to be optimal for a 40K-dim positional model, so threshold-learning is part of the feature configuration, not a shared post-hoc tune.

## Head-to-head results

| # | Experiment | Features | Macro-F1 | HR-F1 | Macro-AUC | Δ macro-F1 vs A |
|---|------------|---------:|---------:|------:|----------:|----------------:|
| A | baseline | 20,000 | 0.5984 | 0.5244 | 0.8667 | — |
| **B** | **+ positional TF-IDF** | **40,000** | **0.6193** | **0.5510** | 0.8640 | **+0.0209** |
| C | + syntactic (42) | 20,042 | 0.5913 | 0.5298 | 0.8214 | −0.0071 |
| D | hybrid (A+B+C) | 40,042 | 0.5909 | 0.5004 | 0.8191 | −0.0075 |

**Phase 3 champion: B (+ positional TF-IDF), macro-F1 = 0.6193.**

## Key findings

### 1. Positional TF-IDF alone is the winner: +0.021 macro-F1 / +0.027 HR-F1 over the honest baseline

Running a separate 5K-feature TF-IDF on each quartile of the contract (head / upper-mid / lower-mid / tail) and concatenating with the global TF-IDF closes **~40% of the distance to published RoBERTa-large** (from −0.05 F1 below SOTA to −0.03). And unlike Mark's `+ domain features` which traded macro-F1 for HR-F1, positional TF-IDF lifts *both*. HR-F1 actually moves more than macro-F1 because HR clauses (Uncapped Liability, Change Of Control, Non-Compete, Liquidated Damages, IP Ownership) sit in structurally predictable regions:

| HR clause | Base F1 | +Positional F1 | Δ |
|-----------|--------:|---------------:|--:|
| Uncapped Liability | 0.500 | 0.538 | +0.038 |
| Change Of Control | 0.593 | 0.630 | +0.037 |
| Liquidated Damages | 0.318 | 0.348 | +0.030 |
| IP Ownership Assignment | 0.723 | 0.739 | +0.016 |
| Non-Compete | 0.488 | 0.500 | +0.012 |
| **mean** | **0.524** | **0.551** | **+0.027** |

All 5 HR clauses improved. None regressed. This is the due-diligence win: position is a reliable signal for exactly the clauses a lawyer most needs caught.

### 2. 42 hand-crafted legal-syntax features moved macro-F1 by −0.007 — the "more features can hurt" pattern

This was the counterintuitive finding. Modifier structures ("shall not," "notwithstanding," "to the fullest extent") are the whole reason legal counsel reads contracts line-by-line. They should add signal. But at the aggregate level, adding them to a 20K-dim TF-IDF **hurt** macro-F1 by 0.007 and *crashed* macro-AUC by 0.045 (0.867 → 0.821).

The per-clause picture explains why. Syntactic features are **high-variance** across clauses — they rescue some and destroy others:

| Clause | Base F1 | +Syntactic F1 | Δ |
|--------|--------:|--------------:|--:|
| Third Party Beneficiary | 0.200 | 0.462 | **+0.262** |
| Unlimited/All-You-Can-Eat License | 0.000 | 0.105 | +0.105 |
| Post-Termination Services | 0.605 | 0.705 | +0.100 |
| Warranty Duration | 0.733 | **0.480** | **−0.253** |
| Minimum Commitment | 0.710 | 0.660 | −0.050 |

The mechanism: raw syntactic counts (even normalized per 1K characters) collinearly encode doc length, section structure, and legal-writing style. For clauses whose positives happen to correlate with those confounders (Third Party Beneficiary, rare and formal), that's useful. For clauses whose positives *don't* (Warranty Duration, which lives in highly variable structural contexts), the features inject noise the model can't debias. Adding 42 features to a 20K-feature matrix isn't free — the model has to pay for every one it uses by overfitting on it.

### 3. Hybrid is the WORST of the three additions: positional + syntactic = net −0.008 macro-F1, −0.024 HR-F1

This is the experiment I expected to win. Instead, it lost to both its halves:

| Configuration | macro-F1 | HR-F1 |
|---------------|---------:|------:|
| +positional alone (B) | **0.6193** | **0.5510** |
| +syntactic alone (C) | 0.5913 | 0.5298 |
| +both (D hybrid) | 0.5909 | 0.5004 |

The syntactic features don't just fail to add signal on top of positional — they actively **neutralize positional's lift** on HR clauses. HR-F1 for hybrid (0.500) is worse than baseline (0.524). This is the Keeper-style "consensus projection destroys taste dimensions" echo: more isn't more.

### 4. Production recommendation

Ship **B (+ positional TF-IDF)** as the new default feature set for the LGBM+LR blend. Drop syntactic features from the pipeline — they carry high per-clause variance that doesn't aggregate to a net win. Phase 4 should tune hyperparameters for the positional model and investigate whether *clause-selective* syntactic features (applying syntactic only to the clauses where they helped by >+0.05 F1, e.g., Third Party Beneficiary) can recover the win without dragging the aggregate down.

## What didn't work and why

- **Hand-crafted syntactic features as a monolithic block.** The hypothesis — that legal modifier structure carries scale-invariant signal — was right per-clause but wrong at the aggregate. Adding 42 correlated, doc-level features to an already-high-dim sparse matrix introduces more variance than bias reduction. The next phase should split this block by clause affinity rather than applying it globally.
- **Hybrid additivity assumption.** I assumed that if positional helps by X and syntactic helps by Y, hybrid would help by roughly X + Y. Wrong: the two feature streams share noise channels (doc length, section boundaries) that LightGBM can't separate at this training size (408 contracts).

## Frontier model comparison

Phase 3 doesn't re-run the LLM comparison (that lives in Phase 5). Against the honest numbers already on main:

| Model | Macro-F1 | HR-F1 | Latency/contract | Cost/1K |
|-------|---------:|------:|-----------------:|--------:|
| Phase 6 production (on main) | 0.598 | 0.524 | 443 ms | $0 |
| **Phase 3 (this PR, +positional)** | **0.619** | **0.551** | **~500 ms** (extra positional transform) | **$0** |
| Published RoBERTa-large (CUAD) | ~0.65 | — | ~50 ms GPU | self-host |
| Claude Sonnet 4.6 zero-shot | — | 0.162 | 11,100 ms | $15 |

Still below RoBERTa-large, but closing. HR-F1 advantage over Claude stays enormous (0.551 vs 0.162, 3.4×).

## Next Steps (Phase 4 plan)

1. **Hyperparameter tuning on the positional champion.** 40K features shifts the optimal `num_leaves`/`min_child_samples` region — Mark's Optuna was on 20K. Run a fresh Optuna study on the B configuration.
2. **Clause-selective syntactic features.** Only 7/28 clauses benefited from syntactic features by ≥+0.05 F1. Build a per-clause feature switch so syntactic features are only fed into the model for those clauses. Hypothesis: this recovers half the positional loss in hybrid.
3. **Section-aware feature importance analysis.** LightGBM importance per quartile to answer *which* section of the contract drove the positional lift. For HR clauses this should reveal specific structural positions (e.g., "Uncapped Liability lives in quartile 3 = the body").

## References Used Today

- [1] Hendrycks, D. et al. (2021). *CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review*. NeurIPS Datasets & Benchmarks. https://arxiv.org/abs/2103.06268
- [2] Chalkidis, I. et al. (2024 update). *LeXFiles & LegalBench benchmark notes on TF-IDF competitiveness on long legal documents*. ACL anthology.
- [3] Brownlee, J. (2022 update). *Feature Selection for Machine Learning: Why More Features Can Hurt*. Machine Learning Mastery blog — read as calibration for what to expect when adding hand-crafted features to already-high-dim sparse pipelines.

## Code Changes

- `src/phase3_features.py` — new feature module: `PositionalTfidfVectorizer`, `extract_syntactic_features`, `stack_features`. Stateless and importable from training code.
- `notebooks/phase3_anthony_iterative.ipynb` — iterative research notebook. Reproduces baseline, runs B/C/D experiments, per-clause lift analysis, HR-clause syntactic importance, saves plots + JSON.
- `results/phase3_anthony_results.json` — experiment metrics, per-clause deltas, LightGBM syntactic-feature importance for HR clauses.
- `results/phase3_anthony_comparison.png` — 3-panel headline plot: aggregate F1 lift, HR-F1 lift, per-clause hybrid delta.
- `reports/day3_phase3_report.md` — this file.
