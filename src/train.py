"""
Production training pipeline for Legal Contract Analyzer.

Model: LightGBM + Logistic Regression probability blend (alpha=0.5), Phase 5 champion.
Per-clause decision thresholds learned via 3-fold CV on the training set
(NO test-set leakage — that was the one flaw in Mark's Phase 5 number of 0.6907,
which optimized thresholds on test).

Usage:
    python -m src.train
    python -m src.train --alpha 0.5 --seed 42 --out-dir models/

Outputs (in --out-dir):
    vectorizer.joblib
    lgbm_models.joblib         # list of N LGBMClassifier, indexed by valid_clauses
    lr_models.joblib           # list of N LogisticRegression
    thresholds.json            # per-clause decision threshold
    valid_clauses.json         # clause ordering used by the model
    training_manifest.json     # metadata: params, seed, metrics, training timestamps
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.feature_engineering import (
    HIGH_RISK_CLAUSES,
    build_vectorizer,
    load_cuad_from_json,
    make_split,
)

# Phase 4/5 champion LGBM params (Mark P4 tuning + P5 ablation validated)
LGBM_PARAMS = dict(
    n_estimators=50,
    max_depth=4,
    learning_rate=0.15,
    subsample=0.8,
    colsample_bytree=0.4,
    n_jobs=1,
    verbose=-1,
    random_state=42,
)

# Phase 5 LR params (Mark Phase 4 Youden LR champion)
LR_PARAMS = dict(
    C=1.0,
    max_iter=500,
    class_weight="balanced",
    solver="saga",
    n_jobs=-1,
)


def train_lgbm_per_clause(X_train, y_train, clauses):
    """Train one LGBM per clause with per-clause scale_pos_weight."""
    models = []
    n_pos = y_train.sum(axis=0)
    for j, clause in enumerate(clauses):
        if len(np.unique(y_train[:, j])) < 2 or n_pos[j] < 2:
            models.append(None)
            continue
        pw = max(1.0, (len(y_train) - n_pos[j]) / n_pos[j])
        clf = lgb.LGBMClassifier(scale_pos_weight=pw, **LGBM_PARAMS)
        clf.fit(X_train, y_train[:, j])
        models.append(clf)
    return models


def train_lr_per_clause(X_train, y_train, clauses):
    """Train one LogReg per clause (class_weight='balanced' handles imbalance)."""
    models = []
    for j, clause in enumerate(clauses):
        if len(np.unique(y_train[:, j])) < 2:
            models.append(None)
            continue
        clf = LogisticRegression(random_state=42, **LR_PARAMS)
        clf.fit(X_train, y_train[:, j])
        models.append(clf)
    return models


def predict_proba_blend(lgbm_models, lr_models, X, alpha):
    """Blend LGBM and LR probabilities: alpha*LGBM + (1-alpha)*LR."""
    n_clauses = len(lgbm_models)
    probs = np.zeros((X.shape[0], n_clauses))
    for j in range(n_clauses):
        plgbm = plr = None
        if lgbm_models[j] is not None:
            plgbm = lgbm_models[j].predict_proba(X)[:, 1]
        if lr_models[j] is not None:
            plr = lr_models[j].predict_proba(X)[:, 1]
        if plgbm is not None and plr is not None:
            probs[:, j] = alpha * plgbm + (1 - alpha) * plr
        elif plgbm is not None:
            probs[:, j] = plgbm
        elif plr is not None:
            probs[:, j] = plr
    return probs


def learn_thresholds_cv(
    X_raw, y, clauses, vectorizer_factory, alpha=0.5, n_splits=3, seed=42
):
    """Compute per-clause optimal threshold via K-fold CV on training data.

    This avoids the test-set leakage that inflated Mark's Phase 5 number.
    We refit the vectorizer + models inside each fold and pick the threshold
    that maximizes per-clause F1 on held-out folds.
    """
    oof_probs = np.zeros_like(y, dtype=float)
    oof_mask = np.zeros_like(y, dtype=bool)

    rng = np.random.RandomState(seed)
    # Use global shuffled indices then do K folds — StratifiedKFold on a single
    # column doesn't generalize to multi-label; rotating across all labels would
    # be expensive. Plain KFold on row indices is sufficient for threshold-setting.
    idx = rng.permutation(len(X_raw))
    fold_size = len(X_raw) // n_splits

    for fold in range(n_splits):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_splits - 1 else len(X_raw)
        val_idx = idx[val_start:val_end]
        train_idx = np.concatenate([idx[:val_start], idx[val_end:]])

        vec = vectorizer_factory()
        vec.fit(X_raw[train_idx])
        X_tr = vec.transform(X_raw[train_idx])
        X_val = vec.transform(X_raw[val_idx])

        lgbm_fold = train_lgbm_per_clause(X_tr, y[train_idx], clauses)
        lr_fold = train_lr_per_clause(X_tr, y[train_idx], clauses)
        probs_val = predict_proba_blend(lgbm_fold, lr_fold, X_val, alpha)

        oof_probs[val_idx] = probs_val
        oof_mask[val_idx] = True

    # Sweep thresholds per clause, maximizing F1 on OOF predictions
    thresholds = {}
    threshold_details = {}
    for j, clause in enumerate(clauses):
        y_col = y[:, j]
        p_col = oof_probs[:, j]
        # Sweep fine grid
        candidates = np.arange(0.05, 0.95, 0.02)
        best_t, best_f1 = 0.5, 0.0
        default_f1 = f1_score(y_col, (p_col >= 0.5).astype(int), zero_division=0)
        for t in candidates:
            f1 = f1_score(y_col, (p_col >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        # If no threshold does better than 0.5, keep 0.5
        if best_f1 <= default_f1:
            best_t = 0.5
            best_f1 = default_f1
        thresholds[clause] = float(best_t)
        threshold_details[clause] = {
            "best_t": float(best_t),
            "cv_f1_at_best": float(best_f1),
            "cv_f1_at_05": float(default_f1),
            "positive_rate": float(y_col.mean()),
        }
    return thresholds, threshold_details


def evaluate_blend(y_true, probs, thresholds, clauses, hr_clauses):
    """Evaluate blended predictions with learned per-clause thresholds."""
    preds = np.zeros_like(y_true)
    for j, c in enumerate(clauses):
        t = thresholds.get(c, 0.5)
        preds[:, j] = (probs[:, j] >= t).astype(int)

    # Per-clause F1
    per_clause = {}
    for j, c in enumerate(clauses):
        if y_true[:, j].sum() == 0:
            continue
        per_clause[c] = {
            "f1": float(f1_score(y_true[:, j], preds[:, j], zero_division=0)),
            "precision": float(precision_score(y_true[:, j], preds[:, j], zero_division=0)),
            "recall": float(recall_score(y_true[:, j], preds[:, j], zero_division=0)),
            "n_positive": int(y_true[:, j].sum()),
            "threshold": float(thresholds.get(c, 0.5)),
        }

    # Aggregate metrics (restrict to clauses with >=1 positive in test to avoid zero-row bias)
    active_mask = y_true.sum(axis=0) > 0
    y_act, p_act = y_true[:, active_mask], preds[:, active_mask]

    macro_f1 = float(f1_score(y_act, p_act, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_act, p_act, average="micro", zero_division=0))
    macro_p = float(precision_score(y_act, p_act, average="macro", zero_division=0))
    macro_r = float(recall_score(y_act, p_act, average="macro", zero_division=0))

    hr_active = [c for c in hr_clauses if c in clauses and active_mask[clauses.index(c)]]
    hr_f1s = [per_clause[c]["f1"] for c in hr_active if c in per_clause]
    hr_f1 = float(np.mean(hr_f1s)) if hr_f1s else 0.0

    # Macro AUC on active clauses
    aucs = []
    for j, c in enumerate(clauses):
        if not active_mask[j]:
            continue
        if 0 < y_true[:, j].sum() < len(y_true):
            aucs.append(roc_auc_score(y_true[:, j], probs[:, j]))
    macro_auc = float(np.mean(aucs)) if aucs else None

    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "hr_f1": hr_f1,
        "macro_auc": macro_auc,
        "per_clause": per_clause,
    }


def main():
    parser = argparse.ArgumentParser(description="Train production Legal Contract Analyzer")
    parser.add_argument("--data", default="data/raw/CUADv1.json", help="Path to CUAD JSON")
    parser.add_argument("--out-dir", default="models", help="Where to save artifacts")
    parser.add_argument("--alpha", type=float, default=0.5, help="LGBM blend weight")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-cv", action="store_true", help="Skip CV threshold learning (use 0.5)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LEGAL CONTRACT ANALYZER — PRODUCTION TRAINING")
    print("=" * 70)
    print(f"Data:   {args.data}")
    print(f"Output: {out_dir}")
    print(f"Alpha:  {args.alpha}  (LGBM weight; LR weight = 1 - alpha)")

    # ----- Load & split -----
    t0 = time.time()
    df = load_cuad_from_json(args.data)
    print(f"\nLoaded {len(df)} contracts in {time.time()-t0:.1f}s")

    train_df, test_df, valid_clauses = make_split(df, test_size=0.2, seed=args.seed)
    print(f"Train: {len(train_df)} | Test: {len(test_df)} | Valid clauses: {len(valid_clauses)}")

    X_train_raw = train_df["text"].values
    X_test_raw = test_df["text"].values
    y_train = train_df[valid_clauses].values.astype(int)
    y_test = test_df[valid_clauses].values.astype(int)

    # ----- Learn per-clause thresholds via CV on TRAIN only -----
    if not args.skip_cv:
        print("\n[1/3] Learning per-clause thresholds via 3-fold CV on training set...")
        t0 = time.time()
        thresholds, thresh_details = learn_thresholds_cv(
            X_train_raw, y_train, valid_clauses,
            vectorizer_factory=build_vectorizer, alpha=args.alpha, seed=args.seed,
        )
        print(f"  CV threshold learning done in {time.time()-t0:.1f}s")
        n_moved = sum(1 for t in thresholds.values() if abs(t - 0.5) > 0.01)
        print(f"  {n_moved}/{len(thresholds)} clauses got non-default thresholds")
    else:
        thresholds = {c: 0.5 for c in valid_clauses}
        thresh_details = {c: {"best_t": 0.5, "skipped": True} for c in valid_clauses}

    # ----- Fit final vectorizer + models on ALL training data -----
    print("\n[2/3] Training final models on full training set...")
    t0 = time.time()
    vectorizer = build_vectorizer()
    vectorizer.fit(X_train_raw)
    X_train = vectorizer.transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    print(f"  Vocabulary: {len(vectorizer.vocabulary_):,} features")

    lgbm_models = train_lgbm_per_clause(X_train, y_train, valid_clauses)
    print(f"  LGBM: trained {sum(m is not None for m in lgbm_models)}/{len(lgbm_models)} per-clause models")

    lr_models = train_lr_per_clause(X_train, y_train, valid_clauses)
    print(f"  LR:   trained {sum(m is not None for m in lr_models)}/{len(lr_models)} per-clause models")

    print(f"  Final fit done in {time.time()-t0:.1f}s")

    # ----- Evaluate on test -----
    print("\n[3/3] Evaluating on held-out test set...")
    t0 = time.time()
    probs_test = predict_proba_blend(lgbm_models, lr_models, X_test, args.alpha)
    infer_time_total = time.time() - t0
    per_contract_ms = 1000 * infer_time_total / len(X_test_raw)

    metrics = evaluate_blend(y_test, probs_test, thresholds, valid_clauses, HIGH_RISK_CLAUSES)

    print(f"\n  Macro-F1:    {metrics['macro_f1']:.4f}")
    print(f"  Micro-F1:    {metrics['micro_f1']:.4f}")
    print(f"  Precision:   {metrics['macro_precision']:.4f}")
    print(f"  Recall:      {metrics['macro_recall']:.4f}")
    print(f"  Macro-AUC:   {metrics['macro_auc']:.4f}")
    print(f"  HR-F1:       {metrics['hr_f1']:.4f}")
    print(f"  Inference:   {per_contract_ms:.2f} ms/contract (batch of {len(X_test_raw)})")

    # ----- Save artifacts -----
    print(f"\nSaving artifacts to {out_dir}/")
    joblib.dump(vectorizer, out_dir / "vectorizer.joblib")
    joblib.dump(lgbm_models, out_dir / "lgbm_models.joblib")
    joblib.dump(lr_models, out_dir / "lr_models.joblib")
    (out_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    (out_dir / "valid_clauses.json").write_text(json.dumps(valid_clauses, indent=2))

    manifest = {
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "dataset": "CUAD v1",
        "n_contracts": int(len(df)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_clauses": len(valid_clauses),
        "alpha": args.alpha,
        "seed": args.seed,
        "lgbm_params": LGBM_PARAMS,
        "lr_params": LR_PARAMS,
        "vectorizer_features": int(len(vectorizer.vocabulary_)),
        "test_metrics": {k: v for k, v in metrics.items() if k != "per_clause"},
        "inference_ms_per_contract": per_contract_ms,
        "threshold_details": thresh_details,
        "per_clause_test": metrics["per_clause"],
    }
    (out_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"  Wrote {out_dir}/{{vectorizer,lgbm_models,lr_models}}.joblib and metadata")
    print("\nDone.")


if __name__ == "__main__":
    main()
