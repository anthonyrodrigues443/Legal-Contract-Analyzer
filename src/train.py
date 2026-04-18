"""
Production training pipeline for Legal Contract Analyzer.

Phase 6 rework (post-Phase 5, PR E):
- Features: word 1-3gram TF-IDF @ 40K features (Phase 4 ablation winner)
- Model:    LightGBM per-clause (Phase 5 blend ablation showed LR HURTS on 40K)
- Thresholds: per-clause class prior (train_positive_rate).
             No CV, no validation split, no test-set touch — the plug-in rule
             from F1-optimization theory for calibrated classifiers. Phase 5
             established this matches RoBERTa-large SOTA on macro-F1 and beats
             every CV-tuned variant on HR-F1 by +0.029.

Why this is simpler AND better than the prior pipeline:
- No LR saga solver (which failed to converge on 5/28 clauses at 40K features).
- No blend alpha hyperparameter.
- No CV threshold learning (SCut overfits on rare labels at n=408; Fan & Lin 2007).
- Thresholds are the training class priors — human-auditable, no fitting.

Artifacts written (in --out-dir, default models/):
    vectorizer.joblib          - fitted TfidfVectorizer (40K word 1-3gram)
    lgbm_models.joblib         - list of per-clause LGBMClassifier (None where no positives)
    thresholds.json            - per-clause class-prior threshold
    valid_clauses.json         - clause ordering used by the model
    training_manifest.json     - metadata: metrics, params, timestamps
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from src.feature_engineering import (
    HIGH_RISK_CLAUSES,
    load_cuad_from_json,
    make_split,
)

# -----------------------------------------------------------------------------
# Config — Phase 5 champion
# -----------------------------------------------------------------------------
TFIDF_MAX_FEATURES = 40_000
TFIDF_NGRAM_RANGE = (1, 3)

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


def build_vectorizer() -> TfidfVectorizer:
    """Phase 5 champion TF-IDF: word 1-3gram @ 40K features."""
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=TFIDF_NGRAM_RANGE,
        max_features=TFIDF_MAX_FEATURES,
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
    )


def train_lgbm_per_clause(X_train, y_train, clauses):
    """Train one LGBMClassifier per clause with per-clause scale_pos_weight.

    Returns a list indexed by `clauses`; entry is None for clauses with <2 positives.
    """
    models = []
    n_pos = y_train.sum(axis=0)
    for j in range(len(clauses)):
        if len(np.unique(y_train[:, j])) < 2 or n_pos[j] < 2:
            models.append(None)
            continue
        pw = max(1.0, (len(y_train) - n_pos[j]) / n_pos[j])
        clf = lgb.LGBMClassifier(scale_pos_weight=pw, **LGBM_PARAMS)
        clf.fit(X_train, y_train[:, j])
        models.append(clf)
    return models


def predict_lgbm(models, X) -> np.ndarray:
    """Per-clause LGBM probability matrix."""
    out = np.zeros((X.shape[0], len(models)))
    for j, m in enumerate(models):
        if m is not None:
            out[:, j] = m.predict_proba(X)[:, 1]
    return out


def class_prior_thresholds(y_train, clauses) -> dict:
    """Phase 5 finding: threshold = training positive rate per clause.

    No CV, no validation, no test-set contact. Theoretically justified as the
    F1-optimal plug-in rule for calibrated classifiers (Lipton & Elkan 2014),
    and empirically the best macro-F1 AND HR-F1 we measured in Phase 5.
    """
    return {c: float(y_train[:, j].mean()) for j, c in enumerate(clauses)}


def evaluate(y_true, probs, thresholds, clauses, hr_clauses):
    preds = np.zeros_like(y_true, dtype=int)
    for j, c in enumerate(clauses):
        preds[:, j] = (probs[:, j] >= thresholds.get(c, 0.5)).astype(int)

    active = y_true.sum(axis=0) > 0

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

    y_act, p_act = y_true[:, active], preds[:, active]
    macro_f1 = float(f1_score(y_act, p_act, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_act, p_act, average="micro", zero_division=0))
    macro_p = float(precision_score(y_act, p_act, average="macro", zero_division=0))
    macro_r = float(recall_score(y_act, p_act, average="macro", zero_division=0))

    hr_active = [c for c in hr_clauses if c in clauses and active[clauses.index(c)]]
    hr_f1 = float(
        np.mean([per_clause[c]["f1"] for c in hr_active if c in per_clause])
    ) if hr_active else 0.0

    aucs = []
    for j, c in enumerate(clauses):
        if not active[j]:
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
    parser = argparse.ArgumentParser(description="Train Phase 6 production pipeline (Phase 5 champion)")
    parser.add_argument("--data", default="data/raw/CUADv1.json", help="CUAD JSON path")
    parser.add_argument("--out-dir", default="models", help="Artifact directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("LEGAL CONTRACT ANALYZER — PRODUCTION TRAINING (Phase 6 rework)")
    print("  features:  word 1-3gram TF-IDF @ 40K (Phase 4 ablation winner)")
    print("  model:     LGBM per-clause (no LR blend — Phase 5 showed it HURTS at 40K)")
    print("  threshold: class prior per clause (no CV, no test leak)")
    print("=" * 72)

    # -------- Load + split --------
    t0 = time.time()
    df = load_cuad_from_json(args.data)
    train_df, test_df, valid_clauses = make_split(df, test_size=0.2, seed=args.seed)
    print(f"\n[1/4] Loaded {len(df)} contracts  ({time.time()-t0:.1f}s)")
    print(f"      train={len(train_df)}  test={len(test_df)}  valid clauses={len(valid_clauses)}")

    X_train_raw = train_df["text"].values
    X_test_raw = test_df["text"].values
    y_train = train_df[valid_clauses].values.astype(int)
    y_test = test_df[valid_clauses].values.astype(int)

    # -------- Vectorize --------
    print("\n[2/4] Fitting TF-IDF (word 1-3gram @ 40K) ...")
    t0 = time.time()
    vec = build_vectorizer()
    vec.fit(X_train_raw)
    X_train = vec.transform(X_train_raw)
    X_test = vec.transform(X_test_raw)
    print(f"      vocabulary: {len(vec.vocabulary_):,}  ({time.time()-t0:.1f}s)")

    # -------- Train LGBM per clause --------
    print("\n[3/4] Training per-clause LGBM classifiers ...")
    t0 = time.time()
    lgbm_models = train_lgbm_per_clause(X_train, y_train, valid_clauses)
    n_fit = sum(m is not None for m in lgbm_models)
    print(f"      {n_fit}/{len(lgbm_models)} trained  ({time.time()-t0:.1f}s)")

    # -------- Class-prior thresholds (no CV, no leak) --------
    thresholds = class_prior_thresholds(y_train, valid_clauses)
    rare = sum(1 for v in thresholds.values() if v < 0.10)
    freq = sum(1 for v in thresholds.values() if v >= 0.30)
    print(f"      thresholds = class priors  ({rare} rare <10%, {freq} frequent ≥30%)")

    # -------- Evaluate --------
    print("\n[4/4] Evaluating on held-out test ...")
    t0 = time.time()
    probs_test = predict_lgbm(lgbm_models, X_test)
    infer_time = time.time() - t0
    per_contract_ms = 1000 * infer_time / len(X_test_raw)

    metrics = evaluate(y_test, probs_test, thresholds, valid_clauses, HIGH_RISK_CLAUSES)

    print()
    print(f"  macro-F1      {metrics['macro_f1']:.4f}")
    print(f"  HR-F1         {metrics['hr_f1']:.4f}")
    print(f"  macro-AUC     {metrics['macro_auc']:.4f}")
    print(f"  precision     {metrics['macro_precision']:.4f}")
    print(f"  recall        {metrics['macro_recall']:.4f}")
    print(f"  inference     {per_contract_ms:.2f} ms/contract")

    sota = 0.65
    macro_f1 = metrics["macro_f1"]
    if macro_f1 >= sota - 0.005:
        verdict = "PARITY/ABOVE"
    else:
        verdict = f"below by {sota - macro_f1:.4f}"
    print()
    print(f"  vs RoBERTa-large SOTA (~{sota}):  {verdict}")

    # -------- Save artifacts --------
    print(f"\nSaving artifacts to {out_dir}/")
    joblib.dump(vec, out_dir / "vectorizer.joblib")
    joblib.dump(lgbm_models, out_dir / "lgbm_models.joblib")
    (out_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    (out_dir / "valid_clauses.json").write_text(json.dumps(valid_clauses, indent=2))

    # Clean up leftover artifacts from previous pipelines
    for old in ("lr_models.joblib", "blend_pipeline.joblib", "training_meta.json"):
        p = out_dir / old
        if p.exists():
            p.unlink()
            print(f"  removed stale artifact: {p.name}")

    manifest = {
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "pipeline": "Phase 6 rework (Phase 5 champion)",
        "notes": (
            "40K word 1-3gram TF-IDF + LGBM per-clause + class-prior thresholds. "
            "No LR blend. No CV. No test-set touch. Thresholds are training positive "
            "rates per clause — plug-in F1-optimal rule for calibrated classifiers."
        ),
        "dataset": "CUAD v1",
        "n_contracts": int(len(df)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_clauses": len(valid_clauses),
        "seed": args.seed,
        "tfidf_max_features": TFIDF_MAX_FEATURES,
        "tfidf_ngram_range": list(TFIDF_NGRAM_RANGE),
        "lgbm_params": LGBM_PARAMS,
        "vectorizer_vocabulary_size": int(len(vec.vocabulary_)),
        "test_metrics": {k: v for k, v in metrics.items() if k != "per_clause"},
        "per_clause_test": metrics["per_clause"],
        "thresholds": thresholds,
        "inference_ms_per_contract": per_contract_ms,
        "sota_reference": {
            "model": "RoBERTa-large on CUAD",
            "macro_f1": 0.65,
            "note": "~0.65 across CUAD literature, ~0.01 variance across papers",
        },
    }
    (out_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"  wrote vectorizer.joblib, lgbm_models.joblib, thresholds.json, valid_clauses.json, training_manifest.json")
    print("\nDone.")


if __name__ == "__main__":
    main()
