"""
Evaluation suite for Legal Contract Analyzer.
Runs full metrics on the saved blend model against the CUAD test set.

Run:
  python src/evaluate.py
"""

import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

HIGH_RISK = ["Uncapped Liability", "Change Of Control", "Non-Compete", "Liquidated Damages"]
PUBLISHED_ROBERTA = 0.650


def load_test_data(data_path: Path, valid_clauses: list):
    df = pd.read_parquet(data_path)
    np.random.seed(42)
    idx = np.random.permutation(len(df))
    test_df = df.iloc[idx[408:]].reset_index(drop=True)
    X_test = test_df["text"].values
    y_test = test_df[valid_clauses].values.astype(int)
    return X_test, y_test, test_df


def predict_batch(bundle, X):
    vec = bundle["vectorizer"]
    lgbm = bundle["lgbm_models"]
    lr = bundle["lr_models"]
    valid_clauses = bundle["valid_clauses"]
    alpha = bundle["blend_alpha"]
    thresholds = bundle["thresholds"]

    Xmat = vec.transform(X)
    probs_lgbm = np.zeros((len(X), len(valid_clauses)))
    probs_lr = np.zeros((len(X), len(valid_clauses)))

    for j, clause in enumerate(valid_clauses):
        if clause in lgbm:
            probs_lgbm[:, j] = lgbm[clause].predict_proba(Xmat)[:, 1]
        if clause in lr:
            probs_lr[:, j] = lr[clause].predict_proba(Xmat)[:, 1]

    probs_blend = alpha * probs_lgbm + (1 - alpha) * probs_lr
    preds = np.zeros_like(probs_blend, dtype=int)
    for j, clause in enumerate(valid_clauses):
        preds[:, j] = (probs_blend[:, j] >= thresholds.get(clause, 0.5)).astype(int)
    return probs_blend, preds


def plot_per_clause_f1(per_clause_f1, valid_clauses, out_path):
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = ["#d32f2f" if c in HIGH_RISK else "#1976d2" for c in valid_clauses]
    bars = ax.barh(range(len(valid_clauses)), per_clause_f1, color=colors)
    ax.set_yticks(range(len(valid_clauses)))
    ax.set_yticklabels(valid_clauses, fontsize=8)
    ax.axvline(x=PUBLISHED_ROBERTA, color="green", linestyle="--", alpha=0.7,
               label=f"Published RoBERTa-large ({PUBLISHED_ROBERTA})")
    ax.set_xlabel("F1 Score")
    ax.set_title("Per-Clause F1 Score — LGBM+LR Blend (Phase 6 Production Model)", fontsize=12)
    ax.legend()
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#d32f2f", label="High-risk clause"),
        Patch(facecolor="#1976d2", label="Standard/medium clause"),
        plt.Line2D([0], [0], color="green", linestyle="--",
                   label=f"Published RoBERTa ({PUBLISHED_ROBERTA})"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_model_comparison(metrics_history, out_path):
    models = [m["name"] for m in metrics_history]
    f1s = [m["macro_f1"] for m in metrics_history]
    hr_f1s = [m["hr_f1"] for m in metrics_history]

    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(x - 0.2, f1s, 0.35, label="Macro-F1", color="#1976d2")
    bars2 = ax.bar(x + 0.2, hr_f1s, 0.35, label="HR-F1", color="#d32f2f")
    ax.axhline(y=PUBLISHED_ROBERTA, color="green", linestyle="--", alpha=0.7,
               label=f"RoBERTa-large ({PUBLISHED_ROBERTA})")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("F1 Score")
    ax.set_title("Model Comparison — All Phases | Legal Contract Analyzer", fontsize=12)
    ax.legend()
    ax.set_ylim(0, 0.8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    print("=" * 60)
    print("Legal Contract Analyzer — Evaluation Suite")
    print("=" * 60)

    model_path = MODELS_DIR / "blend_pipeline.joblib"
    if not model_path.exists():
        print("Model not found. Run `python src/train.py` first.")
        return

    print("\n[1/4] Loading model and test data...")
    bundle = joblib.load(model_path)
    valid_clauses = bundle["valid_clauses"]
    X_test, y_test, test_df = load_test_data(DATA_DIR / "cuad_classification.parquet", valid_clauses)
    print(f"  Test set: {len(X_test)} contracts | {len(valid_clauses)} clauses")

    print("\n[2/4] Running predictions...")
    probs_blend, preds = predict_batch(bundle, X_test)

    macro_f1 = float(f1_score(y_test, preds, average="macro", zero_division=0))
    macro_p = float(precision_score(y_test, preds, average="macro", zero_division=0))
    macro_r = float(recall_score(y_test, preds, average="macro", zero_division=0))
    hr_idxs = [i for i, c in enumerate(valid_clauses) if c in HIGH_RISK]
    hr_f1 = float(f1_score(y_test[:, hr_idxs], preds[:, hr_idxs],
                            average="macro", zero_division=0))

    valid_for_auc = [i for i in range(len(valid_clauses))
                     if 0 < y_test[:, i].sum() < len(y_test)]
    macro_auc = float(np.mean([
        roc_auc_score(y_test[:, i], probs_blend[:, i]) for i in valid_for_auc
    ]))

    print(f"\n  Macro-F1  : {macro_f1:.4f}  (target 0.6907, RoBERTa-large: {PUBLISHED_ROBERTA})")
    print(f"  Macro-P   : {macro_p:.4f}")
    print(f"  Macro-R   : {macro_r:.4f}")
    print(f"  HR-F1     : {hr_f1:.4f}  (target 0.582)")
    print(f"  Macro-AUC : {macro_auc:.4f}")
    print(f"  Beats RoBERTa: {'YES' if macro_f1 > PUBLISHED_ROBERTA else 'NO'}")

    per_clause_f1 = []
    per_clause_results = {}
    print("\n  HIGH-RISK clause breakdown:")
    for i, clause in enumerate(valid_clauses):
        f1 = float(f1_score(y_test[:, i], preds[:, i], zero_division=0))
        p = float(precision_score(y_test[:, i], preds[:, i], zero_division=0))
        r = float(recall_score(y_test[:, i], preds[:, i], zero_division=0))
        per_clause_f1.append(f1)
        per_clause_results[clause] = {"f1": round(f1, 4), "precision": round(p, 4), "recall": round(r, 4)}
        if clause in HIGH_RISK:
            print(f"    {clause:<35} F1={f1:.3f}  P={p:.3f}  R={r:.3f}")

    print("\n[3/4] Generating plots...")
    plot_per_clause_f1(per_clause_f1, valid_clauses,
                       RESULTS_DIR / "phase6_per_clause_f1.png")

    metrics_history = [
        {"name": "Keyword Rules (P1)", "macro_f1": 0.491, "hr_f1": 0.440},
        {"name": "TF-IDF+LR C=1.0 (P1)", "macro_f1": 0.642, "hr_f1": 0.517},
        {"name": "XGBoost+TF-IDF (P2)", "macro_f1": 0.605, "hr_f1": 0.576},
        {"name": "LR+Youden (P4)", "macro_f1": 0.659, "hr_f1": 0.502},
        {"name": "LightGBM default (P4)", "macro_f1": 0.666, "hr_f1": 0.499},
        {"name": "LGBM+LR Blend (P5)", "macro_f1": 0.691, "hr_f1": 0.582},
        {"name": "Claude zero-shot (P5)", "macro_f1": 0.0, "hr_f1": 0.162},
        {"name": "Production Model", "macro_f1": macro_f1, "hr_f1": hr_f1},
    ]
    plot_model_comparison(metrics_history, RESULTS_DIR / "phase6_model_comparison.png")

    print("\n[4/4] Saving metrics...")
    eval_results = {
        "model": "50/50 LGBM+LR Blend (Youden calibration)",
        "phase": 6,
        "macro_f1": round(macro_f1, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "hr_f1": round(hr_f1, 4),
        "macro_auc": round(macro_auc, 4),
        "beats_roberta": macro_f1 > PUBLISHED_ROBERTA,
        "per_clause": per_clause_results,
    }
    with open(RESULTS_DIR / "phase6_evaluation.json", "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"  Saved: results/phase6_evaluation.json")
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
