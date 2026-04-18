"""
Evaluation suite for the production Legal Contract Analyzer.

Runs the held-out test set through a saved model and reports:
  - Aggregate metrics (macro-F1, micro-F1, HR-F1, AUC, precision, recall)
  - Per-clause breakdown (F1, precision, recall, n_positive)
  - Risk-level aggregations (HIGH vs MEDIUM vs LOW)
  - Inference latency (per-contract and batch)
  - Threshold sensitivity (what happens at threshold 0.5 vs learned)

Produces results/phase6_evaluation.json and results/phase6_evaluation.png.

Usage:
    python -m src.evaluate
    python -m src.evaluate --data data/raw/CUADv1.json --models models/
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.feature_engineering import (
    HIGH_RISK_CLAUSES,
    MEDIUM_RISK_CLAUSES,
    load_cuad_from_json,
    make_split,
    risk_level,
)
from src.predict import ContractAnalyzer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def run_eval(data_path: str, models_dir: str, out_dir: str, seed: int = 42):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EVALUATION — Legal Contract Analyzer (production model)")
    print("=" * 70)

    # Load model + data
    analyzer = ContractAnalyzer.load(models_dir)
    df = load_cuad_from_json(data_path)
    _, test_df, valid_clauses = make_split(df, test_size=0.2, seed=seed)
    assert valid_clauses == analyzer.valid_clauses, (
        "Clause ordering mismatch between saved model and eval split"
    )

    X_test = test_df["text"].values
    y_test = test_df[valid_clauses].values.astype(int)
    n_clauses = len(valid_clauses)

    # --- Run inference (measure latency) ---
    t0 = time.time()
    probs = analyzer.predict_proba(X_test)
    batch_ms = 1000 * (time.time() - t0)
    per_ms = batch_ms / len(X_test)

    # Single-document latency (typical use case)
    sample = X_test[0]
    single_times = []
    for _ in range(10):
        t0 = time.time()
        analyzer.predict_proba(sample)
        single_times.append(1000 * (time.time() - t0))
    single_median_ms = float(np.median(single_times))

    # --- Aggregate predictions with learned thresholds ---
    thresholds = analyzer.thresholds
    preds_learned = np.zeros_like(y_test)
    for j, c in enumerate(valid_clauses):
        t = thresholds.get(c, 0.5)
        preds_learned[:, j] = (probs[:, j] >= t).astype(int)

    # Same, with fixed 0.5 (ablation on threshold learning)
    preds_fixed = (probs >= 0.5).astype(int)

    def aggregate(preds, label):
        active_mask = y_test.sum(axis=0) > 0
        y_act = y_test[:, active_mask]
        p_act = preds[:, active_mask]

        macro_f1 = float(f1_score(y_act, p_act, average="macro", zero_division=0))
        micro_f1 = float(f1_score(y_act, p_act, average="micro", zero_division=0))
        macro_p = float(precision_score(y_act, p_act, average="macro", zero_division=0))
        macro_r = float(recall_score(y_act, p_act, average="macro", zero_division=0))
        return dict(
            label=label,
            macro_f1=macro_f1,
            micro_f1=micro_f1,
            macro_precision=macro_p,
            macro_recall=macro_r,
        )

    agg_learned = aggregate(preds_learned, "Learned thresholds (production)")
    agg_fixed = aggregate(preds_fixed, "Fixed 0.5 threshold")

    # Per-clause breakdown (with learned thresholds)
    per_clause_rows = []
    for j, c in enumerate(valid_clauses):
        n_pos = int(y_test[:, j].sum())
        if n_pos == 0:
            continue
        f1 = float(f1_score(y_test[:, j], preds_learned[:, j], zero_division=0))
        p = float(precision_score(y_test[:, j], preds_learned[:, j], zero_division=0))
        r = float(recall_score(y_test[:, j], preds_learned[:, j], zero_division=0))
        auc = None
        if 0 < n_pos < len(y_test):
            auc = float(roc_auc_score(y_test[:, j], probs[:, j]))
        per_clause_rows.append({
            "clause": c,
            "risk_level": risk_level(c),
            "n_positive": n_pos,
            "positive_rate": n_pos / len(y_test),
            "threshold": float(thresholds.get(c, 0.5)),
            "f1": f1,
            "precision": p,
            "recall": r,
            "auc": auc,
        })
    clause_df = pd.DataFrame(per_clause_rows).sort_values("f1", ascending=False)

    # Risk-level aggregation
    by_risk = {}
    for level in ["HIGH", "MEDIUM", "LOW"]:
        subset = clause_df[clause_df["risk_level"] == level]
        if len(subset) == 0:
            continue
        by_risk[level] = {
            "n_clauses": int(len(subset)),
            "mean_f1": float(subset["f1"].mean()),
            "mean_precision": float(subset["precision"].mean()),
            "mean_recall": float(subset["recall"].mean()),
            "mean_auc": float(subset["auc"].dropna().mean()) if subset["auc"].notna().any() else None,
        }

    # --- Print summary ---
    print(f"\nTest contracts: {len(X_test)}   Clauses evaluated: {n_clauses}")
    print("\nAGGREGATE METRICS:")
    print(f"  {'':35s} {'Macro-F1':>10s} {'Micro-F1':>10s} {'Prec':>8s} {'Recall':>8s}")
    for agg in [agg_learned, agg_fixed]:
        print(f"  {agg['label']:35s} {agg['macro_f1']:>10.4f} {agg['micro_f1']:>10.4f} "
              f"{agg['macro_precision']:>8.3f} {agg['macro_recall']:>8.3f}")

    print("\nBY RISK LEVEL (learned thresholds):")
    for level, m in by_risk.items():
        auc_str = f"  AUC={m['mean_auc']:.3f}" if m["mean_auc"] is not None else ""
        print(f"  {level:7s} ({m['n_clauses']:2d} clauses)  mean F1={m['mean_f1']:.3f}  "
              f"Prec={m['mean_precision']:.3f}  Recall={m['mean_recall']:.3f}{auc_str}")

    print(f"\nLATENCY:")
    print(f"  Batch of {len(X_test)}: {batch_ms:.1f} ms total ({per_ms:.2f} ms/contract)")
    print(f"  Single document (median of 10): {single_median_ms:.2f} ms")
    print(f"  vs Claude zero-shot (Mark P5): ~11,100 ms/contract  ->  speedup ~{11100/single_median_ms:.0f}x")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # 1. Per-clause F1 colored by risk
    ax = axes[0]
    cdf = clause_df.sort_values("f1", ascending=True)
    colors = {"HIGH": "#E53935", "MEDIUM": "#FB8C00", "LOW": "#43A047"}
    bar_colors = [colors[r] for r in cdf["risk_level"]]
    ax.barh(cdf["clause"], cdf["f1"], color=bar_colors)
    ax.set_xlabel("F1 Score (learned threshold)")
    ax.set_title(f"Per-clause F1  (Macro-F1={agg_learned['macro_f1']:.3f})")
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xlim(0, 1)
    # Legend
    from matplotlib.patches import Patch
    handles = [Patch(color=v, label=f"{k}-risk") for k, v in colors.items()]
    ax.legend(handles=handles, loc="lower right", fontsize=8)

    # 2. Fixed vs learned thresholds — bar comparison
    ax = axes[1]
    metric_names = ["Macro-F1", "Micro-F1", "Precision", "Recall"]
    learned_vals = [
        agg_learned["macro_f1"], agg_learned["micro_f1"],
        agg_learned["macro_precision"], agg_learned["macro_recall"]
    ]
    fixed_vals = [
        agg_fixed["macro_f1"], agg_fixed["micro_f1"],
        agg_fixed["macro_precision"], agg_fixed["macro_recall"]
    ]
    x = np.arange(len(metric_names))
    w = 0.35
    ax.bar(x - w/2, fixed_vals, w, label="Threshold=0.5", color="#90A4AE")
    ax.bar(x + w/2, learned_vals, w, label="Learned thresholds", color="#1976D2")
    for i, (fx, lr) in enumerate(zip(fixed_vals, learned_vals)):
        ax.text(i - w/2, fx + 0.01, f"{fx:.3f}", ha="center", fontsize=8)
        ax.text(i + w/2, lr + 0.01, f"{lr:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Threshold learning impact")
    ax.legend()

    # 3. Latency comparison (log scale)
    ax = axes[2]
    systems = ["Production\n(LGBM+LR)", "Claude\nzero-shot\n(Mark P5)", "Claude\nfew-shot\n(Mark P5)"]
    latencies_ms = [single_median_ms, 11100, 15400]
    bar_colors = ["#1976D2", "#757575", "#424242"]
    bars = ax.bar(systems, latencies_ms, color=bar_colors)
    ax.set_yscale("log")
    ax.set_ylabel("ms per contract (log scale)")
    ax.set_title("Inference latency vs frontier LLMs")
    for bar, val in zip(bars, latencies_ms):
        ax.text(bar.get_x() + bar.get_width()/2, val * 1.1,
                f"{val:.1f} ms" if val < 1000 else f"{val/1000:.1f} s",
                ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plot_path = out_dir / "phase6_evaluation.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved plot: {plot_path}")

    # --- Save JSON results ---
    results = {
        "phase": 6,
        "model": "LGBM+LR blend (alpha=0.5), CV-learned thresholds",
        "dataset": "CUAD v1",
        "n_test": int(len(X_test)),
        "n_clauses": int(n_clauses),
        "aggregate": {
            "learned_thresholds": agg_learned,
            "fixed_0p5": agg_fixed,
        },
        "by_risk_level": by_risk,
        "per_clause": clause_df.to_dict(orient="records"),
        "latency_ms": {
            "batch_total": batch_ms,
            "batch_per_contract": per_ms,
            "single_median": single_median_ms,
            "claude_zero_shot_mark_p5": 11100,
            "claude_few_shot_mark_p5": 15400,
            "speedup_vs_claude_zero_shot": 11100 / single_median_ms,
        },
    }
    json_path = out_dir / "phase6_evaluation.json"
    json_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"Saved metrics: {json_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate production Legal Contract Analyzer")
    parser.add_argument("--data", default="data/raw/CUADv1.json")
    parser.add_argument("--models", default="models")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_eval(args.data, args.models, args.out_dir, args.seed)


if __name__ == "__main__":
    main()
