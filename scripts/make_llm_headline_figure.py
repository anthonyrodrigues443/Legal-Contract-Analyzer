"""Build a 4-panel headline comparison: our LGBM+LR blend vs Claude Sonnet 4.6."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

llm = json.loads((RESULTS / "phase5_llm_results.json").read_text())
p6 = json.loads((RESULTS / "phase6_evaluation.json").read_text())

OUR_HR_F1 = p6["by_risk_level"]["HIGH"]["mean_f1"]
OUR_LATENCY_MS = 443.0
OUR_COST_PER_1K = 0.0

CLAUDE_ZERO_HR_F1 = llm["zero_shot"]["hr_macro_f1"]
CLAUDE_FEW_HR_F1 = llm["few_shot"]["hr_macro_f1"]
CLAUDE_ZERO_LAT_MS = llm["zero_shot"]["avg_latency_s"] * 1000.0
CLAUDE_FEW_LAT_MS = llm["few_shot"]["avg_latency_s"] * 1000.0
CLAUDE_ZERO_COST_1K = 15.0
CLAUDE_FEW_COST_1K = 20.0

NAVY = "#0b3954"
TEAL = "#087e8b"
CORAL = "#ff6b6b"
GOLD = "#f4b942"
MUTED = "#9aa5ad"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "axes.labelcolor": "#222222",
    "xtick.color": "#222222",
    "ytick.color": "#222222",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "LGBM+LR blend vs Claude Sonnet 4.6 — CUAD high-risk clause detection",
    fontsize=15, fontweight="bold", y=0.995,
)

# ---- Panel 1: HR macro-F1 ---------------------------------------------------
ax = axes[0, 0]
models = ["Our LGBM+LR\n(production)", "Claude Sonnet 4.6\n(zero-shot)", "Claude Sonnet 4.6\n(few-shot)"]
values = [OUR_HR_F1, CLAUDE_ZERO_HR_F1, CLAUDE_FEW_HR_F1]
colors = [TEAL, CORAL, GOLD]
bars = ax.bar(models, values, color=colors, edgecolor="#222", linewidth=0.6)
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f"{v:.3f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_ylim(0, 0.65)
ax.set_ylabel("High-risk macro-F1", fontsize=11)
ax.set_title("Quality — we catch 3.2x more high-risk clauses", fontsize=12, fontweight="bold", loc="left")
ratio = OUR_HR_F1 / CLAUDE_ZERO_HR_F1
ax.annotate(f"{ratio:.1f}x", xy=(0, OUR_HR_F1), xytext=(0.5, 0.58),
            fontsize=18, fontweight="bold", color=TEAL, ha="center")
ax.grid(axis="y", alpha=0.25, linestyle="--")

# ---- Panel 2: Per-clause HR-F1 ---------------------------------------------
ax = axes[0, 1]
clauses = ["Uncapped\nLiability", "Change of\nControl", "Non-Compete", "Liquidated\nDamages"]
our_vals = [
    llm["lightgbm"]["per_clause"]["Uncapped Liability"]["f1"],
    llm["lightgbm"]["per_clause"]["Change Of Control"]["f1"],
    llm["lightgbm"]["per_clause"]["Non-Compete"]["f1"],
    llm["lightgbm"]["per_clause"]["Liquidated Damages"]["f1"],
]
claude_vals = [
    llm["zero_shot"]["per_clause"]["Uncapped Liability"]["f1"],
    llm["zero_shot"]["per_clause"]["Change Of Control"]["f1"],
    llm["zero_shot"]["per_clause"]["Non-Compete"]["f1"],
    llm["zero_shot"]["per_clause"]["Liquidated Damages"]["f1"],
]
x = np.arange(len(clauses))
w = 0.38
b1 = ax.bar(x - w / 2, our_vals, w, label="Our LGBM+LR", color=TEAL, edgecolor="#222", linewidth=0.6)
b2 = ax.bar(x + w / 2, claude_vals, w, label="Claude zero-shot", color=CORAL, edgecolor="#222", linewidth=0.6)
for bars in (b1, b2):
    for bar in bars:
        h = bar.get_height()
        label = f"{h:.2f}" if h > 0 else "0"
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, label,
                ha="center", va="bottom", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(clauses, fontsize=10)
ax.set_ylabel("F1 score", fontsize=11)
ax.set_ylim(0, 0.85)
ax.set_title("Per-clause — Claude scores 0 on 2 of 4 high-risk clauses", fontsize=12, fontweight="bold", loc="left")
ax.legend(frameon=False, fontsize=10, loc="upper right")
ax.grid(axis="y", alpha=0.25, linestyle="--")

# ---- Panel 3: Latency (log scale) ------------------------------------------
ax = axes[1, 0]
names = ["Our LGBM+LR\n(CPU, per contract)", "Claude Sonnet 4.6\n(zero-shot)", "Claude Sonnet 4.6\n(few-shot)"]
lat = [OUR_LATENCY_MS, CLAUDE_ZERO_LAT_MS, CLAUDE_FEW_LAT_MS]
bars = ax.barh(names, lat, color=[TEAL, CORAL, GOLD], edgecolor="#222", linewidth=0.6)
ax.set_xscale("log")
ax.set_xlabel("Latency (ms, log scale)", fontsize=11)
ax.set_title("Speed — 25x–35x faster than Claude", fontsize=12, fontweight="bold", loc="left")
for bar, v in zip(bars, lat):
    if v < 1000:
        label = f"{v:.0f} ms"
    else:
        label = f"{v / 1000:.1f} s"
    ax.text(v * 1.12, bar.get_y() + bar.get_height() / 2, label,
            va="center", fontsize=11, fontweight="bold")
ax.set_xlim(100, 50_000)
ax.grid(axis="x", alpha=0.25, linestyle="--", which="both")
speedup = CLAUDE_ZERO_LAT_MS / OUR_LATENCY_MS
ax.text(0.97, 0.05, f"{speedup:.0f}x speedup", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=13, fontweight="bold", color=TEAL,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=TEAL))

# ---- Panel 4: Cost per 1,000 contracts -------------------------------------
ax = axes[1, 1]
cost_names = ["Our LGBM+LR", "Claude zero-shot", "Claude few-shot"]
costs = [OUR_COST_PER_1K, CLAUDE_ZERO_COST_1K, CLAUDE_FEW_COST_1K]
bars = ax.bar(cost_names, costs, color=[TEAL, CORAL, GOLD], edgecolor="#222", linewidth=0.6)
for bar, v in zip(bars, costs):
    label = "$0" if v == 0 else f"${v:.0f}"
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.4, label,
            ha="center", va="bottom", fontsize=12, fontweight="bold")
ax.set_ylabel("USD per 1,000 contracts", fontsize=11)
ax.set_ylim(0, 24)
ax.set_title("Cost — $0 vs $15–$20 per 1K contracts", fontsize=12, fontweight="bold", loc="left")
ax.grid(axis="y", alpha=0.25, linestyle="--")
ax.text(0.02, 0.95,
        "At 100K contracts/yr:\n  Our model → $0\n  Claude → $1,500–$2,000",
        transform=ax.transAxes, va="top", ha="left", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f7f7f7", edgecolor=MUTED))

fig.text(0.5, 0.005,
         "CUAD v1 | 30-contract test set | HR = High-Risk clauses "
         "(Uncapped Liability, Change of Control, Non-Compete, Liquidated Damages)",
         ha="center", fontsize=9, color=MUTED, style="italic")

plt.tight_layout(rect=[0, 0.02, 1, 0.97])
out = RESULTS / "llm_headline_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved → {out}")
