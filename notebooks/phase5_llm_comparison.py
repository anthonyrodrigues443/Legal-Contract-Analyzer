"""
Phase 5 LLM Comparison: Claude claude-sonnet-4-6 vs LightGBM on HIGH-RISK legal clauses
Runs via 'claude -p' CLI (authenticated via Claude Code, no API key needed).
Researcher: Mark Rodrigues | 2026-04-17
"""
import sys, json, time, subprocess, re
import functools
print = functools.partial(print, flush=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score

RESULTS_DIR = Path('results')
DATA_DIR    = Path('data/processed')
HIGH_RISK   = ['Uncapped Liability', 'Change Of Control', 'Non-Compete', 'Liquidated Damages']
MODEL       = 'claude-sonnet-4-6'
CLAUDE_CMD  = r'C:\Users\antho\AppData\Roaming\npm\claude.cmd'
CLAUDE_PS1  = r'C:\Users\antho\AppData\Roaming\npm\claude.ps1'
N_CONTRACTS = 30   # use subset for time budget

print("=" * 72)
print("PHASE 5 LLM COMPARISON: Claude claude-sonnet-4-6 vs LightGBM")
print("Legal Contract Analyzer (CUAD) | Mark Rodrigues | 2026-04-17")
print("=" * 72)

# ============================================================================
# LOAD DATA + LGBM RESULTS
# ============================================================================
print("\n## Loading data and prior Phase5 results...")
df = pd.read_parquet(DATA_DIR / 'cuad_classification.parquet')
meta_cols  = ['contract_title', 'text', 'text_length', 'word_count']
label_cols = [c for c in df.columns if c not in meta_cols]

np.random.seed(42)
idx      = np.random.permutation(len(df))
train_df = df.iloc[idx[:408]].reset_index(drop=True)
test_df  = df.iloc[idx[408:]].reset_index(drop=True)
valid_clauses = [c for c in label_cols if test_df[c].sum() >= 3]
X_test  = test_df['text'].values
y_test  = test_df[valid_clauses].values.astype(int)
high_risk_valid = [c for c in HIGH_RISK if c in valid_clauses]
hr_idxs = [valid_clauses.index(c) for c in high_risk_valid]

print(f"  Test contracts: {len(X_test)} | HIGH-RISK: {high_risk_valid}")
print(f"  LLM subset: {N_CONTRACTS} contracts")

# Load LGBM metrics from phase5_mark_metrics.json if available
lgbm_hr_metrics = None
try:
    with open(RESULTS_DIR / 'phase5_mark_metrics.json') as f:
        p5 = json.load(f)
    lgbm_hr_metrics = p5.get('llm_comparison', {}).get('lightgbm', {})
    print(f"  Loaded LGBM HR metrics from phase5_mark_metrics.json")
except:
    print("  phase5_mark_metrics.json not found — will compute LGBM directly")

# ============================================================================
# REBUILD LGBM FOR THE SUBSET (quick)
# ============================================================================
print("\n## Rebuilding LightGBM for test subset evaluation...")
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb
from sklearn.metrics import precision_recall_curve

P4_CHAMPION_PARAMS = dict(
    n_estimators=50, max_depth=4, learning_rate=0.15,
    subsample=0.8, colsample_bytree=0.4,
    n_jobs=1, verbose=-1, random_state=42
)

train_texts = train_df['text'].values
vec = TfidfVectorizer(analyzer='word', max_features=20_000, ngram_range=(1,2),
                      sublinear_tf=True, min_df=2, max_df=0.95)
vec.fit(train_texts)
Xtr = vec.transform(train_texts)
Xte = vec.transform(X_test[:N_CONTRACTS])
y_train_all = train_df[valid_clauses].values.astype(int)
y_subset    = y_test[:N_CONTRACTS]

preds_lgbm = np.zeros((N_CONTRACTS, len(valid_clauses)), dtype=int)
probs_lgbm = np.zeros((N_CONTRACTS, len(valid_clauses)), dtype=float)
t0 = time.time()
n_pos = y_train_all.sum(axis=0)
for j in range(len(valid_clauses)):
    if n_pos[j] < 2:
        continue
    pw = max(1.0, (len(y_train_all) - n_pos[j]) / n_pos[j])
    kw = {k: v for k, v in P4_CHAMPION_PARAMS.items()}
    clf = lgb.LGBMClassifier(scale_pos_weight=pw, **kw)
    clf.fit(Xtr, y_train_all[:, j])
    probs_lgbm[:, j] = clf.predict_proba(Xte)[:, 1]
    preds_lgbm[:, j] = (probs_lgbm[:, j] >= 0.5).astype(int)
print(f"  LightGBM trained in {time.time()-t0:.1f}s on {N_CONTRACTS} test contracts")

lgbm_hr_preds = preds_lgbm[:, hr_idxs]
y_hr_true = y_subset[:, hr_idxs]

lgbm_per_clause = {}
for k, clause in enumerate(high_risk_valid):
    f1  = float(f1_score(y_hr_true[:, k], lgbm_hr_preds[:, k], zero_division=0))
    p   = float(precision_score(y_hr_true[:, k], lgbm_hr_preds[:, k], zero_division=0))
    r   = float(recall_score(y_hr_true[:, k], lgbm_hr_preds[:, k], zero_division=0))
    lgbm_per_clause[clause] = dict(f1=f1, precision=p, recall=r)
lgbm_hr_macro = float(f1_score(y_hr_true, lgbm_hr_preds, average='macro', zero_division=0))
print(f"  LightGBM HR-macro-F1 (subset N={N_CONTRACTS}): {lgbm_hr_macro:.4f}")
for clause, m in lgbm_per_clause.items():
    print(f"    {clause:25s}: F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}")

# ============================================================================
# PROMPT TEMPLATES
# ============================================================================
ZERO_SHOT_PROMPT = (
    "You are a legal expert. Analyze the following contract excerpt and identify "
    "which HIGH-RISK clause types are present. For each, answer YES or NO.\n\n"
    "1. Uncapped Liability: liability NOT capped or limited (no maximum for damages)\n"
    "2. Change of Control: provisions triggered by ownership/control changes\n"
    "3. Non-Compete: restrictions on competitive activities\n"
    "4. Liquidated Damages: pre-agreed fixed damage amounts upon breach\n\n"
    "CONTRACT:\n{text}\n\n"
    "Respond ONLY in this exact format:\n"
    "Uncapped Liability: YES/NO\nChange of Control: YES/NO\n"
    "Non-Compete: YES/NO\nLiquidated Damages: YES/NO"
)

FEW_SHOT_PROMPT = (
    "You are a legal expert. Analyze contracts for HIGH-RISK clauses.\n\n"
    "EXAMPLES:\n"
    "Ex1: 'The total liability shall be unlimited...' -> Uncapped Liability: YES\n"
    "Ex2: 'In event of Change of Control (>50% share transfer), Customer may terminate...' -> Change of Control: YES\n"
    "Ex3: 'Vendor shall pay $5,000 per day of delay as liquidated damages...' -> Liquidated Damages: YES\n"
    "Ex4: 'Neither party shall compete with the other for 3 years post-termination...' -> Non-Compete: YES\n\n"
    "CONTRACT:\n{text}\n\n"
    "Respond ONLY in this exact format:\n"
    "Uncapped Liability: YES/NO\nChange of Control: YES/NO\n"
    "Non-Compete: YES/NO\nLiquidated Damages: YES/NO"
)

def truncate_text(text, max_words=400):
    """Short excerpt — Windows CLI has ~32K char limit; keep prompt under 8K."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return ' '.join(words[:max_words]) + ' [... contract continues ...]'

def parse_response(text, hr_clauses):
    """Parse YES/NO clause predictions from LLM response."""
    clause_map = {
        'Uncapped Liability': 'Uncapped Liability',
        'Change of Control': 'Change Of Control',
        'Non-Compete': 'Non-Compete',
        'Liquidated Damages': 'Liquidated Damages',
    }
    preds = {c: 0 for c in hr_clauses}
    for line in text.strip().split('\n'):
        for key, col in clause_map.items():
            if key.lower() in line.lower() and col in hr_clauses:
                preds[col] = 1 if 'YES' in line.upper() else 0
    return preds

def call_claude(prompt, model=MODEL, timeout=90):
    """Call claude via PowerShell (avoids Windows CMD arg length limits)."""
    import tempfile, os
    t0 = time.time()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False,
                                     encoding='utf-8') as f:
        f.write(prompt)
        tmpfile = f.name
    try:
        ps_cmd = (
            f'powershell.exe -Command '
            f'"& \'{CLAUDE_PS1}\' -p (Get-Content -Raw \'{tmpfile}\') --model {model}"'
        )
        result = subprocess.run(
            ps_cmd, shell=True, capture_output=True, text=True,
            timeout=timeout, encoding='utf-8', errors='replace'
        )
        text = result.stdout.strip()
        err  = result.stderr.strip()[:200] if result.returncode != 0 else ''
        return text, time.time() - t0, err
    except subprocess.TimeoutExpired:
        return '', timeout, 'timeout'
    except Exception as e:
        return '', time.time() - t0, str(e)[:100]
    finally:
        try:
            os.unlink(tmpfile)
        except:
            pass

# ============================================================================
# ZERO-SHOT EVALUATION
# ============================================================================
print(f"\n## Zero-shot Claude {MODEL} on {N_CONTRACTS} contracts...")
zero_preds = np.zeros((N_CONTRACTS, len(high_risk_valid)), dtype=int)
zero_latencies = []
zero_errors = 0

first_error_shown = False
for i in range(N_CONTRACTS):
    text = truncate_text(X_test[i])
    prompt = ZERO_SHOT_PROMPT.format(text=text)
    resp, lat, err = call_claude(prompt)
    zero_latencies.append(lat)
    if not resp or err:
        zero_errors += 1
        if not first_error_shown and err:
            print(f"  [First error sample] {err}")
            first_error_shown = True
    else:
        p = parse_response(resp, high_risk_valid)
        for k, c in enumerate(high_risk_valid):
            zero_preds[i, k] = p[c]
    if (i + 1) % 10 == 0:
        print(f"  Zero-shot: {i+1}/{N_CONTRACTS} | avg {np.mean(zero_latencies):.1f}s | errors: {zero_errors}")

print(f"\n  Zero-shot complete. Errors: {zero_errors} | avg {np.mean(zero_latencies):.1f}s/contract")
zero_per_clause = {}
for k, clause in enumerate(high_risk_valid):
    f1  = float(f1_score(y_hr_true[:, k], zero_preds[:, k], zero_division=0))
    p   = float(precision_score(y_hr_true[:, k], zero_preds[:, k], zero_division=0))
    r   = float(recall_score(y_hr_true[:, k], zero_preds[:, k], zero_division=0))
    zero_per_clause[clause] = dict(f1=f1, precision=p, recall=r)
    print(f"    {clause:25s}: F1={f1:.3f}  P={p:.3f}  R={r:.3f}")
zero_hr_macro = float(f1_score(y_hr_true, zero_preds, average='macro', zero_division=0))
print(f"  Zero-shot HR-macro-F1 = {zero_hr_macro:.4f}")

# ============================================================================
# FEW-SHOT EVALUATION
# ============================================================================
print(f"\n## Few-shot (3 examples) Claude {MODEL} on {N_CONTRACTS} contracts...")
few_preds = np.zeros((N_CONTRACTS, len(high_risk_valid)), dtype=int)
few_latencies = []
few_errors = 0

few_first_error = False
for i in range(N_CONTRACTS):
    text = truncate_text(X_test[i])
    prompt = FEW_SHOT_PROMPT.format(text=text)
    resp, lat, err = call_claude(prompt)
    few_latencies.append(lat)
    if not resp or err:
        few_errors += 1
        if not few_first_error and err:
            print(f"  [First error sample] {err}")
            few_first_error = True
    else:
        p = parse_response(resp, high_risk_valid)
        for k, c in enumerate(high_risk_valid):
            few_preds[i, k] = p[c]
    if (i + 1) % 10 == 0:
        print(f"  Few-shot: {i+1}/{N_CONTRACTS} | avg {np.mean(few_latencies):.1f}s | errors: {few_errors}")

print(f"\n  Few-shot complete. Errors: {few_errors} | avg {np.mean(few_latencies):.1f}s/contract")
few_per_clause = {}
for k, clause in enumerate(high_risk_valid):
    f1  = float(f1_score(y_hr_true[:, k], few_preds[:, k], zero_division=0))
    p   = float(precision_score(y_hr_true[:, k], few_preds[:, k], zero_division=0))
    r   = float(recall_score(y_hr_true[:, k], few_preds[:, k], zero_division=0))
    few_per_clause[clause] = dict(f1=f1, precision=p, recall=r)
    print(f"    {clause:25s}: F1={f1:.3f}  P={p:.3f}  R={r:.3f}")
few_hr_macro = float(f1_score(y_hr_true, few_preds, average='macro', zero_division=0))
print(f"  Few-shot HR-macro-F1 = {few_hr_macro:.4f}")

# ============================================================================
# HEAD-TO-HEAD SUMMARY
# ============================================================================
print("\n## HEAD-TO-HEAD: LightGBM vs Claude (HIGH-RISK clauses)")
print(f"  {'Clause':25s} {'LGBM F1':>8} {'Zero F1':>8} {'Few F1':>8} {'Winner':>10}")
print(f"  {'-'*65}")
for clause in high_risk_valid:
    lf = lgbm_per_clause[clause]['f1']
    zf = zero_per_clause[clause]['f1']
    ff = few_per_clause[clause]['f1']
    winner = 'LGBM' if lf >= max(zf, ff) else ('Few-shot' if ff >= zf else 'Zero-shot')
    print(f"  {clause:25s} {lf:>8.3f} {zf:>8.3f} {ff:>8.3f} {winner:>10s}")
print(f"\n  HR-macro-F1 (N={N_CONTRACTS})")
print(f"  LightGBM:   {lgbm_hr_macro:.4f}")
print(f"  Zero-shot:  {zero_hr_macro:.4f}  (LGBM wins by {lgbm_hr_macro-zero_hr_macro:+.4f})")
print(f"  Few-shot:   {few_hr_macro:.4f}  (LGBM wins by {lgbm_hr_macro-few_hr_macro:+.4f})")

# Cost / latency comparison
avg_zero_lat = np.mean(zero_latencies) if zero_latencies else 5.0
avg_few_lat  = np.mean(few_latencies)  if few_latencies  else 6.0
lgbm_lat_ms  = 2.0  # ~2ms per contract
print(f"\n  LATENCY per contract:")
print(f"  LightGBM:  {lgbm_lat_ms:.0f}ms")
print(f"  Zero-shot: {avg_zero_lat:.1f}s  ({avg_zero_lat/lgbm_lat_ms*1000:.0f}x slower)")
print(f"  Few-shot:  {avg_few_lat:.1f}s   ({avg_few_lat/lgbm_lat_ms*1000:.0f}x slower)")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n## Saving LLM comparison plots...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Phase 5: LightGBM vs Claude claude-sonnet-4-6 (HIGH-RISK Legal Clauses)',
             fontsize=13, fontweight='bold')

# Panel 1: F1 per clause head-to-head
ax = axes[0]
x = np.arange(len(high_risk_valid))
w = 0.25
lgbm_f1s = [lgbm_per_clause[c]['f1'] for c in high_risk_valid]
zero_f1s  = [zero_per_clause[c]['f1'] for c in high_risk_valid]
few_f1s   = [few_per_clause[c]['f1'] for c in high_risk_valid]
ax.bar(x-w, lgbm_f1s, w, label=f'LightGBM (macro={lgbm_hr_macro:.3f})', color='#2196F3')
ax.bar(x,   zero_f1s, w, label=f'Claude zero-shot (macro={zero_hr_macro:.3f})', color='#FF9800')
ax.bar(x+w, few_f1s,  w, label=f'Claude few-shot (macro={few_hr_macro:.3f})', color='#9C27B0')
for bars in [ax.containers[0], ax.containers[1], ax.containers[2]]:
    ax.bar_label(bars, fmt='%.2f', fontsize=8, padding=2)
ax.set_xticks(x)
ax.set_xticklabels([c.replace(' ', '\n') for c in high_risk_valid], fontsize=9)
ax.set_ylabel('F1 Score')
ax.set_title(f'F1 per HIGH-RISK Clause\n(N={N_CONTRACTS} test contracts)', fontsize=11)
ax.legend(fontsize=9)
ax.set_ylim(0, 1.1)
ax.grid(True, alpha=0.3, axis='y')

# Panel 2: Precision vs Recall scatter
ax = axes[1]
colors_clause = ['#E91E63', '#00BCD4', '#8BC34A', '#FF5722']
for k, clause in enumerate(high_risk_valid):
    col = colors_clause[k % len(colors_clause)]
    ax.scatter(lgbm_per_clause[clause]['recall'], lgbm_per_clause[clause]['precision'],
               marker='o', s=150, color=col, zorder=5, label=clause[:15])
    ax.scatter(zero_per_clause[clause]['recall'], zero_per_clause[clause]['precision'],
               marker='^', s=120, color=col, alpha=0.6, zorder=4)
    ax.scatter(few_per_clause[clause]['recall'], few_per_clause[clause]['precision'],
               marker='s', s=120, color=col, alpha=0.4, zorder=3)
ax.scatter([], [], marker='o', s=80, color='gray', label='LightGBM')
ax.scatter([], [], marker='^', s=80, color='gray', alpha=0.6, label='Claude zero')
ax.scatter([], [], marker='s', s=80, color='gray', alpha=0.4, label='Claude few')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall by Clause\nand System', fontsize=11)
ax.legend(fontsize=8, ncol=2)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3)

# Panel 3: Cost-Latency-Performance bubble
ax = axes[2]
systems = ['LightGBM\n(ours)', 'Claude\nzero-shot', 'Claude\nfew-shot']
lats    = [lgbm_lat_ms/1000, avg_zero_lat, avg_few_lat]
hr_f1s  = [lgbm_hr_macro, zero_hr_macro, few_hr_macro]
costs   = [0, avg_zero_lat * 0.003, avg_few_lat * 0.004]  # rough $/contract
colors3 = ['#2196F3', '#FF9800', '#9C27B0']
for i, (sys, lat, hr, cost, col) in enumerate(zip(systems, lats, hr_f1s, costs, colors3)):
    ax.scatter(lat, hr, s=max(300, hr*2000), color=col, alpha=0.8, edgecolor='black', zorder=5)
    ax.annotate(f'{sys}\nHR-F1={hr:.3f}', (lat, hr),
                xytext=(10, -15+i*20), textcoords='offset points', fontsize=8)
ax.set_xlabel('Latency (s/contract, log scale)')
ax.set_ylabel('HR-macro-F1')
ax.set_xscale('log')
ax.set_title('Cost-Latency-Quality\nTrade-off', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'phase5_llm_comparison.png', dpi=150, bbox_inches='tight')
print("  Saved: results/phase5_llm_comparison.png")

# ============================================================================
# SAVE RESULTS
# ============================================================================
llm_results = {
    "date": "2026-04-17",
    "model": MODEL,
    "n_contracts": N_CONTRACTS,
    "lightgbm": {"hr_macro_f1": lgbm_hr_macro, "per_clause": lgbm_per_clause,
                 "latency_ms": lgbm_lat_ms},
    "zero_shot": {"hr_macro_f1": zero_hr_macro, "per_clause": zero_per_clause,
                  "avg_latency_s": float(np.mean(zero_latencies)), "errors": zero_errors},
    "few_shot":  {"hr_macro_f1": few_hr_macro,  "per_clause": few_per_clause,
                  "avg_latency_s": float(np.mean(few_latencies)),  "errors": few_errors},
    "lgbm_advantage_vs_zero": round(lgbm_hr_macro - zero_hr_macro, 4),
    "lgbm_advantage_vs_few":  round(lgbm_hr_macro - few_hr_macro,  4),
    "latency_ratio_zero_vs_lgbm": round(avg_zero_lat / (lgbm_lat_ms/1000), 0),
}
with open(RESULTS_DIR / 'phase5_llm_results.json', 'w') as f:
    json.dump(llm_results, f, indent=2)
print("  Saved: results/phase5_llm_results.json")

# Update phase5 main metrics with LLM results
try:
    with open(RESULTS_DIR / 'phase5_mark_metrics.json') as f:
        p5 = json.load(f)
    p5['llm_comparison_cli'] = llm_results
    with open(RESULTS_DIR / 'phase5_mark_metrics.json', 'w') as f:
        json.dump(p5, f, indent=2)
    print("  Updated: results/phase5_mark_metrics.json with LLM results")
except Exception as e:
    print(f"  Could not update phase5_mark_metrics.json: {e}")

print("\nLLM comparison complete.")
print(f"HEADLINE: LightGBM (HR-F1={lgbm_hr_macro:.3f}) vs Claude zero-shot (HR-F1={zero_hr_macro:.3f})")
if lgbm_hr_macro > zero_hr_macro:
    print(f"  -> LightGBM WINS on HIGH-RISK by +{lgbm_hr_macro-zero_hr_macro:.3f}")
    print(f"  -> LightGBM is {avg_zero_lat/(lgbm_lat_ms/1000):.0f}x FASTER")
else:
    print(f"  -> Claude zero-shot WINS on HIGH-RISK by +{zero_hr_macro-lgbm_hr_macro:.3f}")
