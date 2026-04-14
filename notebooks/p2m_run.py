import os, sys
os.chdir(r'C:\Users\antho\OneDrive\Desktop\YC-Portfolio-Projects\Legal-Contract-Analyzer')
import numpy as np, pandas as pd, time, json, warnings, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

df = pd.read_parquet('data/processed/cuad_classification.parquet')
meta = ['contract_title','text','text_length','word_count']
label_cols = [c for c in df.columns if c not in meta]
np.random.seed(42)
idx = np.random.permutation(len(df))
trdf = df.iloc[idx[:408]].reset_index(drop=True)
tedf = df.iloc[idx[408:]].reset_index(drop=True)
vc = [c for c in label_cols if tedf[c].sum() >= 3]
Xt, Xe = trdf['text'].values, tedf['text'].values
yt, ye = trdf[vc].values, tedf[vc].values
print(f"Data: {len(Xt)} train, {len(Xe)} test, {len(vc)} clauses")

def ev(y_true, y_pred, y_prob=None, name=''):
    mf1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    mif1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    mp = precision_score(y_true, y_pred, average='macro', zero_division=0)
    mr = recall_score(y_true, y_pred, average='macro', zero_division=0)
    auc = None
    if y_prob is not None:
        aucs = [roc_auc_score(y_true[:,i], y_prob[:,i]) for i in range(y_true.shape[1]) if 0 < y_true[:,i].sum() < len(y_true)]
        auc = round(float(np.mean(aucs)), 4) if aucs else None
    pc = {}
    for i, c in enumerate(vc):
        if y_true[:,i].sum() > 0:
            pc[c] = {'f1': round(float(f1_score(y_true[:,i], y_pred[:,i], zero_division=0)), 4)}
    return {'model':name,'macro_f1':round(float(mf1),4),'micro_f1':round(float(mif1),4),
            'macro_precision':round(float(mp),4),'macro_recall':round(float(mr),4),
            'macro_auc':auc,'per_clause':pc}

def ovr(Xtr, Xte, ytr, yte, make_fn, name):
    pr = np.zeros_like(yte); pb = np.zeros_like(yte, dtype=float)
    for j in range(ytr.shape[1]):
        if len(np.unique(ytr[:,j])) < 2:
            pr[:,j] = ytr[:,j][0]; pb[:,j] = float(ytr[:,j][0]); continue
        c = make_fn(j); c.fit(Xtr, ytr[:,j]); pr[:,j] = c.predict(Xte)
        try: pb[:,j] = c.predict_proba(Xte)[:,1]
        except: pass
    return ev(yte, pr, pb, name), pr, pb

# 20K TF-IDF (optimal from ablation)
vec = TfidfVectorizer(max_features=20000, ngram_range=(1,2), sublinear_tf=True, min_df=2, max_df=0.95)
Xt20 = vec.fit_transform(Xt); Xe20 = vec.transform(Xe)
print(f"TF-IDF 20K: {Xt20.shape}")

# XGBoost
print("Running XGBoost...")
t0 = time.time()
def mkxgb(j):
    pw = float((yt[:,j]==0).sum())/max(float((yt[:,j]==1).sum()),1.0)
    return xgb.XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.15,
        subsample=0.8, colsample_bytree=0.4, scale_pos_weight=pw,
        eval_metric='logloss', tree_method='hist', verbosity=0, random_state=42)
xr, xp, xpb = ovr(Xt20, Xe20, yt, ye, mkxgb, 'XGBoost+TF-IDF(20K)')
xr['t'] = round(time.time()-t0,1)
print(f"XGBoost: F1={xr['macro_f1']:.4f} P={xr['macro_precision']:.4f} R={xr['macro_recall']:.4f} AUC={xr['macro_auc']} {xr['t']}s")

# LSA + RF
print("Running LSA+RF...")
svd = TruncatedSVD(n_components=256, random_state=42)
Xtl = svd.fit_transform(Xt20); Xel = svd.transform(Xe20)
print(f"  LSA var: {svd.explained_variance_ratio_.sum():.3f}")
t0 = time.time()
def mkrf(j): return RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced_subsample', min_samples_leaf=2, n_jobs=2, random_state=42)
rr, rp, rpb = ovr(Xtl, Xel, yt, ye, mkrf, 'LSA-256+RandomForest')
rr['t'] = round(time.time()-t0,1)
print(f"LSA+RF: F1={rr['macro_f1']:.4f} P={rr['macro_precision']:.4f} R={rr['macro_recall']:.4f} AUC={rr['macro_auc']} {rr['t']}s")

# LR 20K corrected baseline
print("Running LR 20K...")
t0 = time.time()
def mklr(j): return LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, solver='lbfgs')
lr, lp, lpb = ovr(Xt20, Xe20, yt, ye, mklr, 'TF-IDF(20K)+LR')
lr['t'] = round(time.time()-t0,1)
print(f"LR(20K): F1={lr['macro_f1']:.4f} P={lr['macro_precision']:.4f} R={lr['macro_recall']:.4f} AUC={lr['macro_auc']} {lr['t']}s")

# High-risk
HR = ['Uncapped Liability','Ip Ownership Assignment','Change Of Control','Non-Compete','Liquidated Damages','Joint Ip Ownership']
hri = [c for c in HR if c in vc]
print("\n=== HIGH-RISK ANALYSIS ===")
print(f"{'Clause':32} {'LR':>10} {'XGB':>10} {'LSA+RF':>10}")
print("-"*62)
for c in hri:
    ci = vc.index(c)
    lf = float(f1_score(ye[:,ci], lp[:,ci], zero_division=0))
    xf = float(f1_score(ye[:,ci], xp[:,ci], zero_division=0))
    rf = float(f1_score(ye[:,ci], rp[:,ci], zero_division=0))
    print(f"{c:32} {lf:>10.3f} {xf:>10.3f} {rf:>10.3f}")
hi = [vc.index(c) for c in hri]
lh = float(f1_score(ye[:,hi], lp[:,hi], average='macro', zero_division=0))
xh = float(f1_score(ye[:,hi], xp[:,hi], average='macro', zero_division=0))
rh = float(f1_score(ye[:,hi], rp[:,hi], average='macro', zero_division=0))
print(f"{'HR MACRO-F1':32} {lh:>10.3f} {xh:>10.3f} {rh:>10.3f}")

# Summary
print("\n=== MASTER TABLE (All Phase 2) ===")
abl = [
    ('TF-IDF+LR (5K)', 0.5943), ('TF-IDF+LR (10K)', 0.5992),
    ('TF-IDF+LR (20K)', 0.6026), ('TF-IDF+LR (50K)', 0.5836), ('TF-IDF+LR (100K)', 0.5651),
    ('Complement NB', 0.4677), ('Multinomial NB', 0.5493),
    (xr['model'], xr['macro_f1']), (rr['model'], rr['macro_f1']), (lr['model'], lr['macro_f1']),
    ('[Anthony] TF-IDF+LightGBM', 0.5750), ('[Anthony] TF-IDF+SVM', 0.5316),
    ('[Anthony] Legal-BERT CLS+LR', 0.5144), ('[Anthony] SBERT+LR', 0.4721),
    ('[Anthony] Legal-BERT FT', 0.4098), ('[Anthony] BERT-base FT', 0.3501),
]
abl.sort(key=lambda x: x[1], reverse=True)
print(f"{'Rank':>4} {'Model':45} {'Macro-F1':>10}")
print("-"*65)
for rank, (nm, sc) in enumerate(abl, 1):
    print(f"{rank:>4} {nm:45} {sc:>10.4f}")

# Save
best = max([xr, rr, lr], key=lambda x: x['macro_f1'])
out = {
    'phase':'2_mark', 'date':'2026-04-14', 'dataset':'CUAD v1', 'primary_metric':'macro_f1',
    'n_train':408, 'n_test':102, 'n_labels':len(vc),
    'experiments': {
        'vocab_ablation': [{'n_features':k,'macro_f1':v} for k,v in
            [(5000,0.5943),(10000,0.5992),(20000,0.6026),(50000,0.5836),(100000,0.5651)]],
        'complement_nb': {'macro_f1':0.4677,'macro_precision':0.394,'macro_recall':0.898,'macro_auc':0.759},
        'multinomial_nb': {'macro_f1':0.5493},
        'xgboost_20k': {k:v for k,v in xr.items() if k != 'per_clause'},
        'lsa_rf': {k:v for k,v in rr.items() if k != 'per_clause'},
        'lr_20k': {k:v for k,v in lr.items() if k != 'per_clause'},
    },
    'champion': best['model'], 'champion_f1': best['macro_f1'],
    'key_finding': 'Goldilocks vocabulary: 20K bigrams optimal on CUAD. 100K features HURTS by -0.037. Complement NB fails despite imbalance design.',
    'high_risk_f1': {'LR_20K': lh, 'XGBoost': xh, 'LSA_RF': rh},
    'anthony_best': 0.5750,
}
with open('results/phase2_mark_metrics.json', 'w') as f: json.dump(out, f, indent=2)
print("\nSaved: results/phase2_mark_metrics.json")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
all_pts = sorted([
    ('Phase1: TF-IDF+LR', 0.642, '#2ecc71'),
    ('Mark: LR(20K)', lr['macro_f1'], '#27ae60'),
    ('Mark: XGBoost(20K)', xr['macro_f1'], '#e74c3c'),
    ('Mark: LSA-256+RF', rr['macro_f1'], '#e67e22'),
    ('Mark: Complement NB', 0.4677, '#9b59b6'),
    ('Mark: Multinomial NB', 0.5493, '#8e44ad'),
    ('[A] TF-IDF+LightGBM', 0.5750, '#90a4ae'),
    ('[A] TF-IDF+SVM', 0.5316, '#b0bec5'),
    ('[A] Legal-BERT CLS+LR', 0.5144, '#cfd8dc'),
    ('[A] SBERT+LR', 0.4721, '#bdbdbd'),
    ('[A] Legal-BERT FT', 0.4098, '#90a4ae'),
    ('[A] BERT-base FT', 0.3501, '#607d8b'),
], key=lambda x: x[1], reverse=True)
nm2 = [x[0] for x in all_pts]; sc2 = [x[1] for x in all_pts]; co2 = [x[2] for x in all_pts]
bars = axes[0].barh(range(len(nm2)), sc2, color=co2, edgecolor='white', linewidth=0.4)
axes[0].set_yticks(range(len(nm2))); axes[0].set_yticklabels(nm2, fontsize=9)
axes[0].set_xlabel('Macro-F1', fontsize=12)
axes[0].set_title('CUAD Clause Detection -- All Phase 2 Models\n(Mark: green/red/orange | Anthony: grey)', fontsize=11, fontweight='bold')
axes[0].axvline(x=0.650, color='gray', linestyle=':', alpha=0.6, label='Published RoBERTa')
for bar, score in zip(bars, sc2):
    axes[0].text(score+0.003, bar.get_y()+bar.get_height()/2, f'{score:.3f}', va='center', fontsize=8)
axes[0].legend(fontsize=9); axes[0].set_xlim(0, 0.76)
vocab_x = [5000,10000,20000,50000,100000]
f1_y = [0.5943,0.5992,0.6026,0.5836,0.5651]
axes[1].plot(vocab_x, f1_y, 'b-o', linewidth=2.5, markersize=10)
axes[1].axvline(x=20000, color='green', linestyle='--', alpha=0.7, label='Optimal: 20K')
axes[1].set_xscale('log'); axes[1].set_xlabel('TF-IDF Vocabulary Size', fontsize=12); axes[1].set_ylabel('Macro-F1', fontsize=12)
axes[1].set_title('COUNTERINTUITIVE: More Vocabulary = Worse\nGoldilocks zone at 20K bigrams', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10); axes[1].grid(True, alpha=0.3)
for x, y in zip(vocab_x, f1_y):
    axes[1].annotate(f'{y:.3f}', (x,y), textcoords='offset points', xytext=(0,12), ha='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig('results/phase2_mark_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close('all')
print("Saved: results/phase2_mark_model_comparison.png")
print("DONE")
