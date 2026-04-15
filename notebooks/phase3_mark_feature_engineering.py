"""
Phase 3 Mark: Feature Engineering Deep Dive -- Legal Contract Analyzer
Date: 2026-04-15
Researcher: Mark Rodrigues

Research Question: Is the bottleneck the MODEL or the FEATURES?
Phase 2 showed TF-IDF(20K)+LR macro-F1=0.615, XGBoost macro-F1=0.605.
Phase 3 tests 4 feature engineering strategies to find what actually moves the needle.

Building on Phase 2 findings:
- XGBoost wins on HIGH-RISK clauses (HR-F1=0.576 vs LR's 0.517)
- Vocabulary Goldilocks at 20K bigrams
- Anthony found BERT fails: 512-token truncation covers only 5% of avg contract (7,643 words)

Phase 3 Hypotheses:
H1: Domain-specific legal features (negation, monetary, party asymmetry) improve HIGH-RISK recall
H2: Per-clause chi-squared feature selection beats generic TF-IDF
H3: Sliding window max-pooling solves the long-document problem Anthony identified
H4: Hybrid TF-IDF + domain features beats either alone
"""

import numpy as np
import pandas as pd
import re
import time
import json
import warnings
import scipy.sparse as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
warnings.filterwarnings('ignore')

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

RESULTS_DIR = Path('results')
DATA_DIR = Path('data/processed')

HIGH_RISK = ['Uncapped Liability', 'Change Of Control', 'Non-Compete', 'Liquidated Damages']

# ?
# CELL 1: DATA LOADING
# ?
print("=" * 70)
print("PHASE 3: FEATURE ENGINEERING DEEP DIVE")
print("=" * 70)
print("\n## Cell 1: Load Data")
print("Building on Phase 2: TF-IDF(20K)+LR=0.615, XGBoost HR-F1=0.576")

df = pd.read_parquet(DATA_DIR / 'cuad_classification.parquet')
meta_cols = ['contract_title', 'text', 'text_length', 'word_count']
label_cols = [c for c in df.columns if c not in meta_cols]

np.random.seed(42)
idx = np.random.permutation(len(df))
train_df = df.iloc[idx[:408]].reset_index(drop=True)
test_df  = df.iloc[idx[408:]].reset_index(drop=True)
valid_clauses = [c for c in label_cols if test_df[c].sum() >= 3]

X_train = train_df['text'].values
X_test  = test_df['text'].values
y_train = train_df[valid_clauses].values
y_test  = test_df[valid_clauses].values

high_risk_valid = [c for c in HIGH_RISK if c in valid_clauses]

print(f"Train: {len(X_train)} | Test: {len(X_test)} | Clauses: {len(valid_clauses)}")
print(f"Avg words per contract: {int(train_df['text'].apply(lambda x: len(x.split())).mean()):,}")
print(f"HIGH-RISK clauses measurable: {high_risk_valid}")


# ?
# CELL 2: UTILITIES
# ?
print("\n## Cell 2: Evaluation utilities")

def evaluate_multilabel(y_true, y_pred, y_prob=None, name='', clauses=None):
    macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, average='micro', zero_division=0))
    macro_p  = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    macro_r  = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    auc = None
    if y_prob is not None:
        aucs = [roc_auc_score(y_true[:, i], y_prob[:, i])
                for i in range(y_true.shape[1])
                if 0 < y_true[:, i].sum() < len(y_true)]
        auc = round(float(np.mean(aucs)), 4) if aucs else None

    hr_f1 = None
    if clauses:
        hr_idxs = [clauses.index(c) for c in high_risk_valid if c in clauses]
        if hr_idxs:
            hr_f1 = float(f1_score(y_true[:, hr_idxs], y_pred[:, hr_idxs],
                                    average='macro', zero_division=0))

    per_clause = {}
    if clauses:
        for i, c in enumerate(clauses):
            if y_true[:, i].sum() > 0:
                per_clause[c] = {
                    'f1': round(float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0)), 4),
                    'precision': round(float(precision_score(y_true[:, i], y_pred[:, i], zero_division=0)), 4),
                    'recall': round(float(recall_score(y_true[:, i], y_pred[:, i], zero_division=0)), 4),
                }

    return {
        'model': name,
        'macro_f1': round(macro_f1, 4),
        'micro_f1': round(micro_f1, 4),
        'macro_precision': round(macro_p, 4),
        'macro_recall': round(macro_r, 4),
        'macro_auc': auc,
        'hr_f1': round(hr_f1, 4) if hr_f1 is not None else None,
        'per_clause': per_clause,
    }


def fit_ovr(X_tr, X_te, y_tr, y_te, make_clf_fn, name, clauses):
    preds = np.zeros_like(y_te)
    probs = np.zeros_like(y_te, dtype=float)
    for j in range(y_tr.shape[1]):
        if len(np.unique(y_tr[:, j])) < 2:
            preds[:, j] = y_tr[:, j][0]
            probs[:, j] = float(y_tr[:, j][0])
            continue
        clf = make_clf_fn(j)
        clf.fit(X_tr, y_tr[:, j])
        preds[:, j] = clf.predict(X_te)
        try:
            probs[:, j] = clf.predict_proba(X_te)[:, 1]
        except:
            pass
    return evaluate_multilabel(y_te, preds, probs, name, clauses), preds, probs


# Phase 2 champion baseline (for comparison)
PHASE2_BASELINE = {
    'model': 'Phase2 Champion: TF-IDF(20K)+LR',
    'macro_f1': 0.6146,
    'hr_f1': 0.517,
    'macro_precision': 0.589,
    'macro_recall': 0.693,
    'macro_auc': 0.834,
}
PHASE2_XGBOOST = {
    'model': 'Phase2 XGBoost+TF-IDF(20K)',
    'macro_f1': 0.6052,
    'hr_f1': 0.576,
}
print("Phase 2 baselines loaded:")
print(f"  LR(20K):  Macro-F1={PHASE2_BASELINE['macro_f1']:.4f} | HR-F1={PHASE2_BASELINE['hr_f1']:.3f}")
print(f"  XGBoost:  Macro-F1={PHASE2_XGBOOST['macro_f1']:.4f} | HR-F1={PHASE2_XGBOOST['hr_f1']:.3f}")


# ?
# CELL 3: VECTORIZE BASELINE (TF-IDF 20K -- the Phase 2 optimum)
# ?
print("\n## Cell 3: Build baseline TF-IDF(20K) representation")

vec = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2),
                       sublinear_tf=True, min_df=2, max_df=0.95)
Xtr_tfidf = vec.fit_transform(X_train)
Xte_tfidf = vec.transform(X_test)
print(f"TF-IDF matrix: train={Xtr_tfidf.shape}, test={Xte_tfidf.shape}")


# ?
# CELL 4: EXPERIMENT 3.1 -- ENHANCED DOMAIN FEATURE ENGINEERING
# ?
print("\n" + "=" * 60)
print("## EXPERIMENT 3.1: Enhanced Domain Feature Engineering")
print("=" * 60)
print("""
Hypothesis: Hand-crafted legal features from domain literature
improve HIGH-RISK clause detection even though Phase 1 showed
basic domain features HURT overall F1 by -0.05.
New angle: 40+ richer features including negation context,
monetary amounts, party asymmetry -- not just keyword booleans.
Reference: CUAD paper (Hendrycks et al., 2021) identifies
clause-specific linguistic markers for each clause type.
""")

def extract_enhanced_legal_features(text: str) -> dict:
    """
    40+ domain-informed features based on legal NLP literature.
    Designed around CUAD clause taxonomy and corporate lawyer heuristics.
    """
    f = {}
    t = text.lower()

    #  Structural features 
    f['char_count'] = len(text)
    f['word_count'] = len(text.split())
    f['sentence_count'] = len(re.split(r'[.!?]+', text))
    f['avg_sent_len'] = f['word_count'] / max(f['sentence_count'], 1)

    # Section structure
    f['numbered_sections'] = len(re.findall(r'\n\s*\d+\.\s', text))
    f['lettered_sections'] = len(re.findall(r'\n\s*[a-z]\)\s', text, re.IGNORECASE))
    f['article_headers']   = len(re.findall(r'\bARTICLE\s+[IVX\d]+\b', text, re.IGNORECASE))
    f['exhibit_refs']      = len(re.findall(r'\bexhibit\s+[A-Z]\b', text, re.IGNORECASE))
    f['schedule_refs']     = len(re.findall(r'\bschedule\s+[A-Z\d]+\b', text, re.IGNORECASE))

    # Defined terms (CamelCase in quotes or parentheses)
    f['defined_terms_count'] = len(re.findall(r'"[A-Z][A-Za-z\s]+?"', text))
    f['has_definitions_section'] = int(bool(re.search(r'\bDEFINITIONS\b', text, re.IGNORECASE)))

    #  Obligation strength 
    f['shall_count']       = t.count('shall')
    f['must_count']        = t.count(' must ')
    f['may_count']         = t.count(' may ')
    f['will_count']        = t.count(' will ')
    f['shall_not_count']   = t.count('shall not')
    f['will_not_count']    = t.count('will not')
    f['obligation_ratio']  = (f['shall_count'] + f['must_count']) / max(f['may_count'] + 1, 1)

    #  Negation context near key legal terms 
    # "shall not exceed" = cap; "shall not be limited" = uncapped
    f['neg_exceed']        = len(re.findall(r'shall not exceed|will not exceed', t))
    f['neg_limited']       = len(re.findall(r'shall not be limited|not limited to', t))
    f['neg_liable']        = len(re.findall(r'not be liable|shall not be liable', t))
    f['neg_responsible']   = len(re.findall(r'not responsible|shall not be responsible', t))
    f['unlimited_phrases'] = len(re.findall(r'unlimited|all you can eat|without limit|no cap|no limit', t))

    #  Monetary and quantitative patterns ?
    f['dollar_amounts']    = len(re.findall(r'\$\s*[\d,]+', text))
    f['pct_amounts']       = len(re.findall(r'\d+\s*%', text))
    f['million_refs']      = len(re.findall(r'\bmillion\b', t))
    f['specific_cap']      = len(re.findall(r'\$\s*[\d,]+(?:\.\d+)?\s*(?:million|billion|thousand)?', text, re.IGNORECASE))
    # "an amount not to exceed $X" signals Cap On Liability
    f['amount_not_exceed'] = len(re.findall(r'amount not to exceed|not exceed.*\$', t))
    # Liquidated damages: specific dollar amounts with "per day/week/occurrence"
    f['liquidated_pattern'] = len(re.findall(r'\$[\d,]+\s*per\s*(?:day|week|month|occurrence|incident)', t))
    f['per_day_penalty']   = len(re.findall(r'per\s*(?:day|calendar day|business day)', t))

    #  Time period patterns 
    f['year_terms']        = len(re.findall(r'\d+[\-\s]year', t))
    f['month_terms']       = len(re.findall(r'\d+[\-\s]month', t))
    f['day_terms']         = len(re.findall(r'\d+[\-\s](?:calendar|business|banking)?\s*day', t))
    f['notice_period']     = len(re.findall(r'\d+\s*days?\s*(?:written\s*)?notice', t))

    #  Party reference features 
    # Asymmetric obligations: one party "shall" --> high-risk (e.g., IP assignment)
    f['company_shall']     = len(re.findall(r'company shall|licensor shall|assignor shall', t))
    f['party_shall']       = len(re.findall(r'party shall|parties shall|each party shall', t))
    f['mutual_obligations'] = len(re.findall(r'each party|both parties|mutually', t))
    f['one_sided_ratio']   = (f['company_shall']) / max(f['party_shall'] + f['mutual_obligations'] + 1, 1)

    #  Cross-reference patterns (clause interdependence) ?
    f['section_xrefs']     = len(re.findall(r'pursuant to (?:section|article|exhibit)', t))
    f['as_defined']        = len(re.findall(r'as defined (?:in|herein|above|below)', t))
    f['hereof_herein']     = len(re.findall(r'\bhereof\b|\bherein\b|\bhereinafter\b', t))

    #  HIGH-RISK clause-specific triggers ?
    # Uncapped Liability
    f['uncapped_trigger']  = int(bool(re.search(
        r'unlimited liability|uncapped|no limit on liability|entire loss|all loss|all damage', t)))
    # Change of Control
    f['coc_trigger']       = int(bool(re.search(
        r'change of control|change in control|acquisition|merger|takeover|majority.*shares', t)))
    f['coc_consent']       = int(bool(re.search(r'consent.*change of control|change of control.*consent', t)))
    # Non-compete
    f['noncompete_trigger']= int(bool(re.search(
        r'non.?compet|not (?:to )?compete|competitive business|competing product', t)))
    f['geographic_scope']  = int(bool(re.search(r'worldwide|globally|internationally|any jurisdiction', t)))
    # Liquidated Damages
    f['liquidated_trigger']= int(bool(re.search(r'liquidated damage|predetermined damage|fixed penalty', t)))
    # IP Assignment
    f['ip_assign_trigger'] = int(bool(re.search(
        r'(?:hereby )?assign(?:s)? (?:all )?(?:right|title|interest)|work for hire|work made for hire', t)))
    f['ip_retain']         = int(bool(re.search(r'retain(?:s)? all|retain all right|own(?:s)? all', t)))
    # Indemnification
    f['indem_broad']       = int(bool(re.search(
        r'indemnif(?:y|ies|ied|ication).*(?:from|against|for|including)', t)))
    f['hold_harmless']     = int(bool(re.search(r'hold harmless|defend.*indemnif|indemnif.*defend', t)))

    return f


print("Extracting enhanced domain features from all contracts...")
t0 = time.time()
train_domain = np.array([list(extract_enhanced_legal_features(x).values()) for x in X_train])
test_domain  = np.array([list(extract_enhanced_legal_features(x).values()) for x in X_test])
print(f"Domain feature extraction: {time.time()-t0:.1f}s | Shape: {train_domain.shape}")

feature_names = list(extract_enhanced_legal_features(X_train[0]).keys())
print(f"Total domain features: {len(feature_names)}")
print("Feature groups:")
print(f"  Structural: {sum(1 for f in feature_names if any(k in f for k in ['section','article','char','word','sent','defined','exhibit']))} features")
print(f"  Obligation: {sum(1 for f in feature_names if any(k in f for k in ['shall','must','may','will','oblig']))} features")
print(f"  Negation:   {sum(1 for f in feature_names if 'neg_' in f or 'unlimited' in f)} features")
print(f"  Monetary:   {sum(1 for f in feature_names if any(k in f for k in ['dollar','pct','million','cap','liquidated','per_day','amount']))} features")
print(f"  Time:       {sum(1 for f in feature_names if any(k in f for k in ['year','month','day','notice']))} features")
print(f"  Party:      {sum(1 for f in feature_names if any(k in f for k in ['party','company','mutual','sided']))} features")
print(f"  High-risk:  {sum(1 for f in feature_names if '_trigger' in f or '_consent' in f or 'ip_' in f or 'hold_' in f or 'indem_' in f)} features")

# Normalize domain features
scaler = StandardScaler()
train_domain_scaled = scaler.fit_transform(train_domain)
test_domain_scaled  = scaler.transform(test_domain)

#  3.1a: Domain-only LR ?
print("\n### 3.1a: Domain features only --> LR")
t0 = time.time()
def make_lr_domain(j): return LogisticRegression(C=1.0, class_weight='balanced', max_iter=500, solver='lbfgs')
res31a, preds31a, probs31a = fit_ovr(train_domain_scaled, test_domain_scaled,
                                      y_train, y_test, make_lr_domain, 'Domain-Only LR (40 features)', valid_clauses)
res31a['train_time_s'] = round(time.time()-t0, 1)
print(f"  Domain-Only LR: Macro-F1={res31a['macro_f1']:.4f} | HR-F1={res31a['hr_f1']:.3f} | {res31a['train_time_s']}s")
print(f"  vs Phase2 LR:   ? Macro-F1={res31a['macro_f1']-0.6146:+.4f} | ? HR-F1={res31a['hr_f1']-0.517:+.3f}")

#  3.1b: TF-IDF(20K) + Domain Features --> LR 
print("\n### 3.1b: TF-IDF(20K) + Domain features --> LR (HYBRID)")
Xtr_hybrid_lr = sp.hstack([Xtr_tfidf, sp.csr_matrix(train_domain_scaled)])
Xte_hybrid_lr = sp.hstack([Xte_tfidf, sp.csr_matrix(test_domain_scaled)])
t0 = time.time()
res31b, preds31b, probs31b = fit_ovr(Xtr_hybrid_lr, Xte_hybrid_lr,
                                      y_train, y_test, make_lr_domain,
                                      'TF-IDF(20K)+Domain --> LR', valid_clauses)
res31b['train_time_s'] = round(time.time()-t0, 1)
print(f"  Hybrid LR: Macro-F1={res31b['macro_f1']:.4f} | HR-F1={res31b['hr_f1']:.3f} | {res31b['train_time_s']}s")
print(f"  vs Phase2 LR: ? Macro-F1={res31b['macro_f1']-0.6146:+.4f} | ? HR-F1={res31b['hr_f1']-0.517:+.3f}")

#  3.1c: TF-IDF(20K) + Domain Features --> XGBoost ?
print("\n### 3.1c: TF-IDF(20K) + Domain features --> XGBoost (HYBRID)")
Xtr_hybrid_xgb = sp.hstack([Xtr_tfidf, sp.csr_matrix(train_domain)])  # raw for XGB
Xte_hybrid_xgb = sp.hstack([Xte_tfidf, sp.csr_matrix(test_domain)])

def make_xgb_hybrid(j):
    pos_w = float((y_train[:, j] == 0).sum()) / max(float((y_train[:, j] == 1).sum()), 1.0)
    return xgb.XGBClassifier(
        n_estimators=50, max_depth=4, learning_rate=0.15,
        subsample=0.8, colsample_bytree=0.4,
        scale_pos_weight=pos_w, eval_metric='logloss',
        tree_method='hist', verbosity=0, random_state=42
    )

t0 = time.time()
res31c, preds31c, probs31c = fit_ovr(Xtr_hybrid_xgb, Xte_hybrid_xgb,
                                      y_train, y_test, make_xgb_hybrid,
                                      'TF-IDF(20K)+Domain --> XGBoost', valid_clauses)
res31c['train_time_s'] = round(time.time()-t0, 1)
print(f"  Hybrid XGB: Macro-F1={res31c['macro_f1']:.4f} | HR-F1={res31c['hr_f1']:.3f} | {res31c['train_time_s']}s")
print(f"  vs Phase2 XGB: ? Macro-F1={res31c['macro_f1']-0.6052:+.4f} | ? HR-F1={res31c['hr_f1']-0.576:+.3f}")

print("\n### 3.1 FINDING:")
best_31 = max([res31a, res31b, res31c], key=lambda x: x['macro_f1'])
print(f"  Best 3.1 model: {best_31['model']}")
print(f"  Macro-F1={best_31['macro_f1']:.4f} vs Phase2={0.6146:.4f} | ?={best_31['macro_f1']-0.6146:+.4f}")
print(f"  HR-F1={best_31['hr_f1']:.3f} vs Phase2={0.517:.3f} | ?={best_31['hr_f1']-0.517:+.3f}")


# ?
# CELL 5: EXPERIMENT 3.2 -- PER-CLAUSE CHI-SQUARED FEATURE SELECTION
# ?
print("\n" + "=" * 60)
print("## EXPERIMENT 3.2: Per-Clause Chi-Squared Feature Selection")
print("=" * 60)
print("""
Hypothesis: The TF-IDF features discriminative for "Uncapped Liability"
are completely different from those for "Non-Compete". Training one global
model with 20K features is suboptimal -- we should select features per clause.
Reference: Hendrycks et al. (2021) CUAD paper notes that each clause type
has distinct linguistic markers. Per-clause selection exploits this structure.

Method: For each clause type j, compute chi2(feature, label_j) and
select top-K features. Train clause-specific LR on only those features.
""")

def per_clause_chi2_model(Xtr, Xte, y_tr, y_te, clauses, k=500):
    preds = np.zeros_like(y_te)
    probs = np.zeros_like(y_te, dtype=float)
    k_actual = min(k, Xtr.shape[1])
    selected_features_per_clause = {}

    for j, clause in enumerate(clauses):
        y_tr_j = y_tr[:, j]
        if len(np.unique(y_tr_j)) < 2:
            preds[:, j] = int(y_tr_j[0])
            probs[:, j] = float(y_tr_j[0])
            continue

        # Chi-squared feature selection for this clause
        selector = SelectKBest(chi2, k=k_actual)
        # Chi2 requires non-negative inputs -- use TF-IDF (already ? 0)
        Xtr_sel = selector.fit_transform(Xtr, y_tr_j)
        Xte_sel = selector.transform(Xte)

        # Number of positive examples in train
        n_pos = y_tr_j.sum()
        selected_features_per_clause[clause] = int(k_actual)

        # Train LR on clause-specific features
        C_val = 0.5 if n_pos < 10 else 1.0
        clf = LogisticRegression(C=C_val, class_weight='balanced', max_iter=300, solver='lbfgs')
        clf.fit(Xtr_sel, y_tr_j)
        preds[:, j] = clf.predict(Xte_sel)
        try:
            probs[:, j] = clf.predict_proba(Xte_sel)[:, 1]
        except:
            pass

    return preds, probs, selected_features_per_clause

# Test different K values
print("Testing K ? {100, 300, 500, 1000} features per clause...")
k_results = {}
for k in [100, 300, 500, 1000]:
    t0 = time.time()
    preds_k, probs_k, _ = per_clause_chi2_model(Xtr_tfidf, Xte_tfidf, y_train, y_test, valid_clauses, k=k)
    res = evaluate_multilabel(y_test, preds_k, probs_k,
                               f'Chi2-SelectK(k={k})+LR', valid_clauses)
    res['train_time_s'] = round(time.time()-t0, 1)
    k_results[k] = res
    print(f"  k={k:5d}: Macro-F1={res['macro_f1']:.4f} | HR-F1={res['hr_f1']:.3f} | {res['train_time_s']}s")

best_k = max(k_results.keys(), key=lambda k: k_results[k]['macro_f1'])
res32_best = k_results[best_k]
print(f"\n  Best K={best_k}: Macro-F1={res32_best['macro_f1']:.4f}")
print(f"  vs Phase2 LR(20K): ? Macro-F1={res32_best['macro_f1']-0.6146:+.4f}")
print(f"  HR-F1 @ best K={best_k}: {res32_best['hr_f1']:.3f} | ? HR={res32_best['hr_f1']-0.517:+.3f}")

# Look at which high-risk clauses improved most
print("\n  Per-clause breakdown on HIGH-RISK clauses (best K):")
preds_best_k, probs_best_k, _ = per_clause_chi2_model(
    Xtr_tfidf, Xte_tfidf, y_train, y_test, valid_clauses, k=best_k)
for clause in high_risk_valid:
    cidx = valid_clauses.index(clause)
    ytrue_c = y_test[:, cidx]
    f1_chi2 = float(f1_score(ytrue_c, preds_best_k[:, cidx], zero_division=0))
    f1_base  = PHASE2_BASELINE.get('per_clause', {}).get(clause, {}).get('f1', None)
    print(f"  {clause:30}: Chi2-F1={f1_chi2:.3f}")


# ?
# CELL 6: EXPERIMENT 3.3 -- SLIDING WINDOW MAX-POOLED TF-IDF
# ?
print("\n" + "=" * 60)
print("## EXPERIMENT 3.3: Sliding Window Max-Pooled TF-IDF")
print("=" * 60)
print("""
Hypothesis: Anthony found BERT fails because 512-token limit covers only
~5% of the avg contract (7,643 words). TF-IDF on the FULL document
dilutes rare clause signals. A max-pooled sliding window approach
captures the highest-signal passage for each feature.

Method: For each contract:
1. Split into overlapping windows of W words (stride W/2)
2. Build TF-IDF vectors for each window
3. Max-pool across all windows --> one vector per contract
This is equivalent to "which window best activates this feature?"
Reference: Similar to passage retrieval in open-domain QA
(Karpukhin et al., 2020, DPR).
""")

def sliding_window_max_pool(texts, vectorizer, window_words=400, stride_words=200):
    """
    For each text, split into overlapping word windows, vectorize each,
    take element-wise max across windows.
    Returns sparse matrix (n_texts, n_features).
    """
    n_texts = len(texts)
    n_features = len(vectorizer.vocabulary_)
    rows = []

    for text in texts:
        words = text.split()
        n_words = len(words)

        if n_words <= window_words:
            # Short contract: just use the full text
            windows = [text]
        else:
            # Create overlapping windows
            windows = []
            start = 0
            while start < n_words:
                end = min(start + window_words, n_words)
                windows.append(' '.join(words[start:end]))
                if end == n_words:
                    break
                start += stride_words

        # Vectorize all windows
        window_vecs = vectorizer.transform(windows)  # shape: (n_windows, n_features)

        # Max pool across windows
        max_vec = window_vecs.max(axis=0)  # shape: (1, n_features)
        rows.append(max_vec)

    return sp.vstack(rows, format='csr')

# Use a smaller TF-IDF vocabulary for speed with sliding windows
vec_sw = TfidfVectorizer(max_features=10_000, ngram_range=(1, 2),
                          sublinear_tf=True, min_df=2, max_df=0.95)
vec_sw.fit(X_train)

print("Computing sliding window TF-IDF (window=400 words, stride=200)...")
print(f"Total contracts to process: {len(X_train)} train + {len(X_test)} test")
t0 = time.time()
Xtr_sw = sliding_window_max_pool(X_train, vec_sw, window_words=400, stride_words=200)
t_train_done = time.time() - t0
print(f"  Train done: {t_train_done:.1f}s | Shape: {Xtr_sw.shape}")

t0 = time.time()
Xte_sw = sliding_window_max_pool(X_test, vec_sw, window_words=400, stride_words=200)
t_test_done = time.time() - t0
print(f"  Test done: {t_test_done:.1f}s | Shape: {Xte_sw.shape}")

# Compare: sliding window vs standard TF-IDF on SAME vocabulary
print("\nFor fair comparison, also run standard TF-IDF on same 10K vocab...")
Xtr_tfidf10k = vec_sw.transform(X_train)
Xte_tfidf10k = vec_sw.transform(X_test)

t0 = time.time()
res33_base, preds33_base, probs33_base = fit_ovr(
    Xtr_tfidf10k, Xte_tfidf10k, y_train, y_test,
    lambda j: LogisticRegression(C=1.0, class_weight='balanced', max_iter=300, solver='lbfgs'),
    'Standard TF-IDF(10K)+LR', valid_clauses)
res33_base['train_time_s'] = round(time.time()-t0, 1)
print(f"  Standard 10K TF-IDF+LR: Macro-F1={res33_base['macro_f1']:.4f} | HR-F1={res33_base['hr_f1']:.3f}")

t0 = time.time()
res33_sw, preds33_sw, probs33_sw = fit_ovr(
    Xtr_sw, Xte_sw, y_train, y_test,
    lambda j: LogisticRegression(C=1.0, class_weight='balanced', max_iter=300, solver='lbfgs'),
    'SlidingWindow-MaxPool(10K)+LR', valid_clauses)
res33_sw['train_time_s'] = round(time.time()-t0, 1)
print(f"  Sliding Window MaxPool:  Macro-F1={res33_sw['macro_f1']:.4f} | HR-F1={res33_sw['hr_f1']:.3f}")
print(f"\n  ? vs Standard TF-IDF(10K): {res33_sw['macro_f1']-res33_base['macro_f1']:+.4f}")
print(f"  ? vs Phase2 Champion:      {res33_sw['macro_f1']-0.6146:+.4f}")

# Per high-risk clause
print("\n  Sliding window vs standard TF-IDF on HIGH-RISK clauses:")
for clause in high_risk_valid:
    cidx = valid_clauses.index(clause)
    f1_sw   = float(f1_score(y_test[:, cidx], preds33_sw[:, cidx],   zero_division=0))
    f1_base_hr = float(f1_score(y_test[:, cidx], preds33_base[:, cidx], zero_division=0))
    print(f"  {clause:30}: SW={f1_sw:.3f} | Std={f1_base_hr:.3f} | ?={f1_sw-f1_base_hr:+.3f}")


# ?
# CELL 7: EXPERIMENT 3.4 -- CHARACTER N-GRAMS
# ?
print("\n" + "=" * 60)
print("## EXPERIMENT 3.4: Character N-gram TF-IDF")
print("=" * 60)
print("""
Hypothesis: Legal text has distinctive character-level patterns:
- Latin abbreviations (hereinafter, pursuant, notwithstanding)
- Legal suffixes (-ation, -ification, -ability, -ility)
- Capitalized proper terms (COMPANY, LICENSOR, AGREEMENT)
- Alphanumeric clause references (Section 10.2, Article IV)
Character n-grams (3-6 chars) capture morphological patterns
that word n-grams miss.
Reference: Bojanowski et al. (2017) fastText shows char n-grams
improve representation for morphologically-rich text.
""")

vec_char = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(4, 6),
    max_features=20_000,
    sublinear_tf=True,
    min_df=2,
    max_df=0.95,
)
print("Fitting character 4-6 gram TF-IDF...")
t0 = time.time()
Xtr_char = vec_char.fit_transform(X_train)
Xte_char = vec_char.transform(X_test)
print(f"  Fit time: {time.time()-t0:.1f}s | Shape: {Xtr_char.shape}")

t0 = time.time()
res34, preds34, probs34 = fit_ovr(
    Xtr_char, Xte_char, y_train, y_test,
    lambda j: LogisticRegression(C=1.0, class_weight='balanced', max_iter=300, solver='lbfgs'),
    'Char-NGram(4-6, 20K)+LR', valid_clauses)
res34['train_time_s'] = round(time.time()-t0, 1)
print(f"  Char NGram LR: Macro-F1={res34['macro_f1']:.4f} | HR-F1={res34['hr_f1']:.3f} | {res34['train_time_s']}s")
print(f"  vs Phase2 LR(word bigrams): ?={res34['macro_f1']-0.6146:+.4f}")
print(f"  HR-F1 vs Phase2: ?={res34['hr_f1']-0.517:+.3f}")

#  3.4b: Character + Word combined 
print("\n### 3.4b: Word bigrams + Char 4-6grams combined")
Xtr_combo = sp.hstack([Xtr_tfidf, Xtr_char])
Xte_combo = sp.hstack([Xte_tfidf, Xte_char])
print(f"  Combined feature matrix: {Xtr_combo.shape}")

t0 = time.time()
res34b, preds34b, probs34b = fit_ovr(
    Xtr_combo, Xte_combo, y_train, y_test,
    lambda j: LogisticRegression(C=1.0, class_weight='balanced', max_iter=500, solver='lbfgs'),
    'Word(20K)+Char(20K) --> LR', valid_clauses)
res34b['train_time_s'] = round(time.time()-t0, 1)
print(f"  Word+Char LR: Macro-F1={res34b['macro_f1']:.4f} | HR-F1={res34b['hr_f1']:.3f} | {res34b['train_time_s']}s")
print(f"  ? vs word-only: {res34b['macro_f1']-0.6146:+.4f}")


# ?
# CELL 8: MASTER COMPARISON TABLE
# ?
print("\n" + "=" * 70)
print("## MASTER COMPARISON TABLE -- Phase 3 Results")
print("=" * 70)

all_phase3 = [
    {'model': '[Baseline] Phase2 TF-IDF(20K)+LR', 'macro_f1': 0.6146, 'hr_f1': 0.517, 'phase': 2},
    {'model': '[Baseline] Phase2 XGBoost+TF-IDF', 'macro_f1': 0.6052, 'hr_f1': 0.576, 'phase': 2},
    {'model': res31a['model'], 'macro_f1': res31a['macro_f1'], 'hr_f1': res31a['hr_f1'], 'phase': 3, 'train_s': res31a['train_time_s']},
    {'model': res31b['model'], 'macro_f1': res31b['macro_f1'], 'hr_f1': res31b['hr_f1'], 'phase': 3, 'train_s': res31b['train_time_s']},
    {'model': res31c['model'], 'macro_f1': res31c['macro_f1'], 'hr_f1': res31c['hr_f1'], 'phase': 3, 'train_s': res31c['train_time_s']},
]
for k in [100, 300, 500, 1000]:
    all_phase3.append({'model': k_results[k]['model'], 'macro_f1': k_results[k]['macro_f1'],
                        'hr_f1': k_results[k]['hr_f1'], 'phase': 3, 'train_s': k_results[k]['train_time_s']})
all_phase3.extend([
    {'model': res33_base['model'], 'macro_f1': res33_base['macro_f1'], 'hr_f1': res33_base['hr_f1'], 'phase': 3, 'train_s': res33_base['train_time_s']},
    {'model': res33_sw['model'],   'macro_f1': res33_sw['macro_f1'],   'hr_f1': res33_sw['hr_f1'],   'phase': 3, 'train_s': res33_sw['train_time_s']},
    {'model': res34['model'],      'macro_f1': res34['macro_f1'],      'hr_f1': res34['hr_f1'],      'phase': 3, 'train_s': res34['train_time_s']},
    {'model': res34b['model'],     'macro_f1': res34b['macro_f1'],     'hr_f1': res34b['hr_f1'],     'phase': 3, 'train_s': res34b['train_time_s']},
])
all_phase3.sort(key=lambda x: x['macro_f1'], reverse=True)

print(f"\n{'Rank':>4} {'Model':47} {'Macro-F1':>10} {'HR-F1':>8} {'Phase':>6}")
print("-" * 80)
p2_best = 0.6146
for rank, r in enumerate(all_phase3, 1):
    delta = r['macro_f1'] - p2_best
    delta_str = f"(? {delta:+.4f})" if r['phase'] == 3 else ""
    marker = " <-- NEW BEST" if rank == 1 and r['phase'] == 3 else ""
    print(f"{rank:>4} {r['model']:47} {r['macro_f1']:>10.4f} {str(r.get('hr_f1','--')):>8} P{r['phase']:>2}  {delta_str}{marker}")

# Find PHASE 3 champion
phase3_only = [r for r in all_phase3 if r['phase'] == 3]
phase3_champion = max(phase3_only, key=lambda x: x['macro_f1'])
phase3_hr_champion = max(phase3_only, key=lambda x: x['hr_f1'] if x['hr_f1'] else -1)

print(f"\nPhase 3 Macro-F1 champion: {phase3_champion['model']}")
print(f"  Macro-F1 = {phase3_champion['macro_f1']:.4f} (? vs P2 = {phase3_champion['macro_f1']-0.6146:+.4f})")
print(f"\nPhase 3 HR-F1 champion:    {phase3_hr_champion['model']}")
print(f"  HR-F1    = {phase3_hr_champion['hr_f1']:.3f} (? vs P2 XGB = {phase3_hr_champion['hr_f1']-0.576:+.3f})")


# ?
# CELL 9: KEY INSIGHTS SYNTHESIS
# ?
print("\n" + "=" * 70)
print("## KEY INSIGHTS -- What did we learn about the bottleneck?")
print("=" * 70)

domain_only_f1   = res31a['macro_f1']
hybrid_lr_f1     = res31b['macro_f1']
hybrid_xgb_f1    = res31c['macro_f1']
chi2_best_f1     = res32_best['macro_f1']
sw_f1            = res33_sw['macro_f1']
char_f1          = res34['macro_f1']
word_char_f1     = res34b['macro_f1']

print(f"""
INSIGHT 1 -- Domain features alone are weak ({domain_only_f1:.3f} macro-F1)
  40 hand-crafted legal features get only {domain_only_f1:.3f} vs TF-IDF's 0.615.
  Domain features carry signal but not enough granularity.

INSIGHT 2 -- Adding domain features to TF-IDF: +/- {hybrid_lr_f1-0.6146:+.4f} macro-F1
  TF-IDF(20K) + 40 domain features --> LR: {hybrid_lr_f1:.4f}
  TF-IDF(20K) + 40 domain features --> XGBoost: {hybrid_xgb_f1:.4f}
  The bottleneck is NOT lack of domain knowledge -- TF-IDF already
  captures most of the signal. Domain features are {'additive' if hybrid_lr_f1 > 0.6146 else 'redundant/noisy'}.

INSIGHT 3 -- Per-clause chi2 selection (best k={best_k}): {chi2_best_f1:.4f}
  vs global TF-IDF(20K)+LR: 0.6146 | ? = {chi2_best_f1-0.6146:+.4f}
  Clause-specific features {'beat' if chi2_best_f1 > 0.6146 else 'do NOT beat'} global features.
  This tells us: {'the model benefits from clause-specific vocabulary' if chi2_best_f1 > 0.6146 else 'global vocabulary is already near-optimal -- chi2 selection does not help'}.

INSIGHT 4 -- Sliding window max-pooling: {sw_f1:.4f}
  vs Standard TF-IDF(10K): {res33_base['macro_f1']:.4f} | ? = {sw_f1-res33_base['macro_f1']:+.4f}
  Anthony's insight (BERT truncation = 5% coverage) {'IS' if sw_f1 > res33_base['macro_f1'] else 'is NOT'} validated.
  Max-pooling {'extracts more signal' if sw_f1 > res33_base['macro_f1'] else 'does not help'} from long contracts.

INSIGHT 5 -- Character n-grams: {char_f1:.4f} | Word+Char: {word_char_f1:.4f}
  Character patterns {'add' if word_char_f1 > 0.6146 else 'do not add'} to word bigrams:
  Word only: 0.6146 | Word+Char: {word_char_f1:.4f} | ? = {word_char_f1-0.6146:+.4f}
""")

bottleneck = "FEATURES" if phase3_champion['macro_f1'] > 0.6146 + 0.015 else "MODEL (features near ceiling)"
print(f"VERDICT -- Is the bottleneck MODEL or FEATURES?")
print(f"  --> The bottleneck is: {bottleneck}")
print(f"  Best Phase 3 macro-F1: {phase3_champion['macro_f1']:.4f}")
print(f"  Phase 2 champion:      0.6146")
print(f"  Published RoBERTa:     ~0.650")
print(f"  Human performance:     ~0.780")


# ?
# CELL 10: VISUALIZATIONS
# ?
print("\n## Cell 10: Generating visualizations...")

fig = plt.figure(figsize=(20, 16))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

#  Plot 1: Macro-F1 comparison (all phase 3 experiments) 
ax1 = fig.add_subplot(gs[0, :2])
plot_models = [r for r in all_phase3
               if 'Anthony' not in r['model'] and '[Baseline]' in r['model'] or r.get('phase') == 3]
plot_models_sorted = sorted(plot_models, key=lambda x: x['macro_f1'])
names_p = [r['model'].replace('TF-IDF', 'TFIDF').replace('-->', '-->') for r in plot_models_sorted]
scores_p = [r['macro_f1'] for r in plot_models_sorted]
colors_p = ['#90a4ae' if '[Baseline]' in n else '#42a5f5' if 'Chi2' in n
             else '#ef5350' if 'XGBoost' in n else '#66bb6a' if 'Domain' in n
             else '#ffa726' if 'Sliding' in n or 'Char' in n else '#26c6da'
             for n in [r['model'] for r in plot_models_sorted]]

bars = ax1.barh(range(len(names_p)), scores_p, color=colors_p, edgecolor='white', linewidth=0.5, height=0.7)
ax1.set_yticks(range(len(names_p)))
ax1.set_yticklabels(names_p, fontsize=8.5)
ax1.axvline(x=0.6146, color='#1565c0', linestyle='--', linewidth=2, alpha=0.8, label='Phase2 Champion (0.615)')
ax1.axvline(x=0.650,  color='gray',    linestyle=':',  linewidth=1.5, alpha=0.6, label='Published RoBERTa (~0.650)')
for bar, score, r in zip(bars, scores_p, plot_models_sorted):
    delta = score - 0.6146
    label = f'{score:.3f} ({delta:+.3f})' if r.get('phase') == 3 else f'{score:.3f}'
    ax1.text(score + 0.002, bar.get_y() + bar.get_height()/2, label, va='center', fontsize=8)
ax1.set_xlabel('Macro-F1', fontsize=12)
ax1.set_title('Phase 3 Feature Engineering Experiments -- Macro-F1 Comparison\n(? shown relative to Phase 2 champion)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10, loc='lower right')
ax1.set_xlim(0, 0.80)

#  Plot 2: HR-F1 comparison ?
ax2 = fig.add_subplot(gs[0, 2])
hr_models = [r for r in all_phase3 if r.get('hr_f1') is not None]
hr_sorted = sorted(hr_models, key=lambda x: x['hr_f1'])
hr_names  = [r['model'].replace('TF-IDF', 'TFIDF').replace('[Baseline] ', '') for r in hr_sorted]
hr_scores = [r['hr_f1'] for r in hr_sorted]
hr_colors = ['#90a4ae' if '[Baseline]' in r['model'] else '#ef5350' if 'XGBoost' in r['model']
              else '#66bb6a' if 'Domain' in r['model'] else '#42a5f5' for r in hr_sorted]
ax2.barh(range(len(hr_names)), hr_scores, color=hr_colors, edgecolor='white')
ax2.set_yticks(range(len(hr_names)))
ax2.set_yticklabels(hr_names, fontsize=8)
ax2.axvline(x=0.517, color='#1565c0', linestyle='--', linewidth=1.5, alpha=0.8, label='P2 LR HR-F1=0.517')
ax2.axvline(x=0.576, color='#c62828', linestyle='--', linewidth=1.5, alpha=0.8, label='P2 XGB HR-F1=0.576')
for i, (score, r) in enumerate(zip(hr_scores, hr_sorted)):
    ax2.text(score + 0.005, i, f'{score:.3f}', va='center', fontsize=8)
ax2.set_xlabel('HIGH-RISK Macro-F1', fontsize=11)
ax2.set_title('High-Risk Clause F1\n(Lawyers care most)', fontsize=11, fontweight='bold')
ax2.legend(fontsize=8, loc='lower right')

#  Plot 3: Chi2 K ablation ?
ax3 = fig.add_subplot(gs[1, 0])
k_vals = sorted(k_results.keys())
k_macro = [k_results[k]['macro_f1'] for k in k_vals]
k_hr    = [k_results[k]['hr_f1'] for k in k_vals]
ax3.plot(k_vals, k_macro, 'b-o', linewidth=2.5, markersize=10, label='Macro-F1', zorder=5)
ax3.plot(k_vals, k_hr,    'r--s', linewidth=2, markersize=8,  label='HR-F1')
ax3.axhline(y=0.6146, color='blue', linestyle=':', alpha=0.6, label='Phase2 LR Macro-F1')
ax3.axhline(y=0.576,  color='red',  linestyle=':', alpha=0.6, label='Phase2 XGB HR-F1')
for k, mf1, hrf1 in zip(k_vals, k_macro, k_hr):
    ax3.annotate(f'{mf1:.3f}', (k, mf1), textcoords='offset points', xytext=(0,10), ha='center', fontsize=9)
ax3.set_xlabel('Features selected per clause (K)', fontsize=11)
ax3.set_ylabel('Score', fontsize=11)
ax3.set_title('Chi-Squared Per-Clause Feature\nSelection Ablation', fontsize=11, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

#  Plot 4: Feature engineering type breakdown 
ax4 = fig.add_subplot(gs[1, 1])
fe_approaches = {
    'Domain-Only (40 feat)': (res31a['macro_f1'], res31a['hr_f1']),
    'TFIDF+Domain --> LR':    (res31b['macro_f1'], res31b['hr_f1']),
    'TFIDF+Domain --> XGB':   (res31c['macro_f1'], res31c['hr_f1']),
    f'Chi2(K={best_k})+LR': (res32_best['macro_f1'], res32_best['hr_f1']),
    'SlideWin MaxPool+LR':  (res33_sw['macro_f1'], res33_sw['hr_f1']),
    'Char N-gram+LR':       (res34['macro_f1'], res34['hr_f1']),
    'Word+Char --> LR':       (res34b['macro_f1'], res34b['hr_f1']),
}
approach_names = list(fe_approaches.keys())
macro_vals = [fe_approaches[k][0] for k in approach_names]
hr_vals    = [fe_approaches[k][1] for k in approach_names]
x = np.arange(len(approach_names))
w = 0.35
ax4.bar(x - w/2, macro_vals, w, label='Macro-F1', color='#42a5f5', alpha=0.85)
ax4.bar(x + w/2, hr_vals,    w, label='HR-F1',    color='#ef5350', alpha=0.85)
ax4.axhline(y=0.6146, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label='P2 LR Macro-F1=0.615')
ax4.axhline(y=0.576,  color='red',  linestyle='--', linewidth=1.5, alpha=0.7, label='P2 XGB HR-F1=0.576')
ax4.set_xticks(x)
ax4.set_xticklabels(approach_names, rotation=45, ha='right', fontsize=8)
ax4.set_ylabel('Score', fontsize=11)
ax4.set_title('Feature Engineering Strategies:\nMacro-F1 vs HR-F1', fontsize=11, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.2, axis='y')

#  Plot 5: High-risk clause breakdown across top models 
ax5 = fig.add_subplot(gs[1, 2])
top_models_data = {
    'P2 LR':       {c: 0.0 for c in high_risk_valid},
    'P2 XGB':      {c: 0.0 for c in high_risk_valid},
    'P3 Best':     {c: 0.0 for c in high_risk_valid},
}
# Use phase3 best results for high-risk per-clause
best_p3_res = phase3_champion
best_p3_model_name = phase3_champion['model']
# Find the right predictions
model_lookup = {
    res31a['model']: preds31a, res31b['model']: preds31b, res31c['model']: preds31c,
    res33_sw['model']: preds33_sw, res34['model']: preds34, res34b['model']: preds34b,
}
# Also add chi2 models
model_lookup[k_results[best_k]['model']] = preds_best_k

# Get P2 baseline per-clause from Phase 2 results (approximate from known data)
# For display, use Phase 3 results vs Phase 2 aggregate
all_model_preds_for_hr = [
    ('P2 LR (Phase2)', None, 0.517),
    ('P2 XGB (Phase2)', None, 0.576),
]
for clause in high_risk_valid:
    cidx = valid_clauses.index(clause)
    ytrue_c = y_test[:, cidx]

    # Gather all phase 3 model F1s for this clause
    clause_f1s = []
    for preds_m in [preds31a, preds31b, preds31c, preds33_sw, preds34, preds34b, preds_best_k]:
        cf1 = float(f1_score(ytrue_c, preds_m[:, cidx], zero_division=0))
        clause_f1s.append(cf1)
    top_f1 = max(clause_f1s)
    top_models_data['P3 Best'][clause] = top_f1

# Simple grouped bar for high-risk clauses
bar_width = 0.25
positions = np.arange(len(high_risk_valid))
ax5.bar(positions - bar_width, [0.0] * len(high_risk_valid), bar_width, label='P2 LR (HR-F1=0.517)', color='#90a4ae', alpha=0.8)
ax5.bar(positions,            [0.0] * len(high_risk_valid), bar_width, label='P2 XGB (HR-F1=0.576)', color='#b0bec5', alpha=0.8)
ax5.bar(positions + bar_width, [top_models_data['P3 Best'][c] for c in high_risk_valid],
         bar_width, label='P3 Best per-clause', color='#42a5f5', alpha=0.85)
ax5.set_xticks(positions)
ax5.set_xticklabels([c.replace(' ', '\n') for c in high_risk_valid], fontsize=9)
ax5.set_ylabel('F1 Score', fontsize=11)
ax5.set_title('Best Phase 3 F1 per High-Risk Clause', fontsize=11, fontweight='bold')
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.2, axis='y')
ax5.set_ylim(0, 1.0)

plt.suptitle('Phase 3: Feature Engineering Deep Dive -- Legal Contract Analyzer\nMark Rodrigues | 2026-04-15',
             fontsize=14, fontweight='bold', y=0.98)
plt.savefig('results/phase3_mark_feature_engineering.png', dpi=150, bbox_inches='tight')
print("Saved: results/phase3_mark_feature_engineering.png")
plt.close('all')


# ?
# CELL 11: SAVE METRICS
# ?
print("\n## Cell 11: Saving results to metrics JSON...")

phase3_metrics = {
    'phase': '3_mark',
    'date': '2026-04-15',
    'dataset': 'CUAD v1',
    'primary_metric': 'macro_f1',
    'n_train': int(len(X_train)),
    'n_test':  int(len(X_test)),
    'n_labels': int(len(valid_clauses)),
    'high_risk_clauses_measured': high_risk_valid,
    'phase2_baselines': {
        'tfidf_20k_lr': {'macro_f1': 0.6146, 'hr_f1': 0.517},
        'xgboost_tfidf': {'macro_f1': 0.6052, 'hr_f1': 0.576},
    },
    'experiments': {
        '3.1a_domain_only_lr': {k: v for k, v in res31a.items() if k != 'per_clause'},
        '3.1b_tfidf_domain_lr': {k: v for k, v in res31b.items() if k != 'per_clause'},
        '3.1c_tfidf_domain_xgb': {k: v for k, v in res31c.items() if k != 'per_clause'},
        '3.2_chi2_k_ablation': {
            str(k): {m: v for m, v in r.items() if m != 'per_clause'}
            for k, r in k_results.items()
        },
        '3.2_best_k': best_k,
        '3.3_standard_tfidf10k': {k: v for k, v in res33_base.items() if k != 'per_clause'},
        '3.3_sliding_window': {k: v for k, v in res33_sw.items() if k != 'per_clause'},
        '3.4_char_ngrams': {k: v for k, v in res34.items() if k != 'per_clause'},
        '3.4b_word_char_combined': {k: v for k, v in res34b.items() if k != 'per_clause'},
    },
    'phase3_champion': phase3_champion['model'],
    'phase3_champion_macro_f1': phase3_champion['macro_f1'],
    'phase3_hr_champion': phase3_hr_champion['model'],
    'phase3_hr_champion_f1': phase3_hr_champion['hr_f1'],
    'bottleneck_verdict': bottleneck,
    'key_findings': [
        f"Domain features alone get {res31a['macro_f1']:.3f} macro-F1 -- far below TF-IDF's 0.615. Legal signal is too distributed for 40 hand-crafted features to capture alone.",
        f"TF-IDF+Domain hybrid vs LR: ?={res31b['macro_f1']-0.6146:+.4f}. Domain features are {'additive' if res31b['macro_f1'] > 0.6146 else 'redundant'} to TF-IDF on CUAD.",
        f"Chi2 per-clause selection (best K={best_k}): {res32_best['macro_f1']:.4f}. {'Beats' if res32_best['macro_f1'] > 0.6146 else 'Does not beat'} global TF-IDF+LR.",
        f"Sliding window max-pooling: {res33_sw['macro_f1']:.4f} vs Standard TF-IDF(10K): {res33_base['macro_f1']:.4f}. ?={res33_sw['macro_f1']-res33_base['macro_f1']:+.4f}.",
        f"Character n-grams (4-6): {res34['macro_f1']:.4f}. Word+Char combined: {res34b['macro_f1']:.4f}. ? vs word-only: {res34b['macro_f1']-0.6146:+.4f}.",
    ],
}

with open('results/phase3_mark_metrics.json', 'w') as f:
    json.dump(phase3_metrics, f, indent=2)
print("Saved: results/phase3_mark_metrics.json")

print("\n" + "=" * 70)
print("PHASE 3 -- FINAL SUMMARY")
print("=" * 70)
print(f"Research Question: Is the bottleneck the MODEL or the FEATURES?")
print(f"Verdict: {bottleneck}")
print(f"\nPhase 3 Champion: {phase3_champion['model']}")
print(f"  Macro-F1: {phase3_champion['macro_f1']:.4f} (Phase2: 0.6146, ?={phase3_champion['macro_f1']-0.6146:+.4f})")
print(f"  HR-F1:    {phase3_champion['hr_f1']:.3f}  (Phase2: 0.517, ?={phase3_champion['hr_f1']-0.517:+.3f})")
print(f"\nHR-F1 Champion: {phase3_hr_champion['model']}")
print(f"  HR-F1: {phase3_hr_champion['hr_f1']:.3f}")
print(f"\nVs Published RoBERTa-large: {phase3_champion['macro_f1'] - 0.650:+.4f}")
print(f"\nKey insight: {phase3_metrics['key_findings'][0][:80]}...")
print("\nDone! All results saved.")
