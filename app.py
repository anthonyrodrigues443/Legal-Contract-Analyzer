"""
Legal Contract Analyzer — Streamlit UI
Phase 6 Production App | Mark Rodrigues | 2026-04-18

Features:
- Paste or upload contract text
- Per-clause risk detection with color-coded scores
- Feature explanation (which terms drove each prediction)
- Overall risk score + "clauses missing" warning
- Side-by-side model vs Claude comparison reminder

Run:
  streamlit run app.py
"""

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Streamlit import — fail gracefully if not installed
try:
    import streamlit as st
except ImportError:
    print("Streamlit not installed. Run: pip install streamlit")
    sys.exit(1)

try:
    import joblib
except ImportError:
    st.error("joblib not installed. Run: pip install joblib")
    st.stop()

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

HIGH_RISK = ["Uncapped Liability", "Change Of Control", "Non-Compete", "Liquidated Damages"]
MEDIUM_RISK = [
    "Indemnification", "Cap On Liability", "Termination For Convenience",
    "Exclusivity", "No-Solicit Of Employees", "No-Solicit Of Customers",
    "Revenue/Profit Sharing",
]
STANDARD_RISK = [
    "Governing Law", "Renewal Term", "Anti-Assignment", "Audit Rights",
    "Insurance", "Warranty Duration", "License Grant",
]

RISK_COLORS = {
    "HIGH": "#d32f2f",
    "MEDIUM": "#f57c00",
    "STANDARD": "#388e3c",
    "ABSENT": "#9e9e9e",
}

CLAUSE_DESCRIPTIONS = {
    "Uncapped Liability": "No cap on damages — vendor/party can be liable for unlimited amounts.",
    "Change Of Control": "Rights/obligations triggered by merger, acquisition, or ownership change.",
    "Non-Compete": "Restriction on competing business activities within defined scope/duration.",
    "Liquidated Damages": "Pre-specified penalty amount payable upon breach.",
    "Indemnification": "Obligation to compensate for losses, damages, or expenses.",
    "Cap On Liability": "Maximum aggregate liability set for breach of contract.",
    "Termination For Convenience": "Right to terminate without cause (unilateral exit).",
    "Exclusivity": "Restriction on working with competitors during contract term.",
    "No-Solicit Of Employees": "Prohibition on recruiting or hiring the other party's staff.",
    "No-Solicit Of Customers": "Prohibition on soliciting the other party's customer base.",
    "IP Ownership Assignment": "Transfer of IP rights created under the agreement.",
    "Joint IP Ownership": "Both parties share ownership of jointly created IP.",
    "Revenue/Profit Sharing": "Formula for splitting revenues or profits between parties.",
    "Governing Law": "Jurisdiction whose laws govern the contract.",
    "Audit Rights": "Right to audit the other party's records and compliance.",
    "Insurance": "Required insurance coverage during the contract term.",
    "Warranty Duration": "Period during which warranty obligations apply.",
    "Renewal Term": "Automatic or optional renewal conditions.",
    "Anti-Assignment": "Restrictions on transferring contract rights to third parties.",
}

DEMO_CONTRACT = """MASTER SOFTWARE SERVICES AGREEMENT

This Agreement is entered into as of January 1, 2025 between ACME Corp ("Customer")
and TechVendor Inc ("Vendor").

1. SERVICES. Vendor shall provide software development services as described in any
Statement of Work executed by the parties.

2. IP OWNERSHIP. All work product, inventions, and intellectual property created by
Vendor under this Agreement shall be assigned to and become the exclusive property of
Customer upon creation. Vendor hereby assigns all rights, title, and interest in such
work product to Customer. This assignment is irrevocable.

3. NON-COMPETE. During the term of this Agreement and for two (2) years thereafter,
Vendor shall not, directly or indirectly, engage in any business that competes with
Customer's core business activities in the United States and Canada. This restriction
applies to all employees, subcontractors, and affiliated entities of Vendor.

4. UNLIMITED LIABILITY. NOTWITHSTANDING ANYTHING TO THE CONTRARY IN THIS AGREEMENT,
VENDOR'S TOTAL LIABILITY UNDER THIS AGREEMENT SHALL BE UNLIMITED AND SHALL INCLUDE
ALL DIRECT, INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, AND PUNITIVE DAMAGES
WITHOUT ANY CAP OR LIMITATION WHATSOEVER. VENDOR WAIVES ALL DEFENSES BASED ON
LIMITATION OF LIABILITY.

5. CHANGE OF CONTROL. In the event of a Change of Control of Vendor (defined as any
acquisition, merger, reorganization, or transfer of more than 50% of Vendor's voting
equity), Customer may, at its option, terminate this Agreement immediately without
penalty or continue under the same terms.

6. TERMINATION FOR CONVENIENCE. Either party may terminate this Agreement for any
reason or no reason upon thirty (30) days written notice to the other party.

7. GOVERNING LAW. This Agreement shall be governed by the laws of the State of
Delaware, without regard to its conflict of laws principles.

8. INDEMNIFICATION. Vendor shall defend, indemnify, and hold harmless Customer and
its affiliates, officers, and employees from any and all third-party claims, damages,
losses, and expenses (including reasonable attorneys' fees) arising out of or relating
to Vendor's performance of Services under this Agreement.
"""


@st.cache_resource
def load_model():
    path = MODELS_DIR / "blend_pipeline.joblib"
    if not path.exists():
        return None, "Model not found. Run `python src/train.py` to build the model."
    try:
        bundle = joblib.load(path)
        return bundle, None
    except Exception as e:
        return None, str(e)


def predict(text: str, bundle: dict) -> dict:
    from sklearn.metrics import precision_recall_curve
    vec = bundle["vectorizer"]
    lgbm_models = bundle["lgbm_models"]
    lr_models = bundle["lr_models"]
    thresholds = bundle["thresholds"]
    valid_clauses = bundle["valid_clauses"]
    alpha = bundle["blend_alpha"]

    t0 = time.time()
    Xmat = vec.transform([text])
    probs_lgbm = np.zeros(len(valid_clauses))
    probs_lr = np.zeros(len(valid_clauses))

    for j, clause in enumerate(valid_clauses):
        if clause in lgbm_models:
            probs_lgbm[j] = lgbm_models[clause].predict_proba(Xmat)[0, 1]
        if clause in lr_models:
            probs_lr[j] = lr_models[clause].predict_proba(Xmat)[0, 1]

    probs_blend = alpha * probs_lgbm + (1 - alpha) * probs_lr
    latency_ms = (time.time() - t0) * 1000

    results = {}
    for j, clause in enumerate(valid_clauses):
        thr = thresholds.get(clause, 0.5)
        prob = float(probs_blend[j])
        detected = prob >= thr
        risk = "ABSENT"
        if detected:
            risk = ("HIGH" if clause in HIGH_RISK else
                    "MEDIUM" if clause in MEDIUM_RISK else "STANDARD")
        results[clause] = {
            "probability": round(prob, 4),
            "detected": detected,
            "threshold": round(thr, 4),
            "risk_level": risk,
        }
    return results, round(latency_ms, 1)


def get_top_features(clause: str, bundle: dict, top_k: int = 8):
    """Get top positive LR features for a clause."""
    lr_models = bundle.get("lr_models", {})
    clf = lr_models.get(clause)
    if clf is None:
        return []
    feature_names = bundle["vectorizer"].get_feature_names_out()
    coef = clf.coef_[0]
    pos_idx = np.argsort(coef)[::-1][:top_k]
    return [(str(feature_names[i]), round(float(coef[i]), 4))
            for i in pos_idx if coef[i] > 0]


def highlight_contract(text: str, features: list) -> str:
    """Highlight top feature terms in the contract text."""
    highlighted = text
    for feat, _ in features[:5]:
        pattern = re.compile(re.escape(feat), re.IGNORECASE)
        highlighted = pattern.sub(
            f'<mark style="background-color: #fff176; padding: 1px 3px; '
            f'border-radius: 3px;">{feat}</mark>',
            highlighted
        )
    return highlighted


def render_risk_badge(risk_level: str) -> str:
    color = RISK_COLORS.get(risk_level, "#9e9e9e")
    return (f'<span style="background-color:{color}; color:white; padding:3px 8px; '
            f'border-radius:12px; font-size:12px; font-weight:bold;">{risk_level}</span>')


# ============================================================================
# STREAMLIT APP
# ============================================================================

st.set_page_config(
    page_title="Legal Contract Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚖️ Legal Contract Risk Analyzer")
st.markdown(
    "**Phase 6 Production Model** | 50/50 LGBM+LR Blend | "
    "Macro-F1: **0.6907** (beats RoBERTa-large by +0.041) | "
    "Runs in **~12ms** vs Claude's **3s+**"
)

# Sidebar — model info
with st.sidebar:
    st.header("Model Info")
    st.metric("Macro-F1", "0.6907", delta="+0.041 vs RoBERTa-large", delta_color="normal")
    st.metric("HR-F1 (High-Risk)", "0.582", delta="+0.420 vs Claude zero-shot", delta_color="normal")
    st.metric("Latency", "~12ms", delta="-99.7% vs Claude (3s)", delta_color="inverse")

    st.markdown("---")
    st.markdown("**Phase 5 Headline Finding:**")
    st.info(
        "LightGBM beats Claude 3x on high-risk legal clauses (0.499 vs 0.162 HR-F1) "
        "and runs **5,547x faster**. The reason: TF-IDF reads **100% of the contract**. "
        "Claude only read **4.6%** (400-word excerpt)."
    )

    st.markdown("---")
    st.markdown("**Clause Risk Legend:**")
    for level, color in RISK_COLORS.items():
        st.markdown(
            f'<span style="color:{color}; font-weight:bold;">■</span> {level}',
            unsafe_allow_html=True
        )

    st.markdown("---")
    show_features = st.toggle("Show feature explanations", value=True)
    show_highlight = st.toggle("Highlight key terms in contract", value=False)

# Load model
bundle, error = load_model()

if error:
    st.error(f"**Model not available**: {error}")
    st.code("python src/train.py", language="bash")
    st.stop()

# Input area
st.markdown("---")
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Contract Input")
    input_mode = st.radio("Input mode:", ["Paste text", "Upload file", "Use demo contract"])

    if input_mode == "Use demo contract":
        contract_text = DEMO_CONTRACT
        st.success("Demo contract loaded (8 clauses including IP Assignment, Non-Compete, Uncapped Liability).")
    elif input_mode == "Upload file":
        uploaded = st.file_uploader("Upload .txt contract", type=["txt"])
        if uploaded:
            contract_text = uploaded.read().decode("utf-8", errors="replace")
        else:
            contract_text = ""
    else:
        contract_text = st.text_area(
            "Paste contract text:",
            height=300,
            placeholder="Paste the full contract text here...",
        )

    if contract_text:
        words = len(contract_text.split())
        st.caption(f"{words:,} words | {len(contract_text):,} chars")

    analyze_btn = st.button("🔍 Analyze Contract", type="primary",
                             disabled=(not contract_text))

with col2:
    if analyze_btn and contract_text:
        with st.spinner("Analyzing..."):
            results, latency = predict(contract_text, bundle)

        valid = bundle["valid_clauses"]
        detected = {c: v for c, v in results.items() if v["detected"]}
        hr_detected = [c for c in HIGH_RISK if c in detected]

        # Overall risk banner
        if hr_detected:
            st.error(f"🚨 **HIGH RISK** — {len(hr_detected)} high-risk clause(s) detected: "
                     f"{', '.join(hr_detected)}")
        elif any(v["risk_level"] == "MEDIUM" for v in detected.values()):
            st.warning(f"⚠️ **MEDIUM RISK** — {len(detected)} clause(s) detected, review recommended.")
        else:
            st.success(f"✅ **LOW RISK** — No high-risk clauses detected.")

        st.caption(f"Analysis complete in {latency}ms | "
                   f"{len(detected)}/{len(valid)} clauses detected")
        st.markdown("---")

        # HIGH RISK section
        st.subheader("🔴 High-Risk Clauses")
        hr_rows = []
        for clause in HIGH_RISK:
            if clause in results:
                v = results[clause]
                detected_str = "✅ DETECTED" if v["detected"] else "—"
                hr_rows.append({
                    "Clause": clause,
                    "Status": detected_str,
                    "Probability": f"{v['probability']:.1%}",
                    "Description": CLAUSE_DESCRIPTIONS.get(clause, ""),
                })
        if hr_rows:
            df_hr = pd.DataFrame(hr_rows)
            st.dataframe(df_hr, use_container_width=True, hide_index=True)

        # MEDIUM RISK section
        st.subheader("🟡 Medium-Risk Clauses")
        med_rows = []
        for clause in MEDIUM_RISK:
            if clause in results:
                v = results[clause]
                if v["detected"]:
                    med_rows.append({
                        "Clause": clause,
                        "Probability": f"{v['probability']:.1%}",
                        "Description": CLAUSE_DESCRIPTIONS.get(clause, ""),
                    })
        if med_rows:
            st.dataframe(pd.DataFrame(med_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No medium-risk clauses detected.")

        # Other detected clauses
        other_detected = [c for c, v in results.items()
                          if v["detected"] and c not in HIGH_RISK and c not in MEDIUM_RISK]
        if other_detected:
            with st.expander(f"Standard clauses detected ({len(other_detected)})"):
                for c in other_detected:
                    v = results[c]
                    st.markdown(f"**{c}** — {v['probability']:.1%} probability")

        # Feature explanations
        if show_features:
            st.markdown("---")
            st.subheader("🔬 Feature Explanations (Why was each clause flagged?)")
            all_flagged = [c for c in (HIGH_RISK + MEDIUM_RISK) if c in results and results[c]["detected"]]
            if all_flagged:
                tabs = st.tabs(all_flagged[:6])  # limit to 6 tabs
                for tab, clause in zip(tabs, all_flagged[:6]):
                    with tab:
                        top_feats = get_top_features(clause, bundle, top_k=10)
                        st.markdown(f"**{CLAUSE_DESCRIPTIONS.get(clause, '')}**")
                        if top_feats:
                            feat_df = pd.DataFrame(top_feats, columns=["Feature (n-gram)", "Weight"])
                            st.dataframe(feat_df, use_container_width=True, hide_index=True)
                            present = [f for f, _ in top_feats
                                       if f.lower() in contract_text.lower()]
                            st.caption(f"{len(present)}/{len(top_feats)} top features found in this contract: "
                                       f"{', '.join(present[:5])}")
            else:
                st.caption("No high/medium risk clauses detected to explain.")

        # Missing high-risk clauses warning
        missing_hr = [c for c in HIGH_RISK if c not in detected or not results[c]["detected"]]
        if missing_hr:
            st.markdown("---")
            with st.expander("🔎 Clauses NOT detected (may be missing from contract)"):
                st.markdown("Consider whether the following should be present:")
                for c in missing_hr:
                    st.markdown(f"- **{c}**: {CLAUSE_DESCRIPTIONS.get(c, '')}")

        # Contract with highlights
        if show_highlight and hr_detected:
            st.markdown("---")
            st.subheader("📄 Contract with Highlighted Key Terms")
            clause_to_show = st.selectbox("Show highlights for:", hr_detected)
            if clause_to_show:
                top_feats = get_top_features(clause_to_show, bundle, top_k=5)
                highlighted = highlight_contract(contract_text[:3000], top_feats)
                st.markdown(
                    f'<div style="font-family: monospace; font-size: 13px; '
                    f'line-height: 1.6; white-space: pre-wrap; max-height: 400px; '
                    f'overflow-y: auto; padding: 12px; background: #fafafa; '
                    f'border: 1px solid #e0e0e0; border-radius: 4px;">{highlighted}</div>',
                    unsafe_allow_html=True,
                )
                if len(contract_text) > 3000:
                    st.caption("Showing first 3,000 chars. Full contract was analyzed.")

    elif not analyze_btn:
        st.info("Enter contract text and click **Analyze Contract** to see results.")

# Footer
st.markdown("---")
st.markdown(
    "<small>Model: 50/50 LGBM+LR Blend | Dataset: CUAD (510 SEC contracts) | "
    "Training: 408 contracts, Testing: 102 contracts | "
    "39 clause types detected | "
    "Phase 5 research: beats RoBERTa-large by +0.041 macro-F1 and Claude 3x on high-risk clauses</small>",
    unsafe_allow_html=True,
)
