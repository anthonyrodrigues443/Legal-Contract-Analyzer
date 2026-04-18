"""
Legal Contract Analyzer — Streamlit UI.

Paste a contract. Get a risk score (0-100), flagged clauses color-coded by risk
level, evidence snippets that triggered each flag, and a "missing clauses"
warning when high-risk protections are absent.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.feature_engineering import (
    HIGH_RISK_CLAUSES,
    MEDIUM_RISK_CLAUSES,
    load_cuad_from_json,
    risk_level,
)
from src.predict import ContractAnalyzer


# ---------------------------------------------------------------------------
# Page config + CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Legal Contract Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background-color: #f5f7fa;
        border-radius: 8px;
        padding: 14px;
        border-left: 4px solid #1976D2;
        margin-bottom: 10px;
    }
    .risk-high  { background-color: #ffebee; border-left: 4px solid #E53935; padding: 10px; border-radius: 6px; margin-bottom: 6px; }
    .risk-med   { background-color: #fff8e1; border-left: 4px solid #FB8C00; padding: 10px; border-radius: 6px; margin-bottom: 6px; }
    .risk-low   { background-color: #e8f5e9; border-left: 4px solid #43A047; padding: 10px; border-radius: 6px; margin-bottom: 6px; }
    .clause-name { font-weight: 600; font-size: 15px; }
    .snippet { color: #555; font-style: italic; margin-top: 4px; font-size: 13px; }
    .risk-pill-high   { background-color: #E53935; color: white; padding: 2px 10px; border-radius: 10px; font-size: 11px; font-weight: 600; }
    .risk-pill-med    { background-color: #FB8C00; color: white; padding: 2px 10px; border-radius: 10px; font-size: 11px; font-weight: 600; }
    .risk-pill-low    { background-color: #43A047; color: white; padding: 2px 10px; border-radius: 10px; font-size: 11px; font-weight: 600; }
    .missing-box { background-color: #fff3e0; border: 1px dashed #FB8C00; padding: 10px; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_analyzer():
    return ContractAnalyzer.load()


@st.cache_data
def load_example_contracts(n: int = 5) -> dict:
    """Load a few sample contracts from CUAD to seed the 'Try example' dropdown."""
    data_path = Path("data/raw/CUADv1.json")
    if not data_path.exists():
        return {}
    df = load_cuad_from_json(data_path)
    # Pick a diverse set: one short, one long, a few in between
    df = df.sort_values("text_length").reset_index(drop=True)
    picks = [0, len(df) // 4, len(df) // 2, 3 * len(df) // 4, len(df) - 1][:n]
    examples = {}
    for i in picks:
        row = df.iloc[i]
        name = row["contract_title"][:70].strip()
        examples[f"{name}  ({len(row['text']):,} chars)"] = row["text"]
    return examples


@st.cache_data
def load_training_manifest() -> dict:
    p = Path("models/training_manifest.json")
    if p.exists():
        return json.loads(p.read_text())
    return {}


# ---------------------------------------------------------------------------
# Sidebar — model info, how it works
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚖️  Legal Contract Analyzer")
    st.caption("Flag risky clauses in 41-type legal contracts — 700ms, $0/contract.")

    st.markdown("### Model performance")
    manifest = load_training_manifest()
    m = manifest.get("test_metrics", {}) if manifest else {}
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Macro-F1", f"{m.get('macro_f1', 0):.3f}")
        st.metric("HR-F1", f"{m.get('hr_f1', 0):.3f}")
    with col2:
        st.metric("Macro-AUC", f"{m.get('macro_auc', 0):.3f}")
        st.metric("Recall", f"{m.get('macro_recall', 0):.3f}")

    st.markdown("### How it works")
    st.markdown("""
    1. **Vectorize** contract → 20K TF-IDF features (word 1-2 grams)
    2. **Classify** each of 28 clause types with a LightGBM+LogReg probability blend
    3. **Threshold** per-clause using values learned via 3-fold CV (no test-set peek)
    4. **Score** HIGH-risk clauses 10pts, MEDIUM 4pts, LOW 1pt. Cap at 100.
    5. **Evidence**: regex-match the clause passage for the reviewer to verify

    Training data: [CUAD v1](https://www.atticusprojectai.org/cuad) (510 real contracts, 41 clause types).
    """)

    st.markdown("### Benchmarks")
    st.markdown("""
    | System | HR-F1 | Latency | Cost/1K |
    |---|---|---|---|
    | **This model** | **0.524** | **~700 ms** | **$0** |
    | Claude zero-shot* | 0.162 | 11.1 s | $15+ |
    | Published RoBERTa-large | ~0.65** | — | — |

    *Claude numbers from Phase 5 with 400-word excerpts — full-context comparison remains open.
    **RoBERTa reports macro-F1, not HR-F1.
    """)


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("Legal Contract Risk Analyzer")
st.markdown(
    "Paste any commercial contract below. The model will flag the clauses most"
    " likely to matter for due diligence and assign an overall risk band."
)

# --- Input area ---
examples = load_example_contracts()
col_input, col_select = st.columns([3, 1])
with col_select:
    example_name = st.selectbox(
        "Load example from CUAD:",
        options=["— none —"] + list(examples.keys()),
        index=0,
    )

if example_name != "— none —":
    default_text = examples[example_name]
else:
    default_text = ""

contract_text = st.text_area(
    "Contract text",
    value=default_text,
    height=260,
    placeholder="Paste contract text here...",
)

col_a, col_b = st.columns([1, 4])
with col_a:
    analyze_clicked = st.button("🔍 Analyze contract", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

if analyze_clicked:
    if not contract_text or len(contract_text.strip()) < 200:
        st.warning("Please paste a contract (at least 200 characters) or select an example.")
        st.stop()

    analyzer = load_analyzer()
    with st.spinner("Analyzing..."):
        report = analyzer.analyze(contract_text)

    st.markdown("---")

    # --- Top row: Risk score + quick stats ---
    col1, col2, col3, col4, col5 = st.columns(5)

    band_color = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟢"}[report.risk_band]
    with col1:
        st.markdown(f"### {band_color} Risk: **{report.risk_band}**")
        st.caption(f"Score: {report.risk_score:.0f} / 100")
    with col2:
        st.metric("Clauses flagged", report.n_flagged)
    with col3:
        st.metric("HIGH-risk flagged", report.n_high_risk_flagged)
    with col4:
        st.metric("MEDIUM-risk flagged", report.n_medium_risk_flagged)
    with col5:
        st.metric("Inference", f"{report.inference_ms:.0f} ms")

    # --- Risk gauge (plotly) ---
    col_gauge, col_breakdown = st.columns([1, 2])
    with col_gauge:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=report.risk_score,
            title={"text": "Contract Risk Score"},
            number={"suffix": " / 100"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1976D2"},
                "steps": [
                    {"range": [0, 30], "color": "#c8e6c9"},
                    {"range": [30, 70], "color": "#ffe0b2"},
                    {"range": [70, 100], "color": "#ffcdd2"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": report.risk_score,
                },
            },
        ))
        gauge.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(gauge, use_container_width=True)

    with col_breakdown:
        # Per-clause probability bar chart (flagged only)
        if report.flagged_clauses:
            df_flag = pd.DataFrame([
                {
                    "Clause": c.clause,
                    "Probability": c.probability,
                    "Threshold": c.threshold,
                    "Risk": c.risk_level,
                }
                for c in report.flagged_clauses
            ])
            color_map = {"HIGH": "#E53935", "MEDIUM": "#FB8C00", "LOW": "#43A047"}
            fig = px.bar(
                df_flag.sort_values("Probability"),
                x="Probability", y="Clause", color="Risk",
                color_discrete_map=color_map,
                orientation="h",
                title="Flagged clauses (probability & risk level)",
                range_x=[0, 1],
            )
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No clauses flagged above learned thresholds.")

    # --- Flagged clauses detail ---
    st.markdown("### 🚩 Flagged clauses & evidence")
    if not report.flagged_clauses:
        st.info(
            "The model did not flag any clauses above their learned thresholds. "
            "This does NOT mean the contract is risk-free — consider a manual review."
        )
    else:
        for c in report.flagged_clauses:
            cls = {
                "HIGH": "risk-high", "MEDIUM": "risk-med", "LOW": "risk-low"
            }[c.risk_level]
            pill = {
                "HIGH": "risk-pill-high", "MEDIUM": "risk-pill-med", "LOW": "risk-pill-low"
            }[c.risk_level]
            evidence_html = ""
            if c.evidence_snippet:
                escaped = (
                    c.evidence_snippet.replace("<", "&lt;").replace(">", "&gt;")
                )
                evidence_html = f'<div class="snippet">"{escaped}"</div>'
            st.markdown(
                f'<div class="{cls}">'
                f'<span class="clause-name">{c.clause}</span> '
                f'<span class="{pill}">{c.risk_level}</span> '
                f'&nbsp;&nbsp;<code>p={c.probability:.3f}</code> '
                f'<code>threshold={c.threshold:.2f}</code>'
                f"{evidence_html}"
                f'</div>',
                unsafe_allow_html=True,
            )

    # --- Missing HIGH-risk clauses ---
    st.markdown("### 🛡️ HIGH-risk clause coverage")
    hr_in_model = analyzer.all_high_risk_clauses
    flagged_names = {c.clause for c in report.flagged_clauses}
    rows = []
    for c in hr_in_model:
        rows.append({
            "Clause": c,
            "Flagged?": "✅ Yes" if c in flagged_names else "❌ No",
            "Probability": next(
                (cl.probability for cl in report.all_clauses if cl.clause == c), 0.0
            ),
            "Threshold": next(
                (cl.threshold for cl in report.all_clauses if cl.clause == c), 0.5
            ),
        })
    hr_df = pd.DataFrame(rows)
    st.dataframe(
        hr_df.style.format({"Probability": "{:.3f}", "Threshold": "{:.2f}"}),
        use_container_width=True,
        hide_index=True,
    )

    missing_hr = [c for c in hr_in_model if c not in flagged_names]
    if missing_hr:
        st.markdown(
            f'<div class="missing-box"><b>Heads-up:</b> {len(missing_hr)} HIGH-risk clause type(s) '
            f'were not flagged: {", ".join(missing_hr)}. For a buy-side deal, absence of some of '
            f"these (e.g. Cap On Liability, Indemnification) is itself a red flag.</div>",
            unsafe_allow_html=True,
        )

    # --- Full clause table (expandable) ---
    with st.expander("📋 Full 28-clause breakdown"):
        rows = []
        for c in report.all_clauses:
            rows.append({
                "Clause": c.clause,
                "Risk level": c.risk_level,
                "Probability": c.probability,
                "Threshold": c.threshold,
                "Flagged": "✅" if c.flagged else "",
            })
        full_df = pd.DataFrame(rows)
        st.dataframe(
            full_df.style.format({"Probability": "{:.3f}", "Threshold": "{:.2f}"}),
            use_container_width=True,
            hide_index=True,
        )

    # --- JSON report (for downstream integrations) ---
    with st.expander("💾 JSON report (for downstream integration)"):
        st.code(report.to_json(), language="json")


else:
    # When no analysis yet, show placeholder
    st.markdown("#### What you'll see after analysis")
    st.markdown(
        "- **Risk score (0-100)** and band: HIGH / MEDIUM / LOW\n"
        "- **Flagged clauses** color-coded by risk — indemnification, uncapped liability, change-of-control, etc.\n"
        "- **Evidence snippets** showing exactly which passage triggered each flag\n"
        "- **Coverage report**: which HIGH-risk clause types were flagged and which are missing from the contract entirely"
    )
