"""
Inference pipeline for Legal Contract Analyzer.

Phase 6 rework (post-Phase 5, PR E):
  - Loads LGBM-only pipeline (vectorizer + per-clause LGBMClassifier list + class-prior thresholds).
  - NO LR blend — Phase 5 showed it HURTS on 40K word-1-3gram features.
  - Thresholds are per-clause class priors (no CV, no test-set touch).

Usage:
    python -m src.predict --demo
    python -m src.predict --text "Contract text ..."
    python -m src.predict --file path/to/contract.txt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

HIGH_RISK = [
    "Uncapped Liability", "IP Ownership Assignment", "Change Of Control",
    "Non-Compete", "Liquidated Damages", "Joint IP Ownership",
]
MEDIUM_RISK = [
    "Indemnification", "Cap On Liability", "Termination For Convenience",
    "Exclusivity", "No-Solicit Of Employees", "No-Solicit Of Customers",
    "Revenue/Profit Sharing", "Most Favored Nation", "Covenant Not To Sue",
]
RISK_LEVEL = {c: "HIGH" for c in HIGH_RISK}
RISK_LEVEL.update({c: "MEDIUM" for c in MEDIUM_RISK})


def load_model(models_dir: Path | None = None) -> dict:
    """Load the Phase 6 production bundle (LGBM-only, no LR)."""
    d = models_dir or MODELS_DIR
    vec_path = d / "vectorizer.joblib"
    lgbm_path = d / "lgbm_models.joblib"
    thr_path = d / "thresholds.json"
    clauses_path = d / "valid_clauses.json"
    if not vec_path.exists() or not lgbm_path.exists():
        raise FileNotFoundError(
            f"Production artifacts not found in {d}. Run `python -m src.train` first."
        )
    return {
        "vectorizer": joblib.load(vec_path),
        "lgbm_models": joblib.load(lgbm_path),
        "thresholds": json.loads(thr_path.read_text()),
        "valid_clauses": json.loads(clauses_path.read_text()),
    }


def predict(text: str, bundle: dict) -> dict:
    """Run inference on a single contract text. Returns per-clause probabilities + risk summary."""
    vec = bundle["vectorizer"]
    lgbm_models = bundle["lgbm_models"]
    thresholds = bundle["thresholds"]
    valid_clauses = bundle["valid_clauses"]

    t0 = time.time()
    X = vec.transform([text])

    probs = np.zeros(len(valid_clauses))
    for j, clause in enumerate(valid_clauses):
        m = lgbm_models[j] if j < len(lgbm_models) else None
        if m is not None:
            probs[j] = m.predict_proba(X)[0, 1]

    latency_ms = (time.time() - t0) * 1000

    results = {}
    for j, clause in enumerate(valid_clauses):
        thr = thresholds.get(clause, 0.5)
        p = float(probs[j])
        results[clause] = {
            "probability": round(p, 4),
            "detected": bool(p >= thr),
            "threshold": round(thr, 4),
            "risk_level": RISK_LEVEL.get(clause, "STANDARD"),
        }

    detected = [c for c, v in results.items() if v["detected"]]
    high_risk_detected = [c for c in detected if c in HIGH_RISK]
    overall = (
        "HIGH" if high_risk_detected else
        "MEDIUM" if any(RISK_LEVEL.get(c) == "MEDIUM" for c in detected) else
        "LOW"
    )

    return {
        "clauses": results,
        "detected_count": len(detected),
        "high_risk_detected": high_risk_detected,
        "overall_risk": overall,
        "latency_ms": round(latency_ms, 2),
        "word_count": len(text.split()),
    }


def format_report(pred: dict) -> str:
    lines = [
        "=" * 60,
        "LEGAL CONTRACT RISK ANALYSIS (Phase 6 production)",
        f"Overall Risk: {pred['overall_risk']}",
        f"Clauses detected: {pred['detected_count']} / {len(pred['clauses'])}",
        f"High-risk clauses: {len(pred['high_risk_detected'])}",
        f"Latency: {pred['latency_ms']}ms | Words: {pred['word_count']}",
        "=" * 60,
        "",
        "HIGH-RISK CLAUSES:",
    ]
    for c in HIGH_RISK:
        if c in pred["clauses"]:
            v = pred["clauses"][c]
            flag = "[DETECTED]" if v["detected"] else "[absent]  "
            lines.append(f"  {flag} {c:<35} prob={v['probability']:.3f}  thr={v['threshold']:.3f}")

    lines += ["", "MEDIUM-RISK CLAUSES:"]
    for c in MEDIUM_RISK:
        if c in pred["clauses"]:
            v = pred["clauses"][c]
            flag = "[DETECTED]" if v["detected"] else "[absent]  "
            lines.append(f"  {flag} {c:<35} prob={v['probability']:.3f}  thr={v['threshold']:.3f}")

    if pred["high_risk_detected"]:
        lines += ["", "ACTION REQUIRED: review the following high-risk clauses before signing:"]
        for c in pred["high_risk_detected"]:
            lines.append(f"  !  {c}")

    lines += ["", "=" * 60]
    return "\n".join(lines)


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
Customer's core business activities in the United States and Canada.

4. UNLIMITED LIABILITY. NOTWITHSTANDING ANYTHING TO THE CONTRARY IN THIS AGREEMENT,
VENDOR'S TOTAL LIABILITY UNDER THIS AGREEMENT SHALL BE UNLIMITED AND SHALL INCLUDE
ALL DIRECT, INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, AND PUNITIVE DAMAGES
WITHOUT ANY CAP OR LIMITATION WHATSOEVER.

5. CHANGE OF CONTROL. In the event of a Change of Control of Vendor (defined as any
acquisition, merger, reorganization, or transfer of more than 50% of Vendor's voting
equity), Customer may, at its option, terminate this Agreement immediately without
penalty.

6. TERMINATION FOR CONVENIENCE. Either party may terminate this Agreement for any
reason or no reason upon thirty (30) days written notice to the other party.

7. GOVERNING LAW. This Agreement shall be governed by the laws of the State of
Delaware, without regard to its conflict of laws principles.

8. INDEMNIFICATION. Vendor shall defend, indemnify, and hold harmless Customer and
its affiliates, officers, and employees from any and all third-party claims, damages,
losses, and expenses (including reasonable attorneys' fees) arising out of or relating
to Vendor's performance of Services under this Agreement.
"""


def main():
    parser = argparse.ArgumentParser(description="Legal Contract Risk Analyzer")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--text", help="Contract text to analyze")
    group.add_argument("--file", help="Path to contract .txt file")
    group.add_argument("--demo", action="store_true", help="Run on demo contract")
    parser.add_argument("--models-dir", help="Override models directory")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    print("Loading Phase 6 production model (LGBM-only)...")
    bundle = load_model(Path(args.models_dir) if args.models_dir else None)
    print(f"Loaded. Clauses: {len(bundle['valid_clauses'])}  Vocab: {len(bundle['vectorizer'].vocabulary_):,}")

    if args.demo:
        text = DEMO_CONTRACT
        print("\nRunning on DEMO contract...")
    elif args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    else:
        print("Provide --text, --file, or --demo. See --help.")
        return

    pred = predict(text, bundle)
    if args.json:
        print(json.dumps(pred, indent=2))
    else:
        print(format_report(pred))


if __name__ == "__main__":
    main()
