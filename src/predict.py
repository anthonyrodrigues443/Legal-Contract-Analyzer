"""
Inference pipeline for Legal Contract Analyzer.
Loads saved blend model and runs prediction on new contract text.

Usage:
  python src/predict.py --text "Contract text here..."
  python src/predict.py --file path/to/contract.txt
  python src/predict.py --demo   # run on a short sample contract
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

HIGH_RISK = ["Uncapped Liability", "Change Of Control", "Non-Compete", "Liquidated Damages"]
MEDIUM_RISK = [
    "Indemnification", "Cap On Liability", "Termination For Convenience",
    "Exclusivity", "No-Solicit Of Employees", "No-Solicit Of Customers",
]

RISK_LEVEL = {c: "HIGH" for c in HIGH_RISK}
RISK_LEVEL.update({c: "MEDIUM" for c in MEDIUM_RISK})


def load_model(model_path: Path = None):
    path = model_path or MODELS_DIR / "blend_pipeline.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run `python src/train.py` first."
        )
    return joblib.load(path)


def predict(text: str, bundle: dict) -> dict:
    """
    Run the blend pipeline on a single contract text.
    Returns per-clause probabilities, predictions, and risk scores.
    """
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
        results[clause] = {
            "probability": round(prob, 4),
            "detected": bool(detected),
            "threshold": round(thr, 4),
            "risk_level": RISK_LEVEL.get(clause, "STANDARD"),
        }

    detected_clauses = [c for c, v in results.items() if v["detected"]]
    high_risk_detected = [c for c in detected_clauses if c in HIGH_RISK]
    overall_risk = (
        "HIGH" if high_risk_detected else
        "MEDIUM" if any(RISK_LEVEL.get(c) == "MEDIUM" for c in detected_clauses) else
        "LOW"
    )

    return {
        "clauses": results,
        "detected_count": len(detected_clauses),
        "high_risk_detected": high_risk_detected,
        "overall_risk": overall_risk,
        "latency_ms": round(latency_ms, 2),
        "word_count": len(text.split()),
    }


def format_report(prediction: dict) -> str:
    lines = [
        "=" * 60,
        "LEGAL CONTRACT RISK ANALYSIS",
        f"Overall Risk: {prediction['overall_risk']}",
        f"Clauses detected: {prediction['detected_count']} / {len(prediction['clauses'])}",
        f"High-risk clauses: {len(prediction['high_risk_detected'])}",
        f"Latency: {prediction['latency_ms']}ms | Words: {prediction['word_count']}",
        "=" * 60,
        "",
        "HIGH-RISK CLAUSES:",
    ]
    for c in HIGH_RISK:
        if c in prediction["clauses"]:
            v = prediction["clauses"][c]
            flag = "[DETECTED]" if v["detected"] else "[absent]  "
            lines.append(f"  {flag} {c:<35} prob={v['probability']:.3f}")

    lines += ["", "MEDIUM-RISK CLAUSES:"]
    for c in MEDIUM_RISK:
        if c in prediction["clauses"]:
            v = prediction["clauses"][c]
            flag = "[DETECTED]" if v["detected"] else "[absent]  "
            lines.append(f"  {flag} {c:<35} prob={v['probability']:.3f}")

    if prediction["high_risk_detected"]:
        lines += ["", "ACTION REQUIRED: Review the following high-risk clauses before signing:"]
        for c in prediction["high_risk_detected"]:
            lines.append(f"  ⚠  {c}")

    lines += ["", "=" * 60]
    return "\n".join(lines)


DEMO_CONTRACT = """
MASTER SOFTWARE SERVICES AGREEMENT

This Agreement is entered into as of the date last signed below between ACME Corp ("Customer")
and TechVendor Inc ("Vendor").

1. SERVICES. Vendor shall provide software development services as described in any Statement of Work.

2. IP OWNERSHIP. All work product, inventions, and intellectual property created by Vendor under
this Agreement shall be assigned to and become the exclusive property of Customer upon creation.
Vendor hereby assigns all rights, title, and interest in such work product to Customer.

3. NON-COMPETE. During the term of this Agreement and for two (2) years thereafter, Vendor shall
not, directly or indirectly, engage in any business that competes with Customer's core business
activities in the United States and Canada.

4. LIABILITY. NOTWITHSTANDING ANYTHING TO THE CONTRARY, VENDOR'S TOTAL LIABILITY UNDER THIS
AGREEMENT SHALL BE UNLIMITED. Vendor shall indemnify and hold harmless Customer from any and
all claims, damages, losses, and expenses without limitation.

5. TERMINATION. Either party may terminate this Agreement for convenience upon 30 days written notice.

6. GOVERNING LAW. This Agreement shall be governed by the laws of the State of Delaware.
"""


def main():
    parser = argparse.ArgumentParser(description="Legal Contract Risk Analyzer")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--text", help="Contract text to analyze")
    group.add_argument("--file", help="Path to contract .txt file")
    group.add_argument("--demo", action="store_true", help="Run on demo contract")
    parser.add_argument("--model-path", help="Path to saved model bundle")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    print("Loading model...")
    bundle = load_model(Path(args.model_path) if args.model_path else None)
    print(f"Model loaded. Clauses: {len(bundle['valid_clauses'])}")

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

    prediction = predict(text, bundle)

    if args.json:
        print(json.dumps(prediction, indent=2))
    else:
        print(format_report(prediction))


if __name__ == "__main__":
    main()
