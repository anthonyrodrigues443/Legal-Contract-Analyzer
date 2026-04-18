"""
Production inference API for Legal Contract Analyzer.

Loads serialized artifacts from models/ and runs prediction on raw contract text.
Designed to be called from app.py (Streamlit), a CLI, or any downstream service.

Example:
    from src.predict import ContractAnalyzer

    analyzer = ContractAnalyzer.load()
    report = analyzer.analyze(contract_text)
    print(report.risk_score)         # 0-100
    print(report.flagged_clauses)    # list of dicts with probability, risk level, snippet
    print(report.to_json())
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np

from src.feature_engineering import (
    HIGH_RISK_CLAUSES,
    MEDIUM_RISK_CLAUSES,
    extract_clause_snippet,
    risk_level,
)


MODELS_DIR_DEFAULT = Path("models")


@dataclass
class ClausePrediction:
    """Per-clause output of the classifier."""
    clause: str
    probability: float
    threshold: float
    flagged: bool
    risk_level: str  # HIGH / MEDIUM / LOW
    evidence_snippet: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ContractReport:
    """Full analysis report for a single contract."""
    n_flagged: int
    n_high_risk_flagged: int
    n_medium_risk_flagged: int
    risk_score: float           # 0-100
    risk_band: str              # LOW / MEDIUM / HIGH
    inference_ms: float
    word_count: int
    flagged_clauses: List[ClausePrediction] = field(default_factory=list)
    all_clauses: List[ClausePrediction] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_flagged": self.n_flagged,
            "n_high_risk_flagged": self.n_high_risk_flagged,
            "n_medium_risk_flagged": self.n_medium_risk_flagged,
            "risk_score": self.risk_score,
            "risk_band": self.risk_band,
            "inference_ms": self.inference_ms,
            "word_count": self.word_count,
            "flagged_clauses": [c.to_dict() for c in self.flagged_clauses],
            "all_clauses": [c.to_dict() for c in self.all_clauses],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class ContractAnalyzer:
    """Production inference wrapper for the Legal Contract Analyzer model."""

    def __init__(self, vectorizer, lgbm_models, lr_models, thresholds, valid_clauses, alpha=0.5):
        self.vectorizer = vectorizer
        self.lgbm_models = lgbm_models
        self.lr_models = lr_models
        self.thresholds = thresholds
        self.valid_clauses = valid_clauses
        self.alpha = alpha

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, models_dir: str | Path = MODELS_DIR_DEFAULT) -> "ContractAnalyzer":
        models_dir = Path(models_dir)
        if not (models_dir / "vectorizer.joblib").exists():
            raise FileNotFoundError(
                f"No model artifacts found in {models_dir}/. "
                "Run `python -m src.train` first."
            )

        vectorizer = joblib.load(models_dir / "vectorizer.joblib")
        lgbm_models = joblib.load(models_dir / "lgbm_models.joblib")
        lr_models = joblib.load(models_dir / "lr_models.joblib")
        thresholds = json.loads((models_dir / "thresholds.json").read_text())
        valid_clauses = json.loads((models_dir / "valid_clauses.json").read_text())

        manifest_path = models_dir / "training_manifest.json"
        alpha = 0.5
        if manifest_path.exists():
            alpha = json.loads(manifest_path.read_text()).get("alpha", 0.5)

        return cls(
            vectorizer=vectorizer,
            lgbm_models=lgbm_models,
            lr_models=lr_models,
            thresholds=thresholds,
            valid_clauses=valid_clauses,
            alpha=alpha,
        )

    # ------------------------------------------------------------------
    # Prediction primitives
    # ------------------------------------------------------------------
    def predict_proba(self, texts: List[str] | str) -> np.ndarray:
        """Predict clause probabilities for one or many contracts.

        Returns an (n_docs, n_clauses) numpy array aligned with self.valid_clauses.
        """
        if isinstance(texts, str):
            texts = [texts]
        X = self.vectorizer.transform(texts)
        n_clauses = len(self.valid_clauses)
        probs = np.zeros((X.shape[0], n_clauses))

        for j in range(n_clauses):
            lgbm = self.lgbm_models[j]
            lr = self.lr_models[j]
            plgbm = lgbm.predict_proba(X)[:, 1] if lgbm is not None else None
            plr = lr.predict_proba(X)[:, 1] if lr is not None else None
            if plgbm is not None and plr is not None:
                probs[:, j] = self.alpha * plgbm + (1 - self.alpha) * plr
            elif plgbm is not None:
                probs[:, j] = plgbm
            elif plr is not None:
                probs[:, j] = plr
        return probs

    def predict(self, texts: List[str] | str) -> np.ndarray:
        """Binary predictions using learned per-clause thresholds."""
        probs = self.predict_proba(texts)
        preds = np.zeros_like(probs, dtype=int)
        for j, clause in enumerate(self.valid_clauses):
            t = self.thresholds.get(clause, 0.5)
            preds[:, j] = (probs[:, j] >= t).astype(int)
        return preds

    # ------------------------------------------------------------------
    # Rich analysis for the UI
    # ------------------------------------------------------------------
    def analyze(self, text: str) -> ContractReport:
        """Produce a full ContractReport for one contract.

        Includes risk score, per-clause flags, evidence snippets, and latency.
        """
        t0 = time.time()
        probs = self.predict_proba(text)[0]
        infer_ms = 1000 * (time.time() - t0)

        all_clauses: List[ClausePrediction] = []
        for j, clause in enumerate(self.valid_clauses):
            p = float(probs[j])
            thr = float(self.thresholds.get(clause, 0.5))
            flagged = p >= thr
            pred = ClausePrediction(
                clause=clause,
                probability=p,
                threshold=thr,
                flagged=flagged,
                risk_level=risk_level(clause),
                evidence_snippet=extract_clause_snippet(text, clause) if flagged else None,
            )
            all_clauses.append(pred)

        flagged = [c for c in all_clauses if c.flagged]
        n_high = sum(1 for c in flagged if c.risk_level == "HIGH")
        n_med = sum(1 for c in flagged if c.risk_level == "MEDIUM")

        # Risk score: 10 points per HIGH flagged clause, 4 per MEDIUM, 1 per LOW.
        # Capped at 100. Calibrated so: no HIGH + few MED ~ 0-30 (LOW band),
        # 1-2 HIGH or several MED ~ 31-70 (MEDIUM), 3+ HIGH ~ 71-100 (HIGH).
        raw = 10 * n_high + 4 * n_med + 1 * max(0, len(flagged) - n_high - n_med)
        score = min(100.0, raw)
        if score >= 70:
            band = "HIGH"
        elif score >= 30:
            band = "MEDIUM"
        else:
            band = "LOW"

        # Sort flagged clauses: HIGH first, then by probability desc
        risk_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        flagged_sorted = sorted(
            flagged, key=lambda c: (risk_rank[c.risk_level], -c.probability)
        )
        # All clauses sorted similarly (for full breakdown UI)
        all_sorted = sorted(
            all_clauses, key=lambda c: (risk_rank[c.risk_level], -c.probability)
        )

        return ContractReport(
            n_flagged=len(flagged),
            n_high_risk_flagged=n_high,
            n_medium_risk_flagged=n_med,
            risk_score=float(score),
            risk_band=band,
            inference_ms=float(infer_ms),
            word_count=len(text.split()),
            flagged_clauses=flagged_sorted,
            all_clauses=all_sorted,
        )

    # ------------------------------------------------------------------
    # Clause taxonomy accessors (used by app.py for "clauses missing" detection)
    # ------------------------------------------------------------------
    @property
    def all_high_risk_clauses(self) -> List[str]:
        return [c for c in HIGH_RISK_CLAUSES if c in self.valid_clauses]

    @property
    def all_medium_risk_clauses(self) -> List[str]:
        return [c for c in MEDIUM_RISK_CLAUSES if c in self.valid_clauses]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Legal contract risk analyzer CLI")
    parser.add_argument("input", help="Path to a contract .txt file, or '-' for stdin")
    parser.add_argument("--models", default="models", help="Path to models/ directory")
    parser.add_argument("--json", action="store_true", help="Print full JSON report")
    args = parser.parse_args()

    if args.input == "-":
        text = sys.stdin.read()
    else:
        text = Path(args.input).read_text()

    analyzer = ContractAnalyzer.load(args.models)
    report = analyzer.analyze(text)

    if args.json:
        print(report.to_json())
        return

    print(f"RISK SCORE: {report.risk_score:.1f}/100  [{report.risk_band}]")
    print(f"  {report.n_high_risk_flagged} HIGH-risk + {report.n_medium_risk_flagged} MEDIUM-risk clauses flagged")
    print(f"  Inference: {report.inference_ms:.1f} ms  |  {report.word_count:,} words")
    print()
    print("FLAGGED CLAUSES:")
    for c in report.flagged_clauses:
        marker = "[HIGH]" if c.risk_level == "HIGH" else ("[MED] " if c.risk_level == "MEDIUM" else "[LOW] ")
        print(f"  {marker} {c.clause:<38}  p={c.probability:.3f}  (threshold {c.threshold:.2f})")
        if c.evidence_snippet:
            print(f"           evidence: {c.evidence_snippet[:200]}")


if __name__ == "__main__":
    _cli()
