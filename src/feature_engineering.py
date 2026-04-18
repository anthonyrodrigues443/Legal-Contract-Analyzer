"""
Feature engineering + data loading for Legal Contract Analyzer.
Shared across train.py, predict.py, evaluate.py, and app.py.

Phase 6 production choices (based on Phase 1-5 research):
- 20K word TF-IDF (1-2 ngrams), sublinear_tf
- Reason: 20K was Goldilocks zone (Mark P2); Word+Char combo didn't help; domain features hurt LR.
- Class re-weighting is mandatory: removing it drops macro-F1 by -0.08 (Mark P5 ablation).
- Split: fixed random shuffle (seed 42) to match all prior phases.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# Full CUAD clause taxonomy
CUAD_CATEGORIES = [
    "Document Name", "Parties", "Agreement Date", "Effective Date", "Expiration Date",
    "Renewal Term", "Notice Period To Terminate Renewal", "Governing Law",
    "Most Favored Nation", "Non-Compete", "Exclusivity", "No-Solicit Of Customers",
    "No-Solicit Of Employees", "Non-Disparagement", "Termination For Convenience",
    "ROFR/ROFO/ROFN", "Change Of Control", "Anti-Assignment",
    "Revenue/Profit Sharing", "Price Restrictions", "Minimum Commitment",
    "Volume Restriction", "IP Ownership Assignment", "Joint IP Ownership",
    "License Grant", "Non-Transferable License", "Affiliate License-Licensor",
    "Affiliate License-Licensee", "Unlimited/All-You-Can-Eat-License",
    "Irrevocable Or Perpetual License", "Source Code Escrow",
    "Post-Termination Services", "Audit Rights", "Uncapped Liability",
    "Cap On Liability", "Liquidated Damages", "Warranty Duration", "Insurance",
    "Covenant Not To Sue", "Third Party Beneficiary", "Indemnification",
]

# Metadata clauses (not risk-relevant, dropped from modeling)
METADATA_CLAUSES = {
    "Document Name", "Parties", "Agreement Date", "Effective Date", "Expiration Date",
}
RISK_CATEGORIES = [c for c in CUAD_CATEGORIES if c not in METADATA_CLAUSES]

# Legal risk taxonomy — what corporate lawyers flag in due diligence
HIGH_RISK_CLAUSES = [
    "Uncapped Liability",
    "IP Ownership Assignment",
    "Change Of Control",
    "Non-Compete",
    "Liquidated Damages",
    "Joint IP Ownership",
]
MEDIUM_RISK_CLAUSES = [
    "Indemnification",
    "Cap On Liability",
    "Termination For Convenience",
    "Exclusivity",
    "No-Solicit Of Employees",
    "No-Solicit Of Customers",
    "Revenue/Profit Sharing",
    "Most Favored Nation",
    "Covenant Not To Sue",
]


def risk_level(clause: str) -> str:
    """Return HIGH, MEDIUM, or LOW for a clause name."""
    if clause in HIGH_RISK_CLAUSES:
        return "HIGH"
    if clause in MEDIUM_RISK_CLAUSES:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_QUESTION_TO_CATEGORY = {
    "document name": "Document Name",
    "parties": "Parties",
    "agreement date": "Agreement Date",
    "effective date": "Effective Date",
    "expiration date": "Expiration Date",
    "renewal term": "Renewal Term",
    "notice period to terminate renewal": "Notice Period To Terminate Renewal",
    "governing law": "Governing Law",
    "most favored nation": "Most Favored Nation",
    "non-compete": "Non-Compete",
    "exclusivity": "Exclusivity",
    "no-solicit of customers": "No-Solicit Of Customers",
    "no-solicit of employees": "No-Solicit Of Employees",
    "non-disparagement": "Non-Disparagement",
    "termination for convenience": "Termination For Convenience",
    "rofr": "ROFR/ROFO/ROFN",
    "rofo": "ROFR/ROFO/ROFN",
    "rofn": "ROFR/ROFO/ROFN",
    "change of control": "Change Of Control",
    "anti-assignment": "Anti-Assignment",
    "revenue/profit sharing": "Revenue/Profit Sharing",
    "price restrictions": "Price Restrictions",
    "minimum commitment": "Minimum Commitment",
    "volume restriction": "Volume Restriction",
    "ip ownership assignment": "IP Ownership Assignment",
    "joint ip ownership": "Joint IP Ownership",
    "license grant": "License Grant",
    "non-transferable license": "Non-Transferable License",
    "affiliate license-licensor": "Affiliate License-Licensor",
    "affiliate license-licensee": "Affiliate License-Licensee",
    "unlimited/all-you-can-eat": "Unlimited/All-You-Can-Eat-License",
    "irrevocable or perpetual license": "Irrevocable Or Perpetual License",
    "source code escrow": "Source Code Escrow",
    "post-termination services": "Post-Termination Services",
    "audit rights": "Audit Rights",
    "uncapped liability": "Uncapped Liability",
    "cap on liability": "Cap On Liability",
    "liquidated damages": "Liquidated Damages",
    "warranty duration": "Warranty Duration",
    "insurance": "Insurance",
    "covenant not to sue": "Covenant Not To Sue",
    "third party beneficiary": "Third Party Beneficiary",
    "indemnification": "Indemnification",
}


def _map_question(question: str):
    q = question.lower().strip()
    for key, cat in _QUESTION_TO_CATEGORY.items():
        if key in q:
            return cat
    return None


def load_cuad_from_json(path: str | Path) -> pd.DataFrame:
    """Load CUAD SQuAD-format JSON into multi-label DataFrame.

    Each row = one contract. Columns:
      contract_title (str), text (str), text_length (int), <one column per clause type>.
    """
    path = Path(path)
    with open(path) as f:
        raw = json.load(f)

    rows = []
    for entry in raw["data"]:
        title = entry["title"]
        # Context is the same across paragraphs for one contract — take first
        text = entry["paragraphs"][0]["context"]
        labels: Dict[str, int] = {}
        for para in entry["paragraphs"]:
            for qa in para["qas"]:
                cat = _map_question(qa["question"])
                if cat is None:
                    continue
                present = bool(qa["answers"]) and any(
                    len(a["text"].strip()) > 0 for a in qa["answers"]
                )
                labels[cat] = max(labels.get(cat, 0), int(present))

        row = {"contract_title": title, "text": text, "text_length": len(text)}
        for cat in CUAD_CATEGORIES:
            row[cat] = labels.get(cat, 0)
        rows.append(row)

    return pd.DataFrame(rows)


def make_split(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = 42, min_positives: int = 3
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Deterministic shuffle split matching Phase 5 convention.

    Returns (train_df, test_df, valid_clauses) where valid_clauses is the list of
    clause types with >= min_positives positives in the TEST set (filter keeps
    evaluation consistent with prior phases).
    """
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(df))
    n_test = int(round(len(df) * test_size))
    test_idx = idx[-n_test:]
    train_idx = idx[:-n_test]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    valid_clauses = [
        c for c in RISK_CATEGORIES if c in df.columns and test_df[c].sum() >= min_positives
    ]
    return train_df, test_df, valid_clauses


def build_vectorizer(
    max_features: int = 20_000, ngram_range: tuple = (1, 2)
) -> TfidfVectorizer:
    """Production TF-IDF vectorizer (Phase 5 ablation-validated config)."""
    return TfidfVectorizer(
        analyzer="word",
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
    )


# ---------------------------------------------------------------------------
# Clause-matching helpers (for UI: highlight the snippet that triggered)
# ---------------------------------------------------------------------------

# Regex patterns used purely for UI HIGHLIGHTING — NOT for classification.
# Classification = LGBM+LR. These just surface the snippet that most likely
# triggered a positive prediction so a lawyer can verify.
CLAUSE_HIGHLIGHT_PATTERNS = {
    "Uncapped Liability": r"(?:unlimited\s+liability|no\s+(?:cap|limitation)\s+(?:on|for)\s+liability|"
                          r"liability\s+(?:shall|will)\s+not\s+be\s+limited|"
                          r"unlimited\s+(?:damages|obligation))",
    "Cap On Liability": r"(?:cap\s+on\s+liability|aggregate\s+liability\s+shall\s+not\s+exceed|"
                        r"total\s+liability.{0,40}limited|"
                        r"maximum\s+liability.{0,40}(?:exceed|equal))",
    "Liquidated Damages": r"liquidated\s+damages",
    "Non-Compete": r"(?:non[- ]?compete|shall\s+not\s+compete|restrictive\s+covenant)",
    "Change Of Control": r"change\s+of\s+control",
    "Indemnification": r"(?:indemnif(?:y|ication|ies|ied)|hold\s+harmless)",
    "Termination For Convenience": r"terminat(?:e|ion)\s+(?:for|at)\s+(?:convenience|any\s+reason|will)",
    "Exclusivity": r"(?:exclusive\s+(?:right|license|dealer|distributor)|sole\s+and\s+exclusive)",
    "Governing Law": r"govern(?:ing|ed)\s+(?:by\s+the\s+laws|law)",
    "Audit Rights": r"(?:audit\s+right|right\s+to\s+audit)",
    "Anti-Assignment": r"(?:not\s+assign|no\s+assignment|prior\s+written\s+consent.{0,40}assign)",
    "Insurance": r"insurance",
    "IP Ownership Assignment": r"(?:intellectual\s+property.{0,60}(?:assign|own)|"
                                r"assign.{0,60}intellectual\s+property|work\s+(?:product|made\s+for\s+hire))",
    "License Grant": r"(?:grants?\s+(?:to|a).{0,40}license|hereby\s+grants)",
    "ROFR/ROFO/ROFN": r"right\s+of\s+first\s+(?:refusal|offer|negotiation)",
    "Warranty Duration": r"warrant(?:y|ies).{0,40}(?:period|days|months|year)",
    "Covenant Not To Sue": r"covenant\s+not\s+to\s+sue",
    "Most Favored Nation": r"most\s+favored\s+(?:nation|customer)",
}


def extract_clause_snippet(text: str, clause: str, window: int = 180) -> str | None:
    """Return a short excerpt of `text` around the first regex match for `clause`.

    Used in the UI to show WHICH passage most likely triggered a positive prediction.
    Returns None if no pattern for this clause or no match.
    """
    pattern = CLAUSE_HIGHLIGHT_PATTERNS.get(clause)
    if not pattern:
        return None
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if not m:
        return None
    start = max(0, m.start() - window)
    end = min(len(text), m.end() + window)
    snippet = text[start:end]
    # Collapse whitespace for readability
    snippet = re.sub(r"\s+", " ", snippet).strip()
    # Add ellipses for partial windows
    if start > 0:
        snippet = "… " + snippet
    if end < len(text):
        snippet = snippet + " …"
    return snippet
