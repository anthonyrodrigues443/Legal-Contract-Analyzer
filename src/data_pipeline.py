"""
Data pipeline for Legal Contract Analyzer.
Loads CUAD dataset, preprocesses contracts, and prepares features.
"""

import os
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional


# All 41 CUAD clause categories
CUAD_CATEGORIES = [
    "Document Name",
    "Parties",
    "Agreement Date",
    "Effective Date",
    "Expiration Date",
    "Renewal Term",
    "Notice Period To Terminate Renewal",
    "Governing Law",
    "Most Favored Nation",
    "Non-Compete",
    "Exclusivity",
    "No-Solicit Of Customers",
    "No-Solicit Of Employees",
    "Non-Disparagement",
    "Termination For Convenience",
    "ROFR/ROFO/ROFN",
    "Change Of Control",
    "Anti-Assignment",
    "Revenue/Profit Sharing",
    "Price Restrictions",
    "Minimum Commitment",
    "Volume Restriction",
    "IP Ownership Assignment",
    "Joint IP Ownership",
    "License Grant",
    "Non-Transferable License",
    "Affiliate License-Licensor",
    "Affiliate License-Licensee",
    "Unlimited/All-You-Can-Eat-License",
    "Irrevocable Or Perpetual License",
    "Source Code Escrow",
    "Post-Termination Services",
    "Audit Rights",
    "Uncapped Liability",
    "Cap On Liability",
    "Liquidated Damages",
    "Warranty Duration",
    "Insurance",
    "Covenant Not To Sue",
    "Third Party Beneficiary",
    "Indemnification",
]

# Remove "Document Name", "Parties", "Agreement Date", "Effective Date", "Expiration Date"
# as these are metadata, not risk-relevant clauses
RISK_CATEGORIES = [c for c in CUAD_CATEGORIES if c not in [
    "Document Name", "Parties", "Agreement Date", "Effective Date", "Expiration Date"
]]

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
]


def load_cuad_dataset(cache_dir: str = "data/raw") -> dict:
    """Load CUAD dataset from HuggingFace (lazy import of `datasets`)."""
    from datasets import load_dataset  # lazy — tests don't need this
    print("Loading CUAD dataset from HuggingFace...")
    dataset = load_dataset("theatticusproject/cuad", trust_remote_code=True, cache_dir=cache_dir)
    print(f"Dataset loaded: {dataset}")
    return dataset


def cuad_to_classification_df(dataset) -> pd.DataFrame:
    """
    Convert CUAD QA format to multi-label classification format.
    For each contract, determine which of the 41 clause types are present.

    CUAD has one entry per (contract, question) pair.
    We group by contract and determine clause presence via non-empty answers.
    """
    rows = []

    # Process train split (CUAD only has train split, we'll do our own split)
    split = dataset["train"]

    # Group by contract title
    contracts = {}
    for example in split:
        title = example["title"]
        if title not in contracts:
            contracts[title] = {
                "text": example["context"],
                "labels": {}
            }

        # Extract question type (which clause category)
        question = example["question"]
        # Find which category this question maps to
        category = map_question_to_category(question)
        if category:
            # Present = at least one non-empty answer span
            is_present = len(example["answers"]["text"]) > 0 and any(
                len(a.strip()) > 0 for a in example["answers"]["text"]
            )
            contracts[title]["labels"][category] = int(is_present)

    # Build dataframe
    for title, data in contracts.items():
        row = {"contract_title": title, "text": data["text"], "text_length": len(data["text"])}
        for cat in CUAD_CATEGORIES:
            row[cat] = data["labels"].get(cat, 0)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def map_question_to_category(question: str) -> Optional[str]:
    """Map a CUAD question string to a clause category name."""
    question_lower = question.lower().strip()

    # Simple keyword mapping based on CUAD question types
    mapping = {
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

    for key, category in mapping.items():
        if key in question_lower:
            return category

    return None


def extract_text_features(text: str) -> dict:
    """
    Extract domain-informed features from contract text.
    These are features a corporate lawyer would consider:
    - Contract length indicators
    - Key legal keyword presence
    - Structural features
    """
    features = {}

    text_lower = text.lower()

    # Length features
    features["char_count"] = len(text)
    features["word_count"] = len(text.split())
    features["sentence_count"] = len(re.split(r'[.!?]+', text))
    features["avg_sentence_length"] = features["word_count"] / max(features["sentence_count"], 1)

    # Section count (contracts organized by numbered sections)
    features["section_count"] = len(re.findall(r'\n\s*\d+\.', text))
    features["article_count"] = len(re.findall(r'\bARTICLE\b', text, re.IGNORECASE))

    # Capitalized headers (legal documents have ALL CAPS section headers)
    features["caps_headers"] = len(re.findall(r'\n[A-Z][A-Z\s]{5,}\n', text))

    # High-risk clause keywords
    features["has_indemnification_kw"] = int("indemnif" in text_lower)
    features["has_liability_kw"] = int("liabilit" in text_lower)
    features["has_noncompete_kw"] = int("non-compet" in text_lower or "noncompet" in text_lower)
    features["has_ip_assign_kw"] = int("intellectual property" in text_lower and "assign" in text_lower)
    features["has_change_control_kw"] = int("change of control" in text_lower)
    features["has_termination_kw"] = int("terminat" in text_lower)
    features["has_governing_law_kw"] = int("governing law" in text_lower or "choice of law" in text_lower)
    features["has_arbitration_kw"] = int("arbitrat" in text_lower)
    features["has_exclusivity_kw"] = int("exclusiv" in text_lower)
    features["has_warranty_kw"] = int("warrant" in text_lower)
    features["has_liquidated_kw"] = int("liquidated damages" in text_lower)
    features["has_force_majeure_kw"] = int("force majeure" in text_lower)
    features["has_confidential_kw"] = int("confidential" in text_lower)
    features["has_audit_kw"] = int("audit right" in text_lower)

    # Risk indicators
    features["n_shall_mentions"] = text_lower.count("shall")
    features["n_may_mentions"] = text_lower.count(" may ")
    features["n_must_mentions"] = text_lower.count(" must ")
    features["n_will_not_mentions"] = text_lower.count("will not") + text_lower.count("shall not")

    return features


def prepare_classification_data(df: pd.DataFrame,
                                 target_category: str,
                                 text_col: str = "text") -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare X, y for binary classification of a single clause type."""
    y = df[target_category]
    X = df[[text_col, "text_length"] + [c for c in df.columns
                                          if c not in CUAD_CATEGORIES + [text_col, "contract_title", "text_length"]]]
    return X, y


if __name__ == "__main__":
    dataset = load_cuad_dataset()
    df = cuad_to_classification_df(dataset)
    print(f"\nDataset shape: {df.shape}")
    print(f"Contracts: {len(df)}")
    print("\nClause presence rates:")
    for cat in RISK_CATEGORIES:
        if cat in df.columns:
            rate = df[cat].mean()
            print(f"  {cat}: {rate:.1%}")
