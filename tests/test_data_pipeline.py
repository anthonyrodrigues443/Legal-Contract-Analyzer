"""
Tests for data_pipeline.py — clause taxonomy, preprocessing utilities.
Does NOT hit HuggingFace (no network calls).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_pipeline import (
    CUAD_CATEGORIES,
    HIGH_RISK_CLAUSES,
    MEDIUM_RISK_CLAUSES,
    RISK_CATEGORIES,
    map_question_to_category,
)


# ---------------------------------------------------------------------------
# Taxonomy integrity
# ---------------------------------------------------------------------------

def test_cuad_categories_count():
    assert len(CUAD_CATEGORIES) == 41, f"Expected 41 CUAD categories, got {len(CUAD_CATEGORIES)}"


def test_risk_categories_exclude_metadata():
    metadata = {"Document Name", "Parties", "Agreement Date", "Effective Date", "Expiration Date"}
    for m in metadata:
        assert m not in RISK_CATEGORIES, f"Metadata category '{m}' should not be in RISK_CATEGORIES"


def test_risk_categories_subset_of_cuad():
    cuad_set = set(CUAD_CATEGORIES)
    for cat in RISK_CATEGORIES:
        assert cat in cuad_set, f"'{cat}' in RISK_CATEGORIES but not in CUAD_CATEGORIES"


def test_high_risk_clauses_in_cuad():
    cuad_set = set(CUAD_CATEGORIES)
    for clause in HIGH_RISK_CLAUSES:
        assert clause in cuad_set, f"High-risk clause '{clause}' missing from CUAD_CATEGORIES"


def test_medium_risk_clauses_in_cuad():
    cuad_set = set(CUAD_CATEGORIES)
    for clause in MEDIUM_RISK_CLAUSES:
        assert clause in cuad_set, f"Medium-risk clause '{clause}' missing from CUAD_CATEGORIES"


def test_no_overlap_high_medium_risk():
    overlap = set(HIGH_RISK_CLAUSES) & set(MEDIUM_RISK_CLAUSES)
    assert not overlap, f"Clauses appear in both high and medium risk lists: {overlap}"


# ---------------------------------------------------------------------------
# Question → category mapping
# ---------------------------------------------------------------------------

KNOWN_MAPPINGS = [
    ("non-compete", "Non-Compete"),
    ("governing law", "Governing Law"),
    ("indemnification", "Indemnification"),
    ("uncapped liability", "Uncapped Liability"),
    ("liquidated damages", "Liquidated Damages"),
    ("change of control", "Change Of Control"),
    ("termination for convenience", "Termination For Convenience"),
    ("exclusivity", "Exclusivity"),
    ("audit rights", "Audit Rights"),
    ("non-disparagement", "Non-Disparagement"),
]

@pytest.mark.parametrize("question, expected_category", KNOWN_MAPPINGS)
def test_map_question_to_category(question, expected_category):
    result = map_question_to_category(question)
    assert result == expected_category, (
        f"map_question_to_category('{question}') = {result!r}, expected {expected_category!r}"
    )


def test_map_question_unknown_returns_none():
    result = map_question_to_category("does the contract have a unicorn clause?")
    assert result is None, f"Unknown question should return None, got {result!r}"


def test_map_question_case_insensitive():
    assert map_question_to_category("NON-COMPETE") == "Non-Compete"
    assert map_question_to_category("Governing Law") == "Governing Law"


# ---------------------------------------------------------------------------
# Processed data sanity (loads from parquet without HuggingFace)
# ---------------------------------------------------------------------------

PARQUET_PATH = ROOT / "data" / "processed" / "cuad_classification.parquet"

@pytest.mark.skipif(not PARQUET_PATH.exists(), reason="Processed data not available")
class TestProcessedData:
    @pytest.fixture(scope="class")
    def df(self):
        return pd.read_parquet(PARQUET_PATH)

    def test_row_count(self, df):
        assert len(df) == 510, f"Expected 510 contracts, got {len(df)}"

    def test_has_text_column(self, df):
        assert "text" in df.columns
        assert df["text"].notna().all()

    def test_text_length_positive(self, df):
        assert (df["text"].str.len() > 0).all()

    def test_high_risk_columns_present(self, df):
        # Parquet may use title-case for some acronyms (e.g. "Ip" vs "IP").
        # Match case-insensitively.
        lower_cols = {c.lower() for c in df.columns}
        for clause in HIGH_RISK_CLAUSES:
            assert clause.lower() in lower_cols, \
                f"Column matching '{clause}' missing from processed dataframe"

    def test_binary_labels(self, df):
        label_cols = [c for c in df.columns if c in CUAD_CATEGORIES]
        for col in label_cols:
            unique = set(df[col].unique())
            assert unique <= {0, 1}, f"Column '{col}' has non-binary values: {unique}"

    def test_class_imbalance_exists(self, df):
        label_cols = [c for c in df.columns if c in CUAD_CATEGORIES and c in RISK_CATEGORIES]
        positive_rates = {c: df[c].mean() for c in label_cols}
        assert any(r < 0.5 for r in positive_rates.values()), \
            "Expected class imbalance (some clauses should be rare)"

    def test_contract_titles_unique(self, df):
        assert df["contract_title"].nunique() == len(df), "Contract titles should be unique"
