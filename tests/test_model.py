"""
Tests for the trained Phase 5 champion bundle.

Bundle = vectorizer.joblib + lgbm_models.joblib + thresholds.json + valid_clauses.json.
No LR blend (removed in Phase 6 rework — hurt F1 by -0.038 on 40K trigrams).
Loads via src.predict.load_model().
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REQUIRED = [
    ROOT / "models" / "vectorizer.joblib",
    ROOT / "models" / "lgbm_models.joblib",
    ROOT / "models" / "thresholds.json",
    ROOT / "models" / "valid_clauses.json",
]

pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in REQUIRED),
    reason="Production artifacts not found — run `python -m src.train` first",
)


@pytest.fixture(scope="module")
def bundle():
    from src.predict import load_model
    return load_model()


SAMPLE_TEXTS = [
    "This agreement shall be governed by the laws of Delaware.",
    (
        "NON-COMPETE. For two years thereafter, vendor shall not compete. "
        "LIABILITY. Vendor liability is unlimited without cap. "
        "CHANGE OF CONTROL. Upon acquisition, this agreement terminates. "
        "TERMINATION. Either party may terminate for convenience."
    ),
    "The parties agree to the terms set forth herein.",
]


# ---------------------------------------------------------------------------
# Bundle structure
# ---------------------------------------------------------------------------

def test_bundle_has_required_keys(bundle):
    required = {"vectorizer", "lgbm_models", "thresholds", "valid_clauses"}
    missing = required - set(bundle.keys())
    assert not missing, f"Bundle missing keys: {missing}"


def test_valid_clauses_nonempty(bundle):
    assert len(bundle["valid_clauses"]) >= 20, (
        f"Expected ≥20 valid clauses, got {len(bundle['valid_clauses'])}"
    )


def test_thresholds_cover_all_valid_clauses(bundle):
    for clause in bundle["valid_clauses"]:
        assert clause in bundle["thresholds"], f"Missing threshold for '{clause}'"
        t = bundle["thresholds"][clause]
        assert 0.0 <= t <= 1.0, f"Threshold for '{clause}' = {t}, must be in [0,1]"


def test_thresholds_are_class_priors(bundle):
    """Phase 5 champion: thresholds = training positive rate per clause.
    All thresholds should be reasonable priors (between 0.01 and 0.95).
    """
    for clause, t in bundle["thresholds"].items():
        assert 0.01 <= t <= 0.95, (
            f"Threshold for '{clause}' = {t} looks unreasonable for a class prior"
        )


def test_lgbm_models_is_list(bundle):
    """Phase 5 champion stores LGBM models as a list indexed by clause order."""
    assert isinstance(bundle["lgbm_models"], list)
    assert len(bundle["lgbm_models"]) == len(bundle["valid_clauses"])


def test_lgbm_models_mostly_fitted(bundle):
    n_fitted = sum(m is not None for m in bundle["lgbm_models"])
    assert n_fitted >= len(bundle["valid_clauses"]) - 2, (
        f"Only {n_fitted}/{len(bundle['valid_clauses'])} models fitted — too many None entries"
    )


# ---------------------------------------------------------------------------
# Vectorizer — Phase 5 champion config: 40K word 1-3gram
# ---------------------------------------------------------------------------

def test_vectorizer_transforms_text(bundle):
    vec = bundle["vectorizer"]
    X = vec.transform(SAMPLE_TEXTS)
    assert X.shape[0] == len(SAMPLE_TEXTS)
    assert X.shape[1] > 0


def test_vectorizer_vocab_size_is_40k(bundle):
    """Phase 5 ablation winner: 40K word 1-3gram."""
    vocab = bundle["vectorizer"].vocabulary_
    assert 30_000 <= len(vocab) <= 40_000, (
        f"Expected 30K–40K vocab (Phase 5 champion = 40K), got {len(vocab)}"
    )


def test_vectorizer_ngram_range_is_1_to_3(bundle):
    """Phase 4 ablation: trigrams beat positional by +0.025 macro-F1."""
    vec = bundle["vectorizer"]
    assert vec.ngram_range == (1, 3), (
        f"Expected ngram_range=(1,3), got {vec.ngram_range}"
    )


def test_vectorizer_sparse_output(bundle):
    import scipy.sparse
    X = bundle["vectorizer"].transform(SAMPLE_TEXTS[:1])
    assert scipy.sparse.issparse(X), "TF-IDF output should be sparse"


def test_vocabulary_contains_legal_trigrams(bundle):
    """Phase 6 explainability: exact legal trigrams like 'change of control' must be in vocab."""
    vocab = bundle["vectorizer"].vocabulary_
    expected_phrases = ["change of control", "liquidated damages", "intellectual property"]
    for phrase in expected_phrases:
        assert phrase in vocab, f"Legal phrase '{phrase}' missing from 40K vocabulary"


# ---------------------------------------------------------------------------
# Prediction shape and types
# ---------------------------------------------------------------------------

def test_lgbm_predict_proba_shape(bundle):
    vec = bundle["vectorizer"]
    Xmat = vec.transform(SAMPLE_TEXTS)
    fitted = [m for m in bundle["lgbm_models"] if m is not None][:5]
    for m in fitted:
        proba = m.predict_proba(Xmat)
        assert proba.shape == (len(SAMPLE_TEXTS), 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_all_probabilities_in_range(bundle):
    vec = bundle["vectorizer"]
    Xmat = vec.transform(SAMPLE_TEXTS)
    for m in bundle["lgbm_models"]:
        if m is None:
            continue
        proba = m.predict_proba(Xmat)[:, 1]
        assert proba.min() >= 0.0 and proba.max() <= 1.0


# ---------------------------------------------------------------------------
# Training manifest — Phase 5 champion honest metrics
# ---------------------------------------------------------------------------

MANIFEST_PATH = ROOT / "models" / "training_manifest.json"


@pytest.mark.skipif(not MANIFEST_PATH.exists(), reason="training_manifest.json not found")
class TestTrainingManifest:
    @pytest.fixture(scope="class")
    def meta(self):
        import json
        return json.loads(MANIFEST_PATH.read_text())

    def test_pipeline_identifies_as_phase5_champion(self, meta):
        assert "Phase 5 champion" in meta.get("pipeline", "") or \
               "Phase 6 rework" in meta.get("pipeline", ""), (
            f"Expected Phase 5/6 pipeline, got '{meta.get('pipeline')}'"
        )

    def test_tfidf_config_is_40k_trigrams(self, meta):
        assert meta.get("tfidf_max_features") == 40_000
        assert meta.get("tfidf_ngram_range") == [1, 3]

    def test_macro_f1_at_sota_parity(self, meta):
        """Phase 5 honest champion: matches RoBERTa-large SOTA (~0.65) within noise."""
        f1 = meta["test_metrics"]["macro_f1"]
        assert f1 >= 0.63, (
            f"macro-F1 {f1:.4f} dropped below expected Phase 5 champion range (~0.647)"
        )

    def test_macro_auc_high(self, meta):
        """Ranking quality should be strong even when thresholding is imperfect."""
        auc = meta["test_metrics"]["macro_auc"]
        assert auc >= 0.85, f"macro-AUC {auc:.4f} below expected range"

    def test_hr_f1_beats_llm_baseline(self, meta):
        """Phase 5 champion beats Claude zero-shot HR-F1 of 0.162 by ~3.6x."""
        hr_f1 = meta["test_metrics"]["hr_f1"]
        assert hr_f1 >= 0.50, f"HR-F1 {hr_f1:.4f} below expected 0.58"

    def test_n_clauses_matches_valid(self, meta):
        assert meta.get("n_clauses") == 28  # Phase 5 champion produces 28 valid clauses
