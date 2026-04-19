"""
Tests for the trained model bundle — loading, structure, and prediction shape.
Requires models/blend_pipeline.joblib (run src/train.py first).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

MODEL_PATH = ROOT / "models" / "blend_pipeline.joblib"

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Model bundle not found — run `python src/train.py` first",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bundle():
    import joblib
    return joblib.load(MODEL_PATH)


SAMPLE_TEXTS = [
    "This agreement shall be governed by the laws of Delaware.",
    (
        "SERVICES. Vendor will provide software. NON-COMPETE. During the term and "
        "for two years thereafter, vendor shall not compete with customer. "
        "LIABILITY. Vendor liability is unlimited. TERMINATION. Either party may "
        "terminate for convenience upon 30 days notice."
    ),
    "The parties agree to the terms set forth herein. No additional obligations.",
]


# ---------------------------------------------------------------------------
# Bundle structure
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {
    "vectorizer", "lgbm_models", "lr_models", "thresholds",
    "valid_clauses", "blend_alpha", "high_risk",
}

def test_bundle_has_required_keys(bundle):
    missing = REQUIRED_KEYS - set(bundle.keys())
    assert not missing, f"Bundle missing keys: {missing}"


def test_blend_alpha_valid(bundle):
    alpha = bundle["blend_alpha"]
    assert 0.0 < alpha < 1.0, f"blend_alpha must be in (0,1), got {alpha}"


def test_valid_clauses_nonempty(bundle):
    assert len(bundle["valid_clauses"]) > 0


def test_valid_clauses_count(bundle):
    assert len(bundle["valid_clauses"]) >= 20, \
        f"Expected ≥20 valid clauses, got {len(bundle['valid_clauses'])}"


def test_high_risk_in_valid_clauses(bundle):
    valid_set = set(bundle["valid_clauses"])
    for clause in bundle["high_risk"]:
        assert clause in valid_set, f"High-risk clause '{clause}' not in valid_clauses"


def test_lgbm_models_trained(bundle):
    assert len(bundle["lgbm_models"]) > 0, "No LightGBM classifiers in bundle"


def test_lr_models_trained(bundle):
    assert len(bundle["lr_models"]) > 0, "No LogisticRegression classifiers in bundle"


def test_thresholds_per_clause(bundle):
    valid_clauses = bundle["valid_clauses"]
    thresholds = bundle["thresholds"]
    for clause in valid_clauses:
        if clause in thresholds:
            t = thresholds[clause]
            assert 0.0 <= t <= 1.0, f"Threshold for '{clause}' = {t}, must be in [0,1]"


# ---------------------------------------------------------------------------
# Vectorizer
# ---------------------------------------------------------------------------

def test_vectorizer_transforms_text(bundle):
    vec = bundle["vectorizer"]
    X = vec.transform(SAMPLE_TEXTS)
    assert X.shape[0] == len(SAMPLE_TEXTS)
    assert X.shape[1] > 0


def test_vectorizer_vocabulary_size(bundle):
    vocab = bundle["vectorizer"].vocabulary_
    assert 5_000 <= len(vocab) <= 100_000, \
        f"Unexpected vocabulary size: {len(vocab)}"


def test_vectorizer_sparse_output(bundle):
    import scipy.sparse
    vec = bundle["vectorizer"]
    X = vec.transform(SAMPLE_TEXTS[:1])
    assert scipy.sparse.issparse(X), "TF-IDF output should be sparse"


# ---------------------------------------------------------------------------
# Prediction shape and types
# ---------------------------------------------------------------------------

def test_lgbm_predict_proba_shape(bundle):
    vec = bundle["vectorizer"]
    lgbm = bundle["lgbm_models"]
    valid_clauses = bundle["valid_clauses"]
    Xmat = vec.transform(SAMPLE_TEXTS)
    for clause in list(lgbm.keys())[:5]:
        proba = lgbm[clause].predict_proba(Xmat)
        assert proba.shape == (len(SAMPLE_TEXTS), 2), \
            f"LightGBM '{clause}' predict_proba shape {proba.shape} unexpected"
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_lr_predict_proba_shape(bundle):
    vec = bundle["vectorizer"]
    lr = bundle["lr_models"]
    Xmat = vec.transform(SAMPLE_TEXTS)
    for clause in list(lr.keys())[:5]:
        proba = lr[clause].predict_proba(Xmat)
        assert proba.shape == (len(SAMPLE_TEXTS), 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_blend_probabilities_in_range(bundle):
    vec = bundle["vectorizer"]
    lgbm = bundle["lgbm_models"]
    lr = bundle["lr_models"]
    valid_clauses = bundle["valid_clauses"]
    alpha = bundle["blend_alpha"]
    Xmat = vec.transform(SAMPLE_TEXTS)

    probs_lgbm = np.zeros((len(SAMPLE_TEXTS), len(valid_clauses)))
    probs_lr = np.zeros((len(SAMPLE_TEXTS), len(valid_clauses)))
    for j, clause in enumerate(valid_clauses):
        if clause in lgbm:
            probs_lgbm[:, j] = lgbm[clause].predict_proba(Xmat)[:, 1]
        if clause in lr:
            probs_lr[:, j] = lr[clause].predict_proba(Xmat)[:, 1]

    blend = alpha * probs_lgbm + (1 - alpha) * probs_lr
    assert blend.min() >= 0.0 and blend.max() <= 1.0, \
        "Blend probabilities must be in [0, 1]"


# ---------------------------------------------------------------------------
# Training meta
# ---------------------------------------------------------------------------

META_PATH = ROOT / "models" / "training_meta.json"

@pytest.mark.skipif(not META_PATH.exists(), reason="training_meta.json not found")
class TestTrainingMeta:
    @pytest.fixture(scope="class")
    def meta(self):
        import json
        return json.loads(META_PATH.read_text())

    def test_beats_roberta(self, meta):
        assert meta.get("beats_roberta") is True, \
            f"Model should beat RoBERTa-large (macro-F1={meta.get('macro_f1_test')})"

    def test_macro_f1_above_threshold(self, meta):
        f1 = meta.get("macro_f1_test", 0.0)
        assert f1 >= 0.65, f"macro-F1 {f1} below published RoBERTa baseline 0.650"

    def test_n_classifiers_reasonable(self, meta):
        n = meta.get("n_classifiers", 0)
        assert 20 <= n <= 50, f"n_classifiers={n} outside expected range [20, 50]"
