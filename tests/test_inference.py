"""
End-to-end inference tests for the Phase 5 champion pipeline.

Exercises src.predict.predict() with real contract text. Tests output structure,
probability ranges, latency, and semantic correctness (positive contracts get
higher probabilities than benign ones on the relevant clauses).
"""
import sys
import time
from pathlib import Path

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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bundle():
    from src.predict import load_model
    return load_model()


@pytest.fixture(scope="module")
def predict_fn():
    from src.predict import predict
    return predict


HIGH_RISK_CONTRACT = """
MASTER SERVICES AGREEMENT

1. NON-COMPETE. During the term of this Agreement and for two (2) years thereafter,
   Vendor shall not, directly or indirectly, engage in any business that competes
   with Customer's core business activities in the United States.

2. UNLIMITED LIABILITY. NOTWITHSTANDING ANYTHING TO THE CONTRARY, VENDOR'S TOTAL
   LIABILITY UNDER THIS AGREEMENT SHALL BE UNLIMITED. All consequential, indirect,
   and special damages are fully recoverable without limitation or cap.

3. CHANGE OF CONTROL. If Customer undergoes a change of control, merger, or
   acquisition, this Agreement shall automatically terminate unless Vendor provides
   written consent within 15 days.

4. LIQUIDATED DAMAGES. In the event of a material breach, Vendor shall pay
   liquidated damages in the amount of $500,000 per day of non-compliance.

5. IP OWNERSHIP. All work product and intellectual property created by Vendor
   shall be assigned to and become the exclusive property of Customer. Vendor
   hereby assigns all rights, title, and interest to Customer.

6. TERMINATION. Either party may terminate for convenience upon 30 days written
   notice. This Agreement is governed by the laws of the State of Delaware.
"""

BENIGN_CONTRACT = """
MUTUAL NONDISCLOSURE AGREEMENT

This Mutual Non-Disclosure Agreement is entered into between two parties.

1. CONFIDENTIAL INFORMATION. Each party may disclose confidential information to
   the other solely for the purpose of evaluating a potential business relationship.

2. NON-DISCLOSURE. Each party agrees to hold the other's confidential information
   in strict confidence and not disclose it to any third parties.

3. TERM. This Agreement shall remain in effect for one year from execution and
   shall be governed by the laws of California.
"""


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------

def test_predict_returns_dict(bundle, predict_fn):
    assert isinstance(predict_fn(HIGH_RISK_CONTRACT, bundle), dict)


def test_output_has_required_keys(bundle, predict_fn):
    result = predict_fn(HIGH_RISK_CONTRACT, bundle)
    required = {"clauses", "detected_count", "high_risk_detected", "overall_risk",
                "latency_ms", "word_count"}
    assert required.issubset(set(result.keys()))


def test_clauses_dict_nonempty(bundle, predict_fn):
    result = predict_fn(HIGH_RISK_CONTRACT, bundle)
    assert len(result["clauses"]) > 0


def test_per_clause_has_required_fields(bundle, predict_fn):
    result = predict_fn(HIGH_RISK_CONTRACT, bundle)
    for clause, info in result["clauses"].items():
        for field in ("probability", "detected", "threshold", "risk_level"):
            assert field in info, f"Clause '{clause}' missing '{field}'"


def test_probabilities_in_range(bundle, predict_fn):
    result = predict_fn(HIGH_RISK_CONTRACT, bundle)
    for clause, info in result["clauses"].items():
        p = info["probability"]
        assert 0.0 <= p <= 1.0, f"Clause '{clause}' prob {p} out of [0,1]"


def test_detected_flag_consistent_with_probability(bundle, predict_fn):
    result = predict_fn(HIGH_RISK_CONTRACT, bundle)
    for clause, info in result["clauses"].items():
        expected = info["probability"] >= info["threshold"]
        assert info["detected"] == expected


def test_detected_count_matches_clauses(bundle, predict_fn):
    result = predict_fn(HIGH_RISK_CONTRACT, bundle)
    actual = sum(1 for v in result["clauses"].values() if v["detected"])
    assert result["detected_count"] == actual


def test_high_risk_detected_is_subset_of_detected(bundle, predict_fn):
    result = predict_fn(HIGH_RISK_CONTRACT, bundle)
    for clause in result["high_risk_detected"]:
        assert result["clauses"][clause]["detected"]


def test_overall_risk_is_valid_level(bundle, predict_fn):
    result = predict_fn(HIGH_RISK_CONTRACT, bundle)
    assert result["overall_risk"] in {"HIGH", "MEDIUM", "LOW"}


def test_word_count_correct(bundle, predict_fn):
    result = predict_fn(HIGH_RISK_CONTRACT, bundle)
    assert result["word_count"] == len(HIGH_RISK_CONTRACT.split())


def test_risk_level_labels_valid(bundle, predict_fn):
    result = predict_fn(HIGH_RISK_CONTRACT, bundle)
    for clause, info in result["clauses"].items():
        assert info["risk_level"] in {"HIGH", "MEDIUM", "STANDARD"}


# ---------------------------------------------------------------------------
# Semantic correctness
# ---------------------------------------------------------------------------

def test_high_risk_contract_has_more_detections_than_benign(bundle, predict_fn):
    r_high = predict_fn(HIGH_RISK_CONTRACT, bundle)
    r_benign = predict_fn(BENIGN_CONTRACT, bundle)
    assert r_high["detected_count"] >= r_benign["detected_count"]


def test_high_risk_contract_not_low_risk_overall(bundle, predict_fn):
    """A contract with explicit unlimited liability + non-compete + CoC cannot be LOW."""
    result = predict_fn(HIGH_RISK_CONTRACT, bundle)
    assert result["overall_risk"] in {"HIGH", "MEDIUM"}


def test_non_compete_prob_higher_on_non_compete_contract(bundle, predict_fn):
    r_high = predict_fn(HIGH_RISK_CONTRACT, bundle)
    r_benign = predict_fn(BENIGN_CONTRACT, bundle)
    nc_high = r_high["clauses"].get("Non-Compete", {}).get("probability", 0)
    nc_benign = r_benign["clauses"].get("Non-Compete", {}).get("probability", 0)
    assert nc_high > nc_benign, (
        f"Non-Compete prob should rank higher on HR contract ({nc_high:.3f}) "
        f"than on NDA ({nc_benign:.3f})"
    )


def test_change_of_control_prob_higher_on_hr_contract(bundle, predict_fn):
    r_high = predict_fn(HIGH_RISK_CONTRACT, bundle)
    r_benign = predict_fn(BENIGN_CONTRACT, bundle)
    coc_high = r_high["clauses"].get("Change Of Control", {}).get("probability", 0)
    coc_benign = r_benign["clauses"].get("Change Of Control", {}).get("probability", 0)
    assert coc_high > coc_benign


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------

def test_single_contract_latency_under_3s(bundle, predict_fn):
    t0 = time.time()
    predict_fn(HIGH_RISK_CONTRACT, bundle)
    elapsed_ms = (time.time() - t0) * 1000
    assert elapsed_ms < 3000, f"Single-contract took {elapsed_ms:.0f}ms (limit: 3000ms)"


def test_latency_field_is_positive(bundle, predict_fn):
    result = predict_fn(HIGH_RISK_CONTRACT, bundle)
    assert result["latency_ms"] > 0


def test_batch_ten_contracts_under_10s(bundle, predict_fn):
    """Phase 6 benchmark: ~443ms single, 12ms batched; 10 sequential << 10s."""
    t0 = time.time()
    for _ in range(10):
        predict_fn(HIGH_RISK_CONTRACT, bundle)
    elapsed = time.time() - t0
    assert elapsed < 10, f"10 sequential predictions took {elapsed:.1f}s (limit: 10s)"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_string_does_not_crash(bundle, predict_fn):
    result = predict_fn("", bundle)
    assert "clauses" in result
    assert result["word_count"] == 0


def test_very_short_contract(bundle, predict_fn):
    result = predict_fn("Services shall be provided.", bundle)
    assert "overall_risk" in result


def test_predict_is_deterministic(bundle, predict_fn):
    r1 = predict_fn(HIGH_RISK_CONTRACT, bundle)
    r2 = predict_fn(HIGH_RISK_CONTRACT, bundle)
    for clause in r1["clauses"]:
        assert r1["clauses"][clause]["probability"] == r2["clauses"][clause]["probability"]


def test_different_contracts_produce_different_probabilities(bundle, predict_fn):
    r_high = predict_fn(HIGH_RISK_CONTRACT, bundle)
    r_benign = predict_fn(BENIGN_CONTRACT, bundle)
    high_probs = [v["probability"] for v in r_high["clauses"].values()]
    benign_probs = [v["probability"] for v in r_benign["clauses"].values()]
    assert high_probs != benign_probs
