import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from unblind_and_analyze import (  # noqa: E402
    bootstrap_mean_ci,
    classification_metrics,
    paired_permutation_pvalue,
    wilson_interval,
)


def test_wilson_interval_is_bounded_and_contains_estimate():
    low, high = wilson_interval(33, 36)
    assert 0 <= low < 33 / 36 < high <= 1


def test_classification_metrics_match_known_confusion():
    truth = ["n", "n", "c", "c", "g", "g"]
    pred = ["n", "n", "n", "c", "g", "g"]
    result = classification_metrics(truth, pred, ["n", "c", "g"])
    assert result["accuracy"] == pytest.approx(5 / 6)
    assert result["confusion_matrix"]["c"]["n"] == 1
    assert result["per_class"]["g"]["f1"] == 1


def test_exact_paired_permutation_for_all_positive_values():
    assert paired_permutation_pvalue([1, 2, 3], "greater") == pytest.approx(1 / 8)


def test_bootstrap_is_deterministic():
    first = bootstrap_mean_ci([10, 20, 30], reps=1000, seed=7)
    second = bootstrap_mean_ci([10, 20, 30], reps=1000, seed=7)
    assert first == second
