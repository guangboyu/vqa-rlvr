"""Counterfactual pairs: per-pair measures and their resampled differences."""

import numpy as np
import pytest

from scripts.pairs import bootstrap, per_pair


def test_a_pair_is_right_only_with_full_credit_on_both_images():
    cell = {"a": ("yes", 1.0), "b": ("no", 1.0), "c": ("2", 1.0), "d": ("2", 0.0)}
    rows = per_pair(cell, [["a", "b"], ["c", "d"]])
    assert rows.tolist() == [[1.0, 0.0, 1.0], [0.0, 1.0, 0.5]]  # second pair: one answer for both


def test_identical_arms_differ_by_zero():
    rows = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.5]])
    assert all(d == {"delta": 0.0, "low": 0.0, "high": 0.0}
               for d in bootstrap(rows, rows, resamples=50))


def test_differences_are_candidate_minus_reference_in_points():
    candidate = np.array([[1.0, 0.0, 1.0]] * 4)
    reference = np.array([[0.0, 1.0, 0.5]] * 4)
    pair_accuracy, same_answer, accuracy = bootstrap(candidate, reference, resamples=20)
    assert pair_accuracy["delta"] == pytest.approx(100.0)
    assert same_answer["delta"] == pytest.approx(-100.0)
    assert accuracy["delta"] == pytest.approx(50.0)
