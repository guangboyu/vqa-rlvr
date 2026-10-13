"""Calibration measures for the beyond-accuracy tables: majority answer, ECE, selective."""

import pytest

from scripts.beyond_accuracy import calibration_error, majority, selective_accuracy


def line(qid, sample, prediction, score):
    return {"qid": qid, "sample": sample, "prediction": prediction,
            "scores": {"normalized": score}}


def test_the_majority_answer_and_its_share_of_the_first_samples():
    lines = [line("a", i, "Yes" if i < 6 else "no", 1.0 if i < 6 else 0.0) for i in range(8)]
    lines += [line("a", 8, "no", 0.0)]  # beyond the first eight samples: ignored
    lines += [line("b", i, "two" if i < 3 else "3", 0.0 if i < 3 else 1.0) for i in range(8)]
    assert majority(lines) == {"a": (0.75, 1.0), "b": (0.625, 1.0)}


def test_calibration_error_weights_each_bin_by_its_share():
    assert calibration_error([(1.0, 1.0), (1.0, 1.0)]) == 0.0
    # one bin at 0.9 confidence with half right, one at 0.5 with all right
    pairs = [(0.9, 1.0), (0.9, 0.0), (0.5, 1.0), (0.5, 1.0)]
    assert calibration_error(pairs) == pytest.approx(0.5 * 0.4 + 0.5 * 0.5)


def test_selective_accuracy_keeps_the_most_confident_share():
    pairs = [(1.0, 1.0), (0.9, 1.0), (0.6, 0.0), (0.5, 0.0)]
    assert selective_accuracy(pairs, 0.5) == 1.0
    assert selective_accuracy(pairs, 1.0) == 0.5
