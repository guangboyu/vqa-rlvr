"""Paired bootstrap comparisons over questions."""

import json

import pytest

from vqar.eval.paired import paired_bootstrap, question_scores


def test_identical_records_differ_by_exactly_zero():
    scores = {f"q{i}": float(i % 2) for i in range(50)}
    result = paired_bootstrap(scores, dict(scores), resamples=200)
    assert (result["delta"], result["low"], result["high"]) == (0.0, 0.0, 0.0)


def test_a_uniform_gain_is_recovered_with_a_tight_interval():
    reference = {f"q{i}": 0.0 for i in range(100)}
    candidate = {q: 1.0 if i < 30 else 0.0 for i, q in enumerate(reference)}
    result = paired_bootstrap(reference, candidate, resamples=2_000)
    assert result["delta"] == pytest.approx(30.0)
    assert 20.0 < result["low"] < 30.0 < result["high"] < 40.0
    assert paired_bootstrap(reference, candidate, resamples=2_000) == result  # seeded


def test_different_questions_cannot_be_paired():
    with pytest.raises(ValueError, match="same questions"):
        paired_bootstrap({"a": 1.0}, {"b": 1.0})


def test_samples_of_one_question_are_averaged(tmp_path):
    path = tmp_path / "cell.predictions.jsonl"
    rows = [{"qid": "q", "scores": {"official": s}} for s in (1.0, 0.0, 1.0, 1.0)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert question_scores(path, "official") == {"q": 0.75}
