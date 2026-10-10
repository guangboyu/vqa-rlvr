"""Credit rules for rescoring exact-match misses with the judge and its audit."""

import pytest

from scripts.judge_records import is_miss, scorers_for, scores, stratum

ROWS = [
    {"qid": "hit", "prediction": "gray", "official": 1.0},
    {"qid": "partial", "prediction": "grey", "official": 0.3},
    {"qid": "rescued", "prediction": "Grey", "official": 0.0},
    {"qid": "rejected", "prediction": "blue sky above", "official": 0.0},
    {"qid": "empty", "prediction": " ", "official": 0.0},
]
VERDICTS = {"rescued": True, "rejected": False}


def verdict(row):
    return VERDICTS.get(row["qid"])


def test_only_non_empty_zero_score_answers_are_misses():
    assert [row["qid"] for row in ROWS if is_miss(row)] == ["rescued", "rejected"]


def test_official_keeps_every_score():
    assert scores("official", ROWS, verdict, "gqa", None) == {
        "hit": 1.0, "partial": 0.3, "rescued": 0.0, "rejected": 0.0, "empty": 0.0}


def test_adjusted_gives_a_rescued_miss_full_credit_and_nothing_else():
    assert scores("adjusted", ROWS, verdict, "gqa", None) == {
        "hit": 1.0, "partial": 0.3, "rescued": 1.0, "rejected": 0.0, "empty": 0.0}


def test_audited_discounts_accepted_misses_by_their_confirmation_rate():
    rates = {"gqa|accepted|1 word": 0.9, "gqa|rejected": 0.05}
    credit = scores("audited", ROWS, verdict, "gqa", rates)
    assert credit["rescued"] == pytest.approx(0.9)
    assert credit["rejected"] == 0.0  # the judge's rejections are not corrected upward
    assert (credit["hit"], credit["partial"], credit["empty"]) == (1.0, 0.3, 0.0)


def test_strata_split_accepted_misses_by_answer_length():
    assert stratum("vqav2", True, "yes") == "vqav2|accepted|1 word"
    assert stratum("vqav2", True, "a red double decker bus") == "vqav2|accepted|2-5 words"
    assert stratum("vqav2", True, "on the left side of the clock") == "vqav2|accepted|6+ words"
    assert stratum("vqav2", False, "anything") == "vqav2|rejected"


def test_audited_scorer_only_for_datasets_the_audit_sampled():
    rates = {"gqa|accepted|1 word": 0.9, "gqa|rejected": 0.05}
    assert scorers_for("gqa", None) == ("official", "adjusted")
    assert scorers_for("clevr", rates) == ("official", "adjusted")
    assert scorers_for("gqa", rates) == ("official", "adjusted", "audited")
