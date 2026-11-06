"""The pass@k estimator, sampled-record reading, and paired comparisons."""

import itertools
import json
import sys

import pytest

from vqar.eval.examples import Example
from vqar.eval.pass_at_k import default_ks, estimate_pass_at_k, load_sampled, paired_pass_at_k
from vqar.eval.protocol import Protocol, protocol_identity
from vqar.eval.record import RunRecord
from vqar.eval.runner import prediction_line, summarize


def saved(tmp_path, texts, *, dataset="gqa", answers=None, temperature=1.0, label="arm",
          questions=("q1",)):
    """A sampled record: every question gets the same list of sampled texts."""
    protocol = Protocol(dataset, temperature=temperature, samples=len(texts))
    identity = protocol_identity(protocol, {"selection_sha256": "same-cohort"})
    lines = []
    for qid in questions:
        example = Example(dataset, qid, [], "Is it red?", answers or ["yes"], "verify")
        lines += [prediction_line(example, f"image-of-{qid}", i,
                                  {"text": text, "output_tokens": 1, "finish_reason": "stop"},
                                  "test") for i, text in enumerate(texts)]
    model = {"source": label, "label": label, "run": None, "step": None,
             "weights": {"sha256": label}}
    record = RunRecord(model, identity, summarize(lines, dataset, "short"), {})
    return record.save(tmp_path / f"{label}.json", lines)


@pytest.mark.parametrize("correct,k", [(0, 1), (0, 8), (1, 2), (3, 4), (7, 2), (8, 8)])
def test_estimate_matches_the_mean_over_every_subset(correct, k):
    outcomes = [True] * correct + [False] * (8 - correct)
    subsets = list(itertools.combinations(outcomes, k))
    assert estimate_pass_at_k(8, correct, k) == pytest.approx(
        sum(any(subset) for subset in subsets) / len(subsets))
    assert estimate_pass_at_k(8, correct, 1) == pytest.approx(correct / 8)


@pytest.mark.parametrize("n,correct,k", [(0, 0, 1), (8, 9, 1), (8, -1, 1), (8, 2, 9),
                                        (8, 2, 0), (8, 2.5, 1), (True, 1, 1)])
def test_invalid_counts_are_rejected(n, correct, k):
    with pytest.raises(ValueError, match="integer"):
        estimate_pass_at_k(n, correct, k)


def test_default_ks_are_powers_of_two_ending_at_n():
    assert default_ks(32) == [1, 2, 4, 8, 16, 32]
    assert default_ks(12) == [1, 2, 4, 8, 12]


def test_sample_order_does_not_change_the_estimates(tmp_path):
    first = load_sampled(saved(tmp_path, ["yes"] * 3 + ["no"] * 5).path)
    second = load_sampled(saved(tmp_path, ["no"] * 5 + ["yes"] * 3, label="shuffled").path)
    assert first.summary([1, 8])["pass_at_k"] == second.summary([1, 8])["pass_at_k"]
    assert all(row["delta"] == row["low"] == row["high"] == 0
               for row in paired_pass_at_k(first, second, [1, 8], resamples=50))


def test_partial_annotator_credit_is_not_a_pass(tmp_path):
    record = saved(tmp_path, ["yes"] * 8, dataset="vqav2", answers=["yes"] * 2 + ["no"] * 8)
    summary = load_sampled(record.path).summary([1, 8])
    assert summary["mean_score"] == pytest.approx(60)
    assert summary["pass_at_k"] == {1: 0, 8: 0}


def test_variants_score_differently(tmp_path):
    path = saved(tmp_path, ["<answer>Yes</answer>"] * 8).path
    assert load_sampled(path, "official").summary([8])["pass_at_k"][8] == 0  # case-sensitive
    assert load_sampled(path, "normalized").summary([8])["pass_at_k"][8] == 100
    assert load_sampled(path, "reward").summary([8])["pass_at_k"][8] == 0  # reads the full text


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "reference"])
def test_incomplete_or_inconsistent_groups_are_rejected(tmp_path, mutation):
    record = saved(tmp_path, ["yes"] * 8)
    lines = record.predictions()
    if mutation == "missing":
        lines.pop()
    elif mutation == "duplicate":
        lines[-1]["sample"] = 0
    else:
        lines[-1]["answers"] = ["no"]
    record.save(record.path, lines)
    with pytest.raises(ValueError):
        load_sampled(record.path)


def test_greedy_records_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="sampled record"):
        load_sampled(saved(tmp_path, ["yes"], temperature=0.0).path)


def test_different_protocols_or_questions_cannot_be_paired(tmp_path):
    before = load_sampled(saved(tmp_path, ["yes"] * 8, label="before").path)
    hotter = load_sampled(saved(tmp_path, ["yes"] * 8, label="hot", temperature=0.5).path)
    with pytest.raises(ValueError, match="same protocol"):
        paired_pass_at_k(before, hotter, [1], resamples=50)
    other = load_sampled(saved(tmp_path, ["yes"] * 8, label="other", questions=("q2",)).path)
    with pytest.raises(ValueError, match="identical questions"):
        paired_pass_at_k(before, other, [1], resamples=50)


def test_bootstrap_resamples_the_given_images(tmp_path):
    questions = ("q1", "q2", "q3")
    before = load_sampled(saved(tmp_path, ["no"] * 8, label="b", questions=questions).path)
    after = load_sampled(saved(tmp_path, ["yes"] * 8, label="a", questions=questions).path)
    shared = {"q1": "image-a", "q2": "image-a", "q3": "image-b"}
    rows = paired_pass_at_k(before, after, [1, 8], image_groups=shared, resamples=50)
    assert [row["metric"] for row in rows] == ["mean_score", "pass@1", "pass@8"]
    assert all(row["delta"] == 100 and row["image_groups"] == 2 for row in rows)
    assert "image_groups" not in paired_pass_at_k(before, after, [1], resamples=50)[0]


def test_cli_writes_curves_and_paired_differences(tmp_path, monkeypatch):
    import scripts.pass_at_k
    from scripts.pass_at_k import main

    monkeypatch.setattr(scripts.pass_at_k, "image_groups", lambda cell: {"q1": "image-a"})
    before = saved(tmp_path, ["no"] * 4 + ["yes"] * 4, label="before")
    after = saved(tmp_path, ["yes"] * 8, label="after")
    output = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["pass_at_k", "--compare", str(before.path), str(after.path),
                                      "--resamples", "50", "--output", str(output)])
    main()
    report = json.loads(output.read_text())
    assert [row["pass_at_k"]["1"] for row in report["records"]] == [50, 100]
    deltas = {row["metric"]: row["delta"] for row in report["comparisons"][0]["deltas"]}
    assert deltas["pass@1"] == pytest.approx(50) and deltas["pass@8"] == pytest.approx(0)
