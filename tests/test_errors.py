"""Error analysis: flips, the error taxonomy, and answer-form checks on small predictions."""

import json

import pytest
from fixtures import png, save_subset, write_splits

from vqar.eval.errors import answer_form, error_kind, error_taxonomy, flip_counts


def line(prediction, raw=None, official=0.0, normalized=0.0, answers=("blue",),
         finish_reason="stop"):
    return {"prediction": prediction, "raw": prediction if raw is None else raw,
            "scores": {"official": official, "normalized": normalized},
            "answers": list(answers), "finish_reason": finish_reason}


def test_flips_count_fixed_and_broken_questions_by_the_right_threshold():
    # One annotator of ten (0.3) is wrong; two (0.6) are right.
    reference = {"a": 1.0, "b": 0.0, "c": 0.3, "d": 0.6}
    candidate = {"a": 0.0, "b": 1.0, "c": 0.6, "d": 1.0}
    assert flip_counts(reference, candidate) == {
        "n": 4, "wrong_to_right": 2, "right_to_wrong": 1, "right_to_right": 1,
        "wrong_to_wrong": 0}
    with pytest.raises(ValueError, match="same questions"):
        flip_counts({"a": 1.0}, {"b": 1.0})


@pytest.mark.parametrize("prediction, kwargs, kind", [
    ("blue", dict(official=1.0, normalized=1.0), None),
    ("Blue", dict(official=0.0, normalized=1.0), "scorer_only"),
    ("I think", dict(raw="<think>I think", finish_reason="length"), "truncated"),
    ("It is red.", dict(raw="It is red."), "no_answer_tag"),
    ("light blue", dict(raw="<answer>light blue</answer>"), "partial"),
    ("red", dict(raw="<answer>red</answer>"), "wrong_content"),
])
def test_each_wrong_answer_gets_the_first_reason_that_applies(prediction, kwargs, kind):
    assert error_kind(line(prediction, **kwargs), "reasoning") == kind


def test_only_tagged_prompts_can_miss_the_answer_tag():
    untagged = line("red")
    assert error_kind(untagged, "reasoning") == "no_answer_tag"
    assert error_kind(untagged, "short") == "wrong_content"


def test_without_an_official_score_the_normalized_one_decides():
    right = line("Blue", official=None, normalized=1.0)
    assert error_kind(right, "short") is None


def test_the_taxonomy_splits_every_prediction():
    lines = [line("blue", official=1.0, normalized=1.0), line("Blue", normalized=1.0),
             line("red"), line("red")]
    shares = error_taxonomy(lines, "short")
    assert shares == {"right": 0.25, "scorer_only": 0.25, "truncated": 0.0,
                      "no_answer_tag": 0.0, "partial": 0.0, "wrong_content": 0.5}


def test_answer_form_reads_the_answer_the_reward_reads():
    lines = [
        line("red or blue", raw="<think>x</think><answer>red or blue</answer>"),
        line("red", raw="<answer>blue/red</answer> then <answer>red</answer>"),
        line("a cat sitting on a red sofa by the window"),  # untagged: never flagged
        line("a cat sitting on a red sofa by the window",
             raw="<answer>a cat sitting on a red sofa by the window</answer>"),
    ]
    assert answer_form(lines, "reasoning") == {
        "hedged": 0.25, "multiple_answer_tags": 0.25, "long_answer": 0.25}
    # The short prompt's reward reads the whole response.
    assert answer_form(lines[2:3], "short") == {
        "hedged": 0.0, "multiple_answer_tags": 0.0, "long_answer": 1.0}


class Reply:
    """Answers each question from a table keyed by question text."""

    def __init__(self, replies):
        self.replies = replies

    def generate(self, conversations, **kwargs):
        texts = [next(part["text"] for part in messages[0]["content"] if part["type"] == "text")
                 for messages in conversations]
        return [[{"text": next(reply for question, reply in self.replies.items()
                               if text.startswith(question)),
                  "output_tokens": 2, "finish_reason": "stop"}] for text in texts]


MATRIX = """
[[arms]]
label = "base"
name = "Base"
model = "models/base"

[[arms]]
label = "tuned"
name = "Tuned"
model = "models/tuned"

[[protocols]]
datasets = ["gqa"]
prompts = ["short"]

[matrices.main]
arms = ["base", "tuned"]

[[matrices.main.comparisons]]
reference = "base"
candidates = ["tuned"]
"""


def test_every_declared_arm_gets_flips_errors_and_answer_forms(tmp_path):
    from scripts.error_analysis import analyze
    from vqar.eval import ModelRef, Protocol, run_eval

    data, records = tmp_path / "data", tmp_path / "records"
    save_subset(data, "gqa_eval", "gqa", [
        ("g1", png(10), "What color?", ["blue"], "query"),
        ("g2", png(20), "Which side?", ["left"], "choose"),
    ])
    for number, role in enumerate(("sft", "rl")):
        save_subset(data, f"gqa_{role}", "gqa",
                    [(f"{role}{i}", png(100 + 10 * number + i), "Is it red?", ["no"], "")
                     for i in range(3)])
    write_splits(data, ["gqa"])
    replies = {"base": {"What color?": "blue", "Which side?": "right or left"},
               "tuned": {"What color?": "Blue", "Which side?": "left"}}
    for label, answers in replies.items():
        weights = tmp_path / label
        weights.mkdir()
        (weights / "model.safetensors").write_bytes(label.encode())
        run_eval(ModelRef(str(weights), label=label), Protocol("gqa"), Reply(answers),
                 data_dir=data, records_dir=records)
    (tmp_path / "experiments.toml").write_text(MATRIX)

    result = json.loads(json.dumps(analyze(tmp_path / "experiments.toml", records, data)))
    flips = {row["variant"]: row for row in result["flips"]}
    assert (flips["normalized"]["wrong_to_right"], flips["normalized"]["right_to_wrong"]) == (1, 0)
    # GQA's official match is case-sensitive: "Blue" is fixed and broken at once.
    assert (flips["official"]["wrong_to_right"], flips["official"]["right_to_wrong"]) == (1, 1)
    cells = {row["arm"]: row for row in result["cells"]}
    assert cells["base"]["answer_form"]["hedged"] == 0.5
    assert cells["tuned"]["errors"]["scorer_only"] == 0.5
    assert set(result["definitions"]) >= {"right", "error_kinds", "answer_form"}


def test_the_report_tables_read_the_error_analysis():
    from vqar.eval.matrix import Arm
    from vqar.eval.report import answer_form_table, errors_table, flips_table

    arms = [Arm("base", "Base", "m"), Arm("tuned", "Tuned", "n")]
    flip = {"reference": "base", "candidate": "tuned", "dataset": "gqa", "prompt": "reasoning",
            "variant": "normalized", "n": 3000, "wrong_to_right": 1234, "right_to_wrong": 56}
    lines = flips_table(arms, [flip, dict(flip, variant="official")]).splitlines()
    assert len(lines) == 3
    assert lines[2] == "| Tuned − Base | reasoning | — | 1,234 / 56 | — | — |"
    cell = {"arm": "tuned", "dataset": "gqa", "prompt": "reasoning",
            "errors": {"right": 0.5, "scorer_only": 0.1, "truncated": 0.0,
                       "no_answer_tag": 0.2, "partial": 0.05, "wrong_content": 0.15},
            "answer_form": {"hedged": 0.01, "multiple_answer_tags": 0.0, "long_answer": 0.02}}
    assert errors_table(arms, [cell]).splitlines()[2] == (
        "| Tuned | GQA | 50.0 | 10.0 | 0.0 | 20.0 | 5.0 | 15.0 |")
    assert answer_form_table(arms, [cell]).splitlines()[2] == (
        "| Tuned | — | 1.0 · 0.0 · 2.0 | — | — |")
    assert errors_table(arms, [dict(cell, prompt="short")]) == (
        "No error analysis has been computed yet.\n")
