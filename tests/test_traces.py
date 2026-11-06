"""Reasoning traces: choosing one correct response, and training on it."""

import json

from scripts.build_traces import choose_traces, correct_samples
from train_sft import load_targets, to_sft_columns


def line(qid, raw, answers=("red",)):
    return {"qid": qid, "raw": raw, "answers": list(answers)}


def test_only_tagged_right_answers_count_as_correct():
    lines = [line("a", "<think>x</think><answer>red</answer>"), line("a", "<answer>blue</answer>"),
             line("a", "red"), line("b", "<answer>blue</answer>")]
    correct = correct_samples(lines, "gqa", "reasoning")
    assert correct["a"] == ["<think>x</think><answer>red</answer>"]
    assert correct["b"] == []


def test_choice_is_seeded_and_skips_questions_with_nothing_right():
    correct = {"a": ["1", "2", "3"], "b": [], "c": ["4"]}
    assert choose_traces(correct) == choose_traces(dict(reversed(list(correct.items()))))
    assert set(choose_traces(correct)) == {"a", "c"}


def test_targets_replace_the_short_answer_and_the_prompt(tmp_path, monkeypatch):
    path = tmp_path / "traces.jsonl"
    trace = "<think>t</think><answer>7</answer>"
    path.write_text(json.dumps({"qid": "q1", "target": trace}) + "\n")
    targets = load_targets(path)
    batch = {"qid": ["q1"], "question": ["How many?"], "answers": [["7"]], "image": [None]}
    monkeypatch.setattr("train_sft.fit_sft_image", lambda image, *args: image)
    with_targets = to_sft_columns(batch, targets=targets)
    short = to_sft_columns(batch)
    assert with_targets["completion"][0][0]["content"] == trace
    assert "<think>" in with_targets["prompt"][0][0]["content"]
    assert short["completion"][0][0]["content"] == "7"
    assert "single word" in short["prompt"][0][0]["content"]
