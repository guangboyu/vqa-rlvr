"""Paired bootstrap comparisons over questions."""

import json

import pytest
from fixtures import png, save_subset, write_splits

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


class Reply:
    """Answers every question with one fixed text."""

    def __init__(self, text):
        self.text = text

    def generate(self, conversations, **kwargs):
        return [[{"text": self.text, "output_tokens": 1, "finish_reason": "stop"}]
                for _ in conversations]


def test_a_canvas_cell_is_paired_with_its_original_image_cell(tmp_path, monkeypatch):
    import scripts.paired_deltas as script
    from vqar.eval import ModelRef, Protocol, run_eval

    data, records = tmp_path / "data", tmp_path / "records"
    save_subset(data, "gqa_eval", "gqa",
                [(f"g{i}", png(10 * i), "Is it red?", ["yes"], "verify") for i in range(4)])
    for number, role in enumerate(("sft", "rl")):
        save_subset(data, f"gqa_{role}", "gqa",
                    [(f"{role}{i}", png(100 + 10 * number + i), "Is it red?", ["no"], "")
                     for i in range(3)])
    write_splits(data, ["gqa"])
    weights = tmp_path / "model"
    weights.mkdir()
    (weights / "model.safetensors").write_bytes(b"w")
    model = ModelRef(str(weights), label="grpo")
    run_eval(model, Protocol("gqa"), Reply("yes"), data_dir=data, records_dir=records)
    run_eval(model, Protocol("gqa", image_mode="canvas"), Reply("no"), data_dir=data,
             records_dir=records)
    monkeypatch.setattr(script, "RECORDS_DIR", records)
    monkeypatch.setattr(script, "DATA_DIR", data)
    rows = script.resolution_contrasts()
    assert [(row["contrast"], row["variant"], row["delta"]) for row in rows] == [
        ("canvas", "official", -100.0), ("canvas", "normalized", -100.0)]
    assert rows[0]["canvas_size"] == 608
