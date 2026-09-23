"""The evaluation harness through its interface: run_eval with a scripted generator."""

import base64
import io
import json

import pytest
from fixtures import png, save_subset, write_splits
from PIL import Image as PILImage

from vqar.eval import ModelRef, Protocol, RunRecord, find_cell, protocol_identity, run_eval
from vqar.splits import split_path


@pytest.fixture
def data_dir(tmp_path):
    directory = tmp_path / "data"
    save_subset(directory, "vqav2_eval", "vqav2", [
        ("q1", png(10), "What animal?", ["cat"] * 10, "other"),
        ("q2", png(20), "Is it red?", ["yes"] * 10, "yes/no"),
        ("q3", png(30), "How many?", ["2"] * 7 + ["3"] * 3, "number"),
    ])
    save_subset(directory, "gqa_eval", "gqa", [
        ("g1", png(40), "Which side?", ["left"], "choose"),
        ("g2", png(40), "What color?", ["blue"], "query"),
        ("g3", png(50), "Is there a dog?", ["no"], "verify"),
    ])
    for dataset, answer, shade in (("vqav2", "yes", 60), ("gqa", "left", 80)):
        save_subset(directory, f"{dataset}_sft", dataset, [
            (f"{dataset}-s{i}", png(shade + i), "Is it red?", [answer], "") for i in range(3)
        ])
        save_subset(directory, f"{dataset}_rl", dataset, [
            (f"{dataset}-r0", png(shade + 9), "Is it red?", [answer], "")
        ])
    write_splits(directory, ["vqav2", "gqa"])
    return directory


@pytest.fixture
def model(tmp_path):
    directory = tmp_path / "merged-model"
    directory.mkdir()
    (directory / "model.safetensors").write_bytes(b"weights")
    return ModelRef(str(directory), label="tiny")


def image_shade(messages) -> int:
    url = next(part["image_url"]["url"] for part in messages[0]["content"]
               if part["type"] == "image_url")
    with PILImage.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as image:
        return image.convert("L").getpixel((0, 0))


class ScriptedGenerator:
    """Answers each question from a script; can fail after a number of calls."""

    def __init__(self, script, fail_after_calls=None):
        self.script = script
        self.fail_after_calls = fail_after_calls
        self.questions = []

    def generate(self, conversations, *, max_tokens, temperature, samples, seed):
        if self.fail_after_calls is not None and len(self.questions) >= self.fail_after_calls:
            raise RuntimeError("interrupted")
        self.questions.append([])
        outputs = []
        for messages in conversations:
            text = next(part["text"] for part in messages[0]["content"]
                        if part["type"] == "text")
            question = next(q for q in self.script if text.startswith(q))
            self.questions[-1].append(question)
            reply = self.script[question]
            if callable(reply):
                reply = reply(messages)
            outputs.append([{"text": reply, "output_tokens": len(reply.split()),
                             "finish_reason": "length" if reply == "" else "stop"}] * samples)
        return outputs


SCRIPT = {
    "What animal?": "Cat",  # official VQAv2 keeps case on unanimous answers; normalized does not
    "Is it red?": "<answer>yes</answer>",
    "How many?": "",
    "Which side?": "left",
    "What color?": "the blue",  # GQA official is a raw string match
    "Is there a dog?": "<answer>No</answer>",
}


def test_record_carries_both_scorer_variants_and_behavior(data_dir, model, tmp_path):
    record = run_eval(model, Protocol("vqav2"), ScriptedGenerator(SCRIPT),
                      data_dir=data_dir, records_dir=tmp_path / "records")
    official, normalized = record.metrics["official"], record.metrics["normalized"]
    assert official["overall"] == pytest.approx(1 / 3)
    assert normalized["overall"] == pytest.approx(2 / 3)
    assert normalized["by_answer_type"] == {"number": 0.0, "other": 1.0, "yes/no": 1.0}
    assert record.metrics["n"] == 3
    assert record.metrics["answer_tag_fraction"] == pytest.approx(1 / 3)
    assert record.metrics["empty_answer_fraction"] == pytest.approx(1 / 3)
    assert record.metrics["truncated_fraction"] == pytest.approx(1 / 3)
    assert record.metrics["mean_output_tokens"] == pytest.approx(2 / 3)
    lines = [json.loads(line) for line in record.predictions_path().read_text().splitlines()]
    assert [line["qid"] for line in lines] == ["q1", "q2", "q3"]
    assert lines[0]["scores"] == {"official": 0.0, "normalized": 1.0}
    assert set(lines[1]) == {"qid", "sample", "question", "raw", "prediction", "scores",
                             "answers", "answer_type", "image_qid", "output_tokens",
                             "finish_reason"}


def test_gqa_official_scorer_is_a_raw_string_match(data_dir, model, tmp_path):
    record = run_eval(model, Protocol("gqa"), ScriptedGenerator(SCRIPT),
                      data_dir=data_dir, records_dir=tmp_path / "records")
    assert record.metrics["official"]["overall"] == pytest.approx(1 / 3)
    assert record.metrics["normalized"]["overall"] == pytest.approx(1.0)


def test_repeating_a_cell_is_a_cache_hit(data_dir, model, tmp_path):
    first = run_eval(model, Protocol("vqav2"), ScriptedGenerator(SCRIPT),
                     data_dir=data_dir, records_dir=tmp_path / "records")
    generator = ScriptedGenerator(SCRIPT)
    again = run_eval(model, Protocol("vqav2", max_tokens=64), generator,
                     data_dir=data_dir, records_dir=tmp_path / "records")
    assert generator.questions == []
    assert again == first
    assert RunRecord.load(first.path) == first


def test_a_different_model_or_protocol_is_a_new_cell(data_dir, model, tmp_path):
    records = tmp_path / "records"
    run_eval(model, Protocol("vqav2"), ScriptedGenerator(SCRIPT),
             data_dir=data_dir, records_dir=records)
    (tmp_path / "merged-model" / "model.safetensors").write_bytes(b"other weights")
    retrained = ScriptedGenerator(SCRIPT)
    run_eval(model, Protocol("vqav2"), retrained, data_dir=data_dir, records_dir=records)
    reasoning = ScriptedGenerator(SCRIPT)
    run_eval(model, Protocol("vqav2", prompt="reasoning"), reasoning,
             data_dir=data_dir, records_dir=records)
    assert retrained.questions and reasoning.questions
    assert len(list(records.rglob("*.json"))) == 3


def test_interrupted_run_resumes_from_saved_batches(data_dir, model, tmp_path):
    records = tmp_path / "records"
    with pytest.raises(RuntimeError, match="interrupted"):
        run_eval(model, Protocol("vqav2"), ScriptedGenerator(SCRIPT, fail_after_calls=1),
                 data_dir=data_dir, records_dir=records, batch_size=2)
    resumed = ScriptedGenerator(SCRIPT)
    record = run_eval(model, Protocol("vqav2"), resumed,
                      data_dir=data_dir, records_dir=records, batch_size=2)
    assert resumed.questions == [["How many?"]]
    assert record.metrics["n"] == 3
    assert not list(records.rglob("*.progress"))


def test_black_images_keep_size_and_labels(data_dir, model, tmp_path):
    script = dict(SCRIPT, **{q: (lambda m: "black" if image_shade(m) == 0 else "lit")
                             for q in ("What animal?", "Is it red?", "How many?")})
    run_eval(model, Protocol("vqav2"), ScriptedGenerator(script),
             data_dir=data_dir, records_dir=tmp_path / "records")
    record = run_eval(model, Protocol("vqav2", image_mode="black"), ScriptedGenerator(script),
                      data_dir=data_dir, records_dir=tmp_path / "records")
    lines = [json.loads(line) for line in record.predictions_path().read_text().splitlines()]
    assert {line["prediction"] for line in lines} == {"black"}
    assert [line["image_qid"] for line in lines] == ["q1", "q2", "q3"]
    assert lines[0]["answers"] == ["cat"] * 10


def test_swapped_images_come_from_a_different_picture(data_dir, model, tmp_path):
    script = dict(SCRIPT, **{q: (lambda m: str(image_shade(m)))
                             for q in ("Which side?", "What color?", "Is there a dog?")})
    run_eval(model, Protocol("gqa"), ScriptedGenerator(script),
             data_dir=data_dir, records_dir=tmp_path / "records")
    record = run_eval(model, Protocol("gqa", image_mode="swapped"), ScriptedGenerator(script),
                      data_dir=data_dir, records_dir=tmp_path / "records")
    lines = [json.loads(line) for line in record.predictions_path().read_text().splitlines()]
    # g1 and g2 share one picture, so neither may donate to the other.
    assert [line["image_qid"] for line in lines] == ["g3", "g3", "g1"]
    assert [line["prediction"] for line in lines] == ["50", "50", "40"]


def test_smoke_tier_caps_the_question_count(data_dir, model, tmp_path, monkeypatch):
    monkeypatch.setitem(Protocol.TIERS, "smoke", 2)
    record = run_eval(model, Protocol("vqav2", tier="smoke"), ScriptedGenerator(SCRIPT),
                      data_dir=data_dir, records_dir=tmp_path / "records")
    assert record.metrics["n"] == 2
    assert record.protocol["tier"] == "smoke"


def test_protocol_identity_ignores_field_order_and_code_hashes(data_dir, model, tmp_path):
    data = {"fingerprint": "abc", "qids_sha256": "def", "split_sha256": None}
    identity = protocol_identity(Protocol("gqa", prompt="reasoning"), data)
    reordered = dict(reversed(list(identity.items())))
    assert Protocol.id_of(reordered) == Protocol.id_of(identity)
    assert identity["prompt_text"].startswith("{question}\nFirst think")
    assert identity["max_tokens"] == 512
    record = run_eval(model, Protocol("gqa"), ScriptedGenerator(SCRIPT),
                      data_dir=data_dir, records_dir=tmp_path / "records")
    assert "source_sha256" in record.provenance
    assert "source_sha256" not in json.dumps(record.protocol)


def test_protocol_rejects_unknown_or_inconsistent_fields():
    with pytest.raises(ValueError, match="unknown dataset"):
        Protocol("coco")
    with pytest.raises(ValueError, match="no dev role"):
        Protocol("textvqa", role="dev")
    with pytest.raises(ValueError, match="greedy decoding draws one sample"):
        Protocol("gqa", samples=4)
    with pytest.raises(ValueError, match="image mode"):
        Protocol("gqa", image_mode="blurred")


def test_dev_role_reads_the_split_file_and_skips_unscorable_official(data_dir, model,
                                                                     tmp_path):
    split = json.loads(split_path(data_dir, "vqav2").read_text())
    (dev,) = split["roles"]["dev"]
    script = {"Is it red?": "Yes"}
    record = run_eval(model, Protocol("vqav2", role="dev"), ScriptedGenerator(script),
                      data_dir=data_dir, records_dir=tmp_path / "records")
    lines = [json.loads(line) for line in record.predictions_path().read_text().splitlines()]
    assert [line["qid"] for line in lines] == [f"vqav2-s{i}" for i in dev["indices"]]
    # The authors' VQAv2 evaluator needs ten annotators; dev keeps one reference.
    assert record.metrics["official"] is None
    assert record.metrics["normalized"]["overall"] == 1.0
    assert record.protocol["data"]["subsets"] == {
        "vqav2_sft": split["subsets"]["vqav2_sft"]["fingerprint"]}


def test_evaluation_rejects_data_changed_after_the_split(data_dir, model, tmp_path):
    path = split_path(data_dir, "gqa")
    split = json.loads(path.read_text())
    split["subsets"]["gqa_eval"]["qids_sha256"] = "changed"
    path.write_text(json.dumps(split))
    with pytest.raises(ValueError, match="changed since its split file was written"):
        run_eval(model, Protocol("gqa"), ScriptedGenerator(SCRIPT),
                 data_dir=data_dir, records_dir=tmp_path / "records")


def interrupted_cell(data_dir, model, records):
    """Leave one saved batch of two questions behind, as a crash would."""
    with pytest.raises(RuntimeError, match="interrupted"):
        run_eval(model, Protocol("vqav2"), ScriptedGenerator(SCRIPT, fail_after_calls=1),
                 data_dir=data_dir, records_dir=records, batch_size=2)
    return find_cell(model, Protocol("vqav2"), data_dir=data_dir,
                     records_dir=records).path.with_suffix(".progress")


def test_resume_ignores_an_unfinished_write(data_dir, model, tmp_path):
    progress = interrupted_cell(data_dir, model, tmp_path / "records")
    (progress / "batch-00000002.tmp").write_text('[{"qid":')
    record = run_eval(model, Protocol("vqav2"), ScriptedGenerator(SCRIPT),
                      data_dir=data_dir, records_dir=tmp_path / "records", batch_size=2)
    assert record.metrics["n"] == 3


@pytest.mark.parametrize("damage, message", [
    ("identity", "belongs to another evaluation"),
    ("unlabelled", "has no identity"),
    ("gap", "Missing or overlapping"),
    ("qids", "do not match the evaluation subset"),
])
def test_resume_rejects_damaged_progress(data_dir, model, tmp_path, damage, message):
    progress = interrupted_cell(data_dir, model, tmp_path / "records")
    batch = progress / "batch-00000000.json"
    if damage == "identity":
        (progress / "identity.json").write_text('{"model": "another"}')
    elif damage == "unlabelled":
        (progress / "identity.json").unlink()
    elif damage == "gap":
        batch.rename(progress / "batch-00000001.json")
    else:
        lines = json.loads(batch.read_text())
        lines[0]["qid"] = "wrong"
        batch.write_text(json.dumps(lines))
    with pytest.raises(ValueError, match=message):
        run_eval(model, Protocol("vqav2"), ScriptedGenerator(SCRIPT),
                 data_dir=data_dir, records_dir=tmp_path / "records", batch_size=2)


def test_swapping_needs_two_different_pictures(data_dir, model, tmp_path):
    path = split_path(data_dir, "gqa")
    split = json.loads(path.read_text())
    split["roles"]["test"] = [{"subset": "gqa_eval", "indices": [0, 1]}]  # one shared picture
    path.write_text(json.dumps(split))
    with pytest.raises(ValueError, match="at least two different images"):
        run_eval(model, Protocol("gqa", image_mode="swapped"), ScriptedGenerator(SCRIPT),
                 data_dir=data_dir, records_dir=tmp_path / "records")


def test_a_changed_chat_template_is_a_new_model(data_dir, model, tmp_path):
    records = tmp_path / "records"
    run_eval(model, Protocol("gqa"), ScriptedGenerator(SCRIPT),
             data_dir=data_dir, records_dir=records)
    (tmp_path / "merged-model" / "chat_template.jinja").write_text("{{ messages }}")
    generator = ScriptedGenerator(SCRIPT)
    run_eval(model, Protocol("gqa"), generator, data_dir=data_dir, records_dir=records)
    assert generator.questions


def test_generator_must_return_every_sample(data_dir, model, tmp_path):
    class Short(ScriptedGenerator):
        def generate(self, conversations, **kwargs):
            return [samples[:1] for samples in super().generate(conversations, **kwargs)]

    with pytest.raises(ValueError, match="2 samples per question"):
        run_eval(model, Protocol("gqa", temperature=1.0, samples=2), Short(SCRIPT),
                 data_dir=data_dir, records_dir=tmp_path / "records")


def test_casing_diagnostics_separate_capitalization_from_other_misses(data_dir, model, tmp_path):
    record = run_eval(model, Protocol("vqav2"), ScriptedGenerator(SCRIPT),
                      data_dir=data_dir, records_dir=tmp_path / "records")
    # "Cat" fails the official unanimous-case rule only because of its capital letter.
    assert record.metrics["official_on_lowercased_answers"] == pytest.approx(2 / 3)
    assert record.metrics["answers_scored_differently"] == 1


def test_interventions_pair_with_the_original_image_cell(data_dir, model, tmp_path):
    records = tmp_path / "records"
    with pytest.raises(ValueError, match="Evaluate the original-image cell first"):
        run_eval(model, Protocol("gqa", image_mode="black"), ScriptedGenerator(SCRIPT),
                 data_dir=data_dir, records_dir=records)
    run_eval(model, Protocol("vqav2"), ScriptedGenerator(SCRIPT),
             data_dir=data_dir, records_dir=records)
    script = dict(SCRIPT, **{"What animal?": "dog"})
    changed = run_eval(model, Protocol("vqav2", image_mode="black"), ScriptedGenerator(script),
                       data_dir=data_dir, records_dir=records)
    paired = changed.metrics["paired_with_original"]
    assert paired["n"] == 3
    assert paired["original"] == {"official": pytest.approx(1 / 3),
                                  "normalized": pytest.approx(2 / 3)}
    assert paired["normalized_answer_changed_fraction"] == pytest.approx(1 / 3)
