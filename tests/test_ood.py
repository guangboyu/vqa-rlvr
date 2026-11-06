"""Frozen OOD sources, portable rows, primary metrics and image-cluster inference."""

import json
import shutil
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from datasets import load_from_disk
from fixtures import every_dataset, png
from test_eval_harness import ScriptedGenerator

from vqar import ood
from vqar.eval import ModelRef, Protocol, RunRecord, protocol_identity, run_eval
from vqar.eval.paired import paired_bootstrap
from vqar.ood_metrics import official_vizwiz_accuracy, project_case_whitespace_em
from vqar.registry import DATASETS
from vqar.splits import check_all, load_role, split_path


@pytest.fixture
def prepared_inputs(tmp_path, monkeypatch):
    data = tmp_path / "data"
    every_dataset(data)
    source = data / "ood-source" / "realworldqa"
    source.mkdir(parents=True)
    rows = [{"image": {"bytes": png(shade), "path": None}, "question": question,
             "answer": answer} for shade, question, answer in (
                 (240, "Choose A or B.\nReturn just the option letter.", "A"),
                 (240, "Is this duplicate image red?", "no"),
                 (241, "Which side?", "left"))]
    files = list(ood.SOURCE_HASHES["realworldqa"])
    for filename, subset in zip(files, (rows[:2], rows[2:]), strict=True):
        pq.write_table(pa.Table.from_pylist(subset), source / filename)
    monkeypatch.setitem(ood.SOURCE_HASHES, "realworldqa",
                        {name: ood.file_sha256(source / name) for name in files})
    monkeypatch.setitem(DATASETS, "realworldqa",
                        replace(DATASETS["realworldqa"], sizes={"eval": 3}))
    return data, source, tmp_path / "audits"


def test_materialization_is_portable_and_retains_raw_questions_duplicates(prepared_inputs,
                                                                          tmp_path, monkeypatch):
    data, source, audits = prepared_inputs
    original_splits = {name: split_path(data, name).read_bytes() for name in
                       ("vqav2", "gqa", "clevr", "textvqa")}
    audit = ood.prepare_ood("realworldqa", data, source, audits)
    assert audit["rows"] == 3 and audit["unique_images"] == 2
    assert list(audit["duplicate_image_groups"].values()) == [["test-0000", "test-0001"]]
    assert not audit["training_image_overlap"]
    stored = load_from_disk(str(data / "realworldqa_eval"))
    assert stored[0]["image"].startswith("ood-source/realworldqa/images/")
    relocated = tmp_path / "relocated"
    shutil.move(data, relocated)
    monkeypatch.chdir(tmp_path)
    rows = load_role(relocated, ["realworldqa"], "test", seed=None)
    assert rows[0]["question"] == "Choose A or B.\nReturn just the option letter."
    assert rows[0]["answers"] == ["A"]
    assert rows[0]["image"].getpixel((0, 0)) == (240, 240, 240)
    assert len(rows) == 3
    relocated_splits = {name: split_path(relocated, name).read_bytes() for name in original_splits}
    assert relocated_splits == original_splits


def test_overlap_blocks_whole_cohort_without_filtering(prepared_inputs):
    data, source, audits = prepared_inputs
    filename = next(iter(ood.SOURCE_HASHES["realworldqa"]))
    records = pq.read_table(source / filename).to_pylist()
    # VQAv2 RL fixture's native image at index zero is shade 20.
    records[0]["image"]["bytes"] = png(20)
    pq.write_table(pa.Table.from_pylist(records), source / filename)
    ood.SOURCE_HASHES["realworldqa"][filename] = ood.file_sha256(source / filename)
    with pytest.raises(ValueError, match="adoption blocked"):
        ood.prepare_ood("realworldqa", data, source, audits)
    assert not split_path(data, "realworldqa").exists()
    assert not (data / "realworldqa_eval").exists()
    failed = json.loads((audits / "realworldqa-cohort-audit.json").read_text())
    assert failed["status"] == "failed"
    assert failed["rows"] == 3
    assert failed["training_image_overlap"][0]["qid"] == "test-0000"


def test_corrupt_source_is_never_adopted(prepared_inputs):
    data, source, audits = prepared_inputs
    path = source / next(iter(ood.SOURCE_HASHES["realworldqa"]))
    with path.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        ood.prepare_ood("realworldqa", data, source, audits)
    assert not split_path(data, "realworldqa").exists()
    assert not (data / "realworldqa_eval").exists()


def test_optional_absence_is_allowed_but_malformed_presence_fails(prepared_inputs):
    data, _, _ = prepared_inputs
    check_all(data)
    split_path(data, "vizwiz").write_text('{"malformed":true}')
    with pytest.raises((ValueError, KeyError)):
        check_all(data)


@pytest.mark.parametrize("prediction,answer,score", [
    ("", "A", 0), ("  \n", "A", 0), ("A", "A", 1), ("a", "A", 1),
    ("THE  red", "the red", 1), ("red", "the red", 0), ("two", "2", 0),
    ("A.", "A", 0), ("A because it is red", "A", 0),
])
def test_realworld_em_preserves_option_letters_and_content(prediction, answer, score):
    assert project_case_whitespace_em(prediction, [answer]) == score


@pytest.mark.parametrize("prediction,answers,score", [
    ("Yes", ["yes"] * 10, 1), ("Two.", ["2"] * 10, 1),
    ("dont", ["don’t"] * 10, 1), ("dont", ["don't"] * 10, 0),
    ("red", ["red"] * 2 + ["Red"] * 8, .6),
    ("unanswerable", ["unanswerable"] * 10, 1),
])
def test_vizwiz_native_prediction_only_scoring(prediction, answers, score):
    assert official_vizwiz_accuracy(prediction, answers) == pytest.approx(score)


def test_original_protocol_ids_stay_frozen():
    frozen = {"vqav2": "8f35db87bc73ec8d473fd01bd0b70efe5c7754ab9b5153b7064445b510dc3a9d",
              "gqa": "adece6f52a473b45f24f1962b9ef3a7ac4217b2f48bc753d4e8b7d6798882298",
              "clevr": "eec026de1fde6b81dd6e9e33f4e93be6ba23fd73d93ff1d9f7daf5f5b3f972bb",
              "textvqa": "bda644f5256743a1e44844ac2872ae6d654a02dec687a3f548616a142398913f"}
    for name, identity in frozen.items():
        assert Protocol.id_of(protocol_identity(Protocol(name),
                              {"subsets": {}, "selection_sha256": "frozen"})) == identity


def test_project_primary_roundtrip_display_and_paired_ci(prepared_inputs, tmp_path):
    from evaluate import describe
    from scripts.compare_ood import compare

    data, source, audits = prepared_inputs
    audit = ood.prepare_ood("realworldqa", data, source, audits)
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "model.safetensors").write_bytes(b"fixed")
    model = ModelRef(str(weights), label="tiny")
    good = {"Choose A or B.": "a", "Is this duplicate": "no", "Which side?": "left"}
    record = run_eval(model, Protocol("realworldqa"), ScriptedGenerator(good),
                      data_dir=data, records_dir=tmp_path / "records")
    assert RunRecord.load(record.path) == record
    assert record.metrics["official"] is None
    assert record.metrics["project"]["overall"] == 1
    assert record.metrics["normalized"] == record.metrics["project"]
    assert describe(record).startswith("project case/whitespace exact match 1.0000")
    cached = ScriptedGenerator(good)
    assert run_eval(model, Protocol("realworldqa"), cached, data_dir=data,
                    records_dir=tmp_path / "records") == record
    assert not cached.questions
    groups = [[row["qid"], row["image_key"]] for row in audit["image_groups"]]
    wrong_groups = [[groups[0][0], "different image"], *groups[1:]]
    with pytest.raises(ValueError, match="image groups differ"):
        compare(record, record, wrong_groups, resamples=100)
    sampled = replace(record, protocol={**record.protocol, "temperature": 0.7})
    with pytest.raises(ValueError, match="greedy"):
        compare(sampled, sampled, groups, resamples=100)
    result = compare(record, record, groups, resamples=100)
    assert result["primary"] == "project" and result["image_groups"] == 2
    assert (result["delta"], result["low"], result["high"]) == (0, 0, 0)


def test_cluster_bootstrap_keeps_question_weighted_point_estimate():
    reference = {"a": 0, "b": 0, "c": 0}
    candidate = {"a": 1, "b": 1, "c": 0}
    result = paired_bootstrap(reference, candidate, resamples=500,
                              image_groups={"a": "shared", "b": "shared", "c": "other"})
    assert result["delta"] == pytest.approx(200 / 3)
    assert result["low"] == 0 and result["high"] == 100
    assert result["image_groups"] == 2
    with pytest.raises(ValueError, match="exactly"):
        paired_bootstrap(reference, candidate, image_groups={"a": "shared"})


def test_external_image_mutation_is_detected_despite_unchanged_arrow(prepared_inputs):
    data, source, audits = prepared_inputs
    ood.prepare_ood("realworldqa", data, source, audits)
    load_role(data, ["realworldqa"], "test", seed=None)
    (source / "images/test-0000.img").write_bytes(png(239))
    with pytest.raises(ValueError, match="differs from the rows"):
        load_role(data, ["realworldqa"], "test", seed=None)


def test_failed_split_adoption_leaves_no_materialized_cohort(prepared_inputs, monkeypatch):
    from pathlib import Path

    data, source, audits = prepared_inputs
    original = Path.replace

    def fail_split(self, target):
        if target == split_path(data, "realworldqa"):
            raise OSError("simulated split adoption failure")
        return original(self, target)

    monkeypatch.setattr(Path, "replace", fail_split)
    with pytest.raises(OSError, match="adoption failure"):
        ood.prepare_ood("realworldqa", data, source, audits)
    assert not split_path(data, "realworldqa").exists()
    assert not (data / "realworldqa_eval").exists()
    assert json.loads((audits / "realworldqa-cohort-audit.json").read_text())["status"] == "failed"


def test_vizwiz_preserves_native_mixed_integer_float_sum():
    # Native min(1, ...) retains integer saturated terms in the Python 3.12 sum.
    references = ["yes", "no", "no", "no", "yes", "no", "no", "no", "no", "yes"]
    native_terms = [min(1, sum(other == "yes" for j, other in enumerate(references)
                              if j != i) / 3) for i in range(10)]
    assert official_vizwiz_accuracy("yes", references) == sum(native_terms) / 10
