import json

import pytest
from datasets import Dataset, load_from_disk

from evaluate import atomic_json, load_progress, load_subset, summarize, swap_indices
from vqar.splits import VERSION, qid_hash


def test_resume_ignores_incomplete_write_and_rejects_changed_settings(tmp_path):
    identity = {"model": "frozen-model", "image_mode": "original"}
    qids = ["a", "b", "c"]
    assert load_progress(tmp_path, identity, qids, resume=False) == []
    batch = [{"qid": "a"}, {"qid": "b"}]
    atomic_json(tmp_path / "batch-00000000.json", batch)
    (tmp_path / "batch-00000002.tmp").write_text('[{"qid":')
    assert load_progress(tmp_path, identity, qids, resume=True) == batch
    with pytest.raises(ValueError, match="settings or inputs changed"):
        load_progress(tmp_path, dict(identity, image_mode="black"), qids, resume=True)
    with pytest.raises(FileExistsError):
        load_progress(tmp_path, identity, qids, resume=False)


def test_resume_rejects_missing_batches_and_wrong_question_ids(tmp_path):
    load_progress(tmp_path, {}, ["a", "b"], resume=False)
    path = tmp_path / "batch-00000001.json"
    atomic_json(path, [{"qid": "b"}])
    with pytest.raises(ValueError, match="Missing or overlapping"):
        load_progress(tmp_path, {}, ["a", "b"], resume=True)
    path.unlink()
    atomic_json(tmp_path / "batch-00000000.json", [{"qid": "wrong"}])
    with pytest.raises(ValueError, match="Prediction IDs"):
        load_progress(tmp_path, {}, ["a", "b"], resume=True)


def test_swaps_never_reuse_the_same_image_even_with_repeated_questions():
    keys = ["a", "a", "b", "c", "c", "a"]
    donors = swap_indices(keys)
    assert all(keys[i] != keys[j] for i, j in enumerate(donors))
    assert swap_indices(keys) == donors
    with pytest.raises(ValueError, match="at least two"):
        swap_indices(["a", "a"])


def test_output_diagnostics_distinguish_truncation_and_empty_answers():
    records = [
        dict(score=1, answer_type="number", output_tokens=10, finish_reason="stop",
             raw="<answer>2</answer>", prediction="2"),
        dict(score=0, answer_type="number", output_tokens=64, finish_reason="length",
             raw="", prediction=""),
    ]
    metrics = summarize(records)
    assert metrics["overall"] == 0.5
    assert metrics["mean_output_tokens"] == 37
    assert metrics["truncated_fraction"] == 0.5
    assert metrics["empty_answer_fraction"] == 0.5


def test_dev_evaluation_uses_manifest_images_and_checks_source_identity(tmp_path):
    entries = {}
    for name in ("vqav2", "gqa"):
        subset = name + "_sft"
        ds = Dataset.from_dict({"qid": [name + "-a", name + "-b"],
                                "dataset": [name, name]})
        ds.save_to_disk(str(tmp_path / subset))
        ds = load_from_disk(str(tmp_path / subset))
        entries[subset] = {"fingerprint": ds._fingerprint, "qids_sha256": qid_hash(ds),
                           "image_keys": ["train-image", "dev-image"],
                           "retained_indices": [0, 1]}
    manifest = {"version": VERSION, "subsets": entries,
                "sft_dev": {"subsets": list(entries), "seed": 42, "image_keys": ["dev-image"]}}
    path = tmp_path / "split_manifest.json"
    path.write_text(json.dumps(manifest))
    for name in ("vqav2", "gqa"):
        dev, provenance = load_subset(name, "dev", tmp_path)
        assert list(dev["qid"]) == [name + "-b"]
        assert set(dev["image_key"]) == {"dev-image"}
        assert provenance["version"] == VERSION
    with pytest.raises(ValueError, match="only VQAv2 and GQA"):
        load_subset("clevr", "dev", tmp_path)
    entries["gqa_sft"]["qids_sha256"] = "changed"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="changed since the split check"):
        load_subset("gqa", "dev", tmp_path)


def test_official_metric_rejects_single_reference_dev_before_inference(monkeypatch, capsys):
    from evaluate import main

    monkeypatch.setattr("sys.argv", ["evaluate.py", "--preset", "zero_shot_2b", "--split", "dev"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "single-reference answers" in capsys.readouterr().err
