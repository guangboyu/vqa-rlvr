import io
import json

import pytest
from datasets import Dataset, Image, load_from_disk
from PIL import Image as PILImage

from vqar.data import FEATURES
from vqar.splits import (
    VERSION,
    build_split,
    carve_dev,
    image_keys,
    load_role,
    overlap_report,
    qid_hash,
    read_split,
    split_path,
    validate_split,
)


def png(shade: int) -> bytes:
    buffer = io.BytesIO()
    PILImage.new("RGB", (3, 3), (shade, shade, shade)).save(buffer, format="PNG")
    return buffer.getvalue()


def save_subset(data_dir, name, dataset, shades):
    rows = [{"dataset": dataset, "qid": f"{name}-{i}", "image": {"bytes": png(s), "path": None},
             "question": "q?", "answers": ["a"], "answer_type": ""}
            for i, s in enumerate(shades)]
    Dataset.from_list(rows, features=FEATURES).save_to_disk(str(data_dir / name))


def write(data_dir, split):
    path = split_path(data_dir, split["dataset"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split))


def test_duplicate_pixels_survive_different_file_encodings():
    image = PILImage.new("RGB", (5, 7), (3, 40, 100))
    payloads = []
    for format in ("PNG", "BMP"):
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        payloads.append({"bytes": buffer.getvalue(), "path": None})
    dataset = Dataset.from_dict({"image": payloads}).cast_column("image", Image())
    assert len(set(image_keys(dataset))) == 1


def test_dev_keeps_all_questions_for_an_image_together():
    keys = ["a", "b", "a", "c", "b", "d"]
    chosen = carve_dev(keys, list(range(len(keys))), seed=0, target=3)
    held = [i for i, key in enumerate(keys) if key in chosen]
    assert len(held) >= 3 and set(keys) - chosen
    assert carve_dev(keys, list(range(len(keys))), seed=0, target=3) == chosen


@pytest.fixture
def planted(tmp_path):
    """VQAv2-like and GQA-like pools sharing pictures across sources.

    Shade 10 is in VQAv2's test set and also in both GQA training pools; shade 20 is in
    TextVQA's test set and in VQAv2's RL pool.
    """
    save_subset(tmp_path, "vqav2_sft", "vqav2", [1, 1, 2, 3, 4, 5, 6, 7])
    save_subset(tmp_path, "vqav2_rl", "vqav2", [8, 9, 20])
    save_subset(tmp_path, "vqav2_eval", "vqav2", [10, 11])
    save_subset(tmp_path, "textvqa_eval", "textvqa", [20, 21])
    save_subset(tmp_path, "gqa_sft", "gqa", [10, 12, 13, 14, 15])
    save_subset(tmp_path, "gqa_rl", "gqa", [10, 16])
    save_subset(tmp_path, "gqa_eval", "gqa", [17])
    save_subset(tmp_path, "clevr_rl", "clevr", [30, 31])
    save_subset(tmp_path, "clevr_val", "clevr", [32])
    save_subset(tmp_path, "clevr_test", "clevr", [33])
    return tmp_path


def build_all(data_dir, names, dev_target=2):
    splits = []
    for name in names:
        splits.append(build_split(name, data_dir, splits, seed=0, dev_target=dev_target))
        write(data_dir, splits[-1])
    return splits


def keys_of(split, role):
    return {split["subsets"][p["subset"]]["image_keys"][i]
            for p in split["roles"][role] for i in p["indices"]}


def test_new_splits_hold_other_datasets_out_and_leave_existing_files_alone(planted):
    vqav2, textvqa = build_all(planted, ["vqav2", "textvqa"])
    before = split_path(planted, "vqav2").read_bytes()
    (gqa,) = build_all(planted, ["gqa"])
    assert split_path(planted, "vqav2").read_bytes() == before
    vqav2_test = keys_of(vqav2, "test")
    for role in ("sft", "rl"):
        assert not keys_of(gqa, role) & vqav2_test  # shade 10 left both GQA pools
    # TextVQA was built after VQAv2 trained on shade 20, so its test lost that picture.
    assert len(textvqa["roles"]["test"][0]["indices"]) == 1
    report = overlap_report([vqav2, textvqa, gqa])
    assert report["held_out_images"] >= 4


def test_dev_is_carved_from_whole_sft_images_and_never_trained_on(planted):
    (vqav2,) = build_all(planted, ["vqav2"])
    dev, sft, rl = (keys_of(vqav2, role) for role in ("dev", "sft", "rl"))
    assert dev and not dev & (sft | rl)
    rows = vqav2["subsets"]["vqav2_sft"]["image_keys"]
    dev_rows = vqav2["roles"]["dev"][0]["indices"]
    assert all(i in dev_rows for i, key in enumerate(rows) if key in dev)
    # RL reuses every SFT training row besides its own pool.
    assert [p["subset"] for p in vqav2["roles"]["rl"]] == ["vqav2_rl", "vqav2_sft"]


def test_global_check_rejects_a_held_out_image_in_training(planted):
    vqav2, gqa = build_all(planted, ["vqav2", "gqa"])
    leaked = vqav2["roles"]["test"][0]["indices"][0]
    gqa["subsets"]["gqa_rl"]["image_keys"][0] = vqav2["subsets"]["vqav2_eval"]["image_keys"][leaked]
    gqa["roles"]["rl"][0]["indices"] = [0]
    with pytest.raises(ValueError, match="held-out images appear in training pools"):
        overlap_report([vqav2, gqa])


def test_an_ood_dataset_cannot_be_given_a_training_role(planted):
    (textvqa,) = build_all(planted, ["textvqa"])
    textvqa["roles"]["rl"] = [{"subset": "textvqa_eval", "indices": [0]}]
    with pytest.raises(ValueError, match="far-ood; it cannot have a training role"):
        validate_split(textvqa)


def test_trainers_can_only_load_roles_a_dataset_has(planted):
    build_all(planted, ["vqav2", "textvqa", "gqa", "clevr"])
    assert len(load_role(planted, ["vqav2"], "sft")) > 0
    with pytest.raises(ValueError, match="textvqa has no rl role"):
        load_role(planted, ["textvqa"], "rl")


def test_training_refuses_to_start_before_every_dataset_is_checked(planted):
    build_all(planted, ["vqav2"])
    assert len(load_role(planted, ["vqav2"], "test", seed=None)) == 2
    with pytest.raises(FileNotFoundError, match="No split file for gqa"):
        load_role(planted, ["vqav2"], "sft")


def test_a_role_may_only_draw_from_its_registered_subset(planted):
    (vqav2,) = build_all(planted, ["vqav2"])
    vqav2["roles"]["sft"] = [{"subset": "vqav2_eval", "indices": [0]}]
    with pytest.raises(ValueError, match="sft role may only draw from"):
        validate_split(vqav2)


def edit_entry(data_dir, dataset, subset, **fields):
    path = split_path(data_dir, dataset)
    split = json.loads(path.read_text())
    split["subsets"][subset].update(fields)
    path.write_text(json.dumps(split))


def test_loading_rejects_data_changed_after_the_split(planted):
    build_all(planted, ["vqav2"])
    edit_entry(planted, "vqav2", "vqav2_eval", qids_sha256="changed")
    with pytest.raises(ValueError, match="differs from the rows its split file records"):
        load_role(planted, ["vqav2"], "test")


def test_changed_answers_with_the_same_questions_and_images_are_rejected(planted):
    build_all(planted, ["vqav2"])
    edit_entry(planted, "vqav2", "vqav2_eval", fingerprint="elsewhere", rows_sha256="changed")
    with pytest.raises(ValueError, match="differs from the rows its split file records"):
        load_role(planted, ["vqav2"], "test")


def test_a_rebuilt_subset_with_the_same_rows_is_accepted_and_checked_once(planted, monkeypatch):
    # A fresh prepare_data run can write identical rows under a new fingerprint.
    import vqar.splits

    build_all(planted, ["vqav2"])
    edit_entry(planted, "vqav2", "vqav2_eval", fingerprint="recorded-elsewhere")
    expected = [row["qid"] for row in load_from_disk(str(planted / "vqav2_eval"))]
    assert load_role(planted, ["vqav2"], "test", seed=None)["qid"] == expected

    def no_second_check(rows):
        raise AssertionError("the content check should run once per subset")

    monkeypatch.setattr(vqar.splits, "image_keys", no_second_check)
    assert load_role(planted, ["vqav2"], "test", seed=None)["qid"] == expected


def test_unknown_split_version_is_rejected(planted):
    (vqav2,) = build_all(planted, ["vqav2"])
    write(planted, dict(vqav2, version="heldout-images-v2"))
    with pytest.raises(ValueError, match="unsupported split file"):
        read_split(planted, "vqav2")


def test_migration_keeps_every_role_row(tmp_path):
    from scripts.check_data import migrate

    names = {"vqav2_sft": ["a", "a", "b", "c"], "vqav2_rl": ["d", "e"], "vqav2_eval": ["f"],
             "gqa_sft": ["g", "h", "i", "p"], "gqa_rl": ["j"], "gqa_eval": ["k"],
             "clevr_rl": ["l"], "clevr_val": ["m"], "clevr_test": ["n"], "textvqa_eval": ["o"]}
    entries = {}
    for name, keys in names.items():
        rows = Dataset.from_dict({"qid": [f"{name}-{i}" for i in range(len(keys))]})
        rows.save_to_disk(str(tmp_path / name))
        rows = load_from_disk(str(tmp_path / name))
        entries[name] = {"n": len(keys), "fingerprint": rows._fingerprint,
                         "qids_sha256": qid_hash(rows), "image_keys": keys,
                         "retained_indices": list(range(len(keys)))}
    entries["gqa_sft"]["retained_indices"] = [0, 1, 3]  # row 2 overlapped an evaluation image
    manifest = {
        "version": "heldout-images-v2", "subsets": entries,
        "sft_dev": {"seed": 42, "target": 1, "n": 3, "subsets": ["vqav2_sft", "gqa_sft"],
                    "image_keys": ["a", "g"]},
        "rl_views": {"vqav2_rl": [{"subset": "vqav2_rl", "indices": [0, 1]},
                                  {"subset": "vqav2_sft", "indices": [2, 3]}],
                     "gqa_rl": [{"subset": "gqa_rl", "indices": [0]},
                                {"subset": "gqa_sft", "indices": [1, 3]}],
                     "clevr_rl": [{"subset": "clevr_rl", "indices": [0]}]},
    }
    splits = migrate(manifest, "sha")
    vqav2 = splits["vqav2"]["roles"]
    assert vqav2["sft"] == [{"subset": "vqav2_sft", "indices": [2, 3]}]
    assert vqav2["dev"] == [{"subset": "vqav2_sft", "indices": [0, 1]}]
    assert vqav2["rl"] == manifest["rl_views"]["vqav2_rl"]
    assert splits["gqa"]["roles"]["sft"] == [{"subset": "gqa_sft", "indices": [1, 3]}]
    assert splits["gqa"]["roles"]["dev"] == [{"subset": "gqa_sft", "indices": [0]}]
    assert splits["clevr"]["roles"]["dev"] == [{"subset": "clevr_val", "indices": [0]}]
    assert set(splits["textvqa"]["roles"]) == {"test"}
    for split in splits.values():
        assert split["version"] == VERSION and split["migrated_from"]["sha256"] == "sha"
        validate_split(split)
        write(tmp_path, split)
    rl = load_role(tmp_path, ["vqav2", "gqa"], "rl", seed=None)
    assert list(rl["qid"]) == ["vqav2_rl-0", "vqav2_rl-1", "vqav2_sft-2", "vqav2_sft-3",
                               "gqa_rl-0", "gqa_sft-1", "gqa_sft-3"]
