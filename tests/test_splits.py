import io
import json

import pytest
from datasets import Dataset, Image, load_from_disk
from PIL import Image as PILImage

from vqar.splits import (
    VERSION,
    image_keys,
    load_training_subsets,
    qid_hash,
    retained_indices,
    split_dev,
    training_views,
)


def test_duplicate_pixels_survive_different_file_encodings():
    image = PILImage.new("RGB", (5, 7), (3, 40, 100))
    payloads = []
    for format in ("PNG", "BMP"):
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        payloads.append({"bytes": buffer.getvalue(), "path": None})
    dataset = Dataset.from_dict({"image": payloads}).cast_column("image", Image())
    assert len(set(image_keys(dataset))) == 1


def test_mixed_sources_hold_eval_out_of_both_training_pools():
    keys = {
        "vqav2_eval": ["a"],
        "vqav2_sft": ["b", "c"],
        "gqa_sft": ["a", "d"],
        "gqa_rl": ["a", "b", "e"],
    }
    assert retained_indices(keys) == {
        "vqav2_eval": [0],
        "vqav2_sft": [0, 1],
        "gqa_sft": [1],
        "gqa_rl": [1, 2],
    }


def test_sft_dev_keeps_all_questions_for_an_image_together():
    dataset = Dataset.from_dict({"image_key": ["a", "b", "a", "c", "b", "d"]})
    train, dev = split_dev(dataset, target=3)
    assert len(dev) == 4
    assert set(train["image_key"]).isdisjoint(dev["image_key"])


def test_shared_rl_pool_reuses_sft_train_but_never_dev_or_cross_source_eval():
    keys = {
        "a_sft": ["a", "a", "b", "c", "d"],
        "a_rl": ["a", "b", "c", "d", "test", "extra"],
        "b_eval": ["test"],
    }
    kept = retained_indices(keys)
    views = training_views(keys, kept, ["a_sft"], seed=42, dev_target=1)
    dev = set(views["sft_dev"]["image_keys"])
    parts = views["rl_views"]["a_rl"]
    rl = {keys[p["subset"]][i] for p in parts for i in p["indices"]}
    assert rl == {"a", "b", "c", "d", "extra"} - dev
    assert dev.isdisjoint(rl)
    reused = {keys["a_sft"][i] for i in parts[1]["indices"]}
    assert reused == set(keys["a_sft"]) - dev


def test_fixed_dev_membership_survives_a_different_shuffle():
    ds = Dataset.from_dict({"image_key": ["a", "b", "a", "c"],
                            "is_dev": [True, False, True, False]})
    train, dev = split_dev(ds.shuffle(seed=7))
    assert set(train["image_key"]) == {"b", "c"}
    assert set(dev["image_key"]) == {"a"}


def test_manifest_loader_expands_rl_and_preserves_old_view(tmp_path):
    keys = {"a_sft": ["a", "b", "c"], "a_rl": ["a", "test", "new"],
            "b_eval": ["test"]}
    entries = {}
    kept = retained_indices(keys)
    for name, images in keys.items():
        ds = Dataset.from_dict({"qid": [f"{name}-{i}" for i in range(len(images))]})
        ds.save_to_disk(str(tmp_path / name))
        ds = load_from_disk(str(tmp_path / name))
        entries[name] = {"fingerprint": ds._fingerprint, "qids_sha256": qid_hash(ds),
                         "image_keys": images, "retained_indices": kept[name]}
    views = training_views(keys, kept, ["a_sft"], dev_target=1)
    manifest = dict(version=VERSION, subsets=entries, **views)
    path = tmp_path / "split_manifest.json"
    path.write_text(json.dumps(manifest))
    train, dev = split_dev(load_training_subsets(tmp_path, ["a_sft"], seed=7))
    rl = load_training_subsets(tmp_path, ["a_rl"], seed=7)
    assert set(train["qid"]).issubset(rl["qid"])
    assert set(rl["image_key"]).isdisjoint(dev["image_key"])
    assert "test" not in rl["image_key"]
    assert "new" in rl["image_key"]
    # A preserved v1 manifest must not silently acquire the new shared pool.
    entries["a_rl"]["retained_indices"] = [2]
    path.write_text(json.dumps(dict(version="image-disjoint-v1", subsets=entries)))
    old = load_training_subsets(tmp_path, ["a_rl"], seed=7)
    assert list(old["qid"]) == ["a_rl-2"]


def test_unconfigured_sft_dev_fails_before_training():
    ds = Dataset.from_dict({"image_key": ["a", "b"], "is_dev": [False, False]})
    with pytest.raises(ValueError, match="regenerate the manifest"):
        split_dev(ds)
