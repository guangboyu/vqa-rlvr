"""Shared training pools with held-out images; source files stay unchanged."""

import hashlib
import io
import json
from pathlib import Path

from datasets import Dataset, Image, concatenate_datasets, load_from_disk
from PIL import Image as PILImage

VERSION = "heldout-images-v2"


def image_keys(dataset) -> list[str]:
    """Hash decoded RGB pixels so metadata or file encoding cannot hide duplicates."""
    cache, keys = {}, []
    for row in dataset.select_columns(["image"]).cast_column("image", Image(decode=False)):
        payload = row["image"]
        raw = payload["bytes"]
        if raw is None:
            raw = Path(payload["path"]).read_bytes()
        encoded_key = hashlib.sha256(raw).hexdigest()
        if encoded_key not in cache:
            with PILImage.open(io.BytesIO(raw)) as image:
                image = image.convert("RGB")
                pixels = str(image.size).encode() + b":" + image.tobytes()
            cache[encoded_key] = hashlib.sha256(pixels).hexdigest()
        keys.append(cache[encoded_key])
    return keys


def retained_indices(keys: dict[str, list[str]]) -> dict[str, list[int]]:
    """Hold evaluation images out of every training source."""
    evaluation = {
        key for name, values in keys.items() if not name.endswith(("_sft", "_rl")) for key in values
    }
    kept = {}
    for name, values in keys.items():
        kept[name] = (
            [i for i, key in enumerate(values) if key not in evaluation]
            if name.endswith(("_sft", "_rl"))
            else list(range(len(values)))
        )
    return kept


def split_dev(dataset, target: int = 500):
    """Use fixed manifest membership, or select whole image groups for older views."""
    if "is_dev" in dataset.column_names:
        flags = dataset["is_dev"]
        if not any(flags) or all(flags):
            raise ValueError(
                "SFT train/dev must both be nonempty; regenerate the manifest for these sources."
            )
        return (dataset.select([i for i, flag in enumerate(flags) if not flag]),
                dataset.select([i for i, flag in enumerate(flags) if flag]))
    keys = dataset["image_key"]
    counts = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    dev_keys, total = set(), 0
    for key in keys:
        if total >= target:
            break
        if key not in dev_keys:
            dev_keys.add(key)
            total += counts[key]
    train = [i for i, key in enumerate(keys) if key not in dev_keys]
    dev = [i for i, key in enumerate(keys) if key in dev_keys]
    if not train or not dev:
        raise ValueError("Need enough distinct images for nonempty train and dev splits.")
    return dataset.select(train), dataset.select(dev)


def training_views(keys, kept, sft_subsets, seed=42, dev_target=500):
    """Reserve dev once, then reuse SFT training examples in the matching RL pool."""
    sft_keys = [keys[name][i] for name in sft_subsets for i in kept[name]]
    dev_keys = set()
    dev_count = 0
    if sft_keys:
        sft = Dataset.from_dict({"image_key": sft_keys}).shuffle(seed=seed)
        _, dev = split_dev(sft, dev_target)
        dev_keys = set(dev["image_key"])
        dev_count = len(dev)
    views = {}
    for name in keys:
        if not name.endswith("_rl"):
            continue
        sources = [name]
        sft_name = name.removesuffix("_rl") + "_sft"
        if sft_name in sft_subsets:
            sources.append(sft_name)
        views[name] = [
            {"subset": source, "indices": [
                i for i in kept[source] if keys[source][i] not in dev_keys
            ]}
            for source in sources
        ]
    return {
        "sft_dev": {"seed": seed, "target": dev_target, "n": dev_count,
                    "subsets": list(sft_subsets), "image_keys": sorted(dev_keys)},
        "rl_views": views,
    }


def qid_hash(dataset) -> str:
    return hashlib.sha256(",".join(dataset["qid"]).encode()).hexdigest()


def load_training_subsets(data_dir: Path, names, seed: int, legacy: bool = False):
    manifest_path = data_dir / "split_manifest.json"
    if not legacy and not manifest_path.exists():
        raise FileNotFoundError(
            "Run `python -m scripts.check_data` before training to hold out evaluation images. "
            "Use --legacy-splits only to reproduce historical experiments."
        )
    manifest = json.loads(manifest_path.read_text()) if not legacy else None
    if manifest is not None and manifest["version"] not in (VERSION, "image-disjoint-v1"):
        raise ValueError("Unsupported split manifest version; regenerate it.")
    datasets = []
    selections = []
    for name in names:
        if manifest is not None and name in manifest.get("rl_views", {}):
            selections.extend(manifest["rl_views"][name])
        else:
            selections.append({"subset": name, "indices": None})
    dev_keys = set(manifest.get("sft_dev", {}).get("image_keys", [])) if manifest else set()
    for selection in selections:
        name = selection["subset"]
        dataset = load_from_disk(str(data_dir / name))
        if manifest is not None:
            entry = manifest["subsets"][name]
            if (
                entry["qids_sha256"] != qid_hash(dataset)
                or entry["fingerprint"] != dataset._fingerprint
            ):
                raise ValueError(f"{name} changed since the split check; regenerate its manifest.")
            dataset = dataset.add_column("image_key", entry["image_keys"])
            if "sft_dev" in manifest:
                dataset = dataset.add_column("is_dev", [k in dev_keys for k in entry["image_keys"]])
            indices = selection["indices"]
            dataset = dataset.select(entry["retained_indices"] if indices is None else indices)
        datasets.append(dataset)
    return concatenate_datasets(datasets).shuffle(seed=seed)


def split_provenance(data_dir: Path, legacy: bool) -> dict:
    if legacy:
        return {"version": "legacy", "image_disjoint": False}
    return {
        "version": json.loads((data_dir / "split_manifest.json").read_text())["version"],
        "manifest_sha256": hashlib.sha256(
            (data_dir / "split_manifest.json").read_bytes()
        ).hexdigest(),
    }
