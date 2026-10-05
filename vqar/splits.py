"""Per-dataset split files: which materialized rows play each role (sft, rl, dev, test).

Images are identified by a hash of their decoded RGB pixels, so re-encoded copies still
match. The global check keeps every dev and test image out of every training pool across
datasets, and an OOD dataset out of training entirely. Trainers load only their own role.
Files live under data/splits/ and are versioned; source data stays unchanged.
"""

import hashlib
import io
import json
import random
from pathlib import Path

from datasets import Image, Value, concatenate_datasets, load_from_disk
from PIL import Image as PILImage

from vqar.registry import DATASETS, TRAINING_ROLES

VERSION = "split-files-v1"
HELD_OUT_ROLES = frozenset({"dev", "test"})
DEV_TARGET = 500  # questions carved from a training pool when a dataset has no dev subset
SEED = 42
VERIFIED = ".verified-subsets.json"  # content checks already passed, kept beside the data


def image_keys(dataset, image_root: Path | None = None) -> list[str]:
    """Hash decoded RGB pixels so metadata or file encoding cannot hide duplicates."""
    external = isinstance(dataset.features.get("image"), Value)
    cache, keys = {}, []
    for row in dataset.select_columns(["image"]).cast_column("image", Image(decode=False)):
        payload = row["image"]
        raw = payload["bytes"]
        if raw is None:
            path = Path(payload["path"])
            if external and (path.is_absolute() or ".." in path.parts):
                raise ValueError("External OOD image paths must stay relative to the data root")
            if external and image_root is None:
                raise ValueError("External OOD image hashing requires the data root")
            if image_root is not None and not path.is_absolute():
                path = Path(image_root) / path
            raw = path.read_bytes()
        encoded_key = hashlib.sha256(raw).hexdigest()
        if encoded_key not in cache:
            with PILImage.open(io.BytesIO(raw)) as image:
                image = image.convert("RGB")
                pixels = str(image.size).encode() + b":" + image.tobytes()
            cache[encoded_key] = hashlib.sha256(pixels).hexdigest()
        keys.append(cache[encoded_key])
    return keys


def qid_hash(dataset) -> str:
    return hashlib.sha256(",".join(dataset["qid"]).encode()).hexdigest()


def rows_sha256(dataset) -> str:
    """Hash every row's question ID, question, answers, and answer type (not the image)."""
    digest = hashlib.sha256()
    for row in dataset.select_columns(["qid", "question", "answers", "answer_type"]):
        digest.update(json.dumps(row, sort_keys=True).encode() + b"\n")
    return digest.hexdigest()


def verify_subset(data_dir: Path, name: str, entry: dict, rows) -> None:
    """Refuse a materialized subset whose rows differ from what its split file records.

    The recorded fingerprint is only a fast path: a fresh prepare_data run can write the
    same rows under a new fingerprint. Then question IDs, text, and decoded images are
    compared with the split file once, and the match is remembered beside the data.
    """
    external = isinstance(rows.features.get("image"), Value)
    if not external and (rows._fingerprint, qid_hash(rows)) == (
            entry["fingerprint"], entry["qids_sha256"]):
        return
    record = Path(data_dir) / VERIFIED
    verified = json.loads(record.read_text()) if record.exists() else {}
    check = {"fingerprint": rows._fingerprint,
             "entry_sha256": hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()}
    if not external and verified.get(name) == check:
        return
    text = rows_sha256(rows)  # older split files may not record it
    same = (qid_hash(rows) == entry["qids_sha256"]
            and entry.get("rows_sha256", text) == text
            and image_keys(rows, image_root=data_dir) == entry["image_keys"])
    if not same:
        raise ValueError(
            f"{name} differs from the rows its split file records. If you materialized the data "
            "yourself, move data/splits/*.json aside and run python -m scripts.check_data to "
            "build split files for it; its numbers are then not comparable with the recorded "
            "tables.")
    verified[name] = check
    temporary = record.with_suffix(".tmp")
    temporary.write_text(json.dumps(verified, indent=1) + "\n")
    temporary.replace(record)


def split_path(data_dir: Path, dataset: str) -> Path:
    return Path(data_dir) / "splits" / f"{dataset}.json"


def split_sha256(data_dir: Path, dataset: str) -> str:
    return hashlib.sha256(split_path(data_dir, dataset).read_bytes()).hexdigest()


def read_split(data_dir: Path, dataset: str) -> dict:
    path = split_path(data_dir, dataset)
    if not path.exists():
        raise FileNotFoundError(f"No split file for {dataset}; run python -m scripts.check_data")
    split = json.loads(path.read_text())
    validate_split(split)
    return split


def validate_split(split: dict) -> None:
    """A split file must match the registry, and an OOD dataset may never train.

    Each role may draw only from the subset the registry names for it; the rl role may
    also reuse the SFT subset's training rows.
    """
    if split.get("version") != VERSION:
        raise ValueError(f"{split.get('dataset')}: unsupported split file {split.get('version')}")
    spec = DATASETS[split["dataset"]]
    if spec.ladder != "in-domain" and TRAINING_ROLES & set(split["roles"]):
        raise ValueError(f"{spec.name} is {spec.ladder}; it cannot have a training role")
    if set(split["roles"]) != set(spec.subsets):
        raise ValueError(f"{spec.name}: split roles {sorted(split['roles'])} differ from "
                         f"the registry's {sorted(spec.subsets)}")
    for role, parts in split["roles"].items():
        allowed = {spec.subsets[role]}
        if role == "rl" and "sft" in spec.subsets:
            allowed.add(spec.subsets["sft"])
        if any(part["subset"] not in allowed for part in parts):
            raise ValueError(f"{spec.name}: the {role} role may only draw from {sorted(allowed)}")
        if not sum(len(part["indices"]) for part in parts):
            raise ValueError(f"{spec.name}: the {role} role is empty")


def check_all(data_dir: Path) -> dict:
    """Every registered dataset's split file, checked together for held-out leaks."""
    return overlap_report([read_split(data_dir, name) for name, spec in DATASETS.items()
                           if spec.require_split or split_path(data_dir, name).exists()])


def resolve_external_images(rows, data_dir: Path):
    """Small path-only OOD storage resolves against the supplied data root, never the cwd."""
    feature = rows.features.get("image")
    if not isinstance(feature, Value) or feature.dtype != "string":
        return rows

    def resolve_paths(payloads):
        paths = [Path(payload["path"]) for payload in payloads]
        if any(path.is_absolute() or ".." in path.parts for path in paths):
            raise ValueError("External OOD image paths must stay relative to the data root")
        return {"image": [{"bytes": None, "path": str(Path(data_dir).resolve() / path)}
                          for path in paths]}

    rows = rows.cast_column("image", Image(decode=False))
    rows = rows.map(resolve_paths, input_columns=["image"], batched=True, keep_in_memory=True,
                    load_from_cache_file=False)
    return rows.cast_column("image", Image())


def load_role(data_dir: Path, datasets, role: str, seed: int | None = SEED):
    """The rows playing `role` for the datasets, checked against the data on disk.

    Rows are shuffled with `seed` for training; `seed=None` keeps the file's order.
    A dataset without that role is an error, so a trainer cannot load held-out rows, and
    a training role loads only after the global check finds no held-out image in training.
    """
    splits = [read_split(data_dir, dataset) for dataset in datasets]
    for split in splits:
        if role not in split["roles"]:
            raise ValueError(f"{split['dataset']} has no {role} role")
    if role in TRAINING_ROLES:
        check_all(data_dir)
    parts = []
    for split in splits:
        for part in split["roles"][role]:
            name = part["subset"]
            subset = load_from_disk(str(Path(data_dir) / name))
            entry = split["subsets"][name]
            verify_subset(data_dir, name, entry, subset)
            subset = resolve_external_images(subset, data_dir)
            subset = subset.add_column("image_key", entry["image_keys"])
            # Where each row came from, so shuffled rows can still be identified.
            subset = subset.add_column("source_subset", [name] * len(subset))
            subset = subset.add_column("source_row", list(range(len(subset))))
            parts.append(subset.select(part["indices"]))
    rows = concatenate_datasets(parts)
    return rows if seed is None else rows.shuffle(seed=seed)


def provenance(data_dir: Path, datasets) -> dict:
    """Recorded with every training run: which split files defined its data."""
    return {"version": VERSION,
            "split_sha256": {dataset: split_sha256(data_dir, dataset) for dataset in datasets}}


def carve_dev(keys: list[str], candidates: list[int], seed: int, target: int) -> set[str]:
    """Pick whole images, in a seeded order, until at least `target` questions are held out."""
    per_image = {}
    for index in candidates:
        per_image.setdefault(keys[index], []).append(index)
    order = sorted(per_image)
    random.Random(seed).shuffle(order)
    chosen, count = set(), 0
    for key in order:
        if count >= target:
            break
        chosen.add(key)
        count += len(per_image[key])
    if not chosen or len(chosen) == len(per_image):
        raise ValueError("Need enough distinct images for nonempty train and dev pools.")
    return chosen


def build_split(dataset: str, data_dir: Path, other_splits: list[dict],
                seed: int = SEED, dev_target: int = DEV_TARGET) -> dict:
    """A new split file that leaves every existing file unchanged.

    Rows whose images another dataset holds out leave this dataset's training pools and its
    dev role; rows whose images another dataset trains on or uses for dev leave its test
    role. The file records how many rows each role lost this way.
    """
    spec = DATASETS[dataset]
    subsets, keys = {}, {}
    for name in dict.fromkeys(spec.subsets.values()):
        rows = load_from_disk(str(Path(data_dir) / name))
        keys[name] = image_keys(rows, image_root=data_dir)
        subsets[name] = {"n": len(rows), "fingerprint": rows._fingerprint,
                         "qids_sha256": qid_hash(rows), "rows_sha256": rows_sha256(rows),
                         "image_keys": keys[name]}
    others_held = role_keys(other_splits, HELD_OUT_ROLES)
    others_dev = role_keys(other_splits, {"dev"})
    others_train = role_keys(other_splits, TRAINING_ROLES)

    def part(role: str, excluded: set[str]) -> dict:
        name = spec.subsets[role]
        indices = [i for i, key in enumerate(keys[name]) if key not in excluded]
        return {"subset": name, "indices": indices}

    def part_keys(role_part: dict) -> set[str]:
        return {keys[role_part["subset"]][i] for i in role_part["indices"]}

    carved = spec.subsets.get("dev") == spec.subsets.get("sft")
    roles, held = {}, set()
    if "test" in spec.subsets:
        # Another dataset's dev images were used for model selection; keep them out too.
        roles["test"] = [part("test", others_train | others_dev)]
        held |= part_keys(roles["test"][0])
    if "dev" in spec.subsets:
        dev = part("dev", others_train | others_held | held)
        if carved:
            chosen = carve_dev(keys[dev["subset"]], dev["indices"], seed, dev_target)
            dev["indices"] = [i for i in dev["indices"] if keys[dev["subset"]][i] in chosen]
        roles["dev"] = [dev]
        held |= part_keys(dev)
    if "sft" in spec.subsets:
        roles["sft"] = [part("sft", held | others_held)]
    if "rl" in spec.subsets:
        # RL may reuse SFT training examples; it never sees dev or test images.
        roles["rl"] = [part("rl", held | others_held)] + [dict(p) for p in roles.get("sft", [])]
    others = others_held | others_train
    shared = {role: sum(key in others for key in keys[spec.subsets[role]]) for role in roles}
    split = {"version": VERSION, "dataset": dataset, "ladder": spec.ladder,
             "official_splits": sorted(spec.official_splits), "sources": dict(spec.sources),
             "rows_sharing_images_with_other_datasets": shared,
             "subsets": subsets, "roles": roles}
    if carved:
        split["dev_selection"] = {"seed": seed, "target": dev_target,
                                  "method": "whole images from the SFT pool"}
    validate_split(split)
    return split


def roles_sha256(split: dict) -> str:
    return hashlib.sha256(json.dumps(split["roles"], sort_keys=True).encode()).hexdigest()


def role_keys(splits: list[dict], roles) -> set[str]:
    return {
        split["subsets"][part["subset"]]["image_keys"][i]
        for split in splits
        for role, parts in split["roles"].items() if role in roles
        for part in parts
        for i in part["indices"]
    }


def overlap_report(splits: list[dict]) -> dict:
    """Fail on any dev or test image inside any training pool of any dataset.

    Also counts images shared by different training subsets (for example the same COCO
    picture in VQAv2 and GQA); those are allowed and recorded.
    """
    training, held = {}, {}
    for split in splits:
        for role, parts in split["roles"].items():
            for part in parts:
                keys = split["subsets"][part["subset"]]["image_keys"]
                for i in part["indices"]:
                    if role in TRAINING_ROLES:
                        training.setdefault(keys[i], set()).add(part["subset"])
                    else:
                        held.setdefault(keys[i], set()).add(f"{split['dataset']} {role}")
    leaks = held.keys() & training.keys()
    if leaks:
        key = sorted(leaks)[0]
        raise ValueError(f"{len(leaks)} held-out images appear in training pools, e.g. "
                         f"{sorted(held[key])} in {sorted(training[key])}")
    shared = {}
    for subsets in training.values():
        if len(subsets) > 1:
            name = " & ".join(sorted(subsets))
            shared[name] = shared.get(name, 0) + 1
    return {"version": VERSION, "held_out_images": len(held), "training_images": len(training),
            "training_images_shared_across_subsets": dict(sorted(shared.items()))}
