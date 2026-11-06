"""Per-dataset split files: which materialized rows play each role (sft, rl, dev, test, published).

Images are identified by a hash of their decoded RGB pixels, so re-encoded copies still
match. The global check keeps every dev and test image out of every training pool across
datasets, and an OOD dataset out of training entirely. Trainers load only their own role.
A benchmark's `published` role is exempt from all of this: it keeps every row its authors release,
while its `test` role holds the clean rows: each shows an image, shares no image with training
even resized as far as a perceptual hash can tell, and repeats no training question.
Files live under data/splits/ and are versioned; source data stays unchanged.
"""

import functools
import hashlib
import io
import json
import random
import re
from pathlib import Path

import numpy as np
from datasets import Image, Value, concatenate_datasets, load_from_disk
from PIL import Image as PILImage

from vqar.answers import reference_form
from vqar.metrics import normalize_answer
from vqar.registry import DATASETS, TRAINING_ROLES

VERSION = "split-files-v1"
HELD_OUT_ROLES = frozenset({"dev", "test"})
DEV_TARGET = 500  # questions carved from a training pool when a dataset has no dev subset
SEED = 42
VERIFIED = ".verified-subsets.json"  # content checks already passed, kept beside the data


def pixel_key(raw: bytes) -> str:
    with PILImage.open(io.BytesIO(raw)) as image:
        image = image.convert("RGB")
        pixels = str(image.size).encode() + b":" + image.tobytes()
    return hashlib.sha256(pixels).hexdigest()


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
            cache[encoded_key] = pixel_key(raw)
        keys.append(cache[encoded_key])
    return keys


def stored_images(rows, column: str):
    """Each row's stored {bytes, path} payloads, undecoded. This reads Arrow directly: casting to
    an undecoded image list writes a cache as large as the images beside the subset, and these
    checks read other datasets' subsets, which they must leave as they found them."""
    for batch in rows.select_columns([column]).with_format("arrow").iter(batch_size=64):
        for value in batch[column].to_pylist():
            yield value if column == "images" else [value]


def extra_image_keys(rows) -> dict[str, list[str]]:
    """Row index -> keys of the images after the first, for multi-image rows. Only the first image
    has a place in `image_keys`, yet every shown image must stay out of training."""
    if "images" not in rows.column_names:
        return {}
    extras = {}
    for index, payloads in enumerate(stored_images(rows, "images")):
        if len(payloads) > 1:
            extras[str(index)] = [pixel_key(payload["bytes"]) for payload in payloads[1:]]
    return extras


def gray_key(raw: bytes) -> str:
    """A key that also matches a copy differing only by color mode, such as an RGBA diagram saved
    as L: its RGB pixels change, its gray pixels do not. It hashes one channel where `pixel_key`
    hashes three, so the two kinds of key can share a set without colliding."""
    with PILImage.open(io.BytesIO(raw)) as image:
        image = image.convert("L")
        pixels = str(image.size).encode() + b":" + image.tobytes()
    return hashlib.sha256(pixels).hexdigest()


def read_shown_images(data_dir: Path, subset: str, read) -> list[list]:
    """Per row, `read(raw)` of every image it shows, once per distinct encoded image."""
    rows = load_from_disk(str(Path(data_dir) / subset))
    column = "images" if "images" in rows.column_names else "image"
    cache, values = {}, []
    for payloads in stored_images(rows, column):
        row_values = []
        for payload in payloads:
            raw = payload["bytes"] if payload["bytes"] is not None else Path(
                payload["path"]).read_bytes()
            encoded = hashlib.sha256(raw).digest()
            if encoded not in cache:
                cache[encoded] = read(raw)
            row_values.append(cache[encoded])
        values.append(row_values)
    return values


@functools.cache
def subset_gray_keys(data_dir: Path, subset: str) -> tuple[frozenset[str], ...]:
    """Per row, the gray keys of every image it shows, read from the materialized subset: split
    files must stay as they are, so they cannot carry this key. One pass per subset and process."""
    return tuple(frozenset(keys) for keys in read_shown_images(data_dir, subset, gray_key))


# Resized copies: a benchmark's clean rows lose an image whose 64-bit dHash, taken on a 64 x 64 gray
# thumbnail, exactly equals a training image's and whose thumbnail differs from it by under 3 gray
# levels on average. The confirmation keeps hash collisions between similar charts and pages. Any
# copy altered enough to change one hash bit goes unseen, so the count is a lower bound.
NEAR_DUPLICATE_THUMBNAIL = (64, 64)
NEAR_DUPLICATE_MAX_DIFFERENCE = 3
NEAR_DUPLICATE_MIN_SPREAD = 3  # near-uniform images all hash alike and say nothing
# The CT montages take 120 GB of decoding and M3D-VQA's images are black; no public benchmark holds
# either, so the pass does not read them.
NEAR_DUPLICATE_SKIPPED = frozenset({"deeptumor", "deeptumorx", "deeptumorv", "deeptumorff",
                                    "m3dvqa"})


def near_key(raw: bytes) -> tuple[str, bytes] | None:
    """An image's dHash and thumbnail, or None for a near-uniform image."""
    with PILImage.open(io.BytesIO(raw)) as image:
        image.draft("L", (256, 256))
        small = image.convert("L").resize(NEAR_DUPLICATE_THUMBNAIL, PILImage.BILINEAR)
    if np.asarray(small).std() < NEAR_DUPLICATE_MIN_SPREAD:
        return None
    coarse = np.asarray(small.resize((9, 8), PILImage.BILINEAR), dtype=np.int16)
    return np.packbits(coarse[:, 1:] > coarse[:, :-1]).tobytes().hex(), small.tobytes()


@functools.cache
def subset_near_keys(data_dir: Path, subset: str) -> tuple[tuple[tuple[str, bytes], ...], ...]:
    """Per row, the near keys of every image it shows. One pass per subset and process."""
    return tuple(tuple(key for key in keys if key is not None)
                 for keys in read_shown_images(data_dir, subset, near_key))


def near_duplicate_pools(splits: list[dict]) -> list[str]:
    """The subsets behind the training roles the near-duplicate pass reads."""
    return sorted({part["subset"] for split in splits
                   if split["dataset"] not in NEAR_DUPLICATE_SKIPPED
                   for role, parts in split["roles"].items() if role in TRAINING_ROLES
                   for part in parts})


def near_duplicate_index(splits: list[dict], data_dir: Path) -> dict[str, set[bytes]]:
    """dHash -> thumbnails of every image shown in the training roles of the pass's pools."""
    index = {}
    for split in splits:
        if split["dataset"] in NEAR_DUPLICATE_SKIPPED:
            continue
        for role, parts in split["roles"].items():
            if role not in TRAINING_ROLES:
                continue
            for part in parts:
                rows = subset_near_keys(Path(data_dir), part["subset"])
                for i in part["indices"]:
                    for digest, small in rows[i]:
                        index.setdefault(digest, set()).add(small)
    return index


def near_duplicate_rows(rows, index: dict[str, set[bytes]]) -> set[int]:
    """Rows showing an image whose dHash and thumbnail match one in the index."""
    def close(small: bytes, other: bytes) -> bool:
        difference = np.frombuffer(small, np.uint8).astype(int) - np.frombuffer(other, np.uint8)
        return np.abs(difference).mean() < NEAR_DUPLICATE_MAX_DIFFERENCE

    return {i for i, keys in enumerate(rows) for digest, small in keys
            if any(close(small, other) for other in index.get(digest, ()))}


def role_gray_keys(splits: list[dict], roles, data_dir: Path) -> set[str]:
    return {key for split in splits
            for role, parts in split["roles"].items() if role in roles
            for part in parts
            for i in part["indices"]
            for key in subset_gray_keys(Path(data_dir), part["subset"])[i]}


_QUESTION_PREFIX = re.compile(r"(?:^|\n)\s*question:\s*", re.IGNORECASE)
_OPTIONS_START = re.compile(r"\n\s*(?:options?:|choices?:|\(?[A-Z][.):]\s)", re.IGNORECASE)
_IMAGE_TOKEN = re.compile(r"<image(?:\s*\d+)?>", re.IGNORECASE)


def question_stem(question: str) -> str:
    """The question as asked: no hint before it, no options or instructions after, no image
    placeholder. Loaders wrap questions differently, and the wrapping should not hide a repeat."""
    text = _QUESTION_PREFIX.split(question)[-1]
    return _OPTIONS_START.split(_IMAGE_TOKEN.sub(" ", text))[0]


def reference_text(answers: list[str]) -> str:
    """The option's text for a multiple-choice row, else the most common reference."""
    return answers[1] if reference_form(answers) == "choice" else max(answers, key=answers.count)


def question_key(question: str, answers: list[str]) -> str | None:
    """A question repeated from one row to another: its normalized text and reference answer. A
    question that is all placeholder and options (MMMU has some) is compared whole. A yes/no
    reference gives no key: a generic question on a different image is not a leak."""
    if reference_form(answers) == "yes/no":
        return None
    stem = question_stem(question).strip() or question
    return f"{normalize_answer(stem)}\n{normalize_answer(reference_text(answers))}"


def question_keys(data_dir: Path, subset: str) -> list[str | None]:
    rows = load_from_disk(str(Path(data_dir) / subset)).select_columns(["question", "answers"])
    return [question_key(question, answers)
            for question, answers in zip(rows["question"], rows["answers"], strict=True)]


def role_question_keys(splits: list[dict], roles, data_dir: Path) -> set[str]:
    """Question keys of the rows playing `roles`. Split files record no text, so this reads the
    materialized subsets; a missing subset is an error, never a check silently skipped."""
    per_subset, found = {}, set()
    for split in splits:
        for role, parts in split["roles"].items():
            if role not in roles:
                continue
            for part in parts:
                name = part["subset"]
                if name not in per_subset:
                    per_subset[name] = question_keys(data_dir, name)
                found.update(per_subset[name][i] for i in part["indices"])
    found.discard(None)
    return found


def rows_without_images(data_dir: Path, subset: str) -> set[int]:
    """Rows released with no image at all, which only a multi-image subset can hold."""
    rows = load_from_disk(str(Path(data_dir) / subset))
    if "images" not in rows.column_names:
        return set()
    return {i for i, payloads in enumerate(stored_images(rows, "images")) if not payloads}


def clean_rows(row_images: list[set[str]], row_questions: list[str | None], seen_images: set[str],
               seen_questions: set[str], no_image: set[int] = frozenset(),
               near_copies: set[int] = frozenset()) -> tuple[list[int], dict]:
    """A benchmark's clean rows and how many each check caught. `image_overlap` and
    `question_overlap` each count all their rows and `both` the rows in each; `no_image` counts
    questions released without an image. `near_duplicate_lower_bound` counts the rows only the
    near-duplicate pass removes, and more resized copies may remain among the clean rows."""
    indices, by_image, by_question, both, near = [], 0, 0, 0, 0
    for index, (images, question) in enumerate(zip(row_images, row_questions, strict=True)):
        image_hit, question_hit = bool(images & seen_images), question in seen_questions
        by_image += image_hit
        by_question += question_hit
        both += image_hit and question_hit
        if image_hit or question_hit or index in no_image:
            continue
        if index in near_copies:
            near += 1
        else:
            indices.append(index)
    return indices, {"published": len(row_images), "test": len(indices), "image_overlap": by_image,
                     "question_overlap": by_question, "both": both, "no_image": len(no_image),
                     "near_duplicate_lower_bound": near}


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
    if external:
        # The fingerprint does not cover image files stored by path, so the remembered match
        # also covers each file's size and modification time.
        check["files_sha256"] = external_files_signature(rows, data_dir)
    if verified.get(name) == check:
        return
    text = rows_sha256(rows)  # older split files may not record it
    same = (qid_hash(rows) == entry["qids_sha256"]
            and entry.get("rows_sha256", text) == text
            and image_keys(rows, image_root=data_dir) == entry["image_keys"]
            and ("extra_image_keys" not in entry
                 or extra_image_keys(rows) == entry["extra_image_keys"]))
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


def external_files_signature(rows, data_dir: Path) -> str:
    """Path, size and modification time of every image file a path-stored subset reads."""
    digest = hashlib.sha256()
    for relative in rows["image"]:
        stat = (Path(data_dir) / relative).stat()
        digest.update(f"{relative}\t{stat.st_size}\t{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


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
    if "published" in split["roles"]:
        published = {i for part in split["roles"]["published"] for i in part["indices"]}
        if any(i not in published for part in split["roles"]["test"] for i in part["indices"]):
            raise ValueError(f"{spec.name}: clean rows must be published rows")


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

    A benchmark (a dataset with a `published` role) keeps every row there; its test role also
    loses rows repeating a training question, rows without an image and resized copies of training
    images, and a later training pool loses rows repeating a question of a benchmark's test role
    or showing one of its images in another color mode. Other datasets' test roles keep the
    exact-image rule alone, so rebuilding one cannot change a recorded table.

    Register a benchmark before a new training source: the color-mode check runs only when the
    pool is built, because existing split files carry no gray keys and decoding every training
    pool for each new benchmark would take an hour.
    """
    spec = DATASETS[dataset]
    subsets, keys, extras = {}, {}, {}
    for name in dict.fromkeys(spec.subsets.values()):
        rows = load_from_disk(str(Path(data_dir) / name))
        keys[name] = image_keys(rows, image_root=data_dir)
        extras[name] = extra_image_keys(rows)
        subsets[name] = {"n": len(rows), "fingerprint": rows._fingerprint,
                         "qids_sha256": qid_hash(rows), "rows_sha256": rows_sha256(rows),
                         "image_keys": keys[name]}
        if extras[name]:
            subsets[name]["extra_image_keys"] = extras[name]
    others_held = role_keys(other_splits, HELD_OUT_ROLES)
    others_dev = role_keys(other_splits, {"dev"})
    others_train = role_keys(other_splits, TRAINING_ROLES)
    # Only a dataset that trains can repeat a benchmark's question or image into training.
    benchmark_questions, benchmark_grays = set(), set()
    if TRAINING_ROLES & set(spec.subsets):
        benchmarks = [other for other in other_splits if "published" in other["roles"]]
        benchmark_questions = role_question_keys(benchmarks, {"test"}, data_dir)
        benchmark_grays = role_gray_keys(benchmarks, {"test"}, data_dir)

    def part(role: str, excluded: set[str], questions=frozenset(), grays=frozenset()) -> dict:
        name = spec.subsets[role]
        indices = [i for i, key in enumerate(keys[name]) if key not in excluded]
        if questions:
            own = question_keys(data_dir, name)
            indices = [i for i in indices if own[i] not in questions]
        if grays:
            own_grays = subset_gray_keys(Path(data_dir), name)
            indices = [i for i in indices if not own_grays[i] & grays]
        return {"subset": name, "indices": indices}

    def part_keys(role_part: dict) -> set[str]:
        return {keys[role_part["subset"]][i] for i in role_part["indices"]}

    carved = "dev" in spec.subsets and spec.subsets["dev"] == spec.subsets.get("sft")
    roles, held, clean = {}, set(), None
    if "published" in spec.subsets:
        name = spec.subsets["test"]
        row_images = [{key, *extras[name].get(str(i), ())} for i, key in enumerate(keys[name])]
        near_copies = near_duplicate_rows(subset_near_keys(Path(data_dir), name),
                                          near_duplicate_index(other_splits, data_dir))
        indices, clean = clean_rows(row_images, question_keys(data_dir, name),
                                    others_train | others_dev,
                                    role_question_keys(other_splits, TRAINING_ROLES, data_dir),
                                    rows_without_images(data_dir, name), near_copies)
        roles["test"] = [{"subset": name, "indices": indices}]
        roles["published"] = [{"subset": spec.subsets["published"],
                               "indices": list(range(len(keys[name])))}]
        held |= part_keys(roles["test"][0])
    elif "test" in spec.subsets:
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
        roles["sft"] = [part("sft", held | others_held, benchmark_questions, benchmark_grays)]
    if "rl" in spec.subsets:
        # RL may reuse SFT training examples; it never sees dev or test images. A dataset
        # whose RL role reads its SFT subset reuses exactly the SFT rows.
        shares_sft = spec.subsets["rl"] == spec.subsets.get("sft")
        own = ([] if shares_sft
               else [part("rl", held | others_held, benchmark_questions, benchmark_grays)])
        roles["rl"] = own + [dict(p) for p in roles.get("sft", [])]
    others = others_held | others_train
    shared = {role: sum(key in others for key in keys[spec.subsets[role]]) for role in roles}
    split = {"version": VERSION, "dataset": dataset, "ladder": spec.ladder,
             "official_splits": sorted(spec.official_splits), "sources": dict(spec.sources),
             "rows_sharing_images_with_other_datasets": shared,
             "subsets": subsets, "roles": roles}
    if clean:
        split["clean_rows"] = clean
        split["near_duplicate_pass"] = {
            "hash": "64-bit dHash of a 64x64 gray thumbnail, exact match",
            "confirmed_by": f"mean gray difference under {NEAR_DUPLICATE_MAX_DIFFERENCE} "
                            "on the thumbnail",
            "pools": near_duplicate_pools(other_splits),
            "skipped": sorted(NEAR_DUPLICATE_SKIPPED)}
    if carved:
        split["dev_selection"] = {"seed": seed, "target": dev_target,
                                  "method": "whole images from the SFT pool"}
    validate_split(split)
    return split


def roles_sha256(split: dict) -> str:
    return hashlib.sha256(json.dumps(split["roles"], sort_keys=True).encode()).hexdigest()


def part_image_keys(split: dict, part: dict) -> list[str]:
    """Every image a role's part shows, the later images of multi-image rows included."""
    entry = split["subsets"][part["subset"]]
    extras = entry.get("extra_image_keys", {})
    return [key for i in part["indices"]
            for key in (entry["image_keys"][i], *extras.get(str(i), ()))]


def role_keys(splits: list[dict], roles) -> set[str]:
    return {key for split in splits
            for role, parts in split["roles"].items() if role in roles
            for part in parts
            for key in part_image_keys(split, part)}


def overlap_report(splits: list[dict]) -> dict:
    """Fail on any dev or test image inside any training pool of any dataset.

    Also counts images shared by different training subsets (for example the same COCO
    picture in VQAv2 and GQA); those are allowed and recorded. A `published` role may overlap
    training by definition, so it is not checked.
    """
    training, held = {}, {}
    for split in splits:
        for role, parts in split["roles"].items():
            if role == "published":
                continue
            for part in parts:
                for key in part_image_keys(split, part):
                    if role in TRAINING_ROLES:
                        training.setdefault(key, set()).add(part["subset"])
                    else:
                        held.setdefault(key, set()).add(f"{split['dataset']} {role}")
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
