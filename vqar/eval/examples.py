"""The examples a protocol evaluates, in the unified shape, with image interventions."""

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

from datasets import Image, load_from_disk
from PIL import Image as PILImage

from vqar.eval.protocol import Protocol
from vqar.registry import DATASETS
from vqar.splits import image_keys, load_training_subsets, split_dev


@dataclass(frozen=True)
class Example:
    """One question in the unified shape; `images` and `choices` leave room for image-pair
    and multiple-choice datasets. Materialized data stores one image, wrapped here."""

    dataset: str
    qid: str
    images: list[bytes]
    question: str
    answers: list[str]
    answer_type: str
    choices: list[str] | None = None


def to_example(row: dict) -> Example:
    payload = row["image"]
    image = payload["bytes"] if payload["bytes"] is not None else Path(payload["path"]).read_bytes()
    return Example(row["dataset"], row["qid"], [image], row["question"], list(row["answers"]),
                   row["answer_type"], row.get("choices"))


def select_rows(protocol: Protocol, data_dir: Path):
    """The rows a protocol evaluates, in a fixed order, and the identity of that data."""
    spec = DATASETS[protocol.dataset]
    split_sha256 = None
    if protocol.role == "test":
        rows = load_from_disk(str(data_dir / spec.subsets["test"]))
    else:
        rows, split_sha256 = frozen_dev_rows(protocol.dataset, data_dir)
    fingerprint = rows._fingerprint
    limit = Protocol.TIERS[protocol.tier]
    if limit is not None:
        rows = rows.select(range(min(limit, len(rows))))
    rows = rows.cast_column("image", Image(decode=False))
    qids = list(rows["qid"])
    data = {
        "fingerprint": fingerprint,
        "qids_sha256": hashlib.sha256(json.dumps(qids).encode()).hexdigest(),
        "split_sha256": split_sha256,
    }
    return rows, data


def frozen_dev_rows(dataset: str, data_dir: Path):
    """The frozen dev split: whole images held out of the SFT pools by the split check."""
    path = data_dir / "split_manifest.json"
    manifest = json.loads(path.read_text())
    spec = manifest["sft_dev"]
    if DATASETS[dataset].subsets.get("sft") not in spec["subsets"]:
        raise ValueError(f"The frozen dev split has no {DATASETS[dataset].display_name} images.")
    _, dev = split_dev(load_training_subsets(data_dir, spec["subsets"], spec["seed"]))
    rows = dev.select([i for i, name in enumerate(dev["dataset"]) if name == dataset])
    return rows, hashlib.sha256(path.read_bytes()).hexdigest()


def image_donors(rows, image_mode: str) -> list[int]:
    """Which row supplies each question's image: itself, or a row with different pixels."""
    if image_mode != "swapped":
        return list(range(len(rows)))
    keys = image_keys(rows)
    if len(set(keys)) < 2:
        raise ValueError("Swapped-image evaluation needs at least two different images.")
    donors = []
    for i, key in enumerate(keys):
        j = (i + 1) % len(keys)
        while keys[j] == key:
            j = (j + 1) % len(keys)
        donors.append(j)
    return donors


def black_like(image: bytes) -> bytes:
    """A black RGB image with the original size, so the vision-token count is unchanged."""
    with PILImage.open(io.BytesIO(image)) as original:
        blank = PILImage.new("RGB", original.size)
    buffer = io.BytesIO()
    blank.save(buffer, format="PNG")
    return buffer.getvalue()


def shown_images(example: Example, donor: Example, image_mode: str) -> list[bytes]:
    if image_mode == "black":
        return [black_like(image) for image in example.images]
    return donor.images
