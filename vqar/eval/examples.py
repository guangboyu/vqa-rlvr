"""The examples a protocol evaluates, in the unified shape, with image interventions."""

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

from datasets import Image
from PIL import Image as PILImage

from vqar.eval.protocol import Protocol
from vqar.splits import load_role, read_split


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
    """The rows a protocol evaluates, in the split file's order, and the identity of that data.

    The identity names the source subsets and hashes the ordered (subset, row, question id)
    selection, so it changes exactly when the evaluated rows change; editing another role
    of the same split file leaves it alone. Swapped images also pin the donor of each row.
    """
    rows = load_role(data_dir, [protocol.dataset], protocol.role, seed=None)
    split = read_split(data_dir, protocol.dataset)
    parts = split["roles"][protocol.role]
    selection = [[part["subset"], index] for part in parts for index in part["indices"]]
    limit = Protocol.TIERS[protocol.tier]
    if limit is not None:
        rows = rows.select(range(min(limit, len(rows))))
        selection = selection[:limit]
    rows = rows.cast_column("image", Image(decode=False))
    selection = [entry + [qid] for entry, qid in zip(selection, rows["qid"], strict=True)]
    data = {"subsets": {part["subset"]: split["subsets"][part["subset"]]["fingerprint"]
                        for part in parts},
            "selection_sha256": sha256_json(selection)}
    if protocol.image_mode == "swapped":
        data["donors_sha256"] = sha256_json(image_donors(rows, "swapped"))
    return rows, data


def sha256_json(value) -> str:
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


def image_donors(rows, image_mode: str) -> list[int]:
    """Which row supplies each question's image: itself, or a row with different pixels."""
    if image_mode != "swapped":
        return list(range(len(rows)))
    keys = rows["image_key"]
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
