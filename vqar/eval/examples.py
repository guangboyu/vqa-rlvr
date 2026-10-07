"""The examples a protocol evaluates, in the unified shape, with image interventions."""

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

from datasets import Image, Sequence
from PIL import Image as PILImage

from vqar.data import to_uniform_canvas
from vqar.eval.protocol import Protocol
from vqar.registry import DATASETS, TRAINING_ROLES
from vqar.splits import SEED, load_role, read_split


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
    payloads = row.get("images") or [row["image"]]  # multi-image rows list every shown image
    images = [p["bytes"] if p["bytes"] is not None else Path(p["path"]).read_bytes()
              for p in payloads]
    return Example(row["dataset"], row["qid"], images, row["question"], list(row["answers"]),
                   row["answer_type"], row.get("choices"))


def select_rows(protocol: Protocol, data_dir: Path, limit: int | None = None):
    """The rows a protocol evaluates and the identity of that data.

    Dev and test rows keep the split file's order. A training role (read only by the
    learnability pre-check and the screening battery) is shuffled with the training seed,
    so a tier is a random sample of the dataset's pool; a run that mixes datasets shuffles
    the combined pool, so its draw order differs. The identity names the source subsets
    and hashes the ordered (subset, row, question id) selection, so it changes exactly when
    the evaluated rows change; editing another role of the same split file leaves it alone.
    Swapped images also pin the donor of each row. `limit` overrides the tier's question count
    (historical records only).
    """
    order = SEED if protocol.role in TRAINING_ROLES else None
    rows = load_role(data_dir, [protocol.dataset], protocol.role, seed=order)
    split = read_split(data_dir, protocol.dataset)
    parts = split["roles"][protocol.role]
    limit = limit or Protocol.tier_size(protocol.tier)
    if limit is not None:
        rows = rows.select(range(min(limit, len(rows))))
    rows = rows.cast_column("image", Image(decode=False))
    if "images" in rows.column_names:
        rows = rows.cast_column("images", Sequence(Image(decode=False)))
    selection = [list(entry) for entry in zip(rows["source_subset"], rows["source_row"],
                                              rows["qid"], strict=True)]
    data = {"subsets": {part["subset"]: split["subsets"][part["subset"]]["fingerprint"]
                        for part in parts},
            "selection_sha256": sha256_json(selection)}
    if not DATASETS[protocol.dataset].require_split:
        data["image_groups_sha256"] = sha256_json(
            [list(pair) for pair in zip(rows["qid"], rows["image_key"], strict=True)])
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


def black_image(size: tuple[int, int]) -> PILImage.Image:
    """A black RGB image of the given size, so the vision-token count is unchanged."""
    return PILImage.new("RGB", size)


def png_bytes(image: PILImage.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def black_like(image: bytes) -> bytes:
    """The black image with the original's size, PNG-encoded."""
    with PILImage.open(io.BytesIO(image)) as original:
        blank = black_image(original.size)
    return png_bytes(blank)


def canvas_like(image: bytes) -> bytes:
    """The image as RL training shows it: decoded the way the training data loader decodes
    it (EXIF orientation applied), placed on the RL canvas, and PNG-encoded so the engine
    receives exactly those pixels."""
    return png_bytes(to_uniform_canvas(Image().decode_example({"bytes": image, "path": None})))


def shown_images(example: Example, donor: Example, image_mode: str) -> list[bytes]:
    if image_mode == "first":
        return example.images[:1]
    if image_mode == "none":
        return []
    if image_mode == "black":
        return [black_like(image) for image in example.images]
    if image_mode == "canvas":
        return [canvas_like(image) for image in example.images]
    return donor.images
