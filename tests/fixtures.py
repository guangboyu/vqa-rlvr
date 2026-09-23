"""Tiny materialized datasets and split files shared by the harness tests."""

import io
import json

from datasets import Dataset
from PIL import Image as PILImage

from vqar.data import FEATURES
from vqar.splits import build_split, split_path


def png(shade: int, size=(4, 4)) -> bytes:
    buffer = io.BytesIO()
    PILImage.new("RGB", size, (shade, shade, shade)).save(buffer, format="PNG")
    return buffer.getvalue()


def save_subset(data_dir, name, dataset, rows):
    records = [
        {"dataset": dataset, "qid": qid, "image": {"bytes": image, "path": None},
         "question": question, "answers": answers, "answer_type": answer_type}
        for qid, image, question, answers, answer_type in rows
    ]
    Dataset.from_list(records, features=FEATURES).save_to_disk(str(data_dir / name))


def write_splits(directory, datasets):
    """Split files for tiny fixture datasets, built the same way as for real data."""
    built = []
    for dataset in datasets:
        built.append(build_split(dataset, directory, built, seed=0, dev_target=1))
        path = split_path(directory, dataset)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(built[-1]))
