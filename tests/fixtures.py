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


# subset -> (dataset, references per question) for a tiny copy of every registered dataset
LAYOUT = {
    "vqav2_sft": ("vqav2", 1), "vqav2_rl": ("vqav2", 1), "vqav2_eval": ("vqav2", 10),
    "gqa_sft": ("gqa", 1), "gqa_rl": ("gqa", 1), "gqa_eval": ("gqa", 1),
    "clevr_rl": ("clevr", 1), "clevr_val": ("clevr", 1), "clevr_test": ("clevr", 1),
    "textvqa_eval": ("textvqa", 10),
}


def every_dataset(data_dir):
    """Three questions per subset of every registered dataset, with split files."""
    for number, (subset, (dataset, references)) in enumerate(LAYOUT.items()):
        save_subset(data_dir, subset, dataset, [
            (f"{subset}-{i}", png(20 * number + i), "How many?", ["2"] * references, "number")
            for i in range(3)
        ])
    write_splits(data_dir, ["vqav2", "gqa", "clevr", "textvqa"])
