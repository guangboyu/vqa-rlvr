"""Prepare optional OOD test cohorts from pinned local sources without changing training data."""

import hashlib
import json
import shutil
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset, Features, Value, load_from_disk

from vqar.data import FEATURES
from vqar.registry import DATASETS, TRAINING_ROLES
from vqar.splits import (
    VERSION,
    image_keys,
    qid_hash,
    read_split,
    role_keys,
    rows_sha256,
    split_path,
    validate_split,
)

VIZWIZ = DATASETS["vizwiz"]
# Local source files and their checksums. The registry pins VizWiz downloads by URL (stored
# under their file names here) and RealWorldQA by Hub revision, whose two parquet files these are.
SOURCE_HASHES = {
    "vizwiz": {
        **{url.rsplit("/", 1)[-1]: sha256 for url, sha256 in VIZWIZ.sources.items()},
        "native-vqaEval.py": VIZWIZ.metric_metadata["official"]["native_source_sha256"],
    },
    "realworldqa": {
        "test-00000-of-00002.parquet":
            "2cf29e308a34e0255f23520796fc04445a01b8baab18686f1a4b3c3943f695b2",
        "test-00001-of-00002.parquet":
            "dac5b9f7cfb947a4865ea3424abfbdb3449d09e0a0b911a787cdc930cefefdc0",
    },
}

LICENSES = {"vizwiz": "CC-BY-4.0", "realworldqa": "CC-BY-ND-4.0"}
PATH_FEATURES = Features({**FEATURES, "image": Value("string")})


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def checked_sources(dataset: str, source_dir: Path) -> dict:
    files = {}
    for filename, expected in SOURCE_HASHES[dataset].items():
        path = source_dir / filename
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(f"{path}: source checksum differs from the frozen release")
        files[filename] = {"sha256": actual, "bytes": path.stat().st_size}
    return files


def write_source_image(path: Path, payload: bytes) -> None:
    """Retain original encoded bytes; do not re-encode photographs."""
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"{path}: existing extracted image differs from the source")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def vizwiz_rows(source_dir: Path, data_dir: Path):
    annotations = json.loads((source_dir / "VQA_test.json").read_text())
    with zipfile.ZipFile(source_dir / "test.zip") as archive:
        names = {Path(name).name: name for name in archive.namelist() if not name.endswith("/")}
        expected = {row["image"] for row in annotations}
        if len(expected) != len(annotations) or set(names) != expected:
            raise ValueError("VizWiz archive and annotation image identities differ")
        for row in annotations:
            filename = row["image"]
            if Path(filename).name != filename:
                raise ValueError("VizWiz image names must be plain filenames")
            answers = [answer["answer"] for answer in row["answers"]]
            if len(answers) != 10 or any(not isinstance(answer, str) for answer in answers):
                raise ValueError(f"{filename}: expected ten raw string answers")
            image_path = source_dir / "images" / filename
            write_source_image(image_path, archive.read(names[filename]))
            yield {"dataset": "vizwiz", "qid": filename, "question": row["question"],
                   "answers": answers, "answer_type": row["answer_type"],
                   "image": str(image_path.relative_to(data_dir))}


def realworldqa_rows(source_dir: Path, data_dir: Path):
    index = 0
    for filename in SOURCE_HASHES["realworldqa"]:
        parquet = pq.ParquetFile(source_dir / filename)
        for batch in parquet.iter_batches(batch_size=16, columns=["image", "question", "answer"]):
            for row in batch.to_pylist():
                if not isinstance(row["question"], str) or not row["question"].strip():
                    raise ValueError(f"RealWorldQA row {index} has no question")
                if not isinstance(row["answer"], str) or not row["answer"].strip():
                    raise ValueError(f"RealWorldQA row {index} has no reference answer")
                payload = row["image"]["bytes"]
                if payload is None:
                    raise ValueError(f"RealWorldQA row {index} has no embedded source image")
                image_path = source_dir / "images" / f"test-{index:04d}.img"
                write_source_image(image_path, payload)
                yield {"dataset": "realworldqa", "qid": f"test-{index:04d}",
                       "question": row["question"], "answers": [row["answer"]],
                       "answer_type": "", "image": str(image_path.relative_to(data_dir))}
                index += 1


def cohort_audit(rows, keys: list[str], training_keys: set[str]) -> dict:
    by_image = defaultdict(list)
    for qid, key in zip(rows["qid"], keys, strict=True):
        by_image[key].append(qid)
    overlaps = [{"qid": qid, "image_key": key, "source_index": index}
                for index, (qid, key) in enumerate(zip(rows["qid"], keys, strict=True))
                if key in training_keys]
    duplicates = {key: qids for key, qids in by_image.items() if len(qids) > 1}
    return {"rows": len(rows), "unique_images": len(by_image),
            "training_image_overlap": overlaps, "duplicate_image_groups": duplicates,
            "cohort_rule": "all native rows in source order; no duplicates removed",
            "image_groups": [{"qid": qid, "image_key": key}
                             for qid, key in zip(rows["qid"], keys, strict=True)],
            "overlap_scope": "project SFT/RL images; upstream pretraining is unknown",
            "bootstrap_rule": "resample image groups, then compute question-weighted means"}


def prepare_ood(dataset: str, data_dir: Path, source_dir: Path, audit_dir: Path) -> dict:
    """Check source, image isolation and labels before atomically exposing a test split."""
    if dataset not in SOURCE_HASHES:
        raise ValueError(f"Unsupported optional OOD dataset: {dataset}")
    data_dir, source_dir = Path(data_dir).absolute(), Path(source_dir).absolute()
    source_dir.relative_to(data_dir)  # stored paths must relocate with the data directory
    spec = DATASETS[dataset]
    subset_name = spec.subsets["test"]
    target = data_dir / subset_name
    split_file = split_path(data_dir, dataset)
    if target.exists() or split_file.exists():
        raise ValueError(f"{dataset} is already prepared; frozen cohorts are never rewritten")
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / f"{dataset}-cohort-audit.json"
    audit = {"dataset": dataset, "license": LICENSES[dataset], "status": "pending"}
    try:
        audit["source_files"] = checked_sources(dataset, source_dir)
        required = [name for name, entry in DATASETS.items() if entry.require_split]
        audit["posttraining_split_sha256"] = {
            name: file_sha256(split_path(data_dir, name)) for name in required}
        frozen_splits = [read_split(data_dir, name) for name in required]
        training = role_keys(frozen_splits, TRAINING_ROLES)
        loader = vizwiz_rows if dataset == "vizwiz" else realworldqa_rows
        rows = Dataset.from_list(list(loader(source_dir, data_dir)), features=PATH_FEATURES)
        if len(rows) != spec.sizes["eval"] or len(set(rows["qid"])) != len(rows):
            raise ValueError(f"{dataset}: expected {spec.sizes['eval']} distinct source questions")
        keys = image_keys(rows, image_root=data_dir)
        audit.update(cohort_audit(rows, keys, training))
        audit["qids_sha256"] = qid_hash(rows)
        audit["rows_sha256"] = rows_sha256(rows)
        if audit["training_image_overlap"]:
            raise ValueError(f"{dataset}: native cohort overlaps project training images; "
                             "adoption blocked, no cohort filtering performed")
        with tempfile.TemporaryDirectory(prefix=f".{subset_name}-", dir=data_dir) as temporary:
            temporary = Path(temporary)
            rows.save_to_disk(str(temporary))
            stored = load_from_disk(str(temporary))
            split = {"version": VERSION, "dataset": dataset, "ladder": spec.ladder,
                     "official_splits": ["test"], "sources": dict(spec.sources),
                     "source_files": audit["source_files"], "license": LICENSES[dataset],
                     "subsets": {subset_name: {"n": len(stored), "fingerprint": stored._fingerprint,
                         "qids_sha256": audit["qids_sha256"], "rows_sha256": audit["rows_sha256"],
                         "image_keys": keys}},
                     "roles": {"test": [{"subset": subset_name,
                                          "indices": list(range(len(rows)))}]}}
            validate_split(split)
            staged_name = "_split.json"
            (temporary / staged_name).write_text(json.dumps(split, indent=1) + "\n")
            split_file.parent.mkdir(parents=True, exist_ok=True)
            temporary.rename(target)
            try:
                (target / staged_name).replace(split_file)
            except Exception:
                shutil.rmtree(target)  # a failed adoption never leaves a half-prepared cohort
                raise
        audit.update(status="passed", split_file=str(split_file), subset=str(target))
    except Exception as error:
        audit.update(status="failed", error=str(error))
        raise
    finally:
        audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    return audit
