"""Materialize fixed, seeded dataset subsets to data/ so training and eval never
touch the raw sources. Re-running reproduces identical subsets (fixed seeds) and
writes data/manifest.json recording exactly what was sampled.

Usage:
    uv run python prepare_data.py                # all subsets
    uv run python prepare_data.py --only clevr   # one source
"""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
BUILD_CACHE = DATA_DIR / ".build"

# Parallel-chunk downloads + fail fast on stalled connections (must precede datasets import).
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
# Route ALL arrow builds (extraction, from_generator, shuffle indices) to a scratch
# cache wiped after each stage. Interrupted runs otherwise accumulate multi-GB
# partial builds in ~/.cache (this once filled the disk). Hub downloads still land
# in the resumable hub cache.
os.environ["HF_DATASETS_CACHE"] = str(BUILD_CACHE)

from vqar import data  # noqa: E402

# Subset sizes and seeds are fixed project-wide; change = new benchmark version.
SIZES = {
    "vqav2": {"sft": 20_000, "rl": 8_000, "eval": 5_000, "train_seed": 42, "eval_seed": 0},
    "gqa": {"sft": 20_000, "rl": 8_000, "train_seed": 42},  # eval = full testdev_balanced
    "clevr": {"rl": 8_000, "val": 3_000, "test": 3_000, "seed": 42},
    "textvqa": {"eval": 2_000, "eval_seed": 0},
}


def build(only: str | None) -> dict:
    subsets = {}
    if only in (None, "vqav2"):
        cfg = SIZES["vqav2"]
        sft, rl = data.load_vqav2_train(cfg["sft"], cfg["rl"], cfg["train_seed"])
        subsets["vqav2_sft"] = sft
        subsets["vqav2_rl"] = rl
        subsets["vqav2_eval"] = data.load_vqav2_eval(cfg["eval"], cfg["eval_seed"])
    if only in (None, "gqa"):
        cfg = SIZES["gqa"]
        sft, rl = data.load_gqa_train(cfg["sft"], cfg["rl"], cfg["train_seed"])
        subsets["gqa_sft"] = sft
        subsets["gqa_rl"] = rl
        subsets["gqa_eval"] = data.load_gqa_eval()
    if only in (None, "clevr"):
        cfg = SIZES["clevr"]
        splits = data.load_clevr(cfg["rl"], cfg["val"], cfg["test"], cfg["seed"])
        for name, ds in splits.items():
            subsets[f"clevr_{name}"] = ds
    if only in (None, "textvqa"):
        cfg = SIZES["textvqa"]
        subsets["textvqa_eval"] = data.load_textvqa_eval(cfg["eval"], cfg["eval_seed"])
    return subsets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["vqav2", "gqa", "clevr", "textvqa"], default=None)
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    shutil.rmtree(BUILD_CACHE, ignore_errors=True)  # scratch from a killed prior run
    manifest_path = DATA_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.setdefault("sizes", {}).update(
        {k: v for k, v in SIZES.items() if args.only in (None, k)}
    )

    try:
        for name, ds in build(args.only).items():
            out = DATA_DIR / name
            ds.save_to_disk(str(out))
            manifest.setdefault("subsets", {})[name] = {
                "n": len(ds),
                "qids_sha": hashlib.sha256(",".join(ds["qid"]).encode()).hexdigest(),
            }
            print(f"{name}: {len(ds)} examples -> {out}")
    finally:
        shutil.rmtree(BUILD_CACHE, ignore_errors=True)  # scratch is per-run, never kept

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
