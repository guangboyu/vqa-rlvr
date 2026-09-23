"""Materialize fixed, seeded dataset subsets to data/ so training and eval never
touch the raw sources. Source revisions and fixed seeds are recorded in the
manifest. Existing materializations are preserved; select a new output directory
to build another version.

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

from vqar.registry import DATASETS, SOURCE_REVISIONS  # noqa: E402


def materialize(names: list[str]) -> dict:
    """Build every subset of the named datasets with their registered sizes and seeds."""
    subsets = {}
    for name in names:
        spec = DATASETS[name]
        subsets.update(spec.materialize(spec.sizes))
    return subsets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=sorted(DATASETS), default=None)
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    output_dir = args.out_dir
    selected = [name for name in DATASETS if args.only in (None, name)]
    existing = [
        p.name
        for p in output_dir.glob("*")
        if p.is_dir()
        and p.name.startswith(tuple(name + "_" for name in selected))
    ]
    if existing:
        raise SystemExit(f"Already materialized: {existing}. Use check_data or a new --out-dir.")

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(BUILD_CACHE, ignore_errors=True)  # scratch from a killed prior run
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.setdefault("sizes", {}).update(
        {name: dict(DATASETS[name].sizes) for name in selected}
    )

    try:
        for name, ds in materialize(selected).items():
            out = output_dir / name
            ds.save_to_disk(str(out))
            manifest.setdefault("subsets", {})[name] = {
                "n": len(ds),
                "source_revisions": SOURCE_REVISIONS,
                "qids_sha": hashlib.sha256(",".join(ds["qid"]).encode()).hexdigest(),
            }
            print(f"{name}: {len(ds)} examples -> {out}")
    finally:
        shutil.rmtree(BUILD_CACHE, ignore_errors=True)  # scratch is per-run, never kept

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
