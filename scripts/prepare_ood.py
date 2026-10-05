"""Freeze optional OOD test cohorts after source, scorer and training-image checks.

    python -m scripts.prepare_ood --dataset realworldqa --data-dir data \
        --audit-dir runs/closeout-20261005/ood

Pinned raw files must already be in data/ood-source/<dataset>. The original four
datasets and data/manifest.json are never rewritten. An overlap blocks adoption.
"""

import argparse
import json
from pathlib import Path

from vqar.ood import SOURCE_HASHES, prepare_ood


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(SOURCE_HASHES))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--audit-dir", type=Path, required=True)
    args = parser.parse_args()
    source_dir = args.source_dir or args.data_dir / "ood-source" / args.dataset
    audit = prepare_ood(args.dataset, args.data_dir, source_dir, args.audit_dir)
    print(json.dumps({key: audit[key] for key in ("dataset", "status", "rows", "unique_images",
                                                "split_file", "subset")}, indent=2))


if __name__ == "__main__":
    main()
