"""Declare an RL pool from learnability records: the questions a starting model gets right
on some but not all of its sampled answers.

    python -m scripts.learnability --preset grpo_2b_main --label base2 --tier first-2500
    python -m scripts.build_pool --name mixed-base2 results/records/base2-*/*-rl-first-2500-*.json
    python -m scripts.build_pool --name overfit32-chartqa-base2 --limit 32 \
        results/records/base2-*/chartqa-rl-dev-reasoning-t1k8-*.json

Writes data/pools/<name>.json; train_grpo.py --pool <name> trains on it. The selection
model, rule, and source records are part of the pool's definition.
"""

import argparse
import json
from pathlib import Path

from vqar.eval import RunRecord
from vqar.pools import filtered_pool, pool_path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("records", nargs="+", type=Path, help="learnability records")
    parser.add_argument("--name", required=True)
    parser.add_argument("--keep", nargs=2, type=int, default=(1, 7), metavar=("LOW", "HIGH"),
                        help="correct samples per question to keep (default 1 7 of 8)")
    parser.add_argument("--limit", type=int, default=None,
                        help="keep only the first N kept questions (an overfitting check)")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    records = [RunRecord.load(path) for path in args.records]
    pool = filtered_pool(args.name, records, *args.keep, limit=args.limit)
    path = pool_path(args.data_dir, args.name)
    if path.exists():
        raise SystemExit(f"{path} exists; a declared pool is never rewritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pool, indent=1) + "\n")
    scored = sum(source["scored"] for source in pool["selection"]["records"])
    print(f"{len(pool['members'])} of {scored} questions kept -> {path}")


if __name__ == "__main__":
    main()
