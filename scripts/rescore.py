"""Re-score existing runs from their saved raw predictions (no GPU needed).

Used after any metric change: reads results/preds/<run_id>.jsonl, recomputes every
score from the stored raw generation, and rewrites both the preds file and the
metrics block of results/runs/<run_id>.json (bumping git_sha).

Usage:
    uv run python scripts/rescore.py                    # all runs with saved preds
    uv run python scripts/rescore.py zero_shot_2b-vqav2
"""

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

from datasets import load_from_disk

from vqar.inference import extract_answer
from vqar.metrics import score

ROOT = Path(__file__).parent.parent
RUNS = ROOT / "results" / "runs"
PREDS = ROOT / "results" / "preds"


def rescore(run_id: str) -> None:
    run_path = RUNS / f"{run_id}.json"
    preds_path = PREDS / f"{run_id}.jsonl"
    run = json.loads(run_path.read_text())
    dataset = run["eval"]["dataset"]

    type_of = {}
    subset_dir = ROOT / "data" / run["eval"]["subset"]
    if subset_dir.exists():
        ds = load_from_disk(str(subset_dir))
        type_of = dict(zip(ds["qid"], ds["answer_type"], strict=True))

    records, by_type = [], defaultdict(list)
    for line in preds_path.read_text().splitlines():
        r = json.loads(line)
        r["prediction"] = extract_answer(r["raw"])
        r["score"] = score(dataset, r["prediction"], r["answers"])
        records.append(r)
        by_type[type_of.get(r["qid"], "") or "all"].append(r["score"])

    old = run["metrics"]["overall"]
    run["metrics"] = {
        "overall": sum(r["score"] for r in records) / len(records),
        "by_answer_type": {t: sum(v) / len(v) for t, v in sorted(by_type.items())},
    }
    run["git_sha"] = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    run_path.write_text(json.dumps(run, indent=2))
    with preds_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"{run_id}: {old:.4f} -> {run['metrics']['overall']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ids", nargs="*", default=None)
    args = parser.parse_args()
    run_ids = args.run_ids or sorted(
        p.stem for p in RUNS.glob("*.json") if (PREDS / f"{p.stem}.jsonl").exists()
    )
    for run_id in run_ids:
        rescore(run_id)


if __name__ == "__main__":
    main()
