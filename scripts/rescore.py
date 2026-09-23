"""Re-score existing runs from their saved raw predictions (no GPU needed).

Used after any metric change: reads results/preds/<run_id>.jsonl, recomputes every
score from the stored raw generation, and writes a separate result under
results/rescored/<metric>/. Original predictions and generation provenance stay intact.

Usage:
    uv run python scripts/rescore.py                    # all runs with saved preds
    uv run python scripts/rescore.py zero_shot_2b-vqav2
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from datasets import load_from_disk

from vqar.inference import extract_answer
from vqar.registry import score

ROOT = Path(__file__).parent.parent
RUNS = ROOT / "results" / "runs"
PREDS = ROOT / "results" / "preds"


def rescore(run_id: str, metric: str = "official", output_dir: Path | None = None) -> dict:
    run_path = RUNS / f"{run_id}.json"
    preds_path = PREDS / f"{run_id}.jsonl"
    run = json.loads(run_path.read_text())
    dataset = run["eval"]["dataset"]

    type_of, answers_of = {}, {}
    subset_dir = ROOT / "data" / run["eval"]["subset"]
    if subset_dir.exists():
        ds = load_from_disk(str(subset_dir))
        type_of = dict(zip(ds["qid"], ds["answer_type"], strict=True))
        answers_of = dict(zip(ds["qid"], ds["answers"], strict=True))

    records, by_type, seen = [], defaultdict(list), set()
    for line in preds_path.read_text().splitlines():
        r = json.loads(line)
        if r["qid"] in seen:
            raise ValueError(f"Duplicate question {r['qid']} in {preds_path}")
        seen.add(r["qid"])
        if answers_of and answers_of.get(r["qid"]) != r["answers"]:
            raise ValueError(f"References disagree with materialized data: {r['qid']}")
        r["prediction"] = extract_answer(r["raw"])
        r["score"] = score(dataset, r["prediction"], r["answers"], metric)
        records.append(r)
        by_type[type_of.get(r["qid"], "") or "all"].append(r["score"])

    if not records or len(records) != run["eval"]["n"]:
        raise ValueError(f"Prediction count disagrees with run metadata: {run_id}")
    old = run["metrics"]["overall"]
    run["metrics"] = {
        "overall": sum(r["score"] for r in records) / len(records),
        "by_answer_type": {t: sum(v) / len(v) for t, v in sorted(by_type.items())},
    }
    run["eval"]["metric_variant"] = metric
    run["rescore"] = {
        "source_run": str(run_path.relative_to(ROOT)),
        "source_run_sha256": hashlib.sha256(run_path.read_bytes()).hexdigest(),
        "source_predictions_sha256": hashlib.sha256(preds_path.read_bytes()).hexdigest(),
        "qids_sha256": hashlib.sha256(",".join(r["qid"] for r in records).encode()).hexdigest(),
        "scorer_sha256": hashlib.sha256((ROOT / "vqar/metrics.py").read_bytes()).hexdigest(),
        "previous_overall": old,
    }
    output_dir = output_dir or ROOT / "results/rescored" / metric
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{run_id}.json"
    if output.resolve() == run_path.resolve():
        raise ValueError("Rescoring must not overwrite generation metadata.")
    output.write_text(json.dumps(run, indent=2) + "\n")
    print(f"{run_id}: {old:.4f} -> {run['metrics']['overall']:.4f}")
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ids", nargs="*", default=None)
    parser.add_argument("--metric", choices=("official", "normalized"), default="official")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    run_ids = args.run_ids or sorted(
        p.stem for p in RUNS.glob("*.json") if (PREDS / f"{p.stem}.jsonl").exists()
    )
    for run_id in run_ids:
        rescore(run_id, args.metric, args.output_dir)


if __name__ == "__main__":
    main()
