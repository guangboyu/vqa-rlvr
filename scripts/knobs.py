"""Training knobs: one change at a time against a fixed 2B baseline, read off one table.

A knob family in experiments.toml ([knobs.<family>]) names a dataset, the training flags
every run shares, the baseline adapter, and its knobs: one flag change each, with the
direction expected before the run. Each knob trains at the same small budget and its final
adapter is scored greedily on the dataset's full dev role under both prompts; test cells
are kept for knobs that get promoted. A knob reads + or − only when its paired 95% interval
excludes zero and the difference exceeds the baseline repeat's, the run-to-run noise.

    uv run python -m scripts.knobs commands chartqa > runs/knobs-chartqa/queue.sh
    uv run python -m scripts.knobs summarize chartqa   # writes results/knobs-chartqa.json
"""

import argparse
import json
import shlex
import tomllib
from pathlib import Path

from vqar.eval.paired import paired_bootstrap, question_scores
from vqar.eval.record import RunRecord, record_files

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = ("reasoning", "short")
SCORER = "official"
REPEAT = "repeat"  # the knob that reruns the baseline unchanged


def load_family(name: str, toml: Path = ROOT / "experiments.toml") -> dict:
    return tomllib.loads(toml.read_text())["knobs"][name]


def label(family: str, knob: str) -> str:
    return f"knob-{family}-{knob}"


def commands(family: str, spec: dict, only: list[str] | None = None) -> list[str]:
    """Shell lines that train each knob, then score its final adapter and the baseline's."""
    evaluate = ["uv", "run", "--frozen", "python", "evaluate.py", "--model", spec["model"],
                "--revision", spec["revision"], "--datasets", spec["dataset"], "--role", "dev",
                "--tier", "full", "--prompts", *PROMPTS, "--records-dir", spec["records"]]
    lines = [shlex.join(evaluate + ["--adapter", spec["baseline"]["adapter"],
                                    "--label", label(family, "baseline")])]
    for knob in spec["knob"]:
        if only and knob["name"] not in only:
            continue
        run = f"{spec['runs']}/{knob['name']}"
        train = ["uv", "run", "--frozen", "python", "train_grpo.py", *spec["train"],
                 *knob["flags"], "--run-dir", run, "--run-name", label(family, knob["name"])]
        lines.append(shlex.join(train))
        lines.append(shlex.join(evaluate + ["--adapter", f"{run}/checkpoints",
                                            "--label", label(family, knob["name"])]))
    return lines


def dev_records(records_dir: Path, wanted_label: str) -> dict[str, RunRecord]:
    """The label's full-dev records by prompt."""
    found = {}
    for path in record_files(records_dir):
        record = RunRecord.load(path)
        if record.model.get("label") == wanted_label and record.protocol["role"] == "dev":
            found[record.protocol["prompt"]] = record
    return found


def train_hours(run_dir: Path) -> float | None:
    """Training wall time from the run's metrics log (summed over resumed segments)."""
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return None
    seconds = [json.loads(line).get("train_runtime") for line in path.read_text().splitlines()]
    seconds = [s for s in seconds if s is not None]
    return sum(seconds) / 3600 if seconds else None


def verdict(delta: dict, floor: float) -> str:
    """+ or − when the interval excludes zero and the change beats the noise floor."""
    if delta["low"] > 0 and delta["delta"] > floor:
        return "+"
    if delta["high"] < 0 and -delta["delta"] > floor:
        return "−"
    return "0"


def summarize(family: str, spec: dict, root: Path = ROOT) -> dict:
    records_dir = root / spec["records"]
    baseline = dev_records(records_dir, label(family, "baseline"))
    rows = []
    for knob in spec["knob"]:
        cells = dev_records(records_dir, label(family, knob["name"]))
        row = {key: knob[key] for key in ("name", "change", "expect")}
        row["flags"] = knob["flags"]
        row["train_hours"] = train_hours(root / spec["runs"] / knob["name"])
        for prompt in PROMPTS:
            if prompt not in cells or prompt not in baseline:
                continue
            scores = [question_scores(r.predictions_path(), SCORER)
                      for r in (baseline[prompt], cells[prompt])]
            row[prompt] = {"accuracy": 100 * cells[prompt].metrics[SCORER]["overall"],
                           "baseline": 100 * baseline[prompt].metrics[SCORER]["overall"],
                           "tokens": cells[prompt].metrics["mean_output_tokens"],
                           "baseline_tokens": baseline[prompt].metrics["mean_output_tokens"],
                           "n": cells[prompt].metrics["n"], **paired_bootstrap(*scores)}
        rows.append(row)
    repeat = next((r for r in rows if r["name"] == REPEAT and "reasoning" in r), None)
    floor = abs(repeat["reasoning"]["delta"]) if repeat else 0.0
    for row in rows:
        if row["name"] != REPEAT and "reasoning" in row:
            row["verdict"] = verdict(row["reasoning"], floor)
    return {"family": family, "dataset": spec["dataset"], "scorer": SCORER, "role": "dev",
            "baseline": spec["baseline"], "train": spec["train"], "noise_floor": floor,
            "knobs": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("commands", "summarize"))
    parser.add_argument("family")
    parser.add_argument("knobs", nargs="*", help="commands: only these knobs (default: all)")
    args = parser.parse_args()
    spec = load_family(args.family)
    if args.action == "commands":
        print("\n".join(commands(args.family, spec, args.knobs)))
        return
    output = ROOT / "results" / f"knobs-{args.family}.json"
    output.write_text(json.dumps(summarize(args.family, spec), indent=1) + "\n")
    print(f"wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
