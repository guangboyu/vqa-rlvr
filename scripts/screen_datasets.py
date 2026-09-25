"""Screen candidate training datasets: one fixed battery each, hard gates, and a ranking.

    python -m scripts.screen_datasets --arms base2 --dry-run          # list the battery
    python -m scripts.screen_datasets --arms base2                    # run missing cells
    python -m scripts.screen_datasets --arms base2 base8 --from-records   # table only

For each registered dataset, the battery (vqar.eval.screening) evaluates the same questions
from the training pool under the short and reasoning prompts, the short prompt with black
images, and 8 reasoning samples at T = 1. Every cell is an ordinary record under
results/records/, so a finished cell is never run again. The table lists every candidate,
rejects included, as JSON and markdown under --out. `--validate-on-test` screens datasets
without a training pool on their test role, labelled validation only; they always fail the
pool gate. Arms are the labels declared in experiments.toml.
"""

import argparse
import json
from pathlib import Path

from vqar.eval import find_cell, run_eval
from vqar.eval.matrix import load_matrix
from vqar.eval.record import RunRecord
from vqar.eval.runner import DATA_DIR, RECORDS_DIR, ROOT
from vqar.eval.screening import TIER, battery, markdown, pool_questions, rank, screen, summary
from vqar.registry import DATASETS

OUTPUT = ROOT / "results" / "screening"


def batteries(datasets: list[str], tier: str, validate_on_test: bool) -> dict:
    return {dataset: battery(dataset, tier, validate_on_test) for dataset in datasets}


def show(label: str, model, planned: dict, data_dir: Path, records_dir: Path) -> None:
    """Print each dataset's protocols and whether their records exist."""
    for dataset, cells in planned.items():
        if cells is None:
            print(f"{label} {dataset}: no training pool; rejected without a battery "
                  "(--validate-on-test screens its test role, validation only)")
            continue
        for name, protocol in cells.protocols().items():
            cell = find_cell(model, protocol, data_dir=data_dir, records_dir=records_dir)
            state = "cached" if cell.cached else "missing"
            print(f"{label} {dataset} {name}: {protocol.slug()}: {state}, {cell.questions} "
                  f"questions -> {cell.path}")


def collect(label: str, model, planned: dict, data_dir: Path, records_dir: Path) -> list[dict]:
    """One row per dataset from the records that exist."""
    rows = []
    for dataset, cells in planned.items():
        records = {}
        for name, protocol in (cells.protocols().items() if cells else []):
            cell = find_cell(model, protocol, data_dir=data_dir, records_dir=records_dir)
            records[name] = RunRecord.load(cell.path) if cell.cached else None
        rows.append(screen(label, dataset, cells, records, pool_questions(dataset, data_dir)))
    return rows


def run(model, arm, planned: dict, args) -> None:
    from vqar.inference import VLLMGenerator

    generator = VLLMGenerator(model.path, model.revision, args.max_model_len,
                              arm.gpu_memory_utilization)
    for cells in planned.values():
        for protocol in (cells.protocols().values() if cells else []):
            record = run_eval(model, protocol, generator, data_dir=args.data_dir,
                              records_dir=args.records_dir, batch_size=args.batch_size)
            print(f"{arm.label} {protocol.slug()}: n={record.metrics['n']} -> {record.path}",
                  flush=True)


def write(rows: list[dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "screening.json").write_text(json.dumps({**summary(), "rows": rows}, indent=1)
                                        + "\n")
    (out / "screening.md").write_text(markdown(rows))
    print(markdown(rows))
    print(f"wrote {out / 'screening.json'} and {out / 'screening.md'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", nargs="+", required=True,
                        help="model labels from experiments.toml, e.g. base2 base8")
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS),
                        default=list(DATASETS))
    parser.add_argument("--tier", default=TIER, help="pool questions: dev (500) or first-N")
    parser.add_argument("--validate-on-test", action="store_true",
                        help="screen datasets without a training pool on their test role, "
                        "labelled validation only")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="list the battery and which records exist; no GPU")
    mode.add_argument("--from-records", action="store_true",
                      help="build the table from existing records only; no GPU")
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--records-dir", type=Path, default=RECORDS_DIR)
    parser.add_argument("--matrix", type=Path, default=ROOT / "experiments.toml")
    args = parser.parse_args()

    declared = {arm.label: arm for arm in load_matrix(args.matrix)[0]}
    if unknown := [label for label in args.arms if label not in declared]:
        parser.error(f"unknown arms {unknown}; declared: {sorted(declared)}")
    if not (args.dry_run or args.from_records) and len(args.arms) != 1:
        parser.error("generation loads one model per run; pass a single arm")
    try:
        planned = batteries(args.datasets, args.tier, args.validate_on_test)
    except ValueError as error:
        parser.error(str(error))

    rows = []
    for label in args.arms:
        arm = declared[label]
        model = arm.ref(ROOT).pinned()
        if args.dry_run:
            show(label, model, planned, args.data_dir, args.records_dir)
            continue
        if not args.from_records:
            run(model, arm, planned, args)
        rows += collect(label, model, planned, args.data_dir, args.records_dir)
    if not args.dry_run:
        write(rank(rows), args.out)


if __name__ == "__main__":
    main()
