"""Regenerate every report table from the stored RunRecords, or fill the matrix first.

    python -m scripts.report            # tables in README.md and docs/, results/report.json
    python -m scripts.report --check    # list experiment-matrix cells without a record
    python -m scripts.report --run      # evaluate the missing cells (GPU), then the tables

The experiment matrix in experiments.toml declares arms × protocols. A test regenerates the
tables and fails when a committed copy drifts from the records.
"""

import argparse
import json

from vqar.eval import run_eval
from vqar.eval.matrix import load_matrix, missing_cells
from vqar.eval.report import MARKED_FILES, ROOT, build, fill
from vqar.eval.runner import DATA_DIR, RECORDS_DIR


def write_report() -> None:
    blocks, export = build()
    for name in MARKED_FILES:
        path = ROOT / name
        text = path.read_text()
        updated = fill(text, blocks)
        if updated != text:
            path.write_text(updated)
            print(f"updated {name}")
    (ROOT / "results" / "report.json").write_text(json.dumps(export, indent=1) + "\n")
    print("wrote results/report.json")


def run_missing(missing) -> None:
    """One vLLM engine per arm, loaded only when the arm has a missing cell."""
    from vqar.inference import VLLMGenerator

    for arm in dict.fromkeys(arm for arm, _ in missing):
        model = arm.ref(ROOT).pinned()
        generator = VLLMGenerator(model.path, model.revision,
                                  gpu_memory_utilization=arm.gpu_memory_utilization)
        for _, protocol in (cell for cell in missing if cell[0] == arm):
            record = run_eval(model, protocol, generator, data_dir=DATA_DIR,
                              records_dir=RECORDS_DIR)
            print(f"{arm.label} {protocol.slug()} -> {record.path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="list missing matrix cells")
    mode.add_argument("--run", action="store_true", help="evaluate missing cells, then report")
    args = parser.parse_args()
    if args.check or args.run:
        arms, protocols = load_matrix(ROOT / "experiments.toml")
        missing = missing_cells(arms, protocols, RECORDS_DIR, DATA_DIR, ROOT)
        for arm, protocol in missing:
            print(f"missing: {arm.label} {protocol.slug()}")
        print(f"{len(missing)} of {len(arms) * len(protocols)} cells missing")
        if args.check:
            raise SystemExit(1 if missing else 0)
        unavailable = sorted({arm.label for arm, _ in missing if not arm.available(ROOT)})
        if unavailable:
            raise SystemExit(f"Models not on disk for: {', '.join(unavailable)}")
        run_missing(missing)
    write_report()


if __name__ == "__main__":
    main()
