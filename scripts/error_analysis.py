"""Error analysis of every declared arm from its saved predictions.

    python -m scripts.error_analysis     # then python -m scripts.report

For the arms and original-image protocols declared in experiments.toml, writes
results/error-analysis.json with
- flips: per declared comparison, dataset, prompt, and scorer variant, how many questions
  the candidate fixed (wrong_to_right) and broke (right_to_wrong) against its reference;
- cells: per arm, dataset, and prompt, the share of predictions in each error kind and the
  share flagged by each answer-form check;
- definitions: the rules behind both, from vqar/eval/errors.py.
The report turns it into tables. Needs the predictions beside the records, which stay local.
"""

import argparse
import json
from pathlib import Path

from vqar.eval.errors import answer_form, definitions, error_taxonomy, flip_counts
from vqar.eval.matrix import declared_records, load_comparisons, load_matrix
from vqar.eval.paired import question_scores
from vqar.eval.report import ROOT
from vqar.eval.runner import DATA_DIR, RECORDS_DIR
from vqar.registry import SCORER_VARIANTS

OUTPUT = ROOT / "results" / "error-analysis.json"


def flips(records: dict, protocols: list, comparisons: list[tuple[str, str]]) -> list[dict]:
    rows = []
    for reference, candidate in comparisons:
        for protocol in protocols:
            pair = records.get((reference, protocol)), records.get((candidate, protocol))
            if None in pair:
                continue
            for variant in SCORER_VARIANTS:
                if pair[0].metrics[variant] is None:
                    continue
                scores = [question_scores(record.predictions_path(), variant) for record in pair]
                rows.append({"reference": reference, "candidate": candidate,
                             "dataset": protocol.dataset, "prompt": protocol.prompt,
                             "variant": variant, **flip_counts(*scores)})
    return rows


def cells(records: dict) -> list[dict]:
    rows = []
    for (arm, protocol), record in records.items():
        lines = record.predictions()
        rows.append({"arm": arm, "dataset": protocol.dataset, "prompt": protocol.prompt,
                     "n": record.metrics["n"], "errors": error_taxonomy(lines, protocol.prompt),
                     "answer_form": answer_form(lines, protocol.prompt)})
    return rows


def analyze(matrix: Path, records_dir: Path, data_dir: Path) -> dict:
    arms, protocols = load_matrix(matrix)
    protocols = [protocol for protocol in protocols if protocol.image_mode == "original"]
    records = declared_records(arms, protocols, records_dir, data_dir, ROOT)
    return {"definitions": definitions(),
            "flips": flips(records, protocols, load_comparisons(matrix)),
            "cells": cells(records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records-dir", type=Path, default=RECORDS_DIR)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = analyze(ROOT / "experiments.toml", args.records_dir, args.data_dir)
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {args.output}: {len(result['flips'])} flip rows, {len(result['cells'])} cells")


if __name__ == "__main__":
    main()
