"""Paired differences between arms with 95% bootstrap intervals over questions.

    python -m scripts.paired_deltas     # then python -m scripts.report

Reads the comparisons declared in experiments.toml, pairs the two arms' records of every
declared original-image protocol, and writes results/paired-deltas.json, which the report
turns into a table. Needs the predictions beside the records, which stay local.
"""

import json

from vqar.eval.matrix import declared_records, load_comparisons, load_matrix
from vqar.eval.paired import paired_bootstrap, question_scores
from vqar.eval.report import ROOT
from vqar.eval.runner import DATA_DIR, RECORDS_DIR
from vqar.registry import SCORER_VARIANTS

OUTPUT = ROOT / "results" / "paired-deltas.json"


def comparisons() -> list[dict]:
    matrix_path = ROOT / "experiments.toml"
    arms, protocols = load_matrix(matrix_path)
    records = declared_records(arms, protocols, RECORDS_DIR, DATA_DIR, ROOT)
    rows = []
    for reference, candidate in load_comparisons(matrix_path):
        for protocol in protocols:
            pair = records.get((reference, protocol)), records.get((candidate, protocol))
            if protocol.image_mode != "original" or None in pair:
                continue
            for variant in SCORER_VARIANTS:
                if pair[0].metrics[variant] is None:
                    continue
                scores = [question_scores(r.predictions_path(), variant) for r in pair]
                rows.append({"reference": reference, "candidate": candidate,
                             "dataset": protocol.dataset, "prompt": protocol.prompt,
                             "variant": variant, **paired_bootstrap(*scores)})
                print(f"{candidate} vs {reference} {protocol.slug()} {variant}: "
                      f"{rows[-1]['delta']:+.2f} [{rows[-1]['low']:+.2f}, {rows[-1]['high']:+.2f}]",
                      flush=True)
    return rows


def main():
    OUTPUT.write_text(json.dumps(comparisons(), indent=1) + "\n")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
