"""Paired differences between arms with 95% bootstrap intervals over questions.

    python -m scripts.paired_deltas     # then python -m scripts.report

Reads the comparisons declared in experiments.toml, pairs the two arms' records of every
declared original-image protocol, pairs any cell evaluated under a larger image budget with
its default-budget cell and any cell evaluated on the RL canvas with its original-image cell,
pairs each arm of a [transfers.<set>] with its reference on every listed test set,
and writes results/paired-deltas.json, which the report turns into tables. Needs the
predictions beside the records, which stay local.
"""

import json
import tomllib

from vqar.config import MAX_PIXELS
from vqar.eval.matrix import Arm, declared_records, load_comparisons, load_matrix, load_pilot
from vqar.eval.paired import paired_bootstrap, question_scores
from vqar.eval.protocol import Protocol
from vqar.eval.report import ROOT, load_cells
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


def resolution_contrasts() -> list[dict]:
    """Each cell evaluated under a larger image budget, paired with its default-budget cell,
    and each cell evaluated on the RL canvas, paired with its original-image cell."""
    cells = load_cells(RECORDS_DIR, DATA_DIR)
    rows = []
    for cell, record in cells.items():
        if cell.max_pixels != MAX_PIXELS:
            reference = cells.get(cell._replace(max_pixels=MAX_PIXELS))
            kind, change = "max_pixels", f"at {cell.max_pixels} px"
            images = {"reference_max_pixels": MAX_PIXELS, "candidate_max_pixels": cell.max_pixels}
        elif cell.image_mode == "canvas":
            reference = cells.get(cell._replace(image_mode="original"))
            kind, change = "canvas", "on the RL canvas"
            images = {"reference_max_pixels": MAX_PIXELS,
                      "canvas_size": record.protocol["canvas_size"]}
        else:
            continue
        if reference is None:
            continue
        for variant in SCORER_VARIANTS:
            if record.metrics[variant] is None:
                continue
            scores = [question_scores(r.predictions_path(), variant) for r in (reference, record)]
            rows.append({"contrast": kind, "reference": cell.arm, "candidate": cell.arm,
                         "dataset": cell.dataset, "prompt": cell.prompt, "variant": variant,
                         **images, **paired_bootstrap(*scores)})
            print(f"{cell.arm} {cell.dataset} {cell.prompt} {variant} {change}: "
                  f"{rows[-1]['delta']:+.2f} [{rows[-1]['low']:+.2f}, {rows[-1]['high']:+.2f}]")
    return rows


def pilot_comparisons() -> list[dict]:
    """The declared pairs of every single-dataset pilot, on its full test role."""
    matrix_path = ROOT / "experiments.toml"
    rows = []
    for name in tomllib.loads(matrix_path.read_text()).get("pilots", {}):
        arms, protocols, pairs = load_pilot(matrix_path, name)
        records = declared_records(arms, protocols, RECORDS_DIR, DATA_DIR, ROOT)
        for reference, candidate in pairs:
            for protocol in protocols:
                pair = records.get((reference, protocol)), records.get((candidate, protocol))
                if None in pair:
                    continue
                for variant in SCORER_VARIANTS:
                    scores = [question_scores(r.predictions_path(), variant) for r in pair]
                    rows.append({"pilot": name, "reference": reference, "candidate": candidate,
                                 "dataset": protocol.dataset, "prompt": protocol.prompt,
                                 "variant": variant, **paired_bootstrap(*scores)})
    return rows


def transfer_comparisons() -> list[dict]:
    """Each arm of every [transfers.<set>] minus its reference on every listed test set (full
    test role, both prompts, the set's scorer: official unless it names another), in and out of
    its training data."""
    matrix = tomllib.loads((ROOT / "experiments.toml").read_text())
    arms = {arm["label"]: Arm(**arm) for arm in matrix["arms"]}
    rows = []
    for name, spec in matrix.get("transfers", {}).items():
        labels = [spec["reference"], *(row["arm"] for row in spec["rows"])]
        protocols = [Protocol(dataset, prompt=prompt) for dataset in spec["datasets"]
                     for prompt in ("short", "reasoning")]
        records = declared_records([arms[label] for label in labels], protocols, RECORDS_DIR,
                                   DATA_DIR, ROOT)
        for row in spec["rows"]:
            for protocol in protocols:
                pair = (records.get((spec["reference"], protocol)),
                        records.get((row["arm"], protocol)))
                if None in pair:
                    continue
                variant = spec.get("scorer", "official")
                scores = [question_scores(r.predictions_path(), variant) for r in pair]
                rows.append({"transfer": row["trained_on"], "transfer_set": name,
                             "reference": spec["reference"], "candidate": row["arm"],
                             "dataset": protocol.dataset, "prompt": protocol.prompt,
                             "variant": variant, **paired_bootstrap(*scores)})
    return rows


def main():
    rows = comparisons() + resolution_contrasts() + pilot_comparisons() + transfer_comparisons()
    OUTPUT.write_text(json.dumps(rows, indent=1) + "\n")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
