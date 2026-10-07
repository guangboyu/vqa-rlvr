"""Paired comparisons on the extra OOD sets, by their declared primary score.

    python -m scripts.compare_ood    # arms and pairs in experiments.toml [ood] -> results/ood.json

VizWiz is scored with the authors' evaluator (official); RealWorldQA with project-defined
case/whitespace exact match. Questions that share an image are resampled together, and every
estimate still weights questions equally.
"""

import json
import tomllib

from vqar.eval.examples import select_rows, sha256_json
from vqar.eval.matrix import declared_records, load_matrix
from vqar.eval.paired import paired_bootstrap, question_scores
from vqar.eval.protocol import Protocol
from vqar.eval.record import RunRecord
from vqar.eval.report import ROOT
from vqar.eval.runner import DATA_DIR, RECORDS_DIR

OUTPUT = ROOT / "results" / "ood.json"
OOD_DATASETS = ("vizwiz", "realworldqa")


def image_groups(protocol: Protocol) -> list[list[str]]:
    """Ordered (qid, image key) pairs of the cohort, as the record's data identity hashes them."""
    rows, _ = select_rows(protocol, DATA_DIR)
    return [list(pair) for pair in zip(rows["qid"], rows["image_key"], strict=True)]


def primary(record: RunRecord) -> str:
    return record.protocol["metric_metadata"]["primary"]


def compare(reference: RunRecord, candidate: RunRecord, groups: list[list[str]],
            *, resamples: int = 10_000, seed: int = 0) -> dict:
    """candidate − reference on the primary score, resampling image groups."""
    if reference.protocol_id != candidate.protocol_id:
        raise ValueError("OOD comparison needs identical protocols and question populations")
    protocol = reference.protocol
    if protocol["dataset"] not in OOD_DATASETS:
        raise ValueError("This comparison is for the declared OOD cohorts")
    if (protocol["tier"], protocol["prompt"], protocol["image_mode"], protocol["samples"],
            protocol["temperature"]) != ("full", "short", "original", 1, 0):
        raise ValueError("OOD comparisons use full, short-prompt, original-image greedy records")
    if sha256_json(groups) != protocol["data"].get("image_groups_sha256"):
        raise ValueError("The image groups differ from the record's frozen cohort")
    variant = primary(reference)
    scores = [question_scores(record.predictions_path(), variant)
              for record in (reference, candidate)]
    by_qid = dict(groups)
    if any(qids.keys() != by_qid.keys() for qids in scores):
        raise ValueError("Each record needs exactly one prediction per cohort question")
    return {"dataset": protocol["dataset"], "primary": variant,
            **paired_bootstrap(*scores, resamples, seed, image_groups=by_qid)}


def main():
    matrix = ROOT / "experiments.toml"
    config = tomllib.loads(matrix.read_text())["ood"]
    arms, _ = load_matrix(matrix)
    arms = [arm for arm in arms if arm.label in config["arms"]]
    protocols = [Protocol(dataset, seed=config["seed"]) for dataset in config["datasets"]]
    records = declared_records(arms, protocols, RECORDS_DIR, DATA_DIR, ROOT)
    cells = [{"arm": label, "dataset": protocol.dataset, "primary": primary(record),
              "score": 100 * record.metrics[primary(record)]["overall"],
              "normalized": 100 * record.metrics["normalized"]["overall"],
              "n": record.metrics["n"]}
             for (label, protocol), record in records.items()]
    comparisons = []
    for protocol in protocols:
        groups = image_groups(protocol)
        for entry in config["comparisons"]:
            for candidate in entry["candidates"]:
                pair = records.get((entry["reference"], protocol)), records.get((candidate,
                                                                                protocol))
                if None in pair:
                    print(f"skipped {candidate} vs {entry['reference']} {protocol.dataset}: "
                          "no record yet")
                    continue
                comparisons.append({"reference": entry["reference"], "candidate": candidate,
                                    **compare(*pair, groups)})
    OUTPUT.write_text(json.dumps({"cells": cells, "comparisons": comparisons}, indent=1) + "\n")
    print(f"{len(cells)} cells, {len(comparisons)} comparisons -> {OUTPUT}")


if __name__ == "__main__":
    main()
