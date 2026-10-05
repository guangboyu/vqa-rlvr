"""Compare two saved OOD records using their declared primary score and image groups.

python -m scripts.compare_ood reference.json candidate.json --cohort-audit audit.json
The percentile interval resamples image groups and keeps question-weighted scores.
"""

import argparse
import json
from pathlib import Path

from vqar.eval.examples import sha256_json
from vqar.eval.paired import paired_bootstrap, question_scores
from vqar.eval.record import RunRecord


def compare(reference: RunRecord, candidate: RunRecord, audit: dict,
            *, resamples: int = 10_000, seed: int = 0) -> dict:
    if reference.protocol_id != candidate.protocol_id:
        raise ValueError("OOD comparison needs identical protocols and question populations")
    protocol = reference.protocol
    if protocol["dataset"] not in ("vizwiz", "realworldqa"):
        raise ValueError("This comparison is for the declared OOD cohorts")
    if (protocol["role"], protocol["tier"], protocol["prompt"], protocol["image_mode"],
            protocol["samples"], protocol["temperature"]) != (
                "test", "full", "short", "original", 1, 0):
        raise ValueError("OOD closeout compares full, short, original-image greedy records")
    if audit.get("status") != "passed" or audit["dataset"] != protocol["dataset"]:
        raise ValueError("A passed audit for this dataset is required")
    variant = protocol["metric_metadata"]["primary"]
    if reference.metrics.get(variant) is None or candidate.metrics.get(variant) is None:
        raise ValueError(f"Both records must have the declared {variant} primary score")
    ordered_groups = [[row["qid"], row["image_key"]] for row in audit["image_groups"]]
    if sha256_json(ordered_groups) != protocol["data"].get("image_groups_sha256"):
        raise ValueError("The audit image groups differ from the frozen record cohort")
    groups = {row["qid"]: row["image_key"] for row in audit["image_groups"]}
    if len(groups) != audit["rows"]:
        raise ValueError("The audit repeats or omits question identities")
    for record in (reference, candidate):
        lines = record.predictions()
        if (len(lines) != len(groups) or {row["qid"] for row in lines} != groups.keys()
                or any(row["sample"] != 0 for row in lines)):
            raise ValueError("A full greedy record needs exactly one prediction per audited QID")
        if any(row["scores"].get(variant) is None for row in lines):
            raise ValueError(f"A prediction is missing its declared {variant} score")
    baseline = question_scores(reference.predictions_path(), variant)
    trained = question_scores(candidate.predictions_path(), variant)
    result = paired_bootstrap(baseline, trained, resamples, seed, image_groups=groups)
    return {"dataset": protocol["dataset"], "primary": variant,
            "reference_model_id": reference.model_id, "candidate_model_id": candidate.model_id,
            "protocol_id": reference.protocol_id, **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--cohort-audit", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = compare(RunRecord.load(args.reference), RunRecord.load(args.candidate),
                     json.loads(args.cohort_audit.read_text()),
                     resamples=args.resamples, seed=args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
