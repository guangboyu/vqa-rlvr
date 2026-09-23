"""Summarize matched evaluations and image interventions from saved predictions.

Usage: python -m scripts.summarize_comparison results/baselines
Inputs: evaluation/<arm>/{runs,preds} and interventions/<arm>/{runs,preds}.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

from scripts.make_tables import load_matched_results
from vqar.metrics import normalize_answer
from vqar.registry import score


def read_predictions(path, record):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != record["eval"]["n"] or len({r["qid"] for r in rows}) != len(rows):
        raise ValueError(f"{path}: incomplete or duplicate predictions")
    qids_sha = hashlib.sha256(json.dumps([r["qid"] for r in rows]).encode()).hexdigest()
    if qids_sha != record["qids_sha256"]:
        raise ValueError(f"{path}: question order differs from the result record")
    dataset = record["eval"]["dataset"]
    for row in rows:
        expected = score(dataset, row["prediction"], row["answers"], "official")
        if not math.isclose(row["score"], expected, abs_tol=1e-12):
            raise ValueError(f"{path}: saved score differs from official scoring")
    observed = mean(r["score"] for r in rows)
    if not math.isclose(observed, record["metrics"]["overall"], abs_tol=1e-12):
        raise ValueError(f"{path}: predictions disagree with result summary")
    return rows


def score_variants(dataset, rows):
    normalized = [score(dataset, r["prediction"], r["answers"], "normalized")
                  for r in rows]
    lowercase = [score(dataset, r["prediction"].lower(), r["answers"], "official")
                 for r in rows]
    return {
        "normalized": mean(normalized),
        "lowercase_official": mean(lowercase),
        "normalized_changed_questions": sum(
            not math.isclose(r["score"], s, abs_tol=1e-12)
            for r, s in zip(rows, normalized, strict=True)
        ),
    }


def paired_intervention(original, changed):
    reference = original[:len(changed)]
    if [(r["qid"], r["answers"]) for r in reference] != [
        (r["qid"], r["answers"]) for r in changed
    ]:
        raise ValueError("Intervention questions or original references differ")
    return {
        "n": len(changed),
        "original_score": mean(r["score"] for r in reference),
        "changed_image_score": mean(r["score"] for r in changed),
        "normalized_answer_changed_fraction": mean(
            normalize_answer(a["prediction"]) != normalize_answer(b["prediction"])
            for a, b in zip(reference, changed, strict=True)
        ),
    }


def build_report(directory):
    matched = load_matched_results(directory / "evaluation")
    runs, originals, sources = {}, {}, {}
    for path in sorted((directory / "evaluation").glob("*/runs/*.json")):
        record = json.loads(path.read_text())
        evaluation = record["eval"]
        if evaluation["image_mode"] != "original" or evaluation["metric_variant"] != "official":
            raise ValueError(f"{path}: expected original images and official scoring")
        dataset = evaluation["dataset"]
        run_id = record["run_id"].removesuffix(f"-{dataset}")
        assert matched[run_id][dataset] == record
        pred_path = path.parent.parent / "preds" / f"{record['run_id']}.jsonl"
        rows = read_predictions(pred_path, record)
        originals[(run_id, dataset)] = (record, rows)
        run = runs.setdefault(run_id, {"template": evaluation["prompt_template"], "datasets": {}})
        run["datasets"][dataset] = {
            **record["metrics"], "n": len(rows), **score_variants(dataset, rows),
        }
        for source in (path, pred_path):
            sources[str(source.relative_to(directory))] = hashlib.sha256(
                source.read_bytes()).hexdigest()

    interventions = {}
    for path in sorted((directory / "interventions").glob("*/runs/*.json")):
        record = json.loads(path.read_text())
        evaluation = record["eval"]
        dataset, mode = evaluation["dataset"], evaluation["image_mode"]
        if mode not in ("black", "swapped"):
            raise ValueError(f"{path}: expected a black or swapped image intervention")
        run_id = record["run_id"].removesuffix(f"-{mode}-{dataset}")
        original_record, original_rows = originals[(run_id, dataset)]
        for key in ("gen", "source_sha256", "versions", "model", "dataset_fingerprint"):
            if record[key] != original_record[key]:
                raise ValueError(f"{path}: intervention differs in {key}")
        for key in ("prompt_template", "metric_variant", "answer_extraction", "subset"):
            if evaluation[key] != original_record["eval"][key]:
                raise ValueError(f"{path}: intervention differs in {key}")
        pred_path = path.parent.parent / "preds" / f"{record['run_id']}.jsonl"
        changed = read_predictions(pred_path, record)
        result = paired_intervention(original_rows, changed)
        interventions.setdefault(run_id, {}).setdefault(dataset, {})[mode] = result
        for source in (path, pred_path):
            sources[str(source.relative_to(directory))] = hashlib.sha256(
                source.read_bytes()).hexdigest()
    report = {
        "runs": runs,
        "image_interventions": interventions,
        "source_sha256": sources,
        "interpretation": [
            "Scores are on fixed materialized subsets; training uses one seed.",
            "Normalized and lowercase scores reuse the same saved responses.",
            "Output length does not establish reasoning quality.",
            "Interventions retain the original image's references. Their scores measure "
            "agreement with those references, not accuracy on the changed images.",
        ],
    }
    exposure = directory / "training-exposure.json"
    if exposure.exists():
        report["training"] = json.loads(exposure.read_text())
        sources[exposure.name] = hashlib.sha256(exposure.read_bytes()).hexdigest()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    report = build_report(args.directory)
    output = args.directory / "comparison.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{output}: {len(report['runs'])} model/prompt combinations")


if __name__ == "__main__":
    main()
