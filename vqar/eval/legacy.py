"""Historical result files read as RunRecords (`legacy=True`); the files stay unchanged.

Two generations exist:

- The corrected campaign (September 2026) under results/baselines/: one JSON per run plus
  saved predictions, which stay local and in the evidence bundle. Both scorer variants and
  every summary are recomputed from the predictions, so these records need them.
- The July runs under results/runs/ (normalized scorer) and results/rescored/official/
  (the same predictions under the official scorer). Only summaries survive in the tree;
  they are labeled history and never enter headline tables.
"""

import hashlib
import json
import math
from pathlib import Path

from vqar.eval.examples import image_donors, select_rows
from vqar.eval.protocol import Protocol, protocol_identity
from vqar.eval.record import RunRecord
from vqar.eval.runner import paired_with_original, summarize
from vqar.inference import extract_answer
from vqar.registry import SCORER_VARIANTS, score

CAMPAIGN = "corrected campaign, September 2026"
JULY = "July 2026: inert VQAv2 reward, overlapping image pools"
BEHAVIOR = ("mean_output_tokens", "truncated_fraction", "answer_tag_fraction",
            "empty_answer_fraction")


def corrected_records(baselines: Path, data_dir: Path,
                      weights: dict[str, dict]) -> list[tuple[RunRecord, list[dict]]]:
    """Every evaluation and image intervention of the corrected campaign, with its lines.

    `weights` maps each arm (base2, sft2, ...) to the weights identity of its model.
    Interventions pair with the first questions of the matching original-image run.
    """
    records, originals = [], {}
    for path in sorted((baselines / "evaluation").glob("*/runs/*.json")):
        record, lines = campaign_record(path, baselines, data_dir, weights)
        originals[cell_of(record)] = lines
        records.append((record, lines))
    for path in sorted((baselines / "interventions").glob("*/runs/*.json")):
        run = json.loads(path.read_text())
        original = originals[(path.parent.parent.name, run["eval"]["prompt_template"],
                              run["eval"]["dataset"])]
        records.append(campaign_record(path, baselines, data_dir, weights, original))
    return records


def cell_of(record: RunRecord) -> tuple[str, str, str]:
    return record.model["label"], record.protocol["prompt"], record.protocol["dataset"]


def campaign_record(path: Path, baselines: Path, data_dir: Path, weights: dict[str, dict],
                    original: list[dict] | None = None):
    """One campaign run as a record; `original` pairs an image intervention with its run."""
    run = json.loads(path.read_text())
    evaluation = run["eval"]
    arm = path.parent.parent.name
    predictions = path.parent.parent / "preds" / f"{run['run_id']}.jsonl"
    rows = [json.loads(line) for line in predictions.read_text().splitlines()]
    dataset, mode = evaluation["dataset"], evaluation["image_mode"]
    protocol = Protocol(dataset, prompt=evaluation["prompt_template"], image_mode=mode,
                        max_tokens=run["gen"]["max_tokens"])
    partial = evaluation["limit"] is not None
    selected, data = select_rows(protocol, data_dir, limit=evaluation["limit"])
    qids = list(selected["qid"])
    if [row["qid"] for row in rows] != qids or len(rows) != evaluation["n"]:
        raise ValueError(f"{predictions}: questions differ from the current split files")
    if mode == "swapped":
        donors = [qids[i] for i in image_donors(selected, mode)]
        if [row["image_qid"] for row in rows] != donors:
            raise ValueError(f"{predictions}: image donors differ from the harness's")
    identity = protocol_identity(protocol, data)
    if partial:
        identity.update(tier=f"first-{evaluation['limit']}", question_limit=evaluation["limit"])
    lines = [campaign_line(dataset, row, predictions) for row in rows]
    model = {"source": run["model"]["base"], "label": arm, "run": arm, "step": None,
             "weights": weights[arm]}
    sources = {str(p.relative_to(baselines.parent.parent)): sha256(p) for p in (path, predictions)}
    provenance = {"legacy": CAMPAIGN, "code_version": run["git_sha"],
                  "source_sha256": run["source_sha256"], "versions": run["versions"],
                  "timestamp": run["timestamp"], "migrated_from": sources}
    metrics = summarize(lines, dataset)
    if original is not None:
        metrics["paired_with_original"] = paired_with_original(lines, original)
    return RunRecord(model, identity, metrics, provenance, legacy=True), lines


def campaign_line(dataset: str, row: dict, source: Path) -> dict:
    """A saved prediction in the harness's sidecar shape, rescored with both variants."""
    prediction = extract_answer(row["raw"])
    scores = {v: score(dataset, prediction, row["answers"], v) for v in SCORER_VARIANTS}
    if prediction != row["prediction"] or not math.isclose(scores["official"], row["score"],
                                                            abs_tol=1e-12):
        raise ValueError(f"{source}: saved answer or score disagrees with the harness")
    return {"qid": row["qid"], "sample": 0, "question": row["question"], "raw": row["raw"],
            "prediction": prediction, "scores": scores, "answers": row["answers"],
            "answer_type": row["answer_type"], "image_qid": row["image_qid"],
            "output_tokens": row["output_tokens"], "finish_reason": row["finish_reason"]}


def july_records(results: Path) -> list[RunRecord]:
    """The July runs, normalized scores from results/runs and official ones when rescored."""
    records = []
    for path in sorted((results / "runs").glob("*.json")):
        run = json.loads(path.read_text())
        evaluation = run["eval"]
        dataset = evaluation["dataset"]
        name = run["run_id"].removesuffix(f"-{dataset}")
        rescored = results / "rescored" / "official" / path.name
        official = json.loads(rescored.read_text())["metrics"] if rescored.exists() else None
        metrics = {"n": evaluation["n"],
                   "official": summary_of(official),
                   "normalized": summary_of(run["metrics"]),
                   **dict.fromkeys(BEHAVIOR)}
        protocol = {"dataset": dataset, "role": "test", "subset": evaluation["subset"],
                    "question_limit": evaluation["limit"],
                    "prompt": evaluation["prompt_template"],
                    "max_tokens": run["gen"]["max_tokens"],
                    "temperature": run["gen"]["temperature"], "image_mode": "original",
                    "scored_by": {"normalized": "at evaluation",
                                  "official": "rescored from the saved predictions"},
                    "history": JULY}
        model = {"source": run["model"]["base"], "label": name, "run": name, "step": None,
                 "weights": {"legacy_run": name}}
        provenance = {"legacy": JULY, "code_version": run["git_sha"],
                      "timestamp": run["timestamp"],
                      "migrated_from": {str(p.relative_to(results.parent)): sha256(p)
                                        for p in (path, rescored) if p.exists()}}
        records.append(RunRecord(model, protocol, metrics, provenance, legacy=True))
    return records


def summary_of(metrics: dict | None) -> dict | None:
    if metrics is None:
        return None
    return {"overall": metrics["overall"], "by_answer_type": metrics["by_answer_type"]}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
