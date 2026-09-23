"""Historical result files read as RunRecords, and the regression gate for the migration."""

import json
import math
from pathlib import Path

import pytest
from fixtures import png, save_subset, write_splits

from vqar.eval.legacy import corrected_records, july_records
from vqar.eval.record import RunRecord

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = [("q1", "What animal?", ["cat"] * 10, "other"),
             ("q2", "Is it red?", ["yes"] * 10, "yes/no")]


@pytest.fixture
def data_dir(tmp_path):
    directory = tmp_path / "data"
    save_subset(directory, "vqav2_eval", "vqav2",
                [(qid, png(10 * i + 10), q, a, t) for i, (qid, q, a, t) in enumerate(QUESTIONS)])
    save_subset(directory, "vqav2_sft", "vqav2",
                [(f"s{i}", png(100 + i), "Is it red?", ["yes"], "") for i in range(3)])
    save_subset(directory, "vqav2_rl", "vqav2", [("r0", png(120), "Is it red?", ["yes"], "")])
    write_splits(directory, ["vqav2"])
    return directory


def write_campaign_run(baselines, kind, run_id, mode, limit, rows):
    """One run file and its predictions in the corrected campaign's format."""
    directory = baselines / kind / "arm"
    (directory / "runs").mkdir(parents=True, exist_ok=True)
    (directory / "preds").mkdir(parents=True, exist_ok=True)
    run = {"run_id": f"{run_id}-vqav2", "timestamp": "2026-09-13T00:00:00+00:00",
           "git_sha": "unknown", "source_sha256": {}, "versions": {"vllm": "0.23.0"},
           "model": {"base": "/cluster/merged/arm", "preset": "zero_shot_2b"},
           "eval": {"dataset": "vqav2", "n": len(rows), "limit": limit,
                    "prompt_template": "short", "image_mode": mode},
           "gen": {"max_tokens": 64}}
    (directory / "runs" / f"{run_id}-vqav2.json").write_text(json.dumps(run))
    (directory / "preds" / f"{run_id}-vqav2.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows))


def prediction(qid, question, answers, answer_type, raw, score, image_qid=None):
    return {"qid": qid, "question": question, "prediction": raw, "raw": raw, "score": score,
            "answers": answers, "answer_type": answer_type, "image_qid": image_qid or qid,
            "output_tokens": 1, "finish_reason": "stop"}


def test_corrected_campaign_files_become_records_with_both_variants(data_dir, tmp_path):
    baselines = tmp_path / "results" / "baselines"
    (q1, t1, a1, y1), (q2, t2, a2, y2) = [(q, t, a, y) for q, t, a, y in QUESTIONS]
    write_campaign_run(baselines, "evaluation", "arm-short", "original", None, [
        prediction(q1, t1, a1, y1, "Cat", 0.0), prediction(q2, t2, a2, y2, "yes", 1.0)])
    write_campaign_run(baselines, "interventions", "arm-short-black", "black", 1, [
        prediction(q1, t1, a1, y1, "dog", 0.0)])
    (original, _), (black, _) = corrected_records(baselines, data_dir, {"arm": {"sha256": "w"}})
    assert original.legacy and original.metrics["official"]["overall"] == 0.5
    assert original.metrics["normalized"]["overall"] == 1.0
    assert original.metrics["official_on_lowercased_answers"] == 1.0
    assert original.protocol["tier"] == "full" and original.protocol["prompt"] == "short"
    assert black.protocol["tier"] == "first-1" and black.protocol["image_mode"] == "black"
    assert black.metrics["paired_with_original"] == {
        "n": 1, "original": {"official": 0.0, "normalized": 1.0},
        "normalized_answer_changed_fraction": 1.0}


def test_a_saved_score_that_the_harness_disagrees_with_is_rejected(data_dir, tmp_path):
    baselines = tmp_path / "results" / "baselines"
    rows = [prediction(q, t, a, y, "cat", 0.0) for q, t, a, y in QUESTIONS]
    write_campaign_run(baselines, "evaluation", "arm-short", "original", None, rows)
    with pytest.raises(ValueError, match="disagrees with the harness"):
        corrected_records(baselines, data_dir, {"arm": {"sha256": "w"}})


def test_predictions_for_other_questions_are_rejected(data_dir, tmp_path):
    baselines = tmp_path / "results" / "baselines"
    (q1, t1, a1, y1), _ = QUESTIONS
    write_campaign_run(baselines, "evaluation", "arm-short", "original", None, [
        prediction("q9", t1, a1, y1, "cat", 1.0), prediction(q1, t1, a1, y1, "cat", 1.0)])
    with pytest.raises(ValueError, match="questions differ from the current split files"):
        corrected_records(baselines, data_dir, {"arm": {"sha256": "w"}})


def test_july_runs_carry_both_variants_when_rescored(tmp_path):
    results = tmp_path / "results"
    (results / "runs").mkdir(parents=True)
    (results / "rescored" / "official").mkdir(parents=True)
    run = {"run_id": "sft_2b-vqav2", "timestamp": "2026-07-12", "git_sha": "abc",
           "model": {"base": "checkpoints/merged/sft_2b"},
           "eval": {"dataset": "vqav2", "subset": "vqav2_eval", "n": 5000, "limit": None,
                    "prompt_template": "short"},
           "gen": {"max_tokens": 64, "temperature": 0.0},
           "metrics": {"overall": 0.79, "by_answer_type": {"other": 0.7}}}
    (results / "runs" / "sft_2b-vqav2.json").write_text(json.dumps(run))
    rescored = dict(run, metrics={"overall": 0.77, "by_answer_type": {"other": 0.68}})
    (results / "rescored" / "official" / "sft_2b-vqav2.json").write_text(json.dumps(rescored))
    (record,) = july_records(results)
    assert record.legacy and record.protocol["history"].startswith("July 2026")
    assert record.metrics["normalized"]["overall"] == 0.79
    assert record.metrics["official"]["overall"] == 0.77
    assert record.metrics["mean_output_tokens"] is None


CAMPAIGN = ROOT / "results" / "baselines"
SAVED_PREDICTIONS = CAMPAIGN / "evaluation" / "base2" / "preds"


@pytest.mark.skipif(not SAVED_PREDICTIONS.exists(),
                    reason="the corrected campaign's predictions are local and in the bundle")
def test_every_corrected_campaign_number_is_reproduced_from_saved_predictions():
    comparison = json.loads((CAMPAIGN / "comparison.json").read_text())
    stored = {(r.model["label"], r.protocol["prompt"], r.protocol["dataset"],
               r.protocol["image_mode"]): r
              for r in map(RunRecord.load, (ROOT / "results" / "records").rglob("*.json"))
              if r.legacy}
    weights = {label: r.model["weights"] for (label, *_), r in stored.items()}
    fresh = corrected_records(CAMPAIGN, ROOT / "data", weights)
    assert len(fresh) == len(stored) == 120
    checked = 0
    for record, _ in fresh:
        protocol, metrics = record.protocol, record.metrics
        arm = record.model["label"]
        key = (arm, protocol["prompt"], protocol["dataset"], protocol["image_mode"])
        assert json.loads(json.dumps(record.metrics)) == stored[key].metrics
        assert json.loads(json.dumps(record.protocol)) == stored[key].protocol
        run = f"{arm}-{protocol['prompt']}"
        if protocol["image_mode"] == "original":
            expected = comparison["runs"][run]["datasets"][protocol["dataset"]]
            official = metrics["official"]
            observed = {
                "overall": official["overall"], "by_answer_type": official["by_answer_type"],
                "n": metrics["n"], "normalized": metrics["normalized"]["overall"],
                "lowercase_official": metrics["official_on_lowercased_answers"],
                "normalized_changed_questions": metrics["answers_scored_differently"],
                **{k: metrics[k] for k in ("mean_output_tokens", "truncated_fraction",
                                           "answer_tag_fraction", "empty_answer_fraction")},
            }
        else:
            expected = comparison["image_interventions"][run][protocol["dataset"]][
                protocol["image_mode"]]
            paired = metrics["paired_with_original"]
            observed = {"n": paired["n"], "original_score": paired["original"]["official"],
                        "changed_image_score": metrics["official"]["overall"],
                        "normalized_answer_changed_fraction":
                            paired["normalized_answer_changed_fraction"]}
        assert set(observed) == set(expected)
        for name, value in expected.items():
            values = value.items() if isinstance(value, dict) else [(None, value)]
            for part, number in values:
                got = observed[name][part] if part else observed[name]
                # Summation order differs from the original tooling in the last bits.
                assert math.isclose(got, number, rel_tol=0, abs_tol=1e-12), (key, name, part)
                checked += 1
    assert checked == 900
