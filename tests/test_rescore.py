import json

from scripts import rescore


def test_rescoring_preserves_original_predictions_and_generation_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(rescore, "ROOT", tmp_path)
    monkeypatch.setattr(rescore, "RUNS", tmp_path / "results/runs")
    monkeypatch.setattr(rescore, "PREDS", tmp_path / "results/preds")
    rescore.RUNS.mkdir(parents=True)
    rescore.PREDS.mkdir()
    (tmp_path / "vqar").mkdir()
    (tmp_path / "vqar/metrics.py").write_text("scorer source")
    run = {
        "run_id": "model-vqav2",
        "git_sha": "generation-commit",
        "eval": {"dataset": "vqav2", "subset": "vqav2_eval", "n": 1},
        "metrics": {"overall": 1.0},
    }
    record = {
        "qid": "1",
        "prediction": "Yes",
        "raw": "<answer>Yes</answer>",
        "answers": ["yes"] * 10,
        "score": 1.0,
    }
    run_path, pred_path = rescore.RUNS / "model-vqav2.json", rescore.PREDS / "model-vqav2.jsonl"
    run_path.write_text(json.dumps(run))
    pred_path.write_text(json.dumps(record) + "\n")
    before = run_path.read_bytes(), pred_path.read_bytes()
    result = rescore.rescore("model-vqav2")
    assert result["metrics"]["overall"] == 0.0
    assert result["git_sha"] == "generation-commit"
    assert (run_path.read_bytes(), pred_path.read_bytes()) == before
