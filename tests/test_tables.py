import json

import pytest

from scripts.make_tables import print_matched_results


def test_fresh_tables_reject_equal_sized_but_different_eval_questions(tmp_path):
    directory = tmp_path / "runs"
    directory.mkdir()
    record = {
        "run_id": "base-short-gqa",
        "eval": {"dataset": "gqa", "prompt_template": "short", "image_mode": "original", "n": 2},
        "qids_sha256": "questions-a",
        "dataset_fingerprint": "fixed-materialization",
        "gen": {"temperature": 0, "max_tokens": 64},
        "source_sha256": {}, "versions": {},
    }
    (directory / "base.json").write_text(json.dumps(record))
    record.update(run_id="sft-short-gqa", qids_sha256="questions-b")
    (directory / "sft.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="protocol differs"):
        print_matched_results(tmp_path)
