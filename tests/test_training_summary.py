import json

import pytest

from scripts.summarize_training import completed_draws, group_summary


def test_resume_replaces_abandoned_draws_and_excludes_unfinished_update(tmp_path):
    path = tmp_path / "draws.jsonl"
    rows = [
        dict(step_before_update=0, dataset=["gqa"] * 2, qid=["a"] * 2, scores=[1, 1]),
        dict(step_before_update=1, dataset=["gqa"] * 2, qid=["b"] * 2, scores=[0, 0]),
        dict(step_before_update=1, dataset=["gqa"] * 2, qid=["b"] * 2, scores=[0, 1]),
        dict(step_before_update=2, dataset=["gqa"] * 2, qid=["c"] * 2, scores=[1, 1]),
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    report = group_summary(completed_draws(path, steps=2, group_size=2))
    assert report["unique_questions"] == 2
    assert report["responses"] == 4
    assert report["mean_reward"] == 0.75
    assert report["variable_reward_fraction"] == 0.5


def test_missing_update_cannot_be_reported_as_complete(tmp_path):
    path = tmp_path / "draws.jsonl"
    path.write_text(json.dumps(dict(
        step_before_update=1, dataset=["gqa"] * 2, qid=["a"] * 2, scores=[0, 1]
    )))
    with pytest.raises(ValueError, match="missing draws"):
        completed_draws(path, steps=2, group_size=2)


def test_a_refilled_update_is_summarized_by_the_groups_it_kept(tmp_path):
    path = tmp_path / "draws.jsonl"
    rows = [
        dict(step_before_update=0, round=0, dataset=["gqa"] * 4, qid=["a", "a", "b", "b"],
             scores=[1, 1, 0, 1], kept=[False, False, True, True]),
        dict(step_before_update=0, round=1, dataset=["gqa"] * 2, qid=["c", "c"],
             scores=[1, 0], kept=[True, True]),
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    groups = completed_draws(path, steps=1, group_size=2)
    assert [(qid, scores) for _, _, qid, scores in groups] == [("b", [0, 1]), ("c", [1, 0])]
    assert group_summary(groups)["variable_reward_fraction"] == 1.0
