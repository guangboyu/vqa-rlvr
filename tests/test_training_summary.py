import json

import pytest

from scripts.summarize_training import completed_draws, dead_group_windows, group_summary
from vqar.training import dead_group_kind, reward_per_update


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


@pytest.mark.parametrize("scores, kind", [
    ([1.0, 1.0, 1.0], "all_right"),
    ([0.0, 0.0, 0.0], "all_wrong"),
    ([0.3, 0.3, 0.3], "all_partial"),  # VQAv2 soft score: one annotator agreed each time
    ([0.0, 1.0, 1.0], None),
    ([0.9, 1.0, 1.0], None),
])
def test_dead_groups_are_all_right_all_wrong_or_all_partial(scores, kind):
    assert dead_group_kind(scores) == kind


def write_draws(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_every_update_splits_its_dead_groups_by_kind(tmp_path):
    path = tmp_path / "prompt_draws.jsonl"
    write_draws(path, [
        dict(step_before_update=0, dataset=["vqav2"] * 8, qid=list("aabbccdd"),
             scores=[1, 1, 0, 0, 0.6, 0.6, 0, 1]),
        dict(step_before_update=1, dataset=["gqa"] * 4, qid=list("eeff"), scores=[1, 1, 1, 1]),
        # The resumed attempt of update 1 replaces the draw above.
        dict(step_before_update=1, dataset=["gqa"] * 4, qid=list("gghh"), scores=[0, 0, 0, 1]),
    ])
    first, second = reward_per_update(path)[0]
    assert (first["all_right_group_fraction"], first["all_wrong_group_fraction"],
            first["all_partial_group_fraction"]) == (0.25, 0.25, 0.25)
    assert first["zero_variance_group_fraction"] == 0.75
    assert (second["all_right_group_fraction"], second["all_wrong_group_fraction"],
            second["all_partial_group_fraction"]) == (0.0, 0.5, 0.0)


def test_summaries_average_dead_group_shares_over_the_first_and_last_updates(tmp_path):
    updates = [{"all_right_group_fraction": right, "all_wrong_group_fraction": 1 - right,
                "all_partial_group_fraction": 0.0} for right in (0.0, 0.25, 0.5, 1.0)]
    windows = dead_group_windows(updates, size=2)
    assert windows["first_2_updates"] == {"all_right": 0.125, "all_wrong": 0.875,
                                          "all_partial": 0.0}
    assert windows["last_2_updates"]["all_right"] == 0.75
    assert windows["all_updates"]["all_wrong"] == pytest.approx(0.5625)

    path = tmp_path / "draws.jsonl"
    write_draws(path, [dict(step_before_update=0, dataset=["vqav2"] * 6, qid=list("aabbcc"),
                            scores=[0.3, 0.3, 1, 1, 0, 0])])
    report = group_summary(completed_draws(path, steps=1, group_size=2))
    assert (report["all_full_reward_groups"], report["all_zero_reward_groups"],
            report["all_partial_reward_groups"]) == (1, 1, 1)


def test_dead_groups_count_only_the_groups_a_refilled_update_kept(tmp_path):
    path = tmp_path / "prompt_draws.jsonl"
    write_draws(path, [
        # Round 0: a is all right and replaced; b is live. Round 1: the cap is reached, so
        # the all-wrong c fills the batch.
        dict(step_before_update=0, round=0, dataset=["gqa"] * 4, qid=list("aabb"),
             scores=[1, 1, 0, 1], kept=[False, False, True, True]),
        dict(step_before_update=0, round=1, dataset=["gqa"] * 2, qid=list("cc"),
             scores=[0, 0], kept=[True, True]),
    ])
    (update,) = reward_per_update(path)[0]
    assert (update["all_right_group_fraction"], update["all_wrong_group_fraction"],
            update["zero_variance_group_fraction"]) == (0.0, 0.5, 0.5)
