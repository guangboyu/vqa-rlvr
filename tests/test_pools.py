"""RL pools: the whole rl roles, or a declared subset chosen from learnability records."""

import json
from dataclasses import replace

import pytest
from fixtures import every_dataset

from vqar import config
from vqar.pools import VERSION, filtered_pool, load_pool, pool_path, provenance


@pytest.fixture
def data_dir(tmp_path):
    every_dataset(tmp_path / "data")
    return tmp_path / "data"


def declare(data_dir, name, members, datasets=("vqav2", "gqa")):
    path = pool_path(data_dir, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": VERSION, "name": name, "datasets": list(datasets),
                                "selection": {}, "members": members}))


def test_without_a_pool_a_run_draws_from_the_whole_rl_roles(data_dir):
    from vqar.splits import load_role

    rows = load_pool(data_dir, ("vqav2", "gqa"), seed=0)
    assert rows["qid"] == load_role(data_dir, ("vqav2", "gqa"), "rl", 0)["qid"]


def test_a_pool_keeps_its_members_in_the_training_order(data_dir):
    everything = load_pool(data_dir, ("vqav2", "gqa"), seed=0)
    chosen = [{"dataset": d, "qid": q} for d, q in
              list(zip(everything["dataset"], everything["qid"], strict=True))[::3]]
    declare(data_dir, "every-third", chosen)
    rows = load_pool(data_dir, ("vqav2", "gqa"), seed=0, pool="every-third")
    assert list(zip(rows["dataset"], rows["qid"], strict=True)) == [
        (m["dataset"], m["qid"]) for m in chosen]
    assert provenance(data_dir, "every-third")["name"] == "every-third"


def test_a_pool_member_outside_the_rl_roles_is_rejected(data_dir):
    # A test question can never enter training through a pool file.
    declare(data_dir, "leaky", [{"dataset": "vqav2", "qid": "vqav2_eval-0"}])
    with pytest.raises(ValueError, match="not in the rl roles"):
        load_pool(data_dir, ("vqav2", "gqa"), seed=0, pool="leaky")


def test_a_pool_is_used_only_with_the_datasets_it_came_from(data_dir):
    declare(data_dir, "vqav2-only", [{"dataset": "vqav2", "qid": "vqav2_rl-0"}], ("vqav2",))
    with pytest.raises(ValueError, match="was selected from"):
        load_pool(data_dir, ("vqav2", "gqa"), seed=0, pool="vqav2-only")


def test_the_rl_loader_trains_on_the_presets_pool(data_dir):
    from vqar.rl import load_rl_dataset

    declare(data_dir, "one", [{"dataset": "gqa", "qid": "gqa_rl-1"}])
    preset = replace(config.GRPO_PRESETS["grpo_2b_main"], pool="one", rollout_prompt="short")
    rows = load_rl_dataset(preset, data_dir)
    assert len(rows) == 1
    assert rows[0]["rollout_prompt"] == "short"
    assert rows[0]["prompt"][0]["content"].endswith("using a single word or phrase.")


class Learnability:
    def __init__(self, dataset, hits):
        self.protocol = {"dataset": dataset, "tier": "first-3"}
        self.metrics = {"n": len(hits), "samples_per_question": 8,
                        "learnability": {"rule": "reward", "correct_samples": hits}}
        self.model, self.protocol_id, self.path = {"label": "base2"}, "p", f"{dataset}.json"


def test_a_filtered_pool_keeps_questions_with_mixed_outcomes():
    records = [Learnability("vqav2", {"a": 0, "b": 3, "c": 8}),
               Learnability("gqa", {"d": 7, "e": 1})]
    pool = filtered_pool("mixed", records, low=1, high=7)
    assert [(m["dataset"], m["qid"]) for m in pool["members"]] == [
        ("vqav2", "b"), ("gqa", "d"), ("gqa", "e")]
    assert pool["datasets"] == ["gqa", "vqav2"]
    assert pool["selection"]["keep"] == "1 <= correct samples <= 7"
