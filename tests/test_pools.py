"""RL pools: the whole rl roles, or a declared subset chosen from learnability records."""

import json
from dataclasses import replace

import pytest
from fixtures import every_dataset

from vqar import config
from vqar.pools import (
    VERSION,
    RefillStream,
    filtered_pool,
    load_pool,
    pool_path,
    provenance,
    union_pool,
)


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


def test_a_limited_pool_keeps_the_first_mixed_questions():
    records = [Learnability("vqav2", {"a": 0, "b": 3, "c": 8}),
               Learnability("gqa", {"d": 7, "e": 1})]
    pool = filtered_pool("overfit", records, low=1, high=7, limit=2)
    assert [(m["dataset"], m["qid"]) for m in pool["members"]] == [("vqav2", "b"), ("gqa", "d")]
    assert pool["selection"]["first"] == 2


def test_a_union_pool_adds_whole_rl_roles_to_declared_pools_and_loads(data_dir):
    from vqar.splits import load_role

    vqav2 = load_role(data_dir, ["vqav2"], "rl", None)["qid"]
    declare(data_dir, "vqa-two", [{"dataset": "vqav2", "qid": q} for q in vqav2[:2]],
            datasets=("vqav2",))
    pool = union_pool("mix", data_dir, ["vqa-two"], ["gqa"])
    gqa = load_role(data_dir, ["gqa"], "rl", None)["qid"]
    assert len(pool["members"]) == 2 + len(gqa) and pool["datasets"] == ["gqa", "vqav2"]
    assert [part.get("pool") or part["role"] for part in pool["selection"]["union"]] == [
        "vqa-two", "gqa rl"]
    pool_path(data_dir, "mix").write_text(json.dumps(pool))
    assert len(load_pool(data_dir, ("vqav2", "gqa"), seed=0, pool="mix")) == len(pool["members"])
    with pytest.raises(ValueError, match="overlap"):
        union_pool("twice", data_dir, ["vqa-two"], ["vqav2"])


def test_refill_draws_are_seeded_per_update_and_skip_the_updates_own_questions():
    keys = tuple(("gqa", str(i)) for i in range(20))
    stream = RefillStream(keys, seed=42)
    first = list(stream.draws(update=3, exclude=[("gqa", "0"), ("gqa", "1")]))
    assert sorted(first) == list(range(2, 20))  # every other row once, then exhausted
    assert list(stream.draws(update=3, exclude=[("gqa", "0"), ("gqa", "1")])) == first
    assert list(RefillStream(keys, seed=42).draws(update=4))[:5] != first[:5]
    assert list(RefillStream(keys, seed=7).draws(update=3))[:5] != first[:5]


def test_a_resumed_update_redraws_the_same_replacements():
    # Nothing is saved with a checkpoint: the order is a function of seed and update.
    keys = tuple(("vqav2", str(i)) for i in range(1000))
    before = RefillStream(keys, seed=42).draws(update=250)
    before = [next(before) for _ in range(6)]
    after = RefillStream(keys, seed=42).draws(update=250)
    assert [next(after) for _ in range(6)] == before


def test_the_refill_stream_reads_a_pools_ids_without_its_image_transform(data_dir):
    from vqar.rl import load_rl_dataset

    rows = load_rl_dataset(config.GRPO_PRESETS["grpo_2b_main"], data_dir)
    stream = RefillStream.over(rows, seed=42)
    assert len(stream.keys) == len(rows) == 10
    index = next(stream.draws(update=0))
    assert (rows[index]["dataset"], rows[index]["qid"]) == stream.keys[index]
    assert stream.provenance()["pool_rows"] == 10


def test_an_off_refill_leaves_the_recorded_preset_unchanged():
    preset = config.GRPO_PRESETS["grpo_2b_main"]
    assert "refill_rounds" not in config.dump(preset)
    assert config.dump(replace(preset, refill_rounds=2))["refill_rounds"] == 2
    with pytest.raises(ValueError, match="nonnegative"):
        replace(preset, refill_rounds=-1)
