"""Refilling dead groups (DAPO dynamic sampling) with fake rollouts and rewards: no model."""

import json
from collections import defaultdict
from itertools import islice
from types import SimpleNamespace

import pytest

from vqar.pools import RefillStream
from vqar.rl import ScoredGroup, fill_live_groups, refill_draw_records
from vqar.training import reward_per_update

POOL = [{"dataset": "gqa", "qid": f"q{i}"} for i in range(20)]


def fake_score(live_qids):
    """Two answers per question: they disagree for live questions, agree for dead ones."""
    def score(questions):
        return [ScoredGroup(q, [0.0, 1.0] if q["qid"] in live_qids else [1.0, 1.0],
                            q["qid"] in live_qids) for q in questions]
    return score


KEYS = tuple((r["dataset"], r["qid"]) for r in POOL)


def pool_draw(questions, seed=42, update=0):
    """Replacements from the refill stream, excluding the update's own questions."""
    draws = RefillStream(KEYS, seed).draws(update, [(q["dataset"], q["qid"]) for q in questions])
    return lambda count: [POOL[i] for i in islice(draws, count)]


def qids(groups):
    return [group.question["qid"] for group in groups]


def test_dead_groups_are_replaced_until_the_batch_is_live():
    questions = POOL[:4]
    order = [q["qid"] for q in pool_draw(questions)(16)]
    live = {"q0", "q2", order[1], order[3]}
    rounds, batch = fill_live_groups(questions, pool_draw(questions), fake_score(live),
                                     max_rounds=5)
    # Each round draws only as many replacements as live groups are still missing.
    assert [qids(r) for r in rounds] == [
        ["q0", "q1", "q2", "q3"], order[:2], [order[2]], [order[3]]]
    assert sorted(qids(batch)) == sorted(live) and all(group.live for group in batch)


def test_the_cap_bounds_extra_rounds_and_the_earliest_dead_groups_fill_the_batch():
    questions = POOL[:4]
    rounds, batch = fill_live_groups(questions, pool_draw(questions), fake_score({"q1"}),
                                     max_rounds=2)
    assert [len(r) for r in rounds] == [4, 3, 3]
    assert qids(batch) == ["q1", "q0", "q2", "q3"]  # the update's own questions first


def test_zero_rounds_trains_on_the_main_batch_as_before():
    questions = POOL[:4]
    rounds, batch = fill_live_groups(questions, pool_draw(questions), fake_score({"q1"}),
                                     max_rounds=0)
    assert len(rounds) == 1 and sorted(qids(batch)) == ["q0", "q1", "q2", "q3"]


def test_an_exhausted_pool_ends_the_refill():
    tiny = POOL[:3]
    stream = RefillStream(KEYS[:3], 0).draws(0, KEYS[:2])
    draw = lambda count: [tiny[i] for i in islice(stream, count)]  # noqa: E731
    rounds, batch = fill_live_groups(tiny[:2], draw, fake_score(set()), max_rounds=10)
    assert [qids(r) for r in rounds] == [["q0", "q1"], ["q2"]] and len(batch) == 2


def test_refill_order_is_fixed_by_the_seed():
    def refilled(seed):
        rounds, _ = fill_live_groups(POOL[:4], pool_draw(POOL[:4], seed), fake_score(set()),
                                     max_rounds=3)
        return [qids(r) for r in rounds[1:]]

    assert refilled(42) == refilled(42)
    assert refilled(42) != refilled(7)


def test_draw_records_mark_every_attempt_and_what_was_kept():
    questions = POOL[:2]
    rounds, batch = fill_live_groups(questions, pool_draw(questions), fake_score({"q0"}),
                                     max_rounds=1)
    rounds[0][0].scores[0] = float("nan")  # an unscorable answer is logged as null
    records = refill_draw_records(7, rounds, batch)
    assert [(r["step_before_update"], r["round"], len(r["qid"])) for r in records] == [
        (7, 0, 4), (7, 1, 2)]
    assert records[0]["scores"][0] is None
    assert records[0]["kept"] == [True, True, True, True]  # q0 live, q1 pads the batch
    assert records[1]["kept"] == [False, False]


class FakeGRPO:
    """What the refill trainer needs from TRL 1.8, with fake rollouts and rewards: each
    answer's text is its question, and a live question's answers score 0, 1, 0, 1, ..."""

    def __init__(self, live):
        self.live, self.generated = live, []

    def generate(self, trainer, prompts):
        questions = [p[0]["content"][-1]["text"] for p in prompts]
        self.generated.append((questions, trainer._last_loaded_step))
        completion_ids = [[1] * (len(q) + 1) for q in questions]
        completions = [[{"role": "assistant", "content": q}] for q in questions]
        return ([[0]] * len(prompts), completion_ids, None, completions, None,
                [[-0.5] * len(ids) for ids in completion_ids], {}, None, [])

    def rewards(self, trainer, inputs, prompts, completions, completion_ids_list):
        import torch

        return torch.tensor([[float(i % 2) if row["qid"] in self.live else 0.0]
                             for i, row in enumerate(inputs)])

    def score_pass(self, trainer, inputs):
        """TRL's pass on the chosen rows, reduced to the two calls the trainer overrides."""
        from trl.data_utils import prepare_multimodal_messages

        prompts = [prepare_multimodal_messages(r["prompt"], images=[r["image"]])
                   for r in inputs]
        generation = trainer._generate(prompts)
        return {"qids": [row["qid"] for row in inputs], "generation": generation,
                "rewards": trainer._calculate_rewards(inputs, prompts, generation[3],
                                                      generation[1])}


def row(qid):
    return {"prompt": [{"role": "user", "content": qid}], "image": "image",
            "dataset": "gqa", "qid": qid}


@pytest.fixture
def refill_trainer(monkeypatch):
    pytest.importorskip("trl")
    import torch
    from trl import GRPOTrainer

    from vqar.refill_trainer import RefillGRPOTrainer

    def build(live, step=3, rounds=4, training=True):
        fake = FakeGRPO(live)
        # Plain functions, so the trainer binds as `self` like TRL's own methods.
        monkeypatch.setattr(GRPOTrainer, "_generate", lambda self, *a: fake.generate(self, *a))
        monkeypatch.setattr(GRPOTrainer, "_calculate_rewards",
                            lambda self, *a: fake.rewards(self, *a))
        monkeypatch.setattr(GRPOTrainer, "_generate_and_score_completions",
                            lambda self, *a: fake.score_pass(self, *a))
        trainer = RefillGRPOTrainer.__new__(RefillGRPOTrainer)  # no model, no TRL setup
        pool = [row(f"q{i}") for i in range(20)]
        trainer.__dict__.update(
            num_generations=2, model=SimpleNamespace(training=training),
            state=SimpleNamespace(global_step=step), train_dataset=pool,
            refill=RefillStream(tuple((r["dataset"], r["qid"]) for r in pool), seed=42),
            refill_rounds=rounds, draws_path=None, _chosen=None, _last_loaded_step=step,
            _metrics={"train": defaultdict(list)}, reward_weights=torch.ones(1),
            accelerator=SimpleNamespace(device="cpu"))
        return trainer, fake, [row(q) for q in ("q0", "q0", "q1", "q1")]

    return build


def test_the_trainer_trains_on_the_refilled_rows_without_sampling_them_again(
        refill_trainer, tmp_path):
    main = [("gqa", "q0"), ("gqa", "q1")]
    order = [KEYS[i][1] for i in islice(RefillStream(KEYS, 42).draws(3, main), 2)]
    trainer, fake, inputs = refill_trainer(live={"q0", order[1]})  # q1 and order[0] dead
    trainer.record_draws_to(tmp_path / "prompt_draws.jsonl")
    output = trainer._generate_and_score_completions(inputs)

    assert [questions[::2] for questions, _ in fake.generated] == [["q0", "q1"], order[:1],
                                                                  order[1:]]
    assert output["qids"] == ["q0", "q0", order[1], order[1]]
    _, completion_ids, _, completions, tokens, _, *_ = output["generation"]
    assert [c[0]["content"] for c in completions] == output["qids"]  # sampled answers reused
    assert tokens.item() == sum(map(len, completion_ids))
    assert output["rewards"].flatten().tolist() == [0.0, 1.0, 0.0, 1.0]
    # One sampling call per round, each after a forced weight sync; none in TRL's pass.
    assert all(step == -1 for _, step in fake.generated) and trainer._chosen is None
    metrics = trainer._metrics["train"]
    assert metrics == {"refill/attempted_groups": [4], "refill/live_groups": [2],
                       "refill/rounds": [2]}

    records = [json.loads(line) for line in (tmp_path / "prompt_draws.jsonl").open()]
    assert [(r["round"], r["kept"]) for r in records] == [
        (0, [True, True, False, False]), (1, [False, False]), (2, [True, True])]
    (update,), prompts = reward_per_update(tmp_path / "prompt_draws.jsonl")
    assert update == {"update": 4, "reward_mean": 0.5, "reward_std": 0.5,
                      "zero_variance_group_fraction": 0.0, "attempted_groups": 4,
                      "attempted_rollouts": 8, "refill_rounds": 2}
    assert prompts == 2


def test_a_resumed_update_draws_the_same_replacements(refill_trainer):
    first, fake_first, inputs = refill_trainer(live={"q0"}, step=11, rounds=3)
    first._generate_and_score_completions(inputs)
    again, fake_again, inputs = refill_trainer(live={"q0"}, step=11, rounds=3)
    again._generate_and_score_completions(inputs)
    assert [q for q, _ in fake_again.generated] == [q for q, _ in fake_first.generated]
    assert len(fake_first.generated) == 4  # the cap: three extra rounds, all dead


def test_evaluation_passes_straight_through(refill_trainer):
    trainer, fake, inputs = refill_trainer(live=set(), training=False)
    assert trainer._generate_and_score_completions(inputs)["qids"] == [r["qid"] for r in inputs]
    assert len(fake.generated) == 1 and trainer._metrics["train"] == {}


def test_a_refilling_trainer_records_its_own_draws_instead_of_the_reward_wrapper(tmp_path):
    from vqar.training import log_prompt_draws

    def correctness(**kwargs):
        return [1.0]

    paths = []
    trainer = SimpleNamespace(record_draws_to=paths.append, reward_funcs=[correctness])
    log_prompt_draws(trainer, tmp_path / "prompt_draws.jsonl")
    assert paths == [tmp_path / "prompt_draws.jsonl"]
    assert trainer.reward_funcs[0] is correctness  # scoring is not logged twice
