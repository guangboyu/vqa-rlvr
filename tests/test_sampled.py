"""Sampled protocols: avg@k, pass@k, and the learnability pre-check on the RL pool."""

import pytest
from fixtures import every_dataset

from vqar.eval import ModelRef, Protocol, run_eval


class Alternating:
    """Per question, answers alternate across samples: right, wrong, right, ..."""

    def generate(self, conversations, *, max_tokens, temperature, samples, seed):
        assert temperature == 1.0 and seed == 0
        replies = ["<answer>2</answer>", "<answer>3</answer>"]
        return [[{"text": replies[i % 2], "output_tokens": 3, "finish_reason": "stop"}
                 for i in range(samples)] for _ in conversations]


class Untagged:
    """Always right, but never inside answer tags: the reward gives nothing."""

    def generate(self, conversations, *, max_tokens, temperature, samples, seed):
        return [[{"text": "2", "output_tokens": 1, "finish_reason": "stop"}] * samples
                for _ in conversations]


@pytest.fixture
def setup(tmp_path):
    every_dataset(tmp_path / "data")
    weights = tmp_path / "model"
    weights.mkdir()
    (weights / "model.safetensors").write_bytes(b"w")
    return tmp_path / "data", ModelRef(str(weights), label="base"), tmp_path / "records"


def test_sampled_accuracy_averages_samples_and_pass_at_k_needs_one_hit(setup):
    data, model, records = setup
    protocol = Protocol("clevr", prompt="reasoning", role="dev", temperature=1.0, samples=4)
    record = run_eval(model, protocol, Alternating(), data_dir=data, records_dir=records)
    assert record.metrics["normalized"]["overall"] == 0.5  # avg@4
    assert record.metrics["pass_at_k"] == {"official": 1.0, "normalized": 1.0}
    assert record.metrics["samples_per_question"] == 4


def test_learnability_counts_prompts_with_mixed_rewards(setup):
    data, model, records = setup
    protocol = Protocol("gqa", prompt="reasoning", role="rl", temperature=1.0, samples=4)
    record = run_eval(model, protocol, Alternating(), data_dir=data, records_dir=records)
    learnability = record.metrics["learnability"]
    assert learnability["fraction_between_0_and_1"] == 1.0
    assert learnability["correct_samples_histogram"]["2/4"] == record.metrics["n"]


def test_learnability_follows_the_reward_not_the_lenient_scorer(setup):
    data, model, records = setup
    protocol = Protocol("gqa", prompt="reasoning", role="rl", temperature=1.0, samples=2)
    record = run_eval(model, protocol, Untagged(), data_dir=data, records_dir=records)
    assert record.metrics["normalized"]["overall"] == 1.0
    assert record.metrics["learnability"]["correct_samples_histogram"]["0/2"] == \
        record.metrics["n"]
    assert record.metrics["learnability"]["fraction_between_0_and_1"] == 0.0


def test_the_rl_role_is_only_for_sampled_protocols():
    with pytest.raises(ValueError, match="learnability protocols"):
        Protocol("gqa", role="rl")
    with pytest.raises(ValueError, match="at least two responses"):
        Protocol("gqa", role="rl", temperature=1.0, samples=1)
    with pytest.raises(ValueError, match="textvqa has no rl role"):
        Protocol("textvqa", role="rl", temperature=1.0, samples=2)


def test_the_report_lists_learnability_prechecks(setup):
    from vqar.eval.report import learnability_table, load_cells

    data, model, records = setup
    assert learnability_table({}) == "No learnability pre-check has been recorded yet.\n"
    protocol = Protocol("gqa", prompt="reasoning", role="rl", temperature=1.0, samples=4)
    run_eval(model, protocol, Alternating(), data_dir=data, records_dir=records)
    rows = learnability_table(load_cells(records)).splitlines()
    assert rows[2] == "| `base` | GQA | reasoning | 5 | 4 | 100.0 | 0.0 | 0.0 |"


def test_a_sampled_record_has_its_own_table_cell(setup):
    from vqar.eval.report import Cell, load_cells

    data, model, records = setup
    greedy = Protocol("clevr", prompt="reasoning", role="dev")
    sampled = Protocol("clevr", prompt="reasoning", role="dev", temperature=1.0, samples=4)
    run_eval(model, greedy, Untagged(), data_dir=data, records_dir=records)
    run_eval(model, sampled, Alternating(), data_dir=data, records_dir=records)
    cells = load_cells(records)
    assert Cell("base", "clevr", "dev", "full", "reasoning", "original", None) in cells
    assert Cell("base", "clevr", "dev", "full", "reasoning", "original", None, 1.0, 4) in cells
