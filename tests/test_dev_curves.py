"""Dev evaluation of kept adapters, and the training diagnostics file."""

import json
from types import SimpleNamespace

import pytest
from fixtures import png, save_subset, write_splits

from vqar.eval import ModelRef, Protocol, run_eval
from vqar.eval.matrix import Arm
from vqar.eval.report import curves_table, load_cells
from vqar.training import training_diagnostics


def write_adapter(directory, weights: bytes):
    directory.mkdir(parents=True)
    (directory / "adapter_model.safetensors").write_bytes(weights)
    (directory / "adapter_config.json").write_text('{"r": 16}')


def test_scheduled_saves_keep_a_copy_of_their_adapter(tmp_path):
    pytest.importorskip("transformers")
    from vqar.preemption import KeepAdapters

    args = SimpleNamespace(output_dir=str(tmp_path), save_steps=25)
    for step in (25, 31, 60):  # a scheduled save, a time-based save, the final step
        write_adapter(tmp_path / f"checkpoint-{step}", f"step {step}".encode())
        KeepAdapters().on_save(args, SimpleNamespace(global_step=step, max_steps=60), None)
    assert sorted(p.name for p in (tmp_path / "adapters").iterdir()) == ["step-25", "step-60"]
    kept = tmp_path / "adapters" / "step-25"
    assert (kept / "adapter_model.safetensors").read_bytes() == b"step 25"
    (tmp_path / "checkpoint-25" / "adapter_model.safetensors").write_bytes(b"rewritten")
    state = SimpleNamespace(global_step=25, max_steps=60)
    KeepAdapters().on_save(args, state, None)  # an existing copy is never replaced
    assert (kept / "adapter_model.safetensors").read_bytes() == b"step 25"


def test_dev_cells_cover_every_kept_step_prompt_and_dev_dataset(tmp_path):
    from scripts.evaluate_checkpoints import dev_cells

    config = {"policy_model": "Qwen/Qwen3-VL-2B-Instruct",
              "preset": {"name": "grpo_2b_main", "datasets": ["vqav2", "gqa"]},
              "training_args": {"run_name": "grpo-smoke"}}
    for step in (50, 25):
        write_adapter(tmp_path / "adapters" / f"step-{step}", b"w")
    (tmp_path / "adapters" / "step-75.tmp").mkdir()  # an unfinished copy is ignored
    cells = dev_cells(tmp_path, config, ["short", "reasoning"], "dev")
    assert [(m.step, p.dataset, p.prompt) for m, p in cells][:4] == [
        (25, "vqav2", "short"), (25, "vqav2", "reasoning"),
        (25, "gqa", "short"), (25, "gqa", "reasoning")]
    assert len(cells) == 8
    assert {(m.label, m.run) for m, _ in cells} == {("grpo-smoke", "grpo-smoke")}
    assert all(p.role == "dev" and p.tier == "dev" for _, p in cells)


class Echo:
    def __init__(self, reply):
        self.reply = reply

    def generate(self, conversations, *, max_tokens, temperature, samples, seed):
        return [[{"text": self.reply, "output_tokens": 2, "finish_reason": "stop"}]
                for _ in conversations]


def test_each_checkpoint_is_its_own_cell_and_joins_the_curve(tmp_path):
    data = tmp_path / "data"
    save_subset(data, "vqav2_eval", "vqav2", [("q", png(5), "Is it red?", ["yes"] * 10, "")])
    save_subset(data, "vqav2_sft", "vqav2",
                [(f"s{i}", png(100 + i), "Is it red?", ["yes"], "") for i in range(3)])
    save_subset(data, "vqav2_rl", "vqav2", [("r0", png(120), "Is it red?", ["yes"], "")])
    write_splits(data, ["vqav2"])
    base = tmp_path / "base"
    base.mkdir()
    (base / "model.safetensors").write_bytes(b"base")
    records = tmp_path / "records"
    for step, reply in ((25, "no"), (50, "yes")):
        adapter = tmp_path / "adapters" / f"step-{step}"
        write_adapter(adapter, f"step {step}".encode())
        model = ModelRef(str(base), label="run", run="run", step=step, adapter=str(adapter))
        record = run_eval(model, Protocol("vqav2", role="dev", tier="dev"), Echo(reply),
                          data_dir=data, records_dir=records)
        assert record.model["weights"]["adapter_sha256"]
    arms = [Arm("tuned", "Tuned", "merged", run="run"), Arm("other", "Other", "m", run="x")]
    rows = curves_table(arms, load_cells(records)).splitlines()
    assert rows[0] == "| Arm | Dataset | Prompt | 25 | 50 |"
    assert rows[2:] == ["| Tuned | VQAv2 | short | — (0.0) · 2 | — (100.0) · 2 |"]


def test_diagnostics_keep_logged_signals_and_every_updates_reward(tmp_path):
    draws = tmp_path / "prompt_draws.jsonl"
    draws.write_text("".join(json.dumps(record) + "\n" for record in [
        {"step_before_update": 0, "dataset": ["gqa", "gqa", "gqa", "gqa"],
         "qid": ["a", "a", "b", "b"], "scores": [1, 1, 1, 0]},
        {"step_before_update": 1, "dataset": ["gqa", "gqa"], "qid": ["c", "c"],
         "scores": [0, 1]},  # interrupted attempt, replaced by the resumed draw below
        {"step_before_update": 1, "dataset": ["vqav2", "vqav2"], "qid": ["a", "a"],
         "scores": [0, 0]},
    ]))
    logs = [{"step": 10, "loss": 0.1, "reward": 0.6, "reward_std": 0.4,
             "frac_reward_zero_std": 0.5, "entropy": 0.3, "clip_ratio/region_mean": 0.0,
             "completions/mean_length": 130.0, "num_tokens": 1e5},
            {"step": 20, "train_runtime": 99.0}]
    diagnostics = training_diagnostics(logs, draws)
    assert diagnostics["logged"] == [{
        "step": 10, "loss": 0.1, "reward_mean": 0.6, "reward_std": 0.4,
        "zero_variance_group_fraction": 0.5, "entropy": 0.3, "clip_fraction": 0.0,
        "completion_length": 130.0, "tokens_seen": 1e5}]
    first, second = diagnostics["reward_per_update"]
    assert (first["update"], first["reward_mean"], first["zero_variance_group_fraction"]) == (
        1, 0.75, 0.5)
    assert (second["reward_mean"], second["zero_variance_group_fraction"]) == (0.0, 1.0)
    assert diagnostics["distinct_prompts"] == 3
    assert training_diagnostics(logs)["reward_per_update"] is None


def refill_round(step, round_, qids, scores, kept):
    """One generation round's draw record: groups of two answers each."""
    return {"step_before_update": step, "round": round_, "dataset": ["gqa"] * len(qids),
            "qid": qids, "scores": scores, "kept": kept}


def test_refilled_updates_report_the_kept_rows_and_what_they_attempted(tmp_path):
    draws = tmp_path / "prompt_draws.jsonl"
    records = [
        # Update 1: b is dead, replaced in round 1 by c (dead) and in round 2 by d (live).
        refill_round(0, 0, ["a", "a", "b", "b"], [1, 0, 1, 1], [True, True, False, False]),
        refill_round(0, 1, ["c", "c"], [0, 0], [False, False]),
        refill_round(0, 2, ["d", "d"], [0, 1], [True, True]),
        # Update 2: an interrupted attempt, then the resumed one, which hits the cap and
        # keeps its earliest dead group to fill the batch.
        refill_round(1, 0, ["e", "e", "f", "f"], [1, 1, 0, 0], [False] * 4),
        refill_round(1, 0, ["e", "e", "f", "f"], [0, 1, 0, 0], [True, True, True, True]),
        refill_round(1, 1, ["g", "g"], [1, 1], [False, False]),
    ]
    draws.write_text("".join(json.dumps(record) + "\n" for record in records))
    first, second = training_diagnostics([], draws)["reward_per_update"]
    assert first == {"update": 1, "reward_mean": 0.5, "reward_std": 0.5,
                     "zero_variance_group_fraction": 0.0, "all_right_group_fraction": 0.0,
                     "all_wrong_group_fraction": 0.0, "all_partial_group_fraction": 0.0,
                     "attempted_groups": 4, "attempted_rollouts": 8, "refill_rounds": 2}
    assert (second["reward_mean"], second["zero_variance_group_fraction"]) == (0.25, 0.5)
    assert second["all_wrong_group_fraction"] == 0.5  # the kept dead group f
    assert (second["attempted_groups"], second["refill_rounds"]) == (3, 1)
    assert training_diagnostics([], draws)["distinct_prompts"] == 4  # a, d, e, f


class FakeTrainer:
    """Records the wiring run_training performs; trains by jumping to the last step."""

    def __init__(self, max_steps):
        self.callbacks, self.saved = [], None
        self.state = SimpleNamespace(global_step=0, max_steps=max_steps,
                                     log_history=[{"step": 1, "loss": 0.5}])
        self.reward_funcs = [lambda **kwargs: [1.0]]

    def add_callback(self, callback):
        self.callbacks.append(type(callback).__name__)

    def train(self, resume_from_checkpoint):
        self.resumed_from = resume_from_checkpoint
        self.state.global_step = 2

    def save_model(self, path):
        self.saved = path


@pytest.mark.parametrize("max_steps, completed", [(2, True), (5, False)])
def test_training_loop_keeps_adapters_and_writes_diagnostics(tmp_path, max_steps, completed):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from vqar.training import run_training

    trainer = FakeTrainer(max_steps)
    output = tmp_path / "checkpoints"
    output.mkdir()
    assert run_training(trainer, output, tmp_path, log_draws=True) is completed
    assert trainer.callbacks == ["KeepAdapters", "PreemptCallback", "JsonlMetrics"]
    assert trainer.resumed_from is None and trainer.saved == str(output)
    diagnostics = json.loads((output / "diagnostics.json").read_text())
    assert diagnostics["logged"] == [{"step": 1, "loss": 0.5}]
    assert json.loads((output / "run_summary.json").read_text())["completed"] is completed
