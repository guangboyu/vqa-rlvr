from dataclasses import fields, replace

import pytest

from vqar import config
from vqar.training import check_rl_precision


def test_large_gpu_profile_keeps_rollout_geometry():
    original = config.GRPO_PRESETS["grpo_8b_fmt0"]
    resolved = config.for_hardware(original, "80gb")
    for field in fields(original):
        if field.name not in ("quantize_4bit", "vllm_gpu_memory_utilization"):
            assert getattr(resolved, field.name) == getattr(original, field.name)
    assert resolved.vllm_gpu_memory_utilization * 80 == pytest.approx(28.8)


def test_24gb_preserves_existing_sft_qlora_recipe():
    original = config.SFT_PRESETS["sft_8b"]
    assert config.for_hardware(original, "24gb") is original
    assert original.quantize_4bit
    assert not config.for_hardware(original, "96gb").quantize_4bit


def test_memory_profile_uses_recipe_size_instead_of_snapshot_hash():
    snapshot = "checkpoints/snapshots/89644892e4d85e24eaac8bacfd4f463576704203"
    small = replace(config.GRPO_PRESETS["grpo_2b_main"], model=snapshot)
    large = replace(config.GRPO_PRESETS["grpo_8b_fmt0"], model="checkpoints/custom-merge")
    assert config.for_hardware(small, "80gb").vllm_gpu_memory_utilization == 0.15
    assert config.for_hardware(large, "80gb").vllm_gpu_memory_utilization == 0.36


def test_unsafe_nf4_rollout_training_fails_before_loading_weights():
    with pytest.raises(ValueError, match="synchronization"):
        check_rl_precision(replace(config.GRPO_PRESETS["grpo_8b_fmt0"], quantize_4bit=True))


def test_resume_ignores_incomplete_newer_checkpoint(tmp_path):
    from vqar.training import last_complete_checkpoint

    old = tmp_path / "checkpoint-10"
    old.mkdir()
    (old / "complete.json").write_text('{"step": 10}')
    (tmp_path / "checkpoint-20").mkdir()
    assert last_complete_checkpoint(tmp_path) == str(old)


def test_resume_refuses_to_restart_silently_after_incomplete_save(tmp_path):
    from vqar.training import last_complete_checkpoint

    (tmp_path / "checkpoint-1").mkdir()
    with pytest.raises(RuntimeError, match="No completed checkpoint"):
        last_complete_checkpoint(tmp_path)


def test_prompt_draw_log_preserves_rewards_and_records_resume_step(tmp_path):
    import json
    from types import SimpleNamespace

    from vqar.training import log_prompt_draws

    def correctness(**kwargs):
        return [1.0, 0.0]

    trainer = SimpleNamespace(reward_funcs=[correctness], state=SimpleNamespace(global_step=12))
    path = tmp_path / "draws.jsonl"
    log_prompt_draws(trainer, path)
    scores = trainer.reward_funcs[0](dataset=["vqa", "vqa"], qid=["a", "a"])
    assert scores == [1.0, 0.0]
    assert trainer.reward_funcs[0].__name__ == "correctness"
    record = json.loads(path.read_text())
    assert record == dict(step_before_update=12, dataset=["vqa", "vqa"],
                          qid=["a", "a"], scores=scores)


@pytest.mark.parametrize("trainer, presets, name", [
    ("train_grpo", config.GRPO_PRESETS, "grpo_2b_main"),
    ("train_opd", config.OPD_PRESETS, "opd_self_2b_main"),
])
def test_rl_trainers_cannot_load_a_dataset_without_an_rl_role(trainer, presets, name):
    import importlib

    load_rl_dataset = importlib.import_module(trainer).load_rl_dataset
    preset = replace(presets[name], datasets=("textvqa",))
    with pytest.raises(ValueError, match="textvqa has no rl role"):
        load_rl_dataset(preset)


def test_sft_cannot_load_a_dataset_without_an_sft_role():
    pytest.importorskip("trl")
    from train_sft import load_datasets

    preset = replace(config.SFT_PRESETS["sft_2b"], datasets=("clevr",))
    with pytest.raises(ValueError, match="clevr has no sft role"):
        load_datasets(preset)
