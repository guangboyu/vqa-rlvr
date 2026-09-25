"""Each trainer's dry run resolves its preset and loads only its own role (tiny data)."""

import importlib

import pytest
from fixtures import every_dataset


@pytest.fixture
def repo(tmp_path, monkeypatch):
    every_dataset(tmp_path / "data")
    import train_sft
    import vqar.rl

    monkeypatch.setattr(train_sft, "ROOT", tmp_path)
    monkeypatch.setattr(vqar.rl, "ROOT", tmp_path)
    return tmp_path


# SFT trains on 2 of each SFT pool's 3 rows (one image is dev); RL adds each pool's 3 RL
# rows to those 2 SFT training rows, so 5 per dataset.
@pytest.mark.parametrize("module, preset, expected", [
    ("train_sft", "sft_2b", "train=4, dev=2"),
    ("train_grpo", "grpo_2b_main", "dataset ok: 10 prompts"),
    ("train_opd", "opd_self_2b_main", "dataset ok: 10 prompts"),
])
def test_dry_run_loads_the_trainers_role(repo, module, preset, expected, monkeypatch, capsys):
    trainer = importlib.import_module(module)
    monkeypatch.setattr("sys.argv", [f"{module}.py", "--preset", preset, "--hardware", "24gb",
                                     "--dry-run", "--no-wandb"])
    trainer.main()
    assert expected in capsys.readouterr().out


def test_grpo_dry_run_with_the_short_template_rewards_a_plain_answer(repo, monkeypatch, capsys):
    import train_grpo

    monkeypatch.setattr("sys.argv", ["train_grpo.py", "--preset", "grpo_2b_main", "--hardware",
                                     "24gb", "--rollout-prompt", "short", "--dry-run"])
    train_grpo.main()
    out = capsys.readouterr().out
    assert "rollout prompt: short (reward reads the plain answer)" in out
    assert "reward correctness:" in out


def test_a_flag_that_names_a_preset_field_overrides_it():
    from argparse import Namespace

    from vqar import config
    from vqar.training import resolve_preset

    args = Namespace(preset="grpo_2b_main", hardware="24gb", learning_rate=3e-5,
                     gradient_accumulation=16, max_steps=None)
    preset = resolve_preset(config.GRPO_PRESETS, args)
    assert (preset.learning_rate, preset.gradient_accumulation) == (3e-5, 16)
    assert preset.max_steps == config.GRPO_PRESETS["grpo_2b_main"].max_steps


def test_grpo_dry_run_shows_the_refill_stream_when_refilling(repo, monkeypatch, capsys):
    import train_grpo

    argv = ["train_grpo.py", "--preset", "grpo_2b_main", "--hardware", "24gb", "--dry-run"]
    monkeypatch.setattr("sys.argv", argv + ["--refill-rounds", "3"])
    train_grpo.main()
    assert "refill: up to 3 extra rounds per update" in capsys.readouterr().out
    monkeypatch.setattr("sys.argv", argv)
    train_grpo.main()
    assert "refill:" not in capsys.readouterr().out


def test_vllm_sleep_is_opt_out_and_left_out_of_the_record_when_on():
    from argparse import Namespace

    from vqar import config
    from vqar.training import resolve_preset

    default = resolve_preset(config.GRPO_PRESETS, Namespace(preset="grpo_2b_main",
                                                             hardware="96gb", vllm_sleep=None))
    awake = resolve_preset(config.GRPO_PRESETS, Namespace(preset="grpo_2b_main",
                                                           hardware="96gb", vllm_sleep=False))
    assert default.vllm_sleep and not awake.vllm_sleep
    assert "vllm_sleep" not in config.dump(default)
    assert config.dump(awake)["vllm_sleep"] is False
