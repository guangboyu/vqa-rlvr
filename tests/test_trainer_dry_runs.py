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


def test_recipe_knobs_reach_the_grpo_settings_and_stay_out_of_default_records():
    from argparse import Namespace

    from vqar import config
    from vqar.training import resolve_preset

    def resolve(**flags):
        return resolve_preset(config.GRPO_PRESETS,
                              Namespace(preset="grpo_2b_main", hardware="24gb", **flags))

    default = resolve()
    knob = resolve(loss_type="grpo", scale_rewards="none", num_iterations=2, epsilon_high=0.28)
    assert config.rl_recipe(default) == config.RL_RECIPE
    assert {k: config.rl_recipe(knob)[k] for k in config.RECIPE_KNOBS} == {
        "loss_type": "grpo", "scale_rewards": "none", "num_iterations": 2, "epsilon_high": 0.28}
    assert not set(config.RECIPE_KNOBS) & set(config.dump(default))
    assert config.dump(knob)["loss_type"] == "grpo"


def test_the_canvas_size_knob_reaches_the_rl_images():
    from argparse import Namespace

    from PIL import Image

    from vqar import config
    from vqar.rl import to_grpo_columns
    from vqar.training import resolve_preset

    preset = resolve_preset(config.GRPO_PRESETS, Namespace(preset="grpo_2b_main",
                                                            hardware="24gb", canvas_size=896))
    assert preset.canvas_size == 896 and config.dump(preset)["canvas_size"] == 896
    assert "canvas_size" not in config.dump(config.GRPO_PRESETS["grpo_2b_main"])
    batch = {"question": ["q"], "image": [Image.new("RGB", (800, 400))], "answers": [["a"]],
             "dataset": ["chartqa"], "qid": ["1"]}
    assert to_grpo_columns(batch, "short", 896)["image"][0].size == (896, 896)
    assert to_grpo_columns(batch, "short")["image"][0].size == (608, 608)


def test_the_budget_image_fit_keeps_the_shape_within_the_evaluation_budget():
    from argparse import Namespace

    from PIL import Image

    from vqar import config
    from vqar.rl import to_grpo_columns
    from vqar.training import resolve_preset

    preset = resolve_preset(config.GRPO_PRESETS, Namespace(preset="grpo_2b_main",
                                                            hardware="24gb", image_fit="budget"))
    assert config.dump(preset)["image_fit"] == "budget"
    assert "image_fit" not in config.dump(config.GRPO_PRESETS["grpo_2b_main"])
    small, large = Image.new("RGB", (800, 400)), Image.new("L", (2000, 1000))
    batch = {"question": ["q", "q"], "image": [small, large], "answers": [["a"], ["a"]],
             "dataset": ["chartqa"] * 2, "qid": ["1", "2"]}
    fitted = to_grpo_columns(batch, "short", image_fit="budget")["image"]
    assert fitted[0].size == (800, 400)
    width, height = fitted[1].size
    assert width * height <= config.MAX_PIXELS and abs(width / height - 2) < 0.01
    assert fitted[1].mode == "RGB"


def test_vision_lora_adds_the_vision_blocks_and_mergers_to_the_lora_targets():
    import re
    from argparse import Namespace

    from vqar import config
    from vqar.training import lora_targets, resolve_preset

    plain = config.SFT_PRESETS["sft_2b"]
    assert lora_targets(plain) == list(plain.lora_targets)
    preset = resolve_preset(config.SFT_PRESETS, Namespace(preset="sft_2b", hardware="24gb",
                                                          vision_lora=True))
    pattern = lora_targets(preset)
    for name in ("model.visual.blocks.3.attn.qkv", "model.visual.blocks.3.attn.proj",
                 "model.visual.merger.linear_fc2",
                 "model.visual.deepstack_merger_list.0.linear_fc1",
                 "model.language_model.layers.5.self_attn.q_proj"):
        assert re.fullmatch(pattern, name), name
    for name in ("model.language_model.layers.5.self_attn.rotary", "lm_head"):
        assert not re.fullmatch(pattern, name)
    assert config.dump(preset)["vision_lora"] is True
    assert "vision_lora" not in config.dump(plain)


def test_the_upscale_fit_resizes_small_images_up_to_the_budget():
    from PIL import Image

    from vqar import config
    from vqar.rl import fit_rl_image

    small = fit_rl_image(Image.new("RGB", (224, 224)), "upscale")
    assert small.size[0] == small.size[1] and abs(small.size[0] ** 2 - config.MAX_PIXELS) < 2 * 887

