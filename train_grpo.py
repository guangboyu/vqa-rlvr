"""RLVR: GRPO with verifiable rewards, vLLM colocated rollouts on a single GPU.

The policy answers in the `reasoning` template (<think>...</think><answer>...</answer>);
rewards are correctness (official metric / EM, judge-rescued variant available) plus a
weighted format term. vLLM generates rollouts colocated on the training GPU with sleep
mode so training and generation alternate within 24GB.

Usage:
    uv run python train_grpo.py --preset grpo_2b_clevr --dry-run   # no GPU, no model
    uv run python train_grpo.py --preset grpo_2b_clevr
    uv run python train_grpo.py --preset grpo_2b_main --rewards correctness format judge
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

from vqar import config
from vqar.data import PROMPT_REASONING, cap_pixels

os.environ.setdefault("WANDB_PROJECT", "vqa-rlvr")

ROOT = Path(__file__).parent


def to_grpo_columns(batch: dict) -> dict:
    """GRPO rows: conversational `prompt` + `image`; question/answers/dataset ride
    along as extra columns that TRL forwards to every reward function."""
    return {
        "prompt": [
            [{"role": "user", "content": PROMPT_REASONING.format(question=q)}]
            for q in batch["question"]
        ],
        "image": [cap_pixels(img) for img in batch["image"]],
        "question": batch["question"],
        "answers": batch["answers"],
        "dataset": batch["dataset"],
    }


def load_rl_dataset(preset: config.GRPOPreset):
    from datasets import concatenate_datasets, load_from_disk

    ds = concatenate_datasets(
        [load_from_disk(str(ROOT / "data" / name)) for name in preset.subsets]
    ).shuffle(seed=preset.seed)
    return ds.select_columns(["image", "question", "answers", "dataset"]).with_transform(
        to_grpo_columns
    )


def build_rewards(names: tuple[str, ...]):
    from vqar.judge import Judge
    from vqar.rewards import REWARDS, make_judge_reward

    funcs = []
    for name in names:
        if name == "judge":
            funcs.append(make_judge_reward(Judge()))
        else:
            funcs.append(REWARDS[name])
    return funcs


def resolve_model(model: str) -> str:
    """Presets may point at a merged checkpoint dir; fall back to the base model id
    so the RL-only ablation arm can run before/without SFT."""
    path = ROOT / model
    if model.startswith("checkpoints/") and not path.exists():
        raise FileNotFoundError(
            f"{model} not found — run train_sft.py + scripts/merge_adapter.py first, "
            "or pass --model with a base HF id for the RL-only arm"
        )
    return str(path) if model.startswith("checkpoints/") else model


def dry_run(preset: config.GRPOPreset, rewards: tuple[str, ...]) -> None:
    """Validate dataset shape and reward wiring without touching a model or GPU."""
    from vqar.rewards import REWARDS

    ds = load_rl_dataset(preset)
    row = ds[0]
    assert row["prompt"][0]["role"] == "user", row
    assert "<think>" in row["prompt"][0]["content"]
    print(f"dataset ok: {len(ds)} prompts from {preset.subsets}")
    print(f"sample question: {row['question'][:80]}")

    fake_completion = [{"role": "assistant", "content": "<think>x</think><answer>4</answer>"}]
    for name in rewards:
        if name == "judge":
            print("reward judge: skipped in dry run (no API call)")
            continue
        value = REWARDS[name](
            prompts=[row["prompt"]],
            completions=[fake_completion],
            answers=[row["answers"]],
            dataset=[row["dataset"]],
        )
        print(f"reward {name}: {value}")
    print("dry run OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", required=True, choices=sorted(config.GRPO_PRESETS))
    parser.add_argument("--model", default=None, help="override policy model/checkpoint")
    parser.add_argument("--rewards", nargs="+", default=None,
                        help="override reward set, e.g. correctness format judge")
    parser.add_argument("--reward-weights", nargs="+", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--kl-beta", type=float, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    preset = config.GRPO_PRESETS[args.preset]
    rewards = tuple(args.rewards) if args.rewards else preset.rewards
    weights = tuple(args.reward_weights) if args.reward_weights else preset.reward_weights
    if args.rewards and not args.reward_weights:
        weights = tuple(1.0 if r != "format" else 0.2 for r in rewards)
    assert len(rewards) == len(weights), "each reward needs a weight"

    if args.dry_run:
        dry_run(preset, rewards)
        return

    import torch
    from peft import LoraConfig
    from transformers import AutoProcessor, BitsAndBytesConfig
    from trl import GRPOConfig, GRPOTrainer

    run_name = args.run_name or preset.name
    model_path = resolve_model(args.model or preset.model)
    max_steps = args.max_steps or preset.max_steps
    kl_beta = args.kl_beta if args.kl_beta is not None else preset.kl_beta

    processor = AutoProcessor.from_pretrained(model_path)
    processor.image_processor.size["longest_edge"] = config.MAX_PIXELS

    model_kwargs = dict(dtype=torch.bfloat16)
    if preset.quantize_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    output_dir = ROOT / "checkpoints" / run_name
    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        run_name=run_name,
        model_init_kwargs=model_kwargs,
        num_generations=preset.num_generations,
        per_device_train_batch_size=preset.per_device_batch_size,
        gradient_accumulation_steps=preset.gradient_accumulation,
        learning_rate=preset.learning_rate,
        max_steps=max_steps,
        max_completion_length=preset.max_completion_length,
        temperature=1.0,
        beta=kl_beta,
        reward_weights=list(weights),
        use_vllm=True,
        vllm_mode="colocate",
        vllm_enable_sleep_mode=True,
        vllm_gpu_memory_utilization=preset.vllm_gpu_memory_utilization,
        vllm_max_model_length=2048,  # ~850 image tokens + prompt + completion
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit" if preset.quantize_4bit else "adamw_torch",
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        dataloader_num_workers=2,
        dataloader_pin_memory=False,
        seed=preset.seed,
        report_to=[] if args.no_wandb else ["wandb"],
        log_completions=True,
        num_completions_to_print=2,
    )
    peft_config = LoraConfig(
        r=preset.lora_r,
        lora_alpha=preset.lora_alpha,
        target_modules=list(preset.lora_targets),
        task_type="CAUSAL_LM",
    )
    trainer = GRPOTrainer(
        model=model_path,
        reward_funcs=build_rewards(rewards),
        args=grpo_config,
        train_dataset=load_rl_dataset(preset),
        processing_class=processor,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))

    git_sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "preset": config.dump(preset),
                "policy_model": model_path,  # resolved base the adapter merges onto
                "overrides": {
                    "rewards": list(rewards), "reward_weights": list(weights),
                    "max_steps": max_steps, "kl_beta": kl_beta,
                },
                "git_sha": git_sha,
            },
            indent=2,
        )
    )
    print(f"adapter saved -> {output_dir}")


if __name__ == "__main__":
    main()
