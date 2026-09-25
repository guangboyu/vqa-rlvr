"""RLVR: GRPO with verifiable rewards, vLLM colocated rollouts on a single GPU.

The policy answers in the preset's rollout template (`reasoning` by default, which asks for
<think>...</think><answer>...</answer>); the correctness reward reads the answer by that
template's answer rule, optionally with a weighted format term. Prompts come from the
datasets' rl roles or a declared pool (data/pools/). vLLM
generates rollouts colocated on the training GPU with sleep mode, so training and
generation alternate on one card.

Usage:
    uv run python train_grpo.py --preset grpo_2b_main --hardware 24gb --dry-run  # no GPU
    uv run python train_grpo.py --preset grpo_2b_main --hardware 24gb
    uv run python train_grpo.py --preset grpo_2b_main --rewards correctness format
    uv run python train_grpo.py --preset grpo_2b_main --rollout-prompt short --dry-run
"""

import argparse

from vqar import config
from vqar.pools import provenance as pool_provenance
from vqar.registry import PROMPTS
from vqar.rl import (
    build_rewards,
    dry_run,
    grpo_config,
    load_rl_dataset,
    make_chat_rollout,
    resolve_model,
)
from vqar.splits import provenance
from vqar.training import (
    ROOT,
    add_common_arguments,
    check_rl_precision,
    checkpoint_dir,
    evaluate_checkpoints_next,
    load_kwargs,
    lora_config,
    parse_arguments,
    resolve_preset,
    run_training,
    wandb_reporting,
    write_run_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, config.GRPO_PRESETS, save_steps=100)
    # Each flag names a preset field; resolve_preset applies the ones given.
    parser.add_argument("--rewards", nargs="+", default=None,
                        help="override reward set, e.g. correctness format")
    parser.add_argument("--reward-weights", nargs="+", type=float, default=None,
                        help="default: 1.0 per reward, 0.2 for format")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--grad-accum", dest="gradient_accumulation", type=int, default=None)
    parser.add_argument("--batch-size", dest="per_device_batch_size", type=int, default=None)
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="train on these datasets' rl roles (transfer-matrix arms)")
    parser.add_argument("--kl-beta", type=float, default=None)
    parser.add_argument("--rollout-prompt", choices=sorted(PROMPTS), default=None,
                        help="template the rollouts answer; its answer rule scores them")
    parser.add_argument("--pool", default=None,
                        help="train on a declared pool under data/pools/ (default: whole rl roles)")
    args = parse_arguments(parser)
    if args.rewards and not args.reward_weights:
        args.reward_weights = [0.2 if name == "format" else 1.0 for name in args.rewards]
    preset = resolve_preset(config.GRPO_PRESETS, args)

    if args.dry_run:
        print(f"hardware={args.hardware}; recipe={config.RL_RECIPE}")
        dry_run(preset)
        return

    check_rl_precision(preset)

    from transformers import AutoProcessor
    from trl import GRPOTrainer

    run_name = args.run_name or preset.name
    model_path = resolve_model(preset.model)

    # No processor size overrides: colocated vLLM preprocesses with the model's
    # default config, and any asymmetry vs the training processor breaks Qwen3-VL's
    # M-RoPE indexing. RL images are resized onto a fixed canvas in the data instead.
    processor = AutoProcessor.from_pretrained(model_path, padding_side="left")

    output = checkpoint_dir(args.run_dir, run_name)
    training_args = grpo_config(
        preset, output, run_name, load_kwargs(preset.quantize_4bit), save_steps=args.save_steps,
        save_total_limit=args.save_total_limit, report_to=wandb_reporting(args.no_wandb),
    )
    write_run_config(output, {
        "preset": config.dump(preset), "policy_model": model_path,
        "data": provenance(ROOT / "data", preset.datasets),
        "pool": pool_provenance(ROOT / "data", preset.pool), "hardware": args.hardware,
    }, training_args)
    trainer = GRPOTrainer(
        model=model_path,
        reward_funcs=build_rewards(preset.rewards),
        args=training_args,
        train_dataset=load_rl_dataset(preset),
        processing_class=processor,
        peft_config=lora_config(preset),
        rollout_func=make_chat_rollout(),
    )
    completed = run_training(trainer, output, args.run_dir, log_draws=True)
    if completed and not args.no_dev_eval:
        evaluate_checkpoints_next(output)


if __name__ == "__main__":
    main()
