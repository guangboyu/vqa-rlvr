"""RLVR: GRPO with verifiable rewards, vLLM colocated rollouts on a single GPU.

The policy answers in the `reasoning` template (<think>...</think><answer>...</answer>);
rewards are normalized answer correctness and, optionally, a weighted format term. vLLM
generates rollouts colocated on the training GPU with sleep mode, so training and
generation alternate on one card.

Usage:
    uv run python train_grpo.py --preset grpo_2b_main --hardware 24gb --dry-run  # no GPU
    uv run python train_grpo.py --preset grpo_2b_main --hardware 24gb
    uv run python train_grpo.py --preset grpo_2b_main --rewards correctness format
"""

import argparse
from dataclasses import replace

from vqar import config
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
    parser.add_argument("--rewards", nargs="+", default=None,
                        help="override reward set, e.g. correctness format")
    parser.add_argument("--reward-weights", nargs="+", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="train on these datasets' rl roles (transfer-matrix arms)")
    parser.add_argument("--kl-beta", type=float, default=None)
    args = parse_arguments(parser)

    preset = resolve_preset(config.GRPO_PRESETS, args)
    if args.datasets:
        preset = replace(preset, datasets=tuple(args.datasets))
    rewards = tuple(args.rewards) if args.rewards else preset.rewards
    weights = tuple(args.reward_weights) if args.reward_weights else preset.reward_weights
    if args.rewards and not args.reward_weights:
        weights = tuple(1.0 if r != "format" else 0.2 for r in rewards)
    assert len(rewards) == len(weights), "each reward needs a weight"

    if args.dry_run:
        print(f"hardware={args.hardware}; recipe={config.RL_RECIPE}")
        dry_run(preset, rewards)
        return

    check_rl_precision(preset)

    from transformers import AutoProcessor
    from trl import GRPOTrainer

    run_name = args.run_name or preset.name
    model_path = resolve_model(args.model or preset.model)
    max_steps = args.max_steps or preset.max_steps
    kl_beta = args.kl_beta if args.kl_beta is not None else preset.kl_beta

    # No processor size overrides: colocated vLLM preprocesses with the model's
    # default config, and any asymmetry vs the training processor breaks Qwen3-VL's
    # M-RoPE indexing. RL images are resized onto a fixed canvas in the data instead.
    processor = AutoProcessor.from_pretrained(model_path, padding_side="left")

    output = checkpoint_dir(args.run_dir, run_name)
    training_args = grpo_config(
        preset, output, run_name, load_kwargs(preset.quantize_4bit), max_steps=max_steps,
        beta=kl_beta,
        reward_weights=list(weights), batch_size=args.batch_size or preset.per_device_batch_size,
        grad_accum=args.grad_accum or preset.gradient_accumulation, save_steps=args.save_steps,
        save_total_limit=args.save_total_limit, report_to=wandb_reporting(args.no_wandb),
    )
    peft_config = lora_config(preset)
    write_run_config(output, {
        "preset": config.dump(preset), "policy_model": model_path,
        "data": provenance(ROOT / "data", preset.datasets),
        "overrides": {"hardware": args.hardware, "rewards": list(rewards),
                      "reward_weights": list(weights), "max_steps": max_steps,
                      "kl_beta": kl_beta, "batch_size": training_args.per_device_train_batch_size,
                      "grad_accum": training_args.gradient_accumulation_steps},
    }, training_args)
    trainer = GRPOTrainer(
        model=model_path,
        reward_funcs=build_rewards(rewards),
        args=training_args,
        train_dataset=load_rl_dataset(preset),
        processing_class=processor,
        peft_config=peft_config,
        rollout_func=make_chat_rollout(),
    )
    completed = run_training(trainer, output, args.run_dir, log_draws=True)
    if completed and not args.no_dev_eval:
        evaluate_checkpoints_next(output)


if __name__ == "__main__":
    main()
