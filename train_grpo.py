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
    uv run python train_grpo.py --preset grpo_2b_main --refill-rounds 3 --dry-run
    uv run python train_grpo.py --preset grpo_2b_main --rollout-lora --dry-run
    uv run python train_grpo.py --preset grpo_2b_main --loss-type grpo --dry-run
"""

import argparse
from contextlib import nullcontext

from vqar import config
from vqar.pools import RefillStream
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
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-completion-length", type=int, default=None,
                        help="rollout token budget; the short template's is 64")
    parser.add_argument("--grad-accum", dest="gradient_accumulation", type=int, default=None)
    parser.add_argument("--batch-size", dest="per_device_batch_size", type=int, default=None)
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="train on these datasets' rl roles (transfer-matrix arms)")
    parser.add_argument("--kl-beta", type=float, default=None)
    parser.add_argument("--num-generations", type=int, default=None,
                        help="samples per question (group size); raise --grad-accum with it to "
                        "keep 4 questions per update")
    parser.add_argument("--loss-type", choices=("dapo", "grpo", "dr_grpo", "bnpo"), default=None,
                        help="how token losses are averaged: dapo over all tokens in the batch "
                        "(default), grpo per response first, dr_grpo by a fixed constant")
    parser.add_argument("--scale-rewards", choices=("group", "batch", "none"), default=None,
                        help="divide advantages by the group's (default) or batch's reward "
                        "spread, or not at all (Dr. GRPO)")
    parser.add_argument("--num-iterations", type=int, default=None,
                        help="gradient passes per generation batch (default 1: the clip "
                        "never binds)")
    parser.add_argument("--canvas-size", type=int, default=None,
                        help="side of the square canvas RL images are fitted onto (default 608; "
                        "larger keeps chart text legible and costs vision tokens)")
    parser.add_argument("--epsilon-high", type=float, default=None,
                        help="upper clip of the policy ratio (DAPO clip-higher uses 0.28)")
    parser.add_argument("--rollout-prompt", choices=sorted(PROMPTS), default=None,
                        help="template the rollouts answer; its answer rule scores them")
    parser.add_argument("--pool", default=None,
                        help="train on a declared pool under data/pools/ (default: whole rl roles)")
    parser.add_argument("--refill-rounds", type=int, default=None,
                        help="replace dead groups with up to this many extra generation rounds "
                        "per update (DAPO dynamic sampling; default 0: off)")
    parser.add_argument("--no-vllm-sleep", dest="vllm_sleep", action="store_const", const=False,
                        default=None, help="keep vLLM awake during training (large cards)")
    parser.add_argument("--rollout-lora", action="store_true", default=None,
                        help="vLLM serves the LoRA adapter unmerged instead of receiving "
                        "adapter-merged bf16 weights each update (default: merged)")
    args = parse_arguments(parser)
    if args.rewards and not args.reward_weights:
        args.reward_weights = [0.2 if name == "format" else 1.0 for name in args.rewards]
    preset = resolve_preset(config.GRPO_PRESETS, args)

    if args.dry_run:
        print(f"hardware={args.hardware}; recipe={config.rl_recipe(preset)}")
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
    train_dataset = load_rl_dataset(preset)
    refill = RefillStream.over(train_dataset, preset.seed) if preset.refill_rounds else None
    record = {
        "preset": config.dump(preset), "policy_model": model_path,
        "data": provenance(ROOT / "data", preset.datasets),
        "pool": pool_provenance(ROOT / "data", preset.pool), "hardware": args.hardware,
    }
    if refill:  # absent when off, so earlier run configs still match on resume
        record["refill"] = refill.provenance()
    write_run_config(output, record, training_args)
    trainer_args = dict(
        model=model_path,
        reward_funcs=build_rewards(preset.rewards),
        args=training_args,
        train_dataset=train_dataset,
        processing_class=processor,
        peft_config=lora_config(preset),
        rollout_func=make_chat_rollout(serve_adapter=preset.rollout_lora),
    )
    with rollout_engine(preset.rollout_lora):
        if refill:
            from vqar.refill_trainer import RefillGRPOTrainer

            trainer = RefillGRPOTrainer(**trainer_args, refill=refill,
                                        refill_rounds=preset.refill_rounds)
        else:
            trainer = GRPOTrainer(**trainer_args)
    completed = run_training(trainer, output, args.run_dir, log_draws=True)
    if preset.rollout_lora:
        trainer.vllm_generation.adapters.close()
    if completed and not args.no_dev_eval:
        evaluate_checkpoints_next(output)


def rollout_engine(serve_adapter: bool):
    """Where GRPOTrainer is built: with the unmerged-adapter vLLM backend, or TRL's own."""
    if not serve_adapter:
        return nullcontext()
    from vqar.adapter_rollouts import unmerged_adapter_engine

    return unmerged_adapter_engine()


if __name__ == "__main__":
    main()
