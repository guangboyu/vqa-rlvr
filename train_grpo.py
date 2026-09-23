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
import os
from dataclasses import replace
from pathlib import Path

from vqar import config
from vqar.registry import PROMPTS
from vqar.splits import load_role, provenance
from vqar.training import (
    check_rl_precision,
    last_complete_checkpoint,
    log_prompt_draws,
    write_run_config,
    write_training_summary,
)

# Keep the historical image geometry for comparable runs. Preserve aspect ratio
# within a fixed canvas so rollout and training use the same vision-token budget.
RL_IMAGE_SIZE = 608  # 19x19 = 361 vision tokens/image


def to_uniform_canvas(image):
    from PIL import Image as PILImage

    image = image.convert("RGB")
    w, h = image.size
    scale = RL_IMAGE_SIZE / max(w, h)
    image = image.resize((max(1, round(w * scale)), max(1, round(h * scale))))
    canvas = PILImage.new("RGB", (RL_IMAGE_SIZE, RL_IMAGE_SIZE))
    canvas.paste(image, ((RL_IMAGE_SIZE - image.width) // 2, (RL_IMAGE_SIZE - image.height) // 2))
    return canvas

os.environ.setdefault("WANDB_PROJECT", "vqa-rlvr")

ROOT = Path(__file__).parent


def to_grpo_columns(batch: dict) -> dict:
    """GRPO rows: conversational `prompt` + `image`; question/answers/dataset ride
    along as extra columns that TRL forwards to every reward function."""
    return {
        "prompt": [
            [{"role": "user", "content": PROMPTS["reasoning"].text.format(question=q)}]
            for q in batch["question"]
        ],
        "image": [to_uniform_canvas(img) for img in batch["image"]],
        "question": batch["question"],
        "answers": batch["answers"],
        "dataset": batch["dataset"],
        "qid": batch["qid"],
    }


def load_rl_dataset(preset: config.GRPOPreset):
    """The rl role of the preset's datasets; it already reuses their SFT training rows."""
    ds = load_role(ROOT / "data", preset.datasets, "rl", preset.seed)
    return ds.select_columns(["image", "question", "answers", "dataset", "qid"]).with_transform(
        to_grpo_columns
    )


def build_rewards(names: tuple[str, ...]):
    from vqar.rewards import REWARDS

    return [REWARDS[name] for name in names]


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


def make_chat_rollout(preset: config.GRPOPreset):
    """Custom rollout that generates through vLLM's own chat interface.

    TRL 1.8's built-in path hands vLLM processor-pre-expanded prompt token ids PLUS
    the raw image (vllm_generation.py:664); the two preprocessing pipelines must
    agree exactly on image token expansion and don't for our data — producing
    corrupted effective prompts (untagged prose completions, importance-sampling
    ratios ~1e-6) and the M-RoPE crashes. Routing rollouts through llm.chat gives
    one source of truth: vLLM builds prompt ids itself, and the trainer's forward
    re-processes the same PIL deterministically to matching ids.
    """

    def rollout(prompts, trainer):
        from trl.data_utils import prepare_multimodal_messages_vllm
        from vllm import SamplingParams

        generation = trainer.vllm_generation
        conversations = [prepare_multimodal_messages_vllm(p) for p in prompts]
        params = SamplingParams(
            temperature=trainer.args.temperature,
            top_p=trainer.args.top_p,
            max_tokens=trainer.args.max_completion_length,
            logprobs=0,
        )
        if generation.enable_sleep_mode:
            generation.llm.wake_up(tags=["kv_cache"])
        outputs = generation.llm.chat(conversations, params, use_tqdm=False)
        if generation.enable_sleep_mode:
            generation.llm.sleep(level=2)
        completion_ids, logprobs = [], []
        for out in outputs:
            sample = out.outputs[0]
            completion_ids.append(list(sample.token_ids))
            pairs = zip(sample.token_ids, sample.logprobs, strict=True)
            logprobs.append([lps[tid].logprob for tid, lps in pairs])
        return {
            "prompt_ids": [list(out.prompt_token_ids) for out in outputs],
            "completion_ids": completion_ids,
            "logprobs": logprobs,
        }

    return rollout


def dry_run(preset: config.GRPOPreset, rewards: tuple[str, ...]) -> None:
    """Validate dataset shape and reward wiring without touching a model or GPU."""
    from vqar.rewards import REWARDS

    ds = load_rl_dataset(preset)
    row = ds[0]
    assert row["prompt"][0]["role"] == "user", row
    assert "<think>" in row["prompt"][0]["content"]
    print(f"dataset ok: {len(ds)} prompts from the rl roles of {preset.datasets}")
    print(f"sample question: {row['question'][:80]}")

    fake_completion = [{"role": "assistant", "content": "<think>x</think><answer>4</answer>"}]
    for name in rewards:
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
    parser.add_argument("--hardware", choices=config.HARDWARE_PROFILES, default="legacy")
    parser.add_argument("--rewards", nargs="+", default=None,
                        help="override reward set, e.g. correctness format")
    parser.add_argument("--reward-weights", nargs="+", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="train on these datasets' rl roles (transfer-matrix arms)")
    parser.add_argument("--kl-beta", type=float, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--run-dir", default=None,
                        help="cluster mode: checkpoints + metrics.jsonl under this dir "
                        "(auto-resumes from its last checkpoint; see hpc-env/AGENTS.md)")
    parser.add_argument("--save-steps", type=int, default=100,
                        help="checkpoint cadence; on free partitions size to <=15 min")
    parser.add_argument("--save-total-limit", type=int, default=2,
                        help="keep this many saved checkpoints; 0 keeps all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    if args.save_total_limit < 0:
        parser.error("--save-total-limit must be nonnegative")

    preset = config.GRPO_PRESETS[args.preset]
    if args.model:
        preset = replace(preset, model=args.model)
    preset = config.for_hardware(preset, args.hardware)
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

    import torch
    from peft import LoraConfig
    from transformers import AutoProcessor, BitsAndBytesConfig
    from trl import GRPOConfig, GRPOTrainer


    run_name = args.run_name or preset.name
    model_path = resolve_model(args.model or preset.model)
    max_steps = args.max_steps or preset.max_steps
    kl_beta = args.kl_beta if args.kl_beta is not None else preset.kl_beta

    # No processor size overrides: colocated vLLM preprocesses with the model's
    # default config, and any asymmetry vs the training processor breaks Qwen3-VL's
    # M-RoPE indexing. The pixel budget is enforced in the data (cap_pixels).
    processor = AutoProcessor.from_pretrained(model_path, padding_side="left")

    model_kwargs = dict(dtype=torch.bfloat16)
    if preset.quantize_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    output_dir = (Path(args.run_dir) / "checkpoints" if args.run_dir
                  else ROOT / "checkpoints" / run_name)
    grpo_config = GRPOConfig(
        **config.RL_RECIPE,
        output_dir=str(output_dir),
        run_name=run_name,
        model_init_kwargs=model_kwargs,
        num_generations=preset.num_generations,
        per_device_train_batch_size=args.batch_size or preset.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum or preset.gradient_accumulation,
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
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit or None,
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
    write_run_config(output_dir, {
        "preset": config.dump(preset), "policy_model": model_path,
        "data": provenance(ROOT / "data", preset.datasets),
        "overrides": {"hardware": args.hardware, "rewards": list(rewards),
                      "reward_weights": list(weights), "max_steps": max_steps,
                      "kl_beta": kl_beta, "batch_size": grpo_config.per_device_train_batch_size,
                      "grad_accum": grpo_config.gradient_accumulation_steps},
    }, grpo_config)
    trainer = GRPOTrainer(
        model=model_path,
        reward_funcs=build_rewards(rewards),
        args=grpo_config,
        train_dataset=load_rl_dataset(preset),
        processing_class=processor,
        peft_config=peft_config,
        rollout_func=make_chat_rollout(preset),
    )

    # Preemption contract (free HPC3 partitions CANCEL with no warning) — same
    # wiring as train_opd.py; harmless locally on a fresh output_dir.

    from scripts.hf_preempt_callback import JsonlMetrics, PreemptCallback, install_signal_flag

    install_signal_flag()
    trainer.add_callback(PreemptCallback())
    if args.run_dir:
        trainer.add_callback(JsonlMetrics(str(Path(args.run_dir) / "metrics.jsonl")))
        log_prompt_draws(trainer, Path(args.run_dir) / "prompt_draws.jsonl")
    last = last_complete_checkpoint(output_dir) if output_dir.is_dir() else None
    if last:
        print(f"resuming from {last}")
    trainer.train(resume_from_checkpoint=last)
    trainer.save_model(str(output_dir))
    write_training_summary(trainer, output_dir)

    print(f"adapter saved -> {output_dir}")


if __name__ == "__main__":
    main()
