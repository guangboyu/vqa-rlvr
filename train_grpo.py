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
from vqar.data import PROMPT_REASONING

# TRL 1.8 workaround: GRPO's sample<->image-grid handling breaks when image token
# counts differ within a generation batch (M-RoPE shape crash; CLEVR's uniform
# 480x320 images are why the sandbox never hit it). Uniform square canvas =
# identical grids for every sample. Content keeps its aspect ratio (resize long
# side, pad with black) — naive squashing degenerated generation (probe4: the
# model dropped its answer-tag habit entirely on distorted images).
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
            [{"role": "user", "content": PROMPT_REASONING.format(question=q)}]
            for q in batch["question"]
        ],
        "image": [to_uniform_canvas(img) for img in batch["image"]],
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


def _patch_qwen3vl_rope_alignment() -> None:
    """Align M-RoPE position length with the attention mask (verl#856-style).

    Under TRL 1.8's GRPO, Qwen3-VL's get_rope_index intermittently computes a few
    more positions than the sample has unpadded tokens (a small constant, e.g. 6 —
    vision wrapper tokens), which crashes as a shape mismatch whenever prompts in a
    batch aren't identical. Trimming the tail keeps completion positions within ~6
    of exact; training sanity is verified by reward levels at step 10 (mispairing
    would produce near-random correctness).
    """
    from transformers.models.qwen3_vl import modeling_qwen3_vl

    cls = modeling_qwen3_vl.Qwen3VLModel
    original = cls.get_rope_index
    if getattr(cls, "_rope_alignment_patched", False):
        return

    def patched(self, input_ids, image_grid_thw=None, video_grid_thw=None,
                attention_mask=None, **kwargs):
        import torch

        try:
            return original(self, input_ids, image_grid_thw=image_grid_thw,
                            video_grid_thw=video_grid_thw, attention_mask=attention_mask,
                            **kwargs)
        except RuntimeError as err:
            if "shape mismatch" not in str(err) or attention_mask is None:
                raise
        # Per-sample retry (assumes ONE image per sample — true for this project).
        # Samples that still misalign individually fall back to 1D positions.
        batch, seq = input_ids.shape
        position_ids = torch.zeros(3, batch, seq, dtype=input_ids.dtype, device=input_ids.device)
        deltas = []
        per_sample_kwargs = {
            k: v for k, v in kwargs.items() if not hasattr(v, "shape") or v.shape[0] != batch
        }
        for i in range(batch):
            row_kwargs = dict(per_sample_kwargs)
            for k, v in kwargs.items():
                if hasattr(v, "shape") and v.shape[0] == batch:
                    row_kwargs[k] = v[i : i + 1]
            grid = image_grid_thw[i : i + 1] if image_grid_thw is not None else None
            try:
                pos_i, delta_i = original(
                    self, input_ids[i : i + 1], image_grid_thw=grid, video_grid_thw=None,
                    attention_mask=attention_mask[i : i + 1], **row_kwargs,
                )
                position_ids[:, i : i + 1] = pos_i
                deltas.append(delta_i.flatten()[0].item())
            except RuntimeError:
                mask = attention_mask[i].bool()
                n = int(mask.sum())
                pos = torch.arange(n, device=input_ids.device).view(1, -1).expand(3, -1)
                position_ids[:, i, mask] = pos
                deltas.append(0)
        return position_ids, torch.tensor(deltas, device=input_ids.device).unsqueeze(1)

    cls.get_rope_index = patched
    cls._rope_alignment_patched = True


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
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
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

    _patch_qwen3vl_rope_alignment()

    run_name = args.run_name or preset.name
    model_path = resolve_model(args.model or preset.model)
    max_steps = args.max_steps or preset.max_steps
    kl_beta = args.kl_beta if args.kl_beta is not None else preset.kl_beta

    # No processor size overrides: colocated vLLM preprocesses with the model's
    # default config, and any asymmetry vs the training processor breaks Qwen3-VL's
    # M-RoPE indexing. The pixel budget is enforced in the data (cap_pixels).
    processor = AutoProcessor.from_pretrained(model_path)

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
        rollout_func=make_chat_rollout(preset),
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
