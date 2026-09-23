"""What GRPO and on-policy distillation share: RL prompts, rollouts, and the GRPO config.

Heavy libraries (torch, trl, vllm) load inside the functions that need them, so dry runs
and CPU tests work without them.
"""

from pathlib import Path

from vqar import config
from vqar.registry import PROMPTS
from vqar.splits import load_role
from vqar.training import ROOT

# Keep the historical image geometry for comparable runs. Preserve aspect ratio
# within a fixed canvas so rollout and training use the same vision-token budget.
RL_IMAGE_SIZE = 608  # 19x19 = 361 vision tokens/image
ROLLOUT_TEMPERATURE = 1.0


def to_uniform_canvas(image):
    from PIL import Image as PILImage

    image = image.convert("RGB")
    w, h = image.size
    scale = RL_IMAGE_SIZE / max(w, h)
    image = image.resize((max(1, round(w * scale)), max(1, round(h * scale))))
    canvas = PILImage.new("RGB", (RL_IMAGE_SIZE, RL_IMAGE_SIZE))
    canvas.paste(image, ((RL_IMAGE_SIZE - image.width) // 2, (RL_IMAGE_SIZE - image.height) // 2))
    return canvas


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


def load_rl_dataset(preset):
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


def make_chat_rollout():
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


def dry_run(preset, rewards: tuple[str, ...]) -> None:
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


def grpo_config(preset, output_dir: Path, run_name: str, model_init_kwargs: dict, *,
                max_steps: int, beta: float, reward_weights: list[float], batch_size: int,
                grad_accum: int, save_steps: int, save_total_limit: int, report_to: list[str]):
    """The GRPO settings both RL trainers use; only the learning signal differs between them."""
    from trl import GRPOConfig

    return GRPOConfig(
        **config.RL_RECIPE,
        output_dir=str(output_dir),
        run_name=run_name,
        model_init_kwargs=model_init_kwargs,
        num_generations=preset.num_generations,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=preset.learning_rate,
        max_steps=max_steps,
        max_completion_length=preset.max_completion_length,
        temperature=ROLLOUT_TEMPERATURE,
        beta=beta,
        reward_weights=reward_weights,
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
        save_steps=save_steps,
        save_total_limit=save_total_limit or None,
        dataloader_num_workers=2,
        dataloader_pin_memory=False,
        seed=preset.seed,
        report_to=report_to,
        log_completions=True,
        num_completions_to_print=2,
    )
