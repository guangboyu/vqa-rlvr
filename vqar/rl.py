"""What GRPO and on-policy distillation share: RL prompts, rollouts, and the GRPO config.

Heavy libraries (torch, trl, vllm) load inside the functions that need them, so dry runs
and CPU tests work without them.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from itertools import islice
from pathlib import Path

from vqar import config
from vqar.data import to_uniform_canvas
from vqar.pools import RefillStream, load_pool
from vqar.registry import PROMPTS, check_trainable
from vqar.training import ROOT

ROLLOUT_TEMPERATURE = 1.0


def to_grpo_columns(batch: dict, template: str, canvas_size: int = config.RL_IMAGE_SIZE) -> dict:
    """GRPO rows: conversational `prompt` + `image`; question/answers/dataset and the
    rollout template ride along as extra columns that TRL forwards to every reward."""
    return {
        "prompt": [
            [{"role": "user", "content": PROMPTS[template].text.format(question=q)}]
            for q in batch["question"]
        ],
        "image": [to_uniform_canvas(img, canvas_size) for img in batch["image"]],
        "question": batch["question"],
        "answers": batch["answers"],
        "dataset": batch["dataset"],
        "qid": batch["qid"],
        "rollout_prompt": [template] * len(batch["question"]),
    }


def load_rl_dataset(preset, data_dir: Path | None = None):
    """The preset's RL pool (its datasets' rl roles, or a declared subset of them) as GRPO
    rows in the preset's rollout template."""
    check_trainable(preset.datasets)
    rows = load_pool(data_dir or ROOT / "data", preset.datasets, preset.seed, preset.pool)
    columns = rows.select_columns(["image", "question", "answers", "dataset", "qid"])
    return columns.with_transform(partial(
        to_grpo_columns, template=preset.rollout_prompt,
        canvas_size=getattr(preset, "canvas_size", config.RL_IMAGE_SIZE)))


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


def make_chat_rollout(serve_adapter: bool = False):
    """Custom rollout that generates through vLLM's own chat interface.

    TRL 1.8's built-in path hands vLLM processor-pre-expanded prompt token ids PLUS
    the raw image (vllm_generation.py:664); the two preprocessing pipelines must
    agree exactly on image token expansion and don't for our data — producing
    corrupted effective prompts (untagged prose completions, importance-sampling
    ratios ~1e-6) and the M-RoPE crashes. Routing rollouts through llm.chat gives
    one source of truth: vLLM builds prompt ids itself, and the trainer's forward
    re-processes the same PIL deterministically to matching ids.

    With `serve_adapter`, vLLM holds the base weights and applies the policy's adapter
    unmerged (`vqar.adapter_rollouts`): the rollout passes the current update's adapter
    with every request and sleeps at level 1, which keeps the base weights in CPU memory.
    Otherwise TRL merges the adapter into the weights before each update, and level 2
    discards them until that sync.
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
        adapter = None
        if serve_adapter:
            adapter = generation.adapter_request(trainer.state.global_step)
        if generation.enable_sleep_mode:
            generation.llm.wake_up(tags=["kv_cache"])
        outputs = generation.llm.chat(conversations, params, use_tqdm=False,
                                      lora_request=adapter)
        if generation.enable_sleep_mode:
            generation.llm.sleep(level=1 if serve_adapter else 2)
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


@dataclass
class ScoredGroup:
    """One question's sampled answers, scored. `live` means their total rewards differ, so
    the group carries a GRPO gradient; `rollout` holds what the trainer needs to train on it."""

    question: dict  # the pool row the answers respond to
    scores: list[float]  # the first reward per answer, as the draw log records it
    live: bool
    rollout: dict = field(default_factory=dict)


def fill_live_groups(questions: list[dict], draw: Callable[[int], list[dict]],
                     score: Callable[[list[dict]], list[ScoredGroup]],
                     max_rounds: int) -> tuple[list[list[ScoredGroup]], list[ScoredGroup]]:
    """DAPO dynamic sampling for one update.

    Score the update's questions; while fewer groups are live than there are questions and
    extra rounds remain, score as many replacement questions from `draw` as groups are
    missing. The batch is every live group, topped up with the earliest dead groups (the
    update's own questions first) when the cap is reached or the pool runs out, so the
    batch keeps its size. Returns the scored groups of every round, and the batch.
    """
    target = len(questions)
    rounds = [score(questions)]
    live = [group for group in rounds[0] if group.live]
    while len(live) < target and len(rounds) <= max_rounds:
        replacements = draw(target - len(live))
        if not replacements:
            break
        rounds.append(score(replacements))
        live += [group for group in rounds[-1] if group.live]
    dead = [group for scored in rounds for group in scored if not group.live]
    return rounds, live + dead[:target - len(live)]


def refill_draw_records(step: int, rounds: list[list[ScoredGroup]],
                        batch: list[ScoredGroup]) -> list[dict]:
    """One prompt-draw record per generation round: every attempted answer, and whether the
    update trained on it."""
    kept = {id(group) for group in batch}
    records = []
    for number, groups in enumerate(rounds):
        record = {"step_before_update": step, "round": number,
                  "dataset": [], "qid": [], "scores": [], "kept": []}
        for group in groups:
            size = len(group.scores)
            record["dataset"] += [group.question["dataset"]] * size
            record["qid"] += [group.question["qid"]] * size
            record["scores"] += [None if math.isnan(s) else s for s in group.scores]
            record["kept"] += [id(group) in kept] * size
        records.append(record)
    return records


def dry_run(preset) -> None:
    """Validate the pool, the rollout template, and reward wiring without a model or GPU."""
    from vqar.rewards import REWARDS

    ds = load_rl_dataset(preset)
    row = ds[0]
    template = PROMPTS[preset.rollout_prompt]
    assert row["prompt"][0]["content"] == template.text.format(question=row["question"]), row
    pool = preset.pool or "the whole rl roles"
    print(f"dataset ok: {len(ds)} prompts from {pool} of {preset.datasets}")
    print(f"rollout prompt: {template.name} (reward reads the {template.answer_rule} answer)")
    print(f"sample question: {row['question'][:80]}")
    if getattr(preset, "refill_rounds", 0):  # GRPO only
        stream = RefillStream.over(ds, preset.seed)
        first = [stream.keys[i] for i in islice(stream.draws(update=0), 3)]
        print(f"refill: up to {preset.refill_rounds} extra rounds per update replace dead "
              f"groups; first replacements at update 1 (before excluding its own): {first}")
    if getattr(preset, "rollout_lora", False):  # GRPO only
        print(f"rollouts: vLLM serves the rank-{preset.lora_r} adapter unmerged; "
              "sleep level 1 keeps the base weights in CPU memory")

    answer = "<think>x</think><answer>4</answer>" if template.answer_rule == "tagged" else "4"
    for name in preset.rewards:
        value = REWARDS[name](
            prompts=[row["prompt"]],
            completions=[[{"role": "assistant", "content": answer}]],
            answers=[row["answers"]],
            dataset=[row["dataset"]],
            rollout_prompt=[row["rollout_prompt"]],
        )
        print(f"reward {name}: {value}")
    print("dry run OK")


def grpo_config(preset, output_dir: Path, run_name: str, model_init_kwargs: dict, *,
                save_steps: int, save_total_limit: int, report_to: list[str]):
    """The GRPO settings both RL trainers use, all read from the resolved preset; only the
    learning signal differs between them."""
    from trl import GRPOConfig

    return GRPOConfig(
        **config.rl_recipe(preset),
        output_dir=str(output_dir),
        run_name=run_name,
        model_init_kwargs=model_init_kwargs,
        num_generations=preset.num_generations,
        per_device_train_batch_size=preset.per_device_batch_size,
        gradient_accumulation_steps=preset.gradient_accumulation,
        learning_rate=preset.learning_rate,
        max_steps=preset.max_steps,
        max_completion_length=preset.max_completion_length,
        temperature=ROLLOUT_TEMPERATURE,
        beta=preset.kl_beta,
        reward_weights=list(preset.reward_weights),
        use_vllm=True,
        vllm_mode="colocate",
        vllm_enable_sleep_mode=getattr(preset, "vllm_sleep", True),
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
