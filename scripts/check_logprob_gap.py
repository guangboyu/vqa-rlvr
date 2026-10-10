"""Where do vLLM and the trainer disagree on sampled tokens? A per-token log-prob check.

    uv run --frozen python -m scripts.check_logprob_gap --model checkpoints/merged/p1_r3refill \\
        --datasets vqav2 --questions 64 --out runs/gap.json

GRPO logs `sampling/sampling_logp_difference/mean`: the mean |logp_vLLM - logp_trainer| over
sampled tokens. This script reproduces both sides of that number outside training:

1. generate: rollouts exactly as training draws them (`vqar.rl.make_chat_rollout` on RL pool
   rows, TRL's colocate engine settings), saved with vLLM's per-token log-probs to
   <out>.samples.json;
2. score: the trainer's log-probs of the same token ids, built as TRL's GRPO trainer builds
   its no-grad forward (vLLM's prompt ids + completion ids, processor pixel values, TRL's own
   `_get_per_token_logps_and_entropies`), in bf16;
3. report: |delta logp| overall, by answer form (spaced `<answer> x </answer>` or not), by the
   sampled token's vLLM probability, and both crossed; the tokens with the largest mean
   |delta|; the per-response importance-sampling ratio. Printed, and written to --out.

Prompt ids that differ between vLLM and the processor would mean a real bug (the trainer
would score a different prompt), so every mismatch is counted.

With --adapter, --model is the base: vLLM gets the adapter merged into bf16 weights (as TRL's
weight sync does) and the trainer side runs base + unmerged LoRA (as training does). Adding
--unmerged-adapter serves the adapter to vLLM unmerged instead, as a LoRARequest on the base
weights (as `train_grpo.py --rollout-lora` does); the trainer side is unchanged.

Stages: `all` (default) generates in a child process, so the engine's GPU memory is released
before the HF model loads, then scores; `generate` and `score` run one stage each.
"""

import argparse
import json
import math
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from vqar.config import GRPO_PRESETS, RL_RECIPE
from vqar.registry import datasets_with_role
from vqar.rl import ROLLOUT_TEMPERATURE

SPACED_ANSWER = re.compile(r"<answer>\s|\s</answer>")
# (label, lower bound on the sampled token's vLLM probability), highest first.
PROBABILITY_BINS = (("p>0.9", 0.9), ("0.5-0.9", 0.5), ("0.1-0.5", 0.1), ("p<0.1", 0.0))
IS_RATIO_CAP = RL_RECIPE["vllm_importance_sampling_clip_max"]  # sequence_mask drops above it
TRL_MAX_BATCHED_TOKENS = 4096  # TRL 1.8's colocate engine (trl/generation/vllm_generation.py)
TRL_MAX_MODEL_LEN = 2048  # vqar.rl.grpo_config's vllm_max_model_length
TOP_P = 1.0  # GRPOConfig default; the rollout reads trainer.args.top_p


# ---- pure summaries (CPU, tested) ----

def is_spaced(text: str) -> bool:
    """Whether a response writes its answer with whitespace inside the tags."""
    return bool(SPACED_ANSWER.search(text))


def probability_bin(logprob: float) -> str:
    probability = math.exp(logprob)
    for label, low in PROBABILITY_BINS:
        if probability > low:
            return label
    return PROBABILITY_BINS[-1][0]


def first_mismatch(a: list[int], b: list[int]) -> int | None:
    """First position where two id lists differ (a length difference counts), else None."""
    for position, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return position
    return None if len(a) == len(b) else min(len(a), len(b))


def delta_stats(deltas: list[float]) -> dict:
    if not deltas:
        return {"tokens": 0, "mean": None, "p50": None, "p90": None, "max": None}
    ordered = sorted(deltas)

    def quantile(q):
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]

    return {"tokens": len(ordered), "mean": sum(ordered) / len(ordered),
            "p50": quantile(0.5), "p90": quantile(0.9), "max": ordered[-1]}


def is_ratio(vllm_logprobs: list[float], train_logprobs: list[float]) -> float:
    """TRL's sequence-level importance-sampling ratio: exp(sum(train - vLLM))."""
    total = sum(t - v for t, v in zip(train_logprobs, vllm_logprobs, strict=True))
    return math.exp(min(total, 700.0))  # past exp(700) the ratio is masked anyway


def ratio_stats(ratios: list[float]) -> dict:
    if not ratios:
        return {"responses": 0}
    ordered = sorted(ratios)
    return {"responses": len(ordered), "mean": sum(ordered) / len(ordered),
            "median": ordered[len(ordered) // 2], "min": ordered[0], "max": ordered[-1],
            "masked_share": sum(r > IS_RATIO_CAP for r in ordered) / len(ordered)}


def summarize(samples: list[dict], top: int = 10, min_count: int = 5) -> dict:
    """Aggregate scored samples. Each sample needs `text`, `completion_ids`, `tokens` (each id
    decoded), `vllm_logprobs`, `train_logprobs` (None when the trainer could not score it), and
    `prompt_mismatch` (None when vLLM's prompt ids equal the processor's)."""
    deltas = {"all": []}
    ratios = {"all": [], "spaced": [], "unspaced": []}
    by_token = {}
    responses = {"spaced": 0, "unspaced": 0}
    scored = [sample for sample in samples if sample["train_logprobs"] is not None]
    for sample in scored:
        form = "spaced" if is_spaced(sample["text"]) else "unspaced"
        responses[form] += 1
        ratio = is_ratio(sample["vllm_logprobs"], sample["train_logprobs"])
        ratios["all"].append(ratio)
        ratios[form].append(ratio)
        tokens = zip(sample["completion_ids"], sample["tokens"], sample["vllm_logprobs"],
                     sample["train_logprobs"], strict=True)
        for token_id, token, vllm_logprob, train_logprob in tokens:
            delta = abs(train_logprob - vllm_logprob)
            bin_ = probability_bin(vllm_logprob)
            for key in ("all", form, bin_, (form, bin_)):
                deltas.setdefault(key, []).append(delta)
            by_token.setdefault(token_id, {"token": token, "deltas": []})["deltas"].append(delta)

    total = sum(deltas["all"]) or 1.0
    ranked = [{"token_id": token_id, "token": entry["token"], "count": len(entry["deltas"]),
               "mean": sum(entry["deltas"]) / len(entry["deltas"]),
               "share_of_total": sum(entry["deltas"]) / total}
              for token_id, entry in by_token.items() if len(entry["deltas"]) >= min_count]
    ranked.sort(key=lambda row: row["mean"], reverse=True)
    bins = [label for label, _ in PROBABILITY_BINS]
    return {
        "responses": len(samples),
        "unscored": len(samples) - len(scored),
        "responses_by_form": responses,
        "prompt_id_mismatches": sum(s["prompt_mismatch"] is not None for s in samples),
        "overall": delta_stats(deltas["all"]),
        "by_form": {form: delta_stats(deltas.get(form, [])) for form in responses},
        "by_probability": {b: delta_stats(deltas.get(b, [])) for b in bins},
        "by_form_and_probability": {form: {b: delta_stats(deltas.get((form, b), []))
                                           for b in bins} for form in responses},
        "top_tokens": {"min_count": min_count, "rows": ranked[:top]},
        "is_ratio": {key: ratio_stats(values) for key, values in ratios.items()},
    }


def format_report(summary: dict) -> str:
    def cell(stats):
        if not stats["tokens"]:
            return f"{0:>7} {'-':>8} {'-':>8} {'-':>8}"
        return (f"{stats['tokens']:>7} {stats['mean']:>8.4f} {stats['p90']:>8.4f} "
                f"{stats['max']:>8.3f}")

    forms = summary["responses_by_form"]
    lines = [f"responses: {summary['responses']} ({forms['spaced']} spaced, "
             f"{forms['unspaced']} unspaced, {summary['unscored']} unscored); "
             f"prompt-id mismatches vLLM vs processor: {summary['prompt_id_mismatches']}",
             "",
             f"{'|delta logp| over sampled tokens':<34} {'tokens':>7} {'mean':>8} {'p90':>8} "
             f"{'max':>8}",
             f"{'all':<34} {cell(summary['overall'])}"]
    lines += [f"{form:<34} {cell(stats)}" for form, stats in summary["by_form"].items()]
    lines += [f"{'vLLM ' + b:<34} {cell(stats)}"
              for b, stats in summary["by_probability"].items()]
    lines += [f"{form + ', vLLM ' + b:<34} {cell(stats)}"
              for form, row in summary["by_form_and_probability"].items()
              for b, stats in row.items()]
    top = summary["top_tokens"]
    lines += ["", f"tokens with the largest mean |delta| (count >= {top['min_count']}):",
              f"{'token':<20} {'id':>7} {'count':>6} {'mean':>8} {'share':>7}"]
    lines += [f"{row['token']!r:<20} {row['token_id']:>7} {row['count']:>6} "
              f"{row['mean']:>8.4f} {row['share_of_total']:>7.1%}" for row in top["rows"]]
    lines += ["", f"importance-sampling ratio exp(sum(train - vLLM)) per response "
              f"(masked above {IS_RATIO_CAP:g}):"]
    for key, stats in summary["is_ratio"].items():
        if stats["responses"]:
            lines.append(f"{key:<10} n={stats['responses']:<5} mean {stats['mean']:.3f} "
                         f"median {stats['median']:.3f} min {stats['min']:.3g} "
                         f"max {stats['max']:.3g} masked {stats['masked_share']:.1%}")
    return "\n".join(lines)


def batches(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


# ---- GPU stages (heavy imports inside) ----

def load_rows(preset, questions: int, data_dir: Path | None):
    """The first `questions` rows of the preset's RL pool as GRPO rows (image fitted as the
    preset trains, rollout-template prompt), in the training seed's order."""
    from vqar.rl import load_rl_dataset

    dataset = load_rl_dataset(preset, data_dir)
    return [dataset[i] for i in range(min(questions, len(dataset)))]


def repeated_prompts(rows: list[dict], samples: int) -> list:
    """What the trainer hands its rollout function: each question's conversation with its
    image attached, repeated once per sample (TRL's RepeatSampler order)."""
    from trl.data_utils import prepare_multimodal_messages

    return [prepare_multimodal_messages(row["prompt"], images=[row["image"]])
            for row in rows for _ in range(samples)]


def merge_adapter(base: str, revision: str | None, adapter: str, destination: Path) -> None:
    """Merge a LoRA adapter into bf16 base weights on the CPU: the weights TRL's colocate sync
    hands vLLM during training."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(base, revision=revision,
                                                        dtype=torch.bfloat16)
    PeftModel.from_pretrained(model, adapter).merge_and_unload().save_pretrained(destination)
    AutoProcessor.from_pretrained(base, revision=revision).save_pretrained(destination)


def adapter_rank(adapter: str) -> int:
    return json.loads((Path(adapter) / "adapter_config.json").read_text())["r"]


def served_adapter(adapter: str):
    """The adapter as vLLM serves it unmerged; one adapter, so one fixed id."""
    from vllm.lora.request import LoRARequest

    return LoRARequest("adapter-1", 1, str(Path(adapter).resolve()))


def generate(args, preset, model_path: str) -> None:
    import tempfile
    from types import SimpleNamespace

    from vllm import LLM

    from vqar.rl import make_chat_rollout

    rows = load_rows(preset, args.questions, args.data_dir)
    with tempfile.TemporaryDirectory(dir=args.out.parent, prefix="merged-") as scratch:
        engine_model, revision, lora = model_path, args.revision, {}
        if args.unmerged_adapter:
            lora = dict(enable_lora=True, max_lora_rank=adapter_rank(args.adapter),
                        max_loras=1)
        elif args.adapter:
            merge_adapter(model_path, args.revision, args.adapter, Path(scratch))
            engine_model, revision = scratch, None
        per_batch = args.questions_per_batch * args.samples
        # TRL's colocate engine, minus the in-process launcher and sleep mode. Canvas images
        # (608 px) sit under the default pixel budget, so no mm_processor_kwargs, as in training.
        llm = LLM(model=engine_model, revision=revision, max_model_len=TRL_MAX_MODEL_LEN,
                  gpu_memory_utilization=args.gpu_memory_utilization, max_num_seqs=per_batch,
                  max_num_batched_tokens=TRL_MAX_BATCHED_TOKENS,
                  logprobs_mode="processed_logprobs", seed=args.seed,
                  limit_mm_per_prompt={"image": 1}, **lora)
        generation = SimpleNamespace(llm=llm, enable_sleep_mode=False,
                                     adapter_request=lambda step: served_adapter(args.adapter))
        trainer = SimpleNamespace(
            vllm_generation=generation, state=SimpleNamespace(global_step=0),
            args=SimpleNamespace(temperature=ROLLOUT_TEMPERATURE, top_p=TOP_P,
                                 max_completion_length=preset.max_completion_length))
        rollout = make_chat_rollout(serve_adapter=args.unmerged_adapter)
        tokenizer = llm.get_tokenizer()
        samples = []
        for batch_number, batch in enumerate(batches(rows, args.questions_per_batch)):
            output = rollout(repeated_prompts(batch, args.samples), trainer)
            for index, (prompt_ids, completion_ids, logprobs) in enumerate(zip(
                    output["prompt_ids"], output["completion_ids"], output["logprobs"],
                    strict=True)):
                row = batch[index // args.samples]
                samples.append({
                    "batch": batch_number, "dataset": row["dataset"], "qid": row["qid"],
                    "prompt_ids": prompt_ids, "completion_ids": completion_ids,
                    "vllm_logprobs": logprobs,
                    "text": tokenizer.decode(completion_ids, skip_special_tokens=True),
                    "tokens": [tokenizer.decode([token_id]) for token_id in completion_ids],
                })
            print(f"generated {len(samples)} responses", flush=True)
    write_json(samples_path(args.out), {"settings": settings(args, preset, model_path),
                                        "samples": samples})


def load_policy(model_path: str, revision: str | None, adapter: str | None):
    """The trainer-side policy: bf16 weights, plus the unmerged LoRA adapter (its weights
    upcast to fp32, PEFT's default that TRL keeps for bf16 training)."""
    import torch
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(
        model_path, revision=revision, dtype=torch.bfloat16, device_map={"": 0})
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.train()  # as in training: the no-grad scoring pass runs in train mode (no dropout)
    return model


def score_batch(model, processor, rows: list[dict], batch: list[dict],
                forward_batch: int) -> list[dict]:
    """Trainer log-probs for one generation batch, built as GRPOTrainer builds them."""
    import inspect
    from types import SimpleNamespace

    import torch
    from trl import GRPOTrainer
    from trl.data_utils import apply_chat_template, prepare_multimodal_messages
    from trl.trainer.utils import pad

    by_key = {(row["dataset"], row["qid"]): row for row in rows}
    batch_rows = [by_key[(s["dataset"], s["qid"])] for s in batch]
    prompts = [prepare_multimodal_messages(row["prompt"], images=[row["image"]])
               for row in batch_rows]
    images = [[row["image"]] for row in batch_rows]
    texts = [apply_chat_template({"prompt": p}, processor)["prompt"] for p in prompts]
    prompt_inputs = processor(images=images, text=texts, padding=True, return_tensors="pt")
    for sample, ids, mask in zip(batch, prompt_inputs["input_ids"],
                                 prompt_inputs["attention_mask"], strict=True):
        processor_ids = ids[mask.bool()].tolist()
        sample["prompt_mismatch"] = first_mismatch(sample["prompt_ids"], processor_ids)
        if sample["prompt_mismatch"] is not None:  # kept for inspection
            sample["processor_prompt_ids"] = processor_ids

    device = model.device
    pad_id = processor.tokenizer.pad_token_id
    prompt_ids = pad([torch.tensor(s["prompt_ids"]) for s in batch], pad_id, "left")
    prompt_mask = pad([torch.ones(len(s["prompt_ids"]), dtype=torch.long) for s in batch], 0,
                      "left")
    completion_ids = pad([torch.tensor(s["completion_ids"]) for s in batch], pad_id, "right")
    completion_mask = pad([torch.ones(len(s["completion_ids"]), dtype=torch.long)
                           for s in batch], 0, "right")
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1).to(device)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1).to(device)
    forward_kwargs = {k: v.to(device) for k, v in prompt_inputs.items()
                      if k not in ("input_ids", "attention_mask")}
    if "mm_token_type_ids" in forward_kwargs:
        mm_types = forward_kwargs["mm_token_type_ids"]
        if mm_types.shape != prompt_ids.shape:  # only with mismatched prompts; TRL would fail
            mm_types = (prompt_ids == processor.image_token_id).long().to(device)
        forward_kwargs["mm_token_type_ids"] = torch.cat(
            [mm_types, mm_types.new_zeros(completion_ids.shape)], dim=1)

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    trainer = SimpleNamespace(model_kwarg_keys=inspect.signature(base.forward).parameters.keys(),
                              temperature=ROLLOUT_TEMPERATURE)
    per_token_logps = GRPOTrainer._get_per_token_logps_and_entropies.__wrapped__
    try:
        with torch.no_grad():
            logps, _, _ = per_token_logps(
                trainer, model, input_ids, attention_mask, completion_ids.size(1),
                math.gcd(forward_batch, len(batch)), num_images=[1] * len(batch),
                **forward_kwargs)
    except RuntimeError as error:
        if all(sample["prompt_mismatch"] is None for sample in batch):
            raise
        # A mismatched prompt can break the model's image-position bookkeeping, as it would in
        # training; the mismatch count is then the finding.
        print(f"forward failed on a batch with mismatched prompts: {error}", flush=True)
        for sample in batch:
            sample["train_logprobs"] = None
        return batch
    for sample, row in zip(batch, logps.float().cpu(), strict=True):
        sample["train_logprobs"] = row[: len(sample["completion_ids"])].tolist()
    return batch


def score(args, preset, model_path: str) -> dict:
    import torch
    from transformers import AutoProcessor

    saved = json.loads(samples_path(args.out).read_text())
    if saved["settings"] != settings(args, preset, model_path):
        raise ValueError(f"{samples_path(args.out)} was generated with other settings: "
                         f"{saved['settings']}")
    samples = saved["samples"]
    rows = load_rows(preset, args.questions, args.data_dir)
    processor = AutoProcessor.from_pretrained(model_path, revision=args.revision,
                                              padding_side="left")
    model = load_policy(model_path, args.revision, args.adapter)
    by_batch = {}
    for sample in samples:
        by_batch.setdefault(sample["batch"], []).append(sample)
    for number, batch in sorted(by_batch.items()):
        score_batch(model, processor, rows, batch, preset.per_device_batch_size)
        print(f"scored generation batch {number + 1}/{len(by_batch)}", flush=True)
        torch.cuda.empty_cache()
    write_json(samples_path(args.out), saved)
    summary = summarize(samples, top=args.top, min_count=args.min_token_count)
    mismatched = [s for s in samples if s["prompt_mismatch"] is not None]
    if mismatched:
        summary["first_mismatch"] = mismatch_example(mismatched[0], processor.tokenizer)
    return {"settings": saved["settings"], "summary": summary}


def mismatch_example(sample: dict, tokenizer, context: int = 8) -> dict:
    """Both prompts decoded around their first differing position."""
    position = sample["prompt_mismatch"]
    window = slice(max(0, position - context), position + context)
    return {"dataset": sample["dataset"], "qid": sample["qid"], "position": position,
            "vllm": tokenizer.decode(sample["prompt_ids"][window]),
            "processor": tokenizer.decode(sample["processor_prompt_ids"][window])}


def samples_path(out: Path) -> Path:
    return out.with_name(out.stem + ".samples.json")


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def settings(args, preset, model_path: str) -> dict:
    record = {"model": model_path, "revision": args.revision, "adapter": args.adapter,
              "preset": preset.name, "datasets": list(preset.datasets), "pool": preset.pool,
              "seed": preset.seed, "rollout_prompt": preset.rollout_prompt,
              "questions": args.questions, "samples": args.samples,
              "questions_per_batch": args.questions_per_batch,
              "max_completion_length": preset.max_completion_length,
              "temperature": ROLLOUT_TEMPERATURE, "top_p": TOP_P, "engine_seed": args.seed,
              "max_model_len": TRL_MAX_MODEL_LEN}
    if args.unmerged_adapter:  # absent when merged, so earlier sample files still match
        record["vllm_adapter"] = "unmerged"
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None,
                        help="merged checkpoint dir or HF id; the base with --adapter "
                        "(default: the preset's model)")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--adapter", default=None, help="LoRA adapter dir on top of --model")
    parser.add_argument("--unmerged-adapter", action="store_true",
                        help="with --adapter: vLLM serves it unmerged (a LoRARequest, as "
                        "train_grpo.py --rollout-lora does) instead of merged into bf16 weights")
    parser.add_argument("--preset", choices=sorted(GRPO_PRESETS), default="grpo_2b_main",
                        help="the training run to mirror: pool, seed, budget, batch geometry")
    parser.add_argument("--datasets", nargs="+", choices=datasets_with_role("rl"), default=None)
    parser.add_argument("--pool", default=None, help="a declared pool under data/pools/")
    parser.add_argument("--questions", type=int, default=64,
                        help="the first N questions of the pool in the training seed's order")
    parser.add_argument("--samples", type=int, default=None,
                        help="responses per question (default: the preset's group size)")
    parser.add_argument("--questions-per-batch", type=int, default=None,
                        help="questions per generation batch (default: as in training, "
                        "batch size x accumulation / group size)")
    parser.add_argument("--max-completion-length", type=int, default=None)
    parser.add_argument("--image-fit", choices=("canvas", "budget"), default=None,
                        help="RL image geometry, as train_grpo.py --image-fit")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=0, help="vLLM engine seed (TRL: rank 0)")
    parser.add_argument("--top", type=int, default=10, help="token rows to report")
    parser.add_argument("--min-token-count", type=int, default=5,
                        help="rank only tokens sampled at least this often")
    parser.add_argument("--stage", choices=("all", "generate", "score"), default="all")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True,
                        help="summary JSON; samples go to <stem>.samples.json beside it")
    args = parser.parse_args()
    if args.unmerged_adapter and not args.adapter:
        parser.error("--unmerged-adapter needs --adapter")

    from vqar.rl import resolve_model

    preset = GRPO_PRESETS[args.preset]
    overrides = {"datasets": tuple(args.datasets) if args.datasets else None, "pool": args.pool,
                 "max_completion_length": args.max_completion_length,
                 "image_fit": args.image_fit}
    preset = replace(preset, **{k: v for k, v in overrides.items() if v is not None})
    args.samples = args.samples or preset.num_generations
    args.questions_per_batch = args.questions_per_batch or max(
        1, preset.per_device_batch_size * preset.gradient_accumulation // args.samples)
    model_path = resolve_model(args.model or preset.model)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.stage == "all":
        # A child process frees all of vLLM's GPU memory when it exits.
        subprocess.run([sys.executable, "-m", "scripts.check_logprob_gap", *sys.argv[1:],
                        "--stage", "generate"], check=True)
    if args.stage == "generate":
        generate(args, preset, model_path)
        return
    report = score(args, preset, model_path)
    write_json(args.out, report)
    print(format_report(report["summary"]))
    if "first_mismatch" in report["summary"]:
        print(f"first prompt mismatch: {report['summary']['first_mismatch']}")
    print(f"\n-> {args.out} (per-token samples: {samples_path(args.out)})")


if __name__ == "__main__":
    main()
