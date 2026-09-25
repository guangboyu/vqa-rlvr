"""What the three trainers share: arguments, presets, adapters, provenance, and the
preemption-safe training loop."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
from dataclasses import fields, replace
from functools import wraps
from pathlib import Path

from vqar import config

ROOT = Path(__file__).resolve().parent.parent


def add_common_arguments(parser: argparse.ArgumentParser, presets: dict,
                         save_steps: int) -> None:
    parser.add_argument("--preset", required=True, choices=sorted(presets))
    parser.add_argument("--model", default=None, help="override the starting model")
    parser.add_argument("--hardware", choices=config.HARDWARE_PROFILES, default="legacy")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="cluster mode: checkpoints + metrics.jsonl under this dir "
                        "(auto-resumes from its last complete checkpoint)")
    parser.add_argument("--save-steps", type=int, default=save_steps,
                        help="checkpoint cadence; on free partitions size to <=15 min")
    parser.add_argument("--save-total-limit", type=int, default=2,
                        help="keep this many saved checkpoints; 0 keeps all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--no-dev-eval", action="store_true",
                        help="skip evaluating every saved adapter on the dev tier afterwards")


def parse_arguments(parser: argparse.ArgumentParser):
    args = parser.parse_args()
    if args.save_total_limit < 0:
        parser.error("--save-total-limit must be nonnegative")
    return args


def resolve_preset(presets: dict, args):
    """The named preset with every flag that names one of its fields applied, adjusted to
    the hardware profile. The result is the whole run's recipe: trainers read only it."""
    preset = presets[args.preset]
    overrides = {
        field.name: tuple(value) if isinstance(value, list) else value
        for field in fields(preset)
        if field.name != "name" and (value := getattr(args, field.name, None)) is not None
    }
    return config.for_hardware(replace(preset, **overrides), args.hardware)


def checkpoint_dir(run_dir: Path | None, run_name: str) -> Path:
    return run_dir / "checkpoints" if run_dir else ROOT / "checkpoints" / run_name


def wandb_reporting(no_wandb: bool) -> list[str]:
    """The Trainer's report_to list; enabling wandb also sets its default project name."""
    if no_wandb:
        return []
    os.environ.setdefault("WANDB_PROJECT", "vqa-rlvr")
    return ["wandb"]


def load_kwargs(quantize_4bit: bool) -> dict:
    """from_pretrained arguments: bf16 weights, or NF4 for QLoRA."""
    import torch

    kwargs = dict(dtype=torch.bfloat16)
    if quantize_4bit:
        kwargs["quantization_config"] = nf4_quantization()
    return kwargs


def nf4_quantization():
    """QLoRA's 4-bit NF4 weights with bf16 compute."""
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def lora_config(preset):
    """LoRA on the language model's attention and MLP; the vision tower stays frozen."""
    from peft import LoraConfig

    kwargs = dict(r=preset.lora_r, lora_alpha=preset.lora_alpha,
                  target_modules=list(preset.lora_targets), task_type="CAUSAL_LM")
    if hasattr(preset, "lora_dropout"):  # only the SFT recipe sets dropout
        kwargs["lora_dropout"] = preset.lora_dropout
    return LoraConfig(**kwargs)


def run_training(trainer, output: Path, run_dir: Path | None, log_draws: bool = False) -> bool:
    """Train with preemption safety: periodic and signal-triggered complete checkpoints,
    metrics.jsonl in cluster mode, and resumption from the last complete save. Every
    saved adapter is kept for the dev evaluation. Returns whether training completed."""
    from vqar.preemption import JsonlMetrics, KeepAdapters, PreemptCallback, install_signal_flag

    install_signal_flag()
    # Keep the adapter copy before the completion marker, so a complete checkpoint
    # always has its curve point.
    trainer.add_callback(KeepAdapters())
    trainer.add_callback(PreemptCallback())
    if run_dir:
        trainer.add_callback(JsonlMetrics(str(run_dir / "metrics.jsonl")))
        if log_draws:
            log_prompt_draws(trainer, run_dir / "prompt_draws.jsonl")
    last = last_complete_checkpoint(output)
    if last:
        print(f"resuming from {last}")
    trainer.train(resume_from_checkpoint=last)
    trainer.save_model(str(output))
    write_training_summary(trainer, output)
    draws = run_dir / "prompt_draws.jsonl" if run_dir and log_draws else None
    (output / "diagnostics.json").write_text(
        json.dumps(training_diagnostics(trainer.state.log_history, draws), indent=2) + "\n")
    print(f"adapter saved -> {output}")
    return trainer.state.global_step >= trainer.state.max_steps


# Trainer log keys behind each diagnostic; a key appears only when its trainer logs it.
# The Trainer averages over its logging interval (every `logging_steps` updates).
DIAGNOSTICS = {
    "loss": "loss", "eval_loss": "eval_loss", "token_accuracy": "mean_token_accuracy",
    "reward_mean": "reward", "reward_std": "reward_std",
    "zero_variance_group_fraction": "frac_reward_zero_std", "kl": "kl",
    "entropy": "entropy", "clip_fraction": "clip_ratio/region_mean",
    "completion_length": "completions/mean_length", "tokens_seen": "num_tokens",
    "opd_reverse_kl": "opd/reverse_kl_per_token",
}


def training_diagnostics(logs: list[dict], draws: Path | None = None) -> dict:
    """What happened during training: the Trainer's logged signals, and, for RL runs, the
    reward of every update from the saved prompt draws. Reward hacking and dead updates
    (every sampled group scored the same) show up here first."""
    logged = []
    for entry in logs:
        values = {name: entry[key] for name, key in DIAGNOSTICS.items() if key in entry}
        if values:
            logged.append({"step": entry["step"], **values})
    per_update, distinct = None, None
    if draws is not None and draws.exists():
        per_update, distinct = reward_per_update(draws)
    return {"logged": logged, "reward_per_update": per_update, "distinct_prompts": distinct,
            # Filled from the learnability pre-check of the pool sample when one was run.
            "learnability_fraction": None}


def trained_draws(draws: Path) -> dict[int, dict]:
    """The rows each update trained on, keyed by the step before it.

    An update logs one record, or, when it refills dead groups, one per generation round
    (`round` 0, 1, ...) whose `kept` flags mark the rows it trained on. A resumed update
    repeats its draws, so only its last attempt counts: a record without a round, or with
    round 0, starts a new attempt. Each entry also counts what the attempt generated."""
    attempts = {}
    for line in draws.read_text().splitlines():
        record = json.loads(line)
        if record.get("round", 0) == 0:
            attempts[record["step_before_update"]] = []
        attempts[record["step_before_update"]].append(record)
    return {step: kept_rows(records) for step, records in sorted(attempts.items())}


def kept_rows(records: list[dict]) -> dict:
    """The kept rows of one update's records, plus its attempted rows and rounds."""
    kept = {"dataset": [], "qid": [], "scores": []}
    attempted = set()
    for record in records:
        flags = record.get("kept", [True] * len(record["qid"]))
        attempted.update(zip(record["dataset"], record["qid"], strict=True))
        for name, values in kept.items():
            values += [value for value, keep in zip(record[name], flags, strict=True) if keep]
    return {**kept, "attempted_rollouts": sum(len(r["qid"]) for r in records),
            "attempted_groups": len(attempted), "rounds": len(records),
            "refilled": any("round" in r for r in records)}


def reward_per_update(draws: Path) -> tuple[list[dict], int]:
    """Mean and spread of the reward in each update, and the share of prompt groups whose
    samples all scored the same (no GRPO gradient), over the rows the update trained on.
    Updates that refilled dead groups also report what they generated to get there."""
    updates, prompts = [], set()
    for step, rows in trained_draws(draws).items():
        groups = {}
        for dataset, qid, score in zip(rows["dataset"], rows["qid"], rows["scores"],
                                       strict=True):
            groups.setdefault((dataset, qid), []).append(score)
            prompts.add((dataset, qid))
        scores = rows["scores"]
        mean = sum(scores) / len(scores)
        update = {
            "update": step + 1, "reward_mean": mean,
            "reward_std": (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5,
            "zero_variance_group_fraction":
                sum(max(g) - min(g) <= 1e-12 for g in groups.values()) / len(groups),
        }
        if rows["refilled"]:
            update.update(attempted_groups=rows["attempted_groups"],
                          attempted_rollouts=rows["attempted_rollouts"],
                          refill_rounds=rows["rounds"] - 1)
        updates.append(update)
    return updates, len(prompts)


def evaluate_checkpoints_next(output: Path) -> None:
    """Replace this process with the dev evaluation of every kept adapter.

    Exec rather than a subprocess: the training process's GPU memory is released in full
    before the evaluation engine starts, which matters on a 24 GB card.
    """
    import sys

    if "wandb" in sys.modules:
        sys.modules["wandb"].finish()
    sys.stdout.flush()
    sys.stderr.flush()
    script = ROOT / "scripts" / "evaluate_checkpoints.py"
    os.execv(sys.executable, [sys.executable, str(script), str(output)])


def write_run_config(output_dir: Path, record: dict, training_args) -> None:
    """Save resolved settings before training; reject an incompatible auto-resume."""
    root = ROOT
    path = output_dir / "run_config.json"
    record = json.loads(json.dumps(record))
    # Compare the experiment identity, including overrides, before adding provenance.
    if path.exists():
        previous = json.loads(path.read_text())
        changed = [key for key, value in record.items() if previous.get(key) != value]
        if changed:
            raise ValueError(f"Existing run differs in {changed}; choose a new run directory.")
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"{output_dir} has no run_config.json; use a fresh run directory.")

    files = sorted(root.glob("train_*.py")) + sorted((root / "vqar").rglob("*.py"))
    files += [root / "uv.lock"]
    hashes = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
        if p.is_file()
    }
    record = dict(record, training_args=training_args.to_dict(), source_sha256=hashes)
    record["git_sha"] = (
        subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        or None
    )
    record["versions"] = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "trl", "peft", "vllm", "datasets")
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    # Keep the original launch provenance on resume.
    if not path.exists():
        path.write_text(json.dumps(record, indent=2, default=str) + "\n")


def check_rl_precision(preset) -> None:
    if preset.quantize_4bit:
        raise ValueError(
            "NF4 student weights are unsafe with the current TRL/vLLM synchronization. "
            "Use a 2B bf16 preset on 24 GB, or --hardware 80gb/96gb for 8B."
        )


def log_prompt_draws(trainer, path: Path) -> None:
    """Keep actual rollout IDs; resumed attempts can repeat an unfinished update."""
    reward = trainer.reward_funcs[0]

    @wraps(reward)
    def recorded_reward(*args, **kwargs):
        scores = reward(*args, **kwargs)
        record = {
            "step_before_update": trainer.state.global_step,
            "dataset": kwargs["dataset"], "qid": kwargs["qid"], "scores": scores,
        }
        with path.open("a") as file:
            file.write(json.dumps(record) + "\n")
        return scores

    trainer.reward_funcs[0] = recorded_reward


def last_complete_checkpoint(output_dir: Path) -> str | None:
    """Ignore a checkpoint killed during writing; resume only completed saves."""
    checkpoints = sorted(
        (p for p in output_dir.glob("checkpoint-*") if p.name[11:].isdigit()),
        key=lambda p: int(p.name[11:]),
        reverse=True,
    )
    for path in checkpoints:
        if (path / "complete.json").is_file():
            return str(path)
    if checkpoints:
        raise RuntimeError("No completed checkpoint is available; use a fresh run directory.")
    return None


def write_training_summary(trainer, output_dir: Path) -> None:
    """Record actual progress and allocator peaks, including an interrupted run."""
    import torch

    summary = {
        "steps": trainer.state.global_step,
        "planned_steps": trainer.state.max_steps,
        "completed": trainer.state.global_step >= trainer.state.max_steps,
        "logs": trainer.state.log_history,
    }
    if torch.cuda.is_available():
        summary["gpu"] = {
            "name": torch.cuda.get_device_name(),
            "total_bytes": torch.cuda.get_device_properties(0).total_memory,
            "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
