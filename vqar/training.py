"""Run provenance shared by the three training entrypoints."""

import hashlib
import importlib.metadata
import json
import subprocess
from functools import wraps
from pathlib import Path


def write_run_config(output_dir: Path, record: dict, training_args) -> None:
    """Save resolved settings before training; reject an incompatible auto-resume."""
    root = Path(__file__).resolve().parent.parent
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

    files = sorted(root.glob("train_*.py")) + sorted((root / "vqar").glob("*.py"))
    files += [root / "uv.lock", root / "scripts/hf_preempt_callback.py"]
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
            "Use a 2B bf16 preset on 24 GB, or --hardware 80gb/96gb for 8B. "
            "The legacy preset is retained to document the failed configuration."
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
