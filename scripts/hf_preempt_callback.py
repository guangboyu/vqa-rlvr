"""Preemption plumbing for HF Trainer / TRL trainers (SFTTrainer, DPOTrainer…).

Trainer checkpoints contain optimizer, scheduler, RNG state, and global step.
The callback marks a checkpoint complete only after Trainer finishes writing it.
Use vqar.training.last_complete_checkpoint to ignore interrupted saves.

Periodic step and time intervals bound lost work. HPC3 free partitions can cancel instantly,
so signal handling is only useful for time-limit warnings or manual termination.
Long jobs also need the successor mechanism documented in hpc-env/AGENTS.md.
"""

import json
import signal
import time
from pathlib import Path

from transformers import TrainerCallback

_STOP = {"requested": False}


def install_signal_flag():
    def _handler(signum, frame):
        _STOP["requested"] = True

    signal.signal(signal.SIGUSR1, _handler)  # time-limit warning (--signal)
    signal.signal(signal.SIGTERM, _handler)  # manual termination; instant CANCEL sends no warning


class PreemptCallback(TrainerCallback):
    """On Slurm's warning signal: full checkpoint at step end, then clean stop.

    Instant free-partition cancellation sends no warning. Periodic saves are
    essential, and the submission script must arrange a successor separately.
    A completion marker prevents resuming a save interrupted while writing."""

    def __init__(self, save_every_seconds=600):
        if save_every_seconds <= 0:
            raise ValueError("save_every_seconds must be positive")
        self.save_every_seconds = save_every_seconds
        self.last_save = time.monotonic()

    def on_train_begin(self, args, state, control, **kwargs):
        self.last_save = time.monotonic()

    def on_save(self, args, state, control, **kwargs):
        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        marker = checkpoint / "complete.json.tmp"
        marker.write_text(json.dumps({"step": state.global_step}) + "\n")
        marker.replace(checkpoint / "complete.json")
        self.last_save = time.monotonic()

    def on_step_end(self, args, state, control, **kwargs):
        if time.monotonic() - self.last_save >= self.save_every_seconds:
            control.should_save = True
        if _STOP["requested"]:
            control.should_save = True
            control.should_training_stop = True
        return control


class JsonlMetrics(TrainerCallback):
    """Append Trainer logs to metrics.jsonl — progress checks become a tail."""

    def __init__(self, path):
        self.path = path

    def _emit(self, **kv):
        kv.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S"))
        with open(self.path, "a") as f:
            f.write(json.dumps(kv) + "\n")

    def on_train_begin(self, args, state, control, **kwargs):
        self._emit(event="resume" if state.global_step else "start", step=state.global_step)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            nums = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
            self._emit(step=state.global_step, **nums)

    def on_save(self, args, state, control, **kwargs):
        self._emit(
            step=state.global_step,
            event="preempt_checkpoint" if _STOP["requested"] else "checkpoint",
        )

    def on_train_end(self, args, state, control, **kwargs):
        if not _STOP["requested"]:
            event = "done" if state.global_step >= state.max_steps else "stopped"
            self._emit(step=state.global_step, event=event)
