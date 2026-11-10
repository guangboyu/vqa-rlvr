"""Save checkpoints to node-local disk and copy them to the run directory in the background.

On the cluster, a save straight to the shared file system took up to 14 minutes during
storage stalls, and training waited for it (2026-11-08). With VQAR_LOCAL_CHECKPOINTS naming a
node-local directory, the Trainer saves there instead, and one background thread copies each
complete checkpoint, each kept adapter and TRL's completion logs into the run's checkpoint
directory. A copied checkpoint gets its complete.json last, so a successor job only resumes
from a finished copy; a job killed mid-copy loses the checkpoints not yet copied, and the
successor resumes from the last one that was.
"""

import os
import queue
import shutil
import sys
import threading
import time
from pathlib import Path

from transformers import TrainerCallback

ENV = "VQAR_LOCAL_CHECKPOINTS"
MARKER = "complete.json"
RETRY_SECONDS = (5, 30, 120)


def local_root() -> Path | None:
    value = os.environ.get(ENV)
    return Path(value) if value else None


class CheckpointMirror(TrainerCallback):
    """Copies local/checkpoint-N, local/adapters/step-N and local/completions to shared/."""

    def __init__(self, local: Path, shared: Path, save_total_limit: int | None = None,
                 retry_seconds=RETRY_SECONDS, sleep=time.sleep):
        self.local, self.shared = Path(local), Path(shared)
        self.save_total_limit = save_total_limit
        self._retry_seconds, self._sleep = retry_seconds, sleep
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="checkpoint-mirror", daemon=True)
        self._thread.start()

    def on_save(self, args, state, control, **kwargs):
        self._queue.put(state.global_step)

    def flush(self, timeout: float = 1800) -> bool:
        """Wait until every save queued so far is copied or given up on."""
        done = threading.Event()
        self._queue.put(done)
        return done.wait(timeout)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if isinstance(item, threading.Event):
                item.set()
                continue
            for name, copy in (("checkpoint", self._copy_checkpoint),
                               ("adapter", self._copy_adapter),
                               ("completions", self._copy_completions),
                               ("rotation", self._rotate)):
                self._retry(f"{name} {item}", copy, item)

    def _retry(self, what: str, action, step: int) -> None:
        for wait in (*self._retry_seconds, None):
            try:
                action(step)
                return
            except OSError as error:
                if wait is None:
                    try:
                        print(f"[checkpoint mirror] gave up on {what}: {error}", file=sys.stderr)
                    except OSError:
                        pass
                    return
                self._sleep(wait)

    def _copy_checkpoint(self, step: int) -> None:
        source = self.local / f"checkpoint-{step}"
        if not (source / MARKER).is_file():
            return  # an interrupted save is never copied
        target = self.shared / source.name
        if (target / MARKER).is_file():
            return
        copy_atomically(source, target, ignore=shutil.ignore_patterns(MARKER))
        shutil.copy2(source / MARKER, target / MARKER)

    def _copy_adapter(self, step: int) -> None:
        source = self.local / "adapters" / f"step-{step}"
        target = self.shared / "adapters" / source.name
        if source.is_dir() and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            copy_atomically(source, target)

    def _copy_completions(self, step: int) -> None:
        source = self.local / "completions"
        if not source.is_dir():
            return
        target = self.shared / "completions"
        target.mkdir(parents=True, exist_ok=True)
        for path in source.iterdir():
            copied = target / path.name
            if path.is_file() and (not copied.exists()
                                   or copied.stat().st_size != path.stat().st_size):
                shutil.copy2(path, copied)

    def _rotate(self, step: int) -> None:
        """Keep the newest save_total_limit complete copies, as the Trainer does locally;
        with no limit every checkpoint stays."""
        if not self.save_total_limit:
            return
        complete = sorted((path for path in self.shared.glob("checkpoint-*")
                           if path.name[11:].isdigit() and (path / MARKER).is_file()),
                          key=lambda path: int(path.name[11:]))
        for path in complete[:-self.save_total_limit]:
            shutil.rmtree(path)


def copy_atomically(source: Path, target: Path, ignore=None) -> None:
    """Copy a directory under a temporary name, then rename it into place."""
    partial = target.with_name(target.name + ".tmp")
    shutil.rmtree(partial, ignore_errors=True)
    shutil.copytree(source, partial, ignore=ignore)
    if target.exists():
        shutil.rmtree(target)
    partial.rename(target)
