"""Checkpoints saved on local disk reach the run directory complete, or not at all."""

import shutil
from types import SimpleNamespace

from vqar.checkpoint_mirror import MARKER, CheckpointMirror
from vqar.training import last_complete_checkpoint


def save(local, step, complete=True, adapter=False):
    checkpoint = local / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_text(f"weights {step}")
    (checkpoint / "optimizer.pt").write_text("state")
    if complete:
        (checkpoint / MARKER).write_text(f'{{"step": {step}}}\n')
    if adapter:
        kept = local / "adapters" / f"step-{step}"
        kept.mkdir(parents=True)
        (kept / "adapter_model.safetensors").write_text(f"weights {step}")


def mirror_steps(mirror, *steps):
    for step in steps:
        mirror.on_save(None, SimpleNamespace(global_step=step), None)
    assert mirror.flush(timeout=10)


def test_complete_checkpoints_adapters_and_completions_are_copied(tmp_path):
    local, shared = tmp_path / "local", tmp_path / "shared"
    save(local, 250, adapter=True)
    (local / "completions").mkdir()
    (local / "completions" / "completions_00250.parquet").write_text("rows")
    mirror_steps(CheckpointMirror(local, shared), 250)
    assert (shared / "checkpoint-250" / "adapter_model.safetensors").read_text() == "weights 250"
    assert (shared / "checkpoint-250" / MARKER).is_file()
    assert (shared / "adapters" / "step-250" / "adapter_model.safetensors").is_file()
    assert (shared / "completions" / "completions_00250.parquet").read_text() == "rows"
    assert last_complete_checkpoint(shared).endswith("checkpoint-250")


def test_an_interrupted_save_is_not_copied_and_resume_uses_the_last_complete_one(tmp_path):
    local, shared = tmp_path / "local", tmp_path / "shared"
    save(local, 10)
    save(local, 20, complete=False)
    mirror_steps(CheckpointMirror(local, shared), 10, 20)
    assert not (shared / "checkpoint-20").exists()
    assert last_complete_checkpoint(shared).endswith("checkpoint-10")


def test_a_failed_copy_is_retried(tmp_path, monkeypatch):
    local, shared = tmp_path / "local", tmp_path / "shared"
    save(local, 5)
    real, calls = shutil.copytree, []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise OSError(121, "Remote I/O error")
        return real(*args, **kwargs)

    monkeypatch.setattr(shutil, "copytree", flaky)
    waits = []
    mirror_steps(CheckpointMirror(local, shared, retry_seconds=(1,), sleep=waits.append), 5)
    assert (shared / "checkpoint-5" / MARKER).is_file() and waits == [1]


def test_a_save_limit_keeps_only_the_newest_copies(tmp_path):
    local, shared = tmp_path / "local", tmp_path / "shared"
    for step in (10, 20, 30):
        save(local, step)
    mirror_steps(CheckpointMirror(local, shared, save_total_limit=2), 10, 20, 30)
    assert sorted(p.name for p in shared.glob("checkpoint-*")) == ["checkpoint-20",
                                                                   "checkpoint-30"]
