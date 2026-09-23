from types import SimpleNamespace

import pytest

pytest.importorskip("transformers")

from scripts import hf_preempt_callback as checkpointing


def test_elapsed_time_requests_save_and_completed_save_resets_timer(tmp_path, monkeypatch):
    now = [0]
    monkeypatch.setattr(checkpointing.time, "monotonic", lambda: now[0])
    monkeypatch.setitem(checkpointing._STOP, "requested", False)
    callback = checkpointing.PreemptCallback(save_every_seconds=600)
    args = SimpleNamespace(output_dir=str(tmp_path))
    state = SimpleNamespace(global_step=7)
    control = SimpleNamespace(should_save=False, should_training_stop=False)
    callback.on_train_begin(args, state, control)
    now[0] = 599
    callback.on_step_end(args, state, control)
    assert not control.should_save
    now[0] = 600
    callback.on_step_end(args, state, control)
    assert control.should_save and not control.should_training_stop
    (tmp_path / "checkpoint-7").mkdir()
    callback.on_save(args, state, control)
    assert (tmp_path / "checkpoint-7/complete.json").is_file()
    assert not (tmp_path / "checkpoint-7/complete.json.tmp").exists()
    control.should_save = False
    now[0] = 601
    callback.on_step_end(args, state, control)
    assert not control.should_save


def test_stop_signal_requests_save_even_before_time_interval(monkeypatch):
    monkeypatch.setitem(checkpointing._STOP, "requested", True)
    callback = checkpointing.PreemptCallback()
    control = SimpleNamespace(should_save=False, should_training_stop=False)
    callback.on_step_end(None, None, control)
    assert control.should_save and control.should_training_stop
