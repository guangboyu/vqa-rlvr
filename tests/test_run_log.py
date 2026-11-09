"""Run logs on shared storage: appends keep their order, survive failures, never raise."""

from vqar import run_log
from vqar.run_log import LineAppender


def test_lines_arrive_in_order_after_a_flush(tmp_path):
    path = tmp_path / "metrics.jsonl"
    for i in range(50):
        run_log.append(path, f"{i}\n")
    assert run_log.flush(timeout=10)
    assert path.read_text().splitlines() == [str(i) for i in range(50)]


def test_a_failed_append_is_retried(tmp_path, monkeypatch):
    path = tmp_path / "draws.jsonl"
    waits, real_open = [], type(path).open
    failures = iter([OSError(121, "Remote I/O error"), OSError(70, "Communication error")])

    def flaky_open(self, *args, **kwargs):
        error = next(failures, None)
        if error is not None:
            raise error
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(type(path), "open", flaky_open)
    writer = LineAppender(retry_seconds=(1, 2, 3), sleep=waits.append)
    writer.append(path, "kept\n")
    assert writer.flush(timeout=10)
    monkeypatch.undo()
    assert path.read_text() == "kept\n" and waits == [1, 2]


def test_a_line_that_keeps_failing_is_dropped_without_raising(tmp_path, capsys):
    missing = tmp_path / "no-such-dir" / "metrics.jsonl"
    writer = LineAppender(retry_seconds=(0,), sleep=lambda seconds: None)
    writer.append(missing, "lost\n")
    writer.append(tmp_path / "ok.jsonl", "next\n")
    assert writer.flush(timeout=10)
    assert (tmp_path / "ok.jsonl").read_text() == "next\n"
    assert "dropped a line" in capsys.readouterr().err
