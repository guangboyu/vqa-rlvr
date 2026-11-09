"""Append to a run's logs on shared storage without blocking or failing training.

On the cluster the run directory sits on a shared file system that can stall for minutes or
fail a write outright ("Remote I/O error"); on 2026-11-08 one failed append to
prompt_draws.jsonl ended two InfoVQA runs. Lines therefore go through a queue to one
background thread, which retries a failed append and drops the line, with a warning, only
after the last retry. Lines still queued when a job is killed are lost; a resumed run repeats
its unfinished update anyway. Call flush() before reading a log back in the same process.
"""

import atexit
import queue
import sys
import threading
import time
from pathlib import Path

RETRY_SECONDS = (1, 4, 15, 60)


class LineAppender:
    """One writer thread; lines reach each file in the order they were appended."""

    def __init__(self, retry_seconds=RETRY_SECONDS, sleep=time.sleep):
        self._queue: queue.Queue = queue.Queue()
        self._retry_seconds, self._sleep = retry_seconds, sleep
        self._thread = threading.Thread(target=self._run, name="run-log-appender", daemon=True)
        self._thread.start()

    def append(self, path: Path | str, text: str) -> None:
        self._queue.put((Path(path), text))

    def flush(self, timeout: float = 300) -> bool:
        """Wait until every line queued so far is written or dropped."""
        done = threading.Event()
        self._queue.put((None, done))
        return done.wait(timeout)

    def _run(self) -> None:
        while True:
            path, item = self._queue.get()
            if path is None:
                item.set()
            else:
                self._write(path, item)

    def _write(self, path: Path, text: str) -> None:
        for wait in (*self._retry_seconds, None):
            try:
                with path.open("a") as file:
                    file.write(text)
                return
            except OSError as error:
                if wait is None:
                    try:
                        print(f"[run log] dropped a line for {path}: {error}", file=sys.stderr)
                    except OSError:
                        pass
                    return
                self._sleep(wait)


_APPENDER: LineAppender | None = None
_LOCK = threading.Lock()


def _appender() -> LineAppender:
    global _APPENDER
    with _LOCK:
        if _APPENDER is None:
            _APPENDER = LineAppender()
            atexit.register(_APPENDER.flush)
        return _APPENDER


def append(path: Path | str, text: str) -> None:
    """Queue text for the end of a log file; never raises a write error."""
    _appender().append(path, text)


def flush(timeout: float = 300) -> bool:
    return _appender().flush(timeout)
