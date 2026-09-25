"""The training watcher windows logged signals and flags the patterns that mean trouble."""

import json

from scripts.watch_training import flags, logged_lines, windows


def write_metrics(path, entries):
    path.write_text("".join(json.dumps(e) + "\n" for e in entries))


def test_windows_average_logged_lines_and_skip_events(tmp_path):
    path = tmp_path / "metrics.jsonl"
    write_metrics(path, [
        {"event": "start", "step": 0},
        {"step": 10, "reward": 0.4, "entropy": 0.8},
        {"step": 20, "reward": 0.6, "entropy": 0.6},
        {"step": 20, "reward": 0.5, "entropy": 0.7},  # replayed after a resume: last wins
        {"step": 30, "reward": 0.9, "entropy": 0.1},
    ])
    rows = windows(logged_lines(path), size=20)
    assert [r["updates"] for r in rows] == ["1-20", "21-30"]
    assert rows[0]["reward"] == 0.45 and rows[0]["entropy"] == 0.75
    assert "dead" not in rows[0]


def test_flags_name_each_warning_sign():
    rows = [
        {"updates": "1-100", "reward": 0.4, "entropy": 0.8, "logp_gap": 0.03, "dead": 0.5,
         "length": 120},
        {"updates": "101-200", "reward": 0.7, "entropy": 0.1, "logp_gap": 0.2, "dead": 0.9,
         "truncated": 0.1, "clip": 0.01, "length": 5},
    ]
    found = " | ".join(flags(rows))
    for sign in ("log-prob gap", "entropy collapsed", "dead", "truncated", "clipping",
                 "reward moved", "shrank"):
        assert sign in found
    assert flags(rows[:1]) == []
