"""The training figures' series: trailing means and the dead-group split from prompt draws."""

import json

import pytest

from scripts.plot_training import dead_share_series, plot_health, rolling_mean


def test_rolling_mean_keeps_early_points():
    assert rolling_mean([1.0, 3.0, 5.0, 7.0], window=2) == [1.0, 2.0, 4.0, 6.0]


def test_dead_share_series_splits_all_right_from_all_wrong(tmp_path):
    draw = {"step_before_update": 0, "dataset": ["vqav2"] * 4 + ["gqa"] * 4,
            "qid": ["a"] * 4 + ["b"] * 4, "scores": [1.0] * 4 + [0.0] * 4}
    (tmp_path / "prompt_draws.jsonl").write_text(json.dumps(draw) + "\n")
    series = dead_share_series(tmp_path)
    assert series["all_right"] == ([1], [0.5]) and series["all_wrong"] == ([1], [0.5])


def test_health_figure_is_written_from_metrics_alone(tmp_path):
    pytest.importorskip("matplotlib")
    run = tmp_path / "run"
    run.mkdir()
    lines = [{"step": s, "reward": 0.5, "entropy": 0.7, "completions/mean_length": 100,
              "grad_norm": 0.2} for s in (10, 20, 30)]
    (run / "metrics.jsonl").write_text("".join(json.dumps(e) + "\n" for e in lines))
    plot_health([(run, "run")], tmp_path / "health.png", window=10)
    assert (tmp_path / "health.png").stat().st_size > 0
