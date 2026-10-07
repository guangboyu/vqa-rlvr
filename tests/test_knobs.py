"""The knob table: commands per knob, verdicts against the noise floor, and the rendering."""

from scripts import knobs
from vqar.eval.report import knobs_table

SPEC = {"dataset": "chartqa", "model": "base", "revision": "abc", "runs": "runs/k",
        "records": "runs/k/records", "train": ["--max-steps", "300"],
        "baseline": {"adapter": "runs/pilot/checkpoints"},
        "knob": [{"name": "repeat", "flags": []},
                 {"name": "loss-grpo", "flags": ["--loss-type", "grpo"]}]}


def test_each_knob_trains_with_the_shared_flags_plus_its_own_then_scores_its_adapter():
    lines = knobs.commands("chartqa", SPEC)
    assert "--adapter runs/pilot/checkpoints --label knob-chartqa-baseline" in lines[0]
    assert lines[3].endswith("--max-steps 300 --loss-type grpo --run-dir runs/k/loss-grpo "
                             "--run-name knob-chartqa-loss-grpo")
    assert "--role dev --tier full" in lines[4]
    assert lines[4].endswith("--adapter runs/k/loss-grpo/checkpoints "
                             "--label knob-chartqa-loss-grpo")
    assert len(knobs.commands("chartqa", SPEC, ["loss-grpo"])) == 3


def test_a_knob_needs_an_interval_clear_of_zero_and_a_change_beyond_the_repeat():
    assert knobs.verdict({"delta": 2.0, "low": 0.5, "high": 3.5}, floor=1.0) == "+"
    assert knobs.verdict({"delta": 0.8, "low": 0.1, "high": 1.5}, floor=1.0) == "0"
    assert knobs.verdict({"delta": -2.0, "low": -3.5, "high": -0.5}, floor=1.0) == "−"
    assert knobs.verdict({"delta": 2.0, "low": -0.5, "high": 4.5}, floor=0.0) == "0"


def test_the_table_shows_both_prompts_and_marks_missing_cells():
    change = {"delta": 1.5, "low": 0.2, "high": 2.8, "tokens": 230, "baseline_tokens": 201}
    result = {"knobs": [
        {"name": "length-1024", "change": "1,024 cap", "expect": "small +", "train_hours": 1.4,
         "reasoning": change, "verdict": "+"},
        {"name": "kl-0.04", "change": "KL 0.04", "expect": "slight −", "train_hours": None}]}
    lines = knobs_table(result).splitlines()
    assert lines[2].startswith("| length-1024 | 1,024 cap | small + | +1.50 [+0.20, +2.80] | — |")
    assert "201 → 230" in lines[2] and lines[2].endswith("| 1.4 | + |")
    assert lines[3].endswith("| — | — | — | — | — |")
    assert knobs_table(None) == "No knob has been run yet.\n"
