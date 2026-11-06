"""The category table: which cells each arm needs, the macro score, and verdicts."""

from scripts.categories import arm_protocols, load_config, macro_score, scorer, verdict
from vqar.registry import DATASETS


def test_every_arm_gets_its_prompt_and_a_short_black_cell():
    config = load_config()
    spec = next(s for s in config["arms"] if s["label"] == "cqa-grpo2-base-e1")
    cells = {(p.dataset, p.prompt, p.image_mode) for p in arm_protocols(config, spec)}
    for dataset in config["benchmarks"]:
        assert {(dataset, "reasoning", "original"), (dataset, "short", "original"),
                (dataset, "short", "black")} <= cells


def test_originals_come_before_black_images():
    config = load_config()
    modes = [p.image_mode for p in arm_protocols(config, config["arms"][0])]
    assert modes == sorted(modes, reverse=True)


def test_high_resolution_benchmark_keeps_its_budget():
    config = load_config()
    budgets = {p.dataset: p.max_pixels for p in arm_protocols(config, config["arms"][0])}
    assert budgets["hrbench4k"] == 2_097_152 and budgets["mmstar"] < budgets["hrbench4k"]


def test_every_benchmark_has_a_category_and_a_scorer():
    config = load_config()
    for dataset in config["benchmarks"]:
        assert DATASETS[dataset].category
    assert scorer(config, "omnimedx") == "normalized"
    assert scorer(config, "chartqa") == "official"
    assert scorer(config, "mathvista") == "project"


def test_macro_weights_categories_not_benchmarks():
    scores = {"a": 80.0, "b": 60.0, "c": 50.0, "d": 10.0}
    categories = {"a": "X", "b": "X", "c": "Y", "d": "Z"}
    result = macro_score(scores, categories, ["X", "Y"])
    assert result["categories"] == {"X": 70.0, "Y": 50.0}
    assert result["macro"] == 60.0


def test_a_win_needs_an_interval_above_zero():
    assert verdict({"low": 0.1, "high": 2.0}) == "win"
    assert verdict({"low": -2.0, "high": -0.1}) == "loss"
    assert verdict({"low": -0.1, "high": 2.0}) == "tie"
