import pytest

from scripts.gate import macro, macro_difference, verdict


def cells(photo: float, chart: float) -> dict[str, dict[str, float]]:
    """Four questions per dataset with the given share right."""
    def scores(share):
        right = round(4 * share)
        return {f"q{i}": float(i < right) for i in range(4)}
    return {"vqav2": scores(photo), "gqa": scores(photo), "chartqa": scores(chart)}


def test_macro_is_the_mean_of_category_means():
    assert macro(cells(0.5, 1.0)) == {"photos": 50.0, "charts": 100.0, "macro": 75.0}


def test_identical_arms_differ_by_zero_with_a_zero_interval():
    same = cells(0.75, 0.5)
    assert macro_difference(same, same, resamples=200) == {"delta": 0.0, "low": 0.0, "high": 0.0}


def test_the_difference_weights_categories_not_questions():
    result = macro_difference(cells(0.5, 1.0), cells(0.5, 0.5), resamples=200)
    assert result["delta"] == pytest.approx(25.0)


def test_mismatched_questions_are_refused():
    short = cells(0.5, 0.5)
    short["gqa"].pop("q0")
    with pytest.raises(ValueError, match="gqa"):
        macro_difference(short, cells(0.5, 0.5), resamples=10)


def test_the_gate_needs_every_rule():
    above = {"delta": 3.0, "low": 1.2, "high": 4.8}
    close = {"delta": -0.8, "low": -2.0, "high": 0.4}
    assert verdict(above, close, {"photos": -1.0, "charts": -1.5})["passes"]
    routing = verdict(above, {"delta": -2.1, "low": -3.8, "high": -0.5},
                      {"photos": -1.9, "charts": -2.2})
    assert routing == {"beats_instruct": True, "macro_within_margin": False,
                       "categories_within_margin": False, "passes": False}
