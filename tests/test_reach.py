"""Reach: open-answer filtering and the verdict's margins."""

import pytest

from scripts.reach import reach_by_question, verdict
from vqar.eval.errors import reference_form
from vqar.eval.pass_at_k import SampledCell


@pytest.mark.parametrize("answers,form", [
    (["yes"], "yes/no"), (["No."], "yes/no"), (["no", "yes", "no"], "yes/no"),
    (["B", "a red car"], "choice"), (["14"], "number"), (["1,250"], "number"), (["25%"], "number"),
    (["-0.5"], "number"), (["a red car"], "text"), (["a"], "text"), ([], "text"),
])
def test_reference_form(answers, form):
    assert reference_form(answers) == form


def test_reach_leaves_out_questions_chance_can_answer():
    cell = SampledCell("path", {}, {}, "normalized", 8,
                       {"open": (8, 1.0), "closed": (8, 1.0), "choice": (1, 0.1)},
                       {"open": ("q", ("red",), "i1"), "closed": ("q", ("yes",), "i2"),
                        "choice": ("q", ("C", "red"), "i3")})
    assert reach_by_question(cell) == {"open": 1.0}


def row(label, dataset, reach, rl, sft):
    return {"label": label, "dataset": dataset, "prompt": "short", "reach": reach,
            "greedy": {"rl": rl, "sft": sft}}


def test_verdict_needs_rl_near_reach_and_sft_past_it_on_slake():
    assert verdict([row("SLAKE", "slake", 50, 53, 66), row("GQA", "gqa", 70, 50, 55)])["holds"]
    assert not verdict([row("SLAKE", "slake", 50, 56, 66)])["holds"]
    assert not verdict([row("SLAKE", "slake", 50, 53, 54)])["holds"]
