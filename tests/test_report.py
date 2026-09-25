"""The report builder: tables from records, and the drift check on committed copies."""

import json

import pytest
from fixtures import png, save_subset, write_splits

from vqar.eval import ModelRef, Protocol, run_eval
from vqar.eval.report import (
    MARKED_FILES,
    ROOT,
    Arm,
    build,
    curves_table,
    fill,
    interventions_table,
    load_cells,
    matrix_table,
)


class Echo:
    """Answers every question with a fixed reply."""

    def __init__(self, reply):
        self.reply = reply

    def generate(self, conversations, *, max_tokens, temperature, samples, seed):
        return [[{"text": self.reply, "output_tokens": 3, "finish_reason": "stop"}]
                for _ in conversations]


@pytest.fixture
def records(tmp_path):
    data = tmp_path / "data"
    save_subset(data, "vqav2_eval", "vqav2",
                [(f"q{i}", png(10 * i), "Is it red?", ["yes"] * 10, "yes/no") for i in range(2)])
    save_subset(data, "vqav2_sft", "vqav2",
                [(f"s{i}", png(100 + i), "Is it red?", ["yes"], "") for i in range(3)])
    save_subset(data, "vqav2_rl", "vqav2", [("r0", png(120), "Is it red?", ["yes"], "")])
    write_splits(data, ["vqav2"])
    weights = tmp_path / "model"
    weights.mkdir()
    (weights / "model.safetensors").write_bytes(b"w")
    model = ModelRef(str(weights), label="arm")
    directory = tmp_path / "records"
    run_eval(model, Protocol("vqav2"), Echo("Yes"), data_dir=data, records_dir=directory)
    run_eval(model, Protocol("vqav2", image_mode="black"), Echo("no"), data_dir=data,
             records_dir=directory)
    return load_cells(directory)


ARMS = [Arm("arm", "Instruct", "model"), Arm("missing", "Not evaluated", "model")]


def test_matrix_puts_the_normalized_scorer_beside_the_official_one(records):
    rows = matrix_table(ARMS, records).splitlines()
    assert rows[0] == ("| Arm | Prompt | VQAv2 | GQA | CLEVR counting | TextVQA (far-OOD) |")
    # "Yes" misses the official unanimous-case rule but matches after normalization.
    assert rows[2] == "| Instruct | short | 0.00 (100.00) | — | — | — |"
    assert rows[3] == "| Instruct | reasoning | — | — | — | — |"
    assert rows[4].startswith("| Not evaluated | short | — |")


def test_interventions_show_original_to_changed_scores(records):
    rows = interventions_table(ARMS, records).splitlines()
    assert rows[2] == "| Instruct | short | black | 0.0 → 0.0 (100%) | — | — | — |"


def test_curves_say_so_when_no_checkpoint_was_evaluated(records):
    assert curves_table(records) == "No checkpoint dev evaluations have been recorded yet.\n"


def test_fill_rejects_an_unknown_block():
    with pytest.raises(KeyError, match="Unknown report block"):
        fill("<!-- report:nope -->\nold\n<!-- /report:nope -->", {})


def test_committed_tables_and_export_match_the_records():
    blocks, export = build(ROOT)
    for name in MARKED_FILES:
        text = (ROOT / name).read_text()
        assert fill(text, blocks) == text, f"{name} drifted; run python -m scripts.report"
    committed = json.loads((ROOT / "results" / "report.json").read_text())
    assert committed == json.loads(json.dumps(export)), "run python -m scripts.report"


def test_every_generated_block_has_a_home_in_the_marked_files():
    blocks, _ = build(ROOT)
    text = "".join((ROOT / name).read_text() for name in MARKED_FILES)
    missing = [name for name in blocks if f"<!-- report:{name} -->" not in text]
    assert not missing, f"blocks without markers: {missing}"


def test_two_records_for_one_cell_are_an_error(records, tmp_path):
    first = next(iter(records.values())).path
    twin = first.parent.parent / "arm-twin" / first.name
    twin.parent.mkdir()
    twin.write_text(first.read_text())
    with pytest.raises(ValueError, match="Two records claim one table cell"):
        load_cells(first.parent.parent)


def test_a_record_of_superseded_data_leaves_the_tables_but_stays_on_disk(records, tmp_path):
    from vqar.eval.record import RunRecord

    first = next(iter(records.values()))
    stale = dict(first.protocol, data={"subsets": {"vqav2_eval": "an-older-fingerprint"},
                                       "selection_sha256": "older"})
    old = RunRecord(first.model, stale, first.metrics, first.provenance)
    old.save(first.path.parent / "vqav2-test-full-short-older.json", [])
    directory = first.path.parent.parent
    with pytest.raises(ValueError, match="Two records claim one table cell"):
        load_cells(directory)
    current = load_cells(directory, tmp_path / "data")
    assert [record.path for record in current.values()] == [r.path for r in records.values()]


def test_paired_differences_show_the_interval_per_dataset():
    from vqar.eval.matrix import Arm
    from vqar.eval.report import deltas_table

    arms = [Arm("base", "Base", "m"), Arm("tuned", "Tuned", "n")]
    row = {"reference": "base", "candidate": "tuned", "dataset": "gqa", "prompt": "short",
           "variant": "official", "n": 3, "delta": 1.5, "low": -0.25, "high": 3.0}
    lines = deltas_table(arms, [row]).splitlines()
    assert lines[2].startswith("| Tuned − Base | short | official | — | +1.50 [-0.25, +3.00] |")
    assert deltas_table(arms, []) == "No paired comparison has been computed yet.\n"


def test_a_larger_image_budget_is_reported_as_a_paired_change():
    from vqar.eval.matrix import Arm
    from vqar.eval.report import deltas_table, resolution_table

    arms = [Arm("base", "Base", "m")]
    row = {"contrast": "max_pixels", "reference": "base", "candidate": "base",
           "dataset": "textvqa", "prompt": "short", "variant": "official",
           "reference_max_pixels": 786432, "candidate_max_pixels": 16777216,
           "n": 5, "delta": 2.0, "low": 1.0, "high": 3.0}
    lines = resolution_table(arms, [row]).splitlines()
    assert lines[2] == "| Base | TextVQA | short | official | 0.79 → 16.78 | +2.00 [+1.00, +3.00] |"
    assert deltas_table(arms, [row]) == "No paired comparison has been computed yet.\n"


def test_the_rl_canvas_is_reported_beside_other_image_budgets():
    from vqar.eval.matrix import Arm
    from vqar.eval.report import deltas_table, resolution_table

    arms = [Arm("grpo", "GRPO", "m")]
    row = {"contrast": "canvas", "reference": "grpo", "candidate": "grpo", "dataset": "gqa",
           "prompt": "reasoning", "variant": "official", "reference_max_pixels": 786432,
           "canvas_size": 608, "n": 5, "delta": -1.0, "low": -2.0, "high": 0.5}
    lines = resolution_table(arms, [row]).splitlines()
    assert lines[2] == ("| GRPO | GQA | reasoning | official | 0.79 → 0.37 (608×608 RL canvas) "
                        "| -1.00 [-2.00, +0.50] |")
    assert deltas_table(arms, [row]) == "No paired comparison has been computed yet.\n"
