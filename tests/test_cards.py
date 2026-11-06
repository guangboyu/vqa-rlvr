"""Experiment cards: declarations, the committed rule, and the ledger table."""

from scripts.cards import commands, held, label, load_cards, summarize_card
from vqar.config import GRPO_PRESETS, dump, for_hardware
from vqar.eval.report import cards_table

REQUIRED = {"id", "node", "question", "recipe", "baseline", "change", "train", "predicted",
            "holds_if", "if_holds", "if_not", "status"}
STATUSES = {"experimental", "adopted", "retired", "null"}


def test_every_card_is_declared_in_full():
    cards = load_cards()
    assert len({card["id"] for card in cards}) == len(cards)
    for card in cards:
        assert REQUIRED <= card.keys(), card["id"]
        assert card["status"] in STATUSES


def test_one_id_names_the_run_and_the_label():
    card = {"id": "x", "train": ["--preset", "grpo_2b_v1"]}
    assert label(card) == "card-x"
    assert commands(card) == ["uv run --frozen python train_grpo.py --preset grpo_2b_v1 "
                              "--run-dir runs/cards/x --run-name card-x"]
    assert "train_sft.py" in commands({**card, "entry": "train_sft.py"})[0]


def test_the_rule_reads_length_and_accuracy():
    rule = {"min_token_ratio": 0.5, "max_accuracy_drop": 2.0}
    base = {"baseline_tokens": 145}
    assert held({**base, "tokens": 100, "delta": -1.9}, rule)
    assert not held({**base, "tokens": 70, "delta": 0.0}, rule)
    assert not held({**base, "tokens": 140, "delta": -2.1}, rule)


def test_a_card_without_records_reads_as_running(tmp_path):
    row = summarize_card({"id": "x", "change": "c", "predicted": "p"}, 0.5, tmp_path)
    assert row["final"] is None and row["held"] is None
    assert "running" in cards_table({"cards": [row]})


def test_the_table_reports_a_finished_card():
    row = {"id": "x", "change": "c", "predicted": "Holds. Because.", "held": False,
           "train_hours": 9.54, "final": {"accuracy": 65.0, "baseline": 75.8, "tokens": 10,
                                          "baseline_tokens": 145}}
    line = cards_table({"cards": [row]}).splitlines()[2]
    assert line == "| x | c | Holds | 75.8 → 65.0, 145 → 10 tokens | did not hold | 9.5 |"


def test_recipe_v1_is_the_saved_final_rerun_recipe():
    import json
    from pathlib import Path

    saved = json.loads((Path(__file__).resolve().parent.parent / "results" / "recipes"
                        / "v1.0.json").read_text())["preset"]
    # Name and memory placement are not the recipe; the saved run also fixed its seed.
    ignored = {"name", "vllm_gpu_memory_utilization", "vllm_sleep", "seed"}
    current = json.loads(json.dumps(dump(for_hardware(GRPO_PRESETS["grpo_2b_v1"], "80gb"))))
    assert ({k: v for k, v in current.items() if k not in ignored}
            == {k: v for k, v in saved.items() if k not in ignored})
