"""Experiment cards: one change against a frozen recipe, with the prediction committed first.

A card in experiments.toml ([[cards]]) holds the question, the change, the baseline, what was
predicted and what each outcome means. One id names everything: run directory runs/cards/<id>,
run name and record label card-<id>. Trainers score their kept adapters on dev rows when they
finish, so a card reads dev rows only; test rows stay for reported arms.

    python -m scripts.cards commands <id>   # the training line for the GPU
    python -m scripts.cards summarize       # results/ledger.json, from saved dev records
"""

import argparse
import json
import shlex
import tomllib
from pathlib import Path

from scripts.knobs import train_hours, verdict
from vqar.eval.paired import paired_bootstrap, question_scores
from vqar.eval.record import RunRecord, record_files

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "results" / "ledger.json"
SCORER = "official"
PROMPT = "reasoning"
TEXT_FIELDS = ("id", "node", "question", "recipe", "baseline", "change", "predicted", "holds_if",
               "if_holds", "if_not", "status", "why", "implies")


def load_cards(toml: Path = ROOT / "experiments.toml") -> list[dict]:
    return tomllib.loads(toml.read_text()).get("cards", [])


def run_dir(card: dict, root: Path = ROOT) -> Path:
    return root / "runs" / "cards" / card["id"]


def label(card: dict) -> str:
    return f"card-{card['id']}"


def commands(card: dict) -> list[str]:
    """The training line; the trainer scores its adapters on dev rows at the end."""
    entry = card.get("entry", "train_grpo.py")
    return [shlex.join(["uv", "run", "--frozen", "python", entry, *card["train"],
                        "--run-dir", f"runs/cards/{card['id']}", "--run-name", label(card)])]


def dev_records(run: Path, dataset: str | None = None) -> dict[tuple[int, str], RunRecord]:
    """(step, prompt) -> the run's dev record, on one dataset when several were scored."""
    directory = run / "checkpoints" / "dev-records"
    found = {}
    if not directory.exists():
        return found
    for path in record_files(directory):
        record = RunRecord.load(path)
        if dataset is None or record.protocol["dataset"] == dataset:
            found[(record.model["step"], record.protocol["prompt"])] = record
    return found


def curve(records: dict[tuple[int, str], RunRecord], prompt: str = PROMPT) -> list[dict]:
    return [{"step": step, "accuracy": 100 * record.metrics[SCORER]["overall"],
             "tokens": record.metrics["mean_output_tokens"]}
            for (step, kind), record in sorted(records.items()) if kind == prompt]


def compare(reference: RunRecord, candidate: RunRecord) -> dict:
    """Candidate against reference at their final steps, paired over the same dev questions."""
    scores = [question_scores(r.predictions_path(), SCORER) for r in (reference, candidate)]
    return {"accuracy": 100 * candidate.metrics[SCORER]["overall"],
            "baseline": 100 * reference.metrics[SCORER]["overall"],
            "tokens": candidate.metrics["mean_output_tokens"],
            "baseline_tokens": reference.metrics["mean_output_tokens"],
            **paired_bootstrap(*scores)}


def held(final: dict, rule: dict) -> bool:
    """Whether the card's committed rule read true at the end of the run."""
    return (final["tokens"] >= rule["min_token_ratio"] * final["baseline_tokens"]
            and final["delta"] >= -rule["max_accuracy_drop"])


def summarize_card(card: dict, floor: float, root: Path = ROOT) -> dict:
    row = {key: card.get(key, "") for key in TEXT_FIELDS}
    run = run_dir(card, root)
    records = dev_records(run)
    baseline = dev_records(root / card["baseline_run"]) if "baseline_run" in card else {}
    finals = [step for step, prompt in records if prompt == PROMPT]
    if not finals or not baseline:
        return {**row, "final": None, "held": None, "train_hours": train_hours(run)}
    last_baseline = max(step for step, prompt in baseline if prompt == PROMPT)
    final = compare(baseline[(last_baseline, PROMPT)], records[(max(finals), PROMPT)])
    return {**row, "final": final, "final_step": max(finals), "baseline_step": last_baseline,
            "curve": curve(records), "baseline_curve": curve(baseline),
            "held": held(final, card["holds"]) if "holds" in card else None,
            "verdict": verdict(final, floor), "noise_floor": floor,
            "train_hours": train_hours(run)}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("commands", "summarize"))
    parser.add_argument("card", nargs="?", help="commands: the card's id")
    args = parser.parse_args()
    cards = load_cards()
    if args.action == "commands":
        print("\n".join(commands(next(c for c in cards if c["id"] == args.card))))
        return
    # The repeat of the ChartQA knob family: a same-seed retrain's reasoning-prompt difference.
    floor = json.loads((ROOT / "results" / "knobs-chartqa.json").read_text())["noise_floor"]
    LEDGER.write_text(json.dumps({"cards": [summarize_card(card, floor) for card in cards]},
                                 indent=1) + "\n")
    print(f"wrote {LEDGER.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
