"""Experiment cards: one change against a frozen recipe, with the prediction committed first.

A card in experiments.toml ([[cards]]) holds the question, the change, the baseline, what was
predicted and what each outcome means. One id names everything: run directory runs/cards/<id>,
run name and record label card-<id>. Trainers score their kept adapters on dev rows when they
finish, so a card reads dev rows only; test rows stay for reported arms. The baseline is another
run (baseline_run) or a starting model's records in results/records (baseline_label, read as
step 0); cards compare under the reasoning prompt unless they name another (prompt), and the
baseline under the same prompt unless it trained under another (baseline_prompt). A starting
model scored at several token budgets names the one to read (baseline_max_tokens).

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
from vqar.registry import DATASETS

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "results" / "ledger.json"
SCORER = "official"  # unless the dataset's metric metadata names another primary scorer
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
    """The training line (the trainer scores its adapters on dev rows at the end), or an
    evaluation card's own command."""
    if card.get("kind") == "evaluation":
        return [card["command"]]
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


def primary_scorer(dataset: str | None) -> str:
    """The dataset's reported column: official, or the project rule where no author evaluator
    exists (the math sources)."""
    if dataset is None:
        return SCORER
    return (DATASETS[dataset].metric_metadata or {}).get("primary", SCORER)


def curve(records: dict[tuple[int, str], RunRecord], prompt: str = PROMPT,
          scorer: str = SCORER) -> list[dict]:
    return [{"step": step, "accuracy": 100 * record.metrics[scorer]["overall"],
             "tokens": record.metrics["mean_output_tokens"]}
            for (step, kind), record in sorted(records.items()) if kind == prompt]


def compare(reference: RunRecord, candidate: RunRecord, scorer: str = SCORER) -> dict:
    """Candidate against reference at their final steps, paired over the same dev questions."""
    scores = [question_scores(r.predictions_path(), scorer) for r in (reference, candidate)]
    return {"accuracy": 100 * candidate.metrics[scorer]["overall"],
            "baseline": 100 * reference.metrics[scorer]["overall"],
            "tokens": candidate.metrics["mean_output_tokens"],
            "baseline_tokens": reference.metrics["mean_output_tokens"],
            **paired_bootstrap(*scores)}


def labeled_dev_records(records_dir: Path, wanted: str, dataset: str,
                        max_tokens: int | None = None) -> dict:
    """A starting model's greedy dev records on original images, by prompt, as step 0."""
    found = {}
    for path in record_files(records_dir):
        record = RunRecord.load(path)
        protocol = record.protocol
        if (record.model.get("label") == wanted and protocol["dataset"] == dataset
                and protocol["role"] == "dev" and protocol["tier"] == "dev"
                and protocol["samples"] == 1 and protocol["image_mode"] == "original"
                and (max_tokens is None or protocol["max_tokens"] == max_tokens)):
            key = (0, protocol["prompt"])
            if key in found:
                raise ValueError(f"{wanted} has several {dataset} dev records under "
                                 f"{protocol['prompt']}; name baseline_max_tokens")
            found[key] = record
    return found


def baseline_records(card: dict, root: Path = ROOT) -> dict:
    if "baseline_run" in card:
        return dev_records(root / card["baseline_run"], card.get("dataset"))
    if "baseline_label" in card:
        return labeled_dev_records(root / "results" / "records", card["baseline_label"],
                                   card["dataset"], card.get("baseline_max_tokens"))
    return {}


def held(final: dict, rule: dict) -> bool:
    """Whether the card's committed rule read true at the end of the run."""
    checks = {"min_token_ratio": lambda v: final["tokens"] >= v * final["baseline_tokens"],
              "max_accuracy_drop": lambda v: final["delta"] >= -v,
              "min_delta": lambda v: final["delta"] >= v,
              "max_delta": lambda v: final["delta"] <= v}
    return all(checks[name](value) for name, value in rule.items())


def summarize_card(card: dict, floor: float, root: Path = ROOT) -> dict:
    row = {key: card.get(key, "") for key in TEXT_FIELDS}
    run = run_dir(card, root)
    records = dev_records(run, card.get("dataset"))
    baseline = baseline_records(card, root)
    prompt = card.get("prompt", PROMPT)
    base_prompt = card.get("baseline_prompt", prompt)
    finals = [step for step, kind in records if kind == prompt]
    bases = [step for step, kind in baseline if kind == base_prompt]
    if not finals or not bases:
        return {**row, "final": None, "held": None, "train_hours": train_hours(run)}
    scorer = primary_scorer(card.get("dataset"))
    final = compare(baseline[(max(bases), base_prompt)], records[(max(finals), prompt)], scorer)
    return {**row, "final": final, "final_step": max(finals), "baseline_step": max(bases),
            "scorer": scorer, "curve": curve(records, prompt, scorer),
            "baseline_curve": curve(baseline, base_prompt, scorer),
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
