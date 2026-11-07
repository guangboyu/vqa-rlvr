"""What each training candidate became: results/candidate-sources.json, from raw sources and splits.

    python -m scripts.candidate_sources

Per candidate: source rows and what its drop rule removed, rows other datasets took out by rule,
the final role sizes, the reference-form mix of the rl role, repeated questions between dev and
training, and for the document sources how many numeric references the reward pays one digit off.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from datasets import load_from_disk

from vqar.answers import reference_form
from vqar.benchmark_metrics import anls_match
from vqar.data import dropped_rows
from vqar.registry import DATASETS
from vqar.splits import question_keys, read_split

CANDIDATES = ("geoqa", "mmk12", "virl39k", "docvqat", "infovqa")
OUTPUT = Path(__file__).resolve().parent.parent / "results" / "candidate-sources.json"
_NUMERIC = re.compile(r"[$\d.,%-]*\d[$\d.,%-]*")


def one_digit_off(reference: str) -> str:
    """The reference with its last digit changed."""
    digits = [i for i, char in enumerate(reference) if char.isdigit()]
    i = digits[-1]
    return f"{reference[:i]}{(int(reference[i]) + 1) % 10}{reference[i + 1:]}"


def numeric_leniency(answers_per_row: list[list[str]]) -> dict:
    """Rows whose references are all numbers, and those that pay a number one digit off."""
    numeric = [answers for answers in answers_per_row
               if all(_NUMERIC.fullmatch(answer.strip()) for answer in answers)]
    passed = sum(anls_match(one_digit_off(answers[0]), answers) > 0 for answers in numeric)
    return {"numeric_rows": len(numeric), "pay_one_digit_off": passed}


def role_rows(split: dict, role: str) -> dict[str, list[int]]:
    return {part["subset"]: part["indices"] for part in split["roles"][role]}


def dev_question_repeats(split: dict, data_dir: Path) -> int:
    """Dev rows whose question and reference also appear in training: whole images do not stop a
    question repeated on another image."""
    (dev_subset, dev), (train_subset, train) = (next(iter(role_rows(split, role).items()))
                                                for role in ("dev", "rl"))
    dev_keys = question_keys(data_dir, dev_subset)
    train_keys = question_keys(data_dir, train_subset)
    trained = {train_keys[i] for i in train} - {None}
    return sum(dev_keys[i] in trained for i in dev)


def summarize(dataset: str, data_dir: Path) -> dict:
    split = read_split(data_dir, dataset)
    (subset, rl), = role_rows(split, "rl").items()
    answers = load_from_disk(str(data_dir / subset)).select_columns(["answers"])["answers"]
    in_pool = [answers[i] for i in rl]
    row = {**dropped_rows(dataset), "materialized_rows": len(answers),
           "overlap": split["pool_losses"][subset],
           "roles": {role: sum(len(part["indices"]) for part in split["roles"][role])
                     for role in ("sft", "rl", "dev")},
           "reference_forms": dict(Counter(reference_form(row) for row in in_pool).most_common()),
           "distinct_rl_images": len({split["subsets"][subset]["image_keys"][i] for i in rl}),
           "dev_rows_repeating_a_training_question": dev_question_repeats(split, data_dir)}
    if DATASETS[dataset].normalized is anls_match:
        row["numeric_references"] = numeric_leniency(in_pool)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = {dataset: summarize(dataset, args.data_dir) for dataset in CANDIDATES}
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
