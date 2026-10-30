"""How far DeepTumorVQA's answer options go without the image: a text-only baseline.

    python -m scripts.option_priors              # released options
    python -m scripts.option_priors --debiased   # the deeptumorx questions

Writes results/screening/<dataset>-option-prior.json. From the training questions (benchmark
patients excluded), per question subtype: the letter most often correct, the answer most often
correct, and, for questions whose options are all numbers, the rank the correct value most
often takes among the sorted options (per option count). On the test questions, a numeric
question is answered with its subtype's most common rank, any other with the option that was
correct most often in training (its subtype's most common letter when no option was). Also
reported: chance (one over the option count), the most-common-letter rule alone, and the rank
distribution of the correct value on four-option measurement questions.
"""

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from vqar.data import (
    DEEPTUMOR_ROOT,
    DEEPTUMOR_ROOT_DEFAULT,
    deeptumor_balance,
    deeptumor_debias,
    deeptumor_dev_patient,
    deeptumor_record,
)
from vqar.eval.report import ROOT
from vqar.registry import DATASETS

OUTPUT_DIR = ROOT / "results" / "screening"


def value_rank(record: dict) -> tuple[int, int] | None:
    """(option count, rank of the correct value among the sorted values), for numeric options."""
    try:
        values = [float(text) for text in record["options"]]
    except ValueError:
        return None
    return len(values), sorted(values).index(values[record["correct"]])


def questions(root: Path, debiased: bool) -> tuple[list[dict], list[dict]]:
    """Training and test records, as the registered dataset builds them (before sampling)."""
    sizes = DATASETS["deeptumorx"].sizes
    seed = sizes["seed"]

    def build(record):
        if record is None or not debiased:
            return record
        return deeptumor_debias(record, seed)

    rows = [json.loads(line) for line in
            (root / "benchmark" / "test_qa.jsonl").read_text().splitlines()]
    test_patients = {row["image_id"] for row in rows}
    test = [build(deeptumor_record(row["qid"], row["image_id"], row["mc_question"],
                                   row["correct_option"], row["answer"], row["question_type"],
                                   "", row["question_subtype"])) for row in rows]
    train = []
    with open(root / "train" / "train_qa.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            patient = row["Image ID"]
            if patient in test_patients or (
                    debiased and deeptumor_dev_patient(patient, sizes["dev_patient_percent"],
                                                       seed)):
                continue
            train.append(build(deeptumor_record(
                row["Question ID"], patient, row["multiple-choice question"],
                row["correct option"], row["answer"], row["question type"], "",
                row["question subtype"])))
    test, train = [r for r in test if r], [r for r in train if r]
    if debiased:
        test = deeptumor_balance(test, sizes["min_test_per_answer"], seed)
        train = deeptumor_balance(train, sizes["min_train_per_answer"], seed)
    return train, test


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--debiased", action="store_true")
    args = parser.parse_args()
    root = Path(os.environ.get(DEEPTUMOR_ROOT, DEEPTUMOR_ROOT_DEFAULT))
    train, test = questions(root, args.debiased)
    letters, answers, ranks = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    for record in train:
        subtype = record["subtype"]
        letters[subtype][record["answers"][0]] += 1
        answers[subtype][record["answers"][1]] += 1
        if rank := value_rank(record):
            ranks[(subtype, rank[0])][rank[1]] += 1
    by_type = defaultdict(lambda: {"questions": 0, "prior": 0, "letter": 0, "chance": 0.0})
    measurement = Counter()
    for record in test:
        subtype, rank = record["subtype"], value_rank(record)
        if rank and ranks[(subtype, rank[0])]:
            hit = ranks[(subtype, rank[0])].most_common(1)[0][0] == rank[1]
        elif any(answers[subtype][option] for option in record["options"]):
            hit = max(record["options"], key=lambda o: answers[subtype][o]) == record["answers"][1]
        else:
            hit = bool(letters[subtype]) and \
                letters[subtype].most_common(1)[0][0] == record["answers"][0]
        letter_hit = bool(letters[subtype]) and \
            letters[subtype].most_common(1)[0][0] == record["answers"][0]
        if rank and rank[0] == 4 and record["answer_type"] == "measurement":
            measurement[rank[1] + 1] += 1
        for key in (record["answer_type"], "all"):
            by_type[key]["questions"] += 1
            by_type[key]["prior"] += hit
            by_type[key]["letter"] += letter_hit
            by_type[key]["chance"] += 1 / len(record["options"])
    summary = {key: {"questions": v["questions"],
                     "prior_accuracy": round(100 * v["prior"] / v["questions"], 2),
                     "letter_prior_accuracy": round(100 * v["letter"] / v["questions"], 2),
                     "chance": round(100 * v["chance"] / v["questions"], 2)}
               for key, v in by_type.items()}
    total = sum(measurement.values())
    dataset = "deeptumorx" if args.debiased else "deeptumor"
    report = {"dataset": dataset,
              "source": "tumor-vqa/DeepTumorVQA_2.0 a8b9db5133fda02567476dcb49498ec51f6aed37",
              "training_questions": len(train), "benchmark": summary,
              "measurement_four_options_correct_rank_share": {
                  str(rank): round(100 * count / total, 1)
                  for rank, count in sorted(measurement.items())}}
    out = OUTPUT_DIR / f"{dataset}-option-prior.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
