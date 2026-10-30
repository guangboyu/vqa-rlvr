"""How far DeepTumorVQA's answer options go without the image: a text-only baseline.

    python -m scripts.option_priors     # writes results/screening/deeptumor-option-prior.json

From the training file (benchmark patients excluded), per question subtype: the letter most
often correct, and, for questions whose options are all numbers, the rank the correct value
most often takes among the sorted options (per option count). On the benchmark, a numeric
question is answered with its subtype's most common rank, any other with its subtype's most
common letter. Also reported: chance (one over the option count) and the rank distribution of
the correct value on four-option measurement questions.
"""

import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from vqar.data import DEEPTUMOR_ROOT, DEEPTUMOR_ROOT_DEFAULT, deeptumor_question
from vqar.eval.report import ROOT

OUTPUT = ROOT / "results" / "screening" / "deeptumor-option-prior.json"


def value_rank(texts: list[str], correct: int) -> tuple[int, int] | None:
    """(option count, rank of the correct value among the sorted values), for numeric options."""
    try:
        values = [float(text) for text in texts]
    except ValueError:
        return None
    return len(values), sorted(values).index(values[correct])


def main():
    root = Path(os.environ.get(DEEPTUMOR_ROOT, DEEPTUMOR_ROOT_DEFAULT))
    test = [json.loads(line) for line in
            (root / "benchmark" / "test_qa.jsonl").read_text().splitlines()]
    test_patients = {row["image_id"] for row in test}
    letters, ranks = defaultdict(Counter), defaultdict(Counter)
    with open(root / "train" / "train_qa.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            parsed = deeptumor_question(row["multiple-choice question"])
            correct = row["correct option"]
            if row["Image ID"] in test_patients or parsed is None or len(correct) != 1:
                continue
            subtype = row["question subtype"]
            letters[subtype][correct] += 1
            if rank := value_rank(parsed[1], ord(correct) - ord("A")):
                ranks[(subtype, rank[0])][rank[1]] += 1
    by_type = defaultdict(lambda: {"questions": 0, "prior": 0, "chance": 0.0})
    measurement = Counter()
    for row in test:
        parsed = deeptumor_question(row["mc_question"])
        if parsed is None:
            continue
        texts, correct = parsed[1], ord(row["correct_option"]) - ord("A")
        subtype, rank = row["question_subtype"], value_rank(texts, correct)
        if rank and ranks[(subtype, rank[0])]:
            hit = ranks[(subtype, rank[0])].most_common(1)[0][0] == rank[1]
        else:
            hit = bool(letters[subtype]) and \
                letters[subtype].most_common(1)[0][0] == row["correct_option"]
        if rank and rank[0] == 4 and row["question_type"] == "measurement":
            measurement[rank[1] + 1] += 1
        for key in (row["question_type"], "all"):
            by_type[key]["questions"] += 1
            by_type[key]["prior"] += hit
            by_type[key]["chance"] += 1 / len(texts)
    summary = {key: {"questions": v["questions"],
                     "prior_accuracy": round(100 * v["prior"] / v["questions"], 2),
                     "chance": round(100 * v["chance"] / v["questions"], 2)}
               for key, v in by_type.items()}
    total = sum(measurement.values())
    report = {"source": "tumor-vqa/DeepTumorVQA_2.0 a8b9db5133fda02567476dcb49498ec51f6aed37",
              "benchmark": summary,
              "measurement_four_options_correct_rank_share": {
                  str(rank): round(100 * count / total, 1)
                  for rank, count in sorted(measurement.items())}}
    OUTPUT.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
