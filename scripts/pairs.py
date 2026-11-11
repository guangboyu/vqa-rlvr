"""Counterfactual pairs: does a model follow the image when it contradicts the usual answer?

VQAv2 asks one question of two similar images whose answers differ (its complementary pairs,
Goyal et al. 2017). A model that answers from the question alone gives both images the same
answer, which a black-image test cannot catch when the usual answer happens to be right. The
1,000 pairs were sampled once (data/pairs/vqav2-complementary.json); a pair that loses a
question to the overlap check drops out whole. Each arm answers under the prompt it trained
with, and Instruct under both.

    python -m scripts.pairs commands       # the evaluation lines for the GPU
    python -m scripts.pairs run <label>    # one arm's missing cells
    python -m scripts.pairs summarize      # results/pairs.json from the records

Per arm: pair accuracy (both answers get full VQA credit), same-answer rate (one normalized
answer for both images) and mean VQA accuracy, each with its difference from Instruct under
the same prompt and a 95% interval from resampling pairs.
"""

import argparse
import json
import tomllib
from pathlib import Path

import numpy as np

from vqar.eval.matrix import Arm, declared_records
from vqar.eval.protocol import Protocol
from vqar.eval.report import ROOT
from vqar.eval.runner import DATA_DIR, RECORDS_DIR, run_eval
from vqar.metrics import normalize_answer
from vqar.registry import PAIRS_FILE

DATASET = "vqav2pairs"
INSTRUCT = "base2"
OUTPUT = ROOT / "results" / "pairs.json"
RESAMPLES, SEED = 10_000, 0
# Each arm under its training prompt; Instruct under both, as the reference for each.
CELLS = {"base2": ("short", "reasoning"), "sft2": ("short",), "p1-f-short2-base": ("short",),
         "p1-f-short2-sft": ("short",), "p1-f-grpo2-base": ("reasoning",)}


def load_arms(toml: Path = ROOT / "experiments.toml") -> dict[str, Arm]:
    declared = tomllib.loads(toml.read_text())
    return {arm["label"]: Arm(**arm) for arm in declared["arms"] if arm["label"] in CELLS}


def protocol(prompt: str) -> Protocol:
    return Protocol(DATASET, prompt=prompt, role="test", tier="full")


def records(arms: dict[str, Arm]) -> dict:
    wanted = [protocol(prompt) for prompt in dict.fromkeys(p for ps in CELLS.values() for p in ps)]
    return declared_records(list(arms.values()), wanted, RECORDS_DIR, DATA_DIR, ROOT)


def missing(arms: dict[str, Arm]) -> list[tuple[str, Protocol]]:
    found = records(arms)
    return [(label, protocol(prompt)) for label, prompts in CELLS.items() for prompt in prompts
            if (label, protocol(prompt)) not in found]


def run(arms: dict[str, Arm], label: str) -> None:
    from vqar.inference import VLLMGenerator

    arm = arms[label]
    model = arm.ref(ROOT).pinned()
    generator = VLLMGenerator(model.path, model.revision, 4096, arm.gpu_memory_utilization)
    for name, cell in missing(arms):
        if name == label:
            record = run_eval(model, cell, generator, data_dir=DATA_DIR, records_dir=RECORDS_DIR)
            print(f"{label} {cell.slug()} -> {record.path}", flush=True)


def answers(predictions: Path) -> dict[str, tuple[str, float]]:
    """Each question's normalized answer and official VQA credit (greedy, one sample)."""
    found = {}
    for line in predictions.read_text().splitlines():
        row = json.loads(line)
        found[row["qid"]] = (normalize_answer(row["prediction"] or ""), row["scores"]["official"])
    return found


def per_pair(cell: dict[str, tuple[str, float]], pairs: list[list[str]]) -> np.ndarray:
    """Rows of (both full credit, same answer, mean credit), one per pair."""
    rows = []
    for first, second in pairs:
        (answer_a, credit_a), (answer_b, credit_b) = cell[first], cell[second]
        rows.append((float(credit_a == 1 and credit_b == 1), float(answer_a == answer_b),
                     (credit_a + credit_b) / 2))
    return np.array(rows)


def bootstrap(candidate: np.ndarray, reference: np.ndarray, resamples: int = RESAMPLES,
              seed: int = SEED) -> list[dict]:
    """Candidate minus reference per measure, in points, resampling pairs."""
    differences = candidate - reference
    rng = np.random.default_rng(seed)
    means = np.stack([differences[rng.integers(0, len(differences), len(differences))].mean(0)
                      for _ in range(resamples)])
    low, high = np.percentile(means, [2.5, 97.5], axis=0)
    return [{"delta": 100 * float(d), "low": 100 * float(lo), "high": 100 * float(hi)}
            for d, lo, hi in zip(differences.mean(0), low, high, strict=True)]


MEASURES = ("pair_accuracy", "same_answer_rate", "accuracy")


def summarize(arms: dict[str, Arm]) -> dict:
    found = records(arms)
    cells = {(label, cell.prompt): answers(record.predictions_path())
             for (label, cell), record in found.items()}
    pairs = [[str(a), str(b)] for a, b in json.loads(PAIRS_FILE.read_text())["pairs"]]
    pairs = [pair for pair in pairs if all(qid in cell for cell in cells.values() for qid in pair)]
    rows = []
    for label, prompts in CELLS.items():
        for prompt in prompts:
            if (label, prompt) not in cells:
                continue
            values = per_pair(cells[(label, prompt)], pairs)
            row = {"arm": label, "name": arms[label].name, "prompt": prompt,
                   **{name: 100 * float(v) for name, v in zip(MEASURES, values.mean(0),
                                                             strict=True)}}
            if label != INSTRUCT and (INSTRUCT, prompt) in cells:
                reference = per_pair(cells[(INSTRUCT, prompt)], pairs)
                row["vs_instruct"] = dict(zip(MEASURES, bootstrap(values, reference),
                                              strict=True))
            rows.append(row)
    return {"dataset": DATASET, "pairs": len(pairs), "resamples": RESAMPLES, "seed": SEED,
            "right": "full VQA credit (at least 3 of 10 annotators gave the answer)",
            "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("commands", "run", "summarize"))
    parser.add_argument("label", nargs="?", help="run: the arm to evaluate")
    args = parser.parse_args()
    arms = load_arms()
    if args.action == "commands":
        for label in dict.fromkeys(label for label, _ in missing(arms)):
            print(f"uv run --frozen python -m scripts.pairs run {label}")
    elif args.action == "run":
        run(arms, args.label)
    else:
        OUTPUT.write_text(json.dumps(summarize(arms), indent=1) + "\n")
        print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
