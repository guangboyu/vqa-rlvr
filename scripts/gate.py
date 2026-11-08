"""The 8B gate: is a mixed 2B recipe close enough to the specialists to be worth scaling?

Declared before the mixture cards ran. On dev rows, each arm under its training prompt:
1. the mixed arm's macro beats Instruct's, with the paired 95% interval above zero;
2. its macro is at most 1 point below the specialists' macro, and no category is more than
   2 points below its specialist.
The macro is the mean over the trained categories; a category is the mean of its datasets.
The bootstrap resamples questions within each dataset.

    uv run python -m scripts.gate   # writes results/gate.json
"""

import json
from pathlib import Path

import numpy as np

from vqar.eval.paired import RESAMPLES, SEED, question_scores
from vqar.eval.record import RunRecord, record_files

ROOT = Path(__file__).resolve().parent.parent
SCORER = "normalized"  # the only scorer every dev role has (VQAv2 dev lacks official scores)
CATEGORIES = {"photos": ("vqav2", "gqa"), "charts": ("chartqa",)}
MACRO_MARGIN = 1.0
CATEGORY_MARGIN = 2.0

CARDS = "runs/cards/{}/checkpoints/dev-records"
INSTRUCT = {"photos": ("results/records", "base2", None),
            "charts": ("runs/cards/trace-source-probe/records", "base2", None)}
SPECIALISTS = {"photos": ("results/records", "p1f-short2-base", 1000, "short"),
               "charts": (CARDS.format("chart-length-control"), "card-chart-length-control",
                          2844, "reasoning")}
# Each candidate is scored under its training prompts; a row with "gate": False is a side look.
ROUTED = {"photos": "short", "charts": "reasoning"}
SHARED = {"photos": "reasoning", "charts": "reasoning"}
CANDIDATES = [
    {"card": "prompt-routing", "step": 2844, "prompts": ROUTED},
    {"card": "mix-ratio-90", "step": 2844, "prompts": SHARED},
    {"card": "mix-ratio-90", "step": 2844, "prompts": ROUTED, "gate": False},
]


def dev_scores(directory: Path, label: str, dataset: str, prompt: str,
               step: int | None) -> dict[str, float]:
    """Per-question scores of one greedy dev record on original images."""
    for path in record_files(directory):
        record = RunRecord.load(path)
        protocol = record.protocol
        if (record.model.get("label") == label and record.model.get("step") == step
                and protocol["dataset"] == dataset and protocol["prompt"] == prompt
                and protocol["role"] == "dev" and protocol["tier"] == "dev"
                and protocol["samples"] == 1 and protocol["image_mode"] == "original"):
            return question_scores(record.predictions_path(), SCORER)
    raise FileNotFoundError(f"no dev record for {label} step {step}: {dataset} {prompt}")


def macro(cells: dict[str, dict[str, float]]) -> dict[str, float]:
    """Category means and their macro, in points."""
    means = {name: 100 * float(np.mean([np.mean(list(cells[d].values())) for d in datasets]))
             for name, datasets in CATEGORIES.items()}
    return {**means, "macro": float(np.mean(list(means.values())))}


def macro_difference(candidate: dict, reference: dict, resamples: int = RESAMPLES,
                     seed: int = SEED) -> dict:
    """candidate − reference on the macro, with a 95% interval from per-dataset resampling."""
    weights, differences = {}, {}
    for datasets in CATEGORIES.values():
        for dataset in datasets:
            if candidate[dataset].keys() != reference[dataset].keys():
                raise ValueError(f"{dataset}: the two arms scored different questions")
            qids = sorted(reference[dataset])
            differences[dataset] = np.array([candidate[dataset][q] - reference[dataset][q]
                                             for q in qids])
            weights[dataset] = 1 / (len(CATEGORIES) * len(datasets))
    rng = np.random.default_rng(seed)
    draws = np.zeros(resamples)
    for dataset, values in differences.items():
        picks = rng.integers(0, len(values), size=(resamples, len(values)))
        draws += weights[dataset] * values[picks].mean(axis=1)
    point = sum(weights[d] * values.mean() for d, values in differences.items())
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"delta": 100 * float(point), "low": 100 * float(low), "high": 100 * float(high)}


def verdict(vs_instruct: dict, vs_specialists: dict, gaps: dict[str, float]) -> dict:
    rules = {"beats_instruct": vs_instruct["low"] > 0,
             "macro_within_margin": vs_specialists["delta"] >= -MACRO_MARGIN,
             "categories_within_margin": all(gap >= -CATEGORY_MARGIN for gap in gaps.values())}
    return {**rules, "passes": all(rules.values())}


def load_cells(directories: dict, label_for: dict, prompts: dict, steps: dict) -> dict:
    return {dataset: dev_scores(ROOT / directories[name], label_for[name], dataset,
                                prompts[name], steps[name])
            for name, datasets in CATEGORIES.items() for dataset in datasets}


def evaluate_candidate(spec: dict, specialists: dict) -> dict:
    prompts, names = spec["prompts"], list(CATEGORIES)
    cells = load_cells({n: CARDS.format(spec["card"]) for n in names},
                       {n: f"card-{spec['card']}" for n in names}, prompts,
                       {n: spec["step"] for n in names})
    instruct = load_cells({n: INSTRUCT[n][0] for n in names}, {n: INSTRUCT[n][1] for n in names},
                          prompts, {n: INSTRUCT[n][2] for n in names})
    scores, specialist_scores = macro(cells), macro(specialists)
    gaps = {n: scores[n] - specialist_scores[n] for n in names}
    vs_instruct = macro_difference(cells, instruct)
    vs_specialists = macro_difference(cells, specialists)
    row = {"card": spec["card"], "step": spec["step"], "prompts": prompts,
           "gate_row": spec.get("gate", True), "scores": scores, "instruct": macro(instruct),
           "vs_instruct": vs_instruct, "vs_specialists": vs_specialists, "category_gaps": gaps}
    return {**row, **verdict(vs_instruct, vs_specialists, gaps)}


def main():
    names = list(CATEGORIES)
    specialists = load_cells({n: SPECIALISTS[n][0] for n in names},
                             {n: SPECIALISTS[n][1] for n in names},
                             {n: SPECIALISTS[n][3] for n in names},
                             {n: SPECIALISTS[n][2] for n in names})
    rows = [evaluate_candidate(spec, specialists) for spec in CANDIDATES]
    output = {"scorer": SCORER, "role": "dev", "macro_margin": MACRO_MARGIN,
              "category_margin": CATEGORY_MARGIN,
              "specialists": {"arms": {n: SPECIALISTS[n][1] for n in names},
                              "scores": macro(specialists)},
              "questions": {d: len(specialists[d]) for d in specialists}, "candidates": rows}
    path = ROOT / "results" / "gate.json"
    path.write_text(json.dumps(output, indent=1) + "\n")
    for row in rows:
        side = "" if row["gate_row"] else " (side look)"
        print(f"{row['card']} {row['prompts']['photos']} photos{side}: macro "
              f"{row['scores']['macro']:.1f}, vs Instruct {row['vs_instruct']['delta']:+.1f} "
              f"[{row['vs_instruct']['low']:.1f}, {row['vs_instruct']['high']:.1f}], "
              f"vs specialists {row['vs_specialists']['delta']:+.1f} "
              f"[{row['vs_specialists']['low']:.1f}, {row['vs_specialists']['high']:.1f}], gaps "
              + ", ".join(f"{n} {g:+.1f}" for n, g in row["category_gaps"].items())
              + f" -> {'passes' if row['passes'] else 'fails'}")
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
