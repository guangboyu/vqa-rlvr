"""The category table: declared arms on every benchmark, grouped by category.

    python -m scripts.categories check             # cells without a record
    python -m scripts.categories commands          # one line per GPU engine, for a queue
    python -m scripts.categories run ARM PIXELS    # evaluate one arm's missing cells at one budget
    python -m scripts.categories summarize         # results/categories.json

Each arm reads under the prompt it trained with, against Instruct under the same prompt; a win or
a loss needs a paired 95% interval that excludes zero. Image gain is short-prompt accuracy minus
black-image accuracy on the same questions. Paired differences need the local predictions.
"""

import argparse
import json
import tomllib
from pathlib import Path

from vqar.config import MAX_PIXELS
from vqar.eval.matrix import Arm, declared_records
from vqar.eval.paired import paired_bootstrap, question_scores
from vqar.eval.protocol import Protocol
from vqar.eval.report import ROOT
from vqar.eval.runner import DATA_DIR, RECORDS_DIR, run_eval
from vqar.registry import DATASETS

OUTPUT = ROOT / "results" / "categories.json"
INSTRUCT = "base2"
MODEL_LEN = 8192  # room for several full images, with the reasoning budget


def load_config(toml: Path = ROOT / "experiments.toml") -> dict:
    declared = tomllib.loads(toml.read_text())
    arms = {arm["label"]: Arm(**arm) for arm in declared["arms"]}
    config = declared["categories"]
    return {**config, "models": {spec["label"]: arms[spec["label"]] for spec in config["arms"]}}


def budget(config: dict, dataset: str) -> int:
    return config["max_pixels"].get(dataset, MAX_PIXELS)


def arm_protocols(config: dict, spec: dict) -> list[Protocol]:
    """Originals before black images: a black cell pairs with its original."""
    originals = [Protocol(dataset, prompt=prompt, max_pixels=budget(config, dataset))
                 for dataset in config["benchmarks"]
                 for prompt in dict.fromkeys(["short", *spec["prompts"]])]
    black = [Protocol(dataset, prompt="short", image_mode="black",
                      max_pixels=budget(config, dataset)) for dataset in config["benchmarks"]]
    return originals + black


def records_for(config: dict) -> dict:
    protocols = list(dict.fromkeys(p for spec in config["arms"]
                                   for p in arm_protocols(config, spec)))
    return declared_records(list(config["models"].values()), protocols, RECORDS_DIR, DATA_DIR,
                            ROOT)


def missing(config: dict) -> list[tuple[str, Protocol]]:
    found = records_for(config)
    return [(spec["label"], p) for spec in config["arms"] for p in arm_protocols(config, spec)
            if (spec["label"], p) not in found]


def commands(config: dict) -> list[str]:
    needed = dict.fromkeys((label, p.max_pixels) for label, p in missing(config))
    return [f"uv run --frozen python -m scripts.categories run {label} {pixels}"
            for label, pixels in needed]


def run(config: dict, label: str, pixels: int) -> None:
    from vqar.inference import VLLMGenerator

    cells = [p for name, p in missing(config) if name == label and p.max_pixels == pixels]
    arm = config["models"][label]
    model = arm.ref(ROOT).pinned()
    generator = VLLMGenerator(model.path, model.revision, MODEL_LEN, arm.gpu_memory_utilization,
                              max_pixels=pixels,
                              max_images=max(DATASETS[p.dataset].max_images for p in cells))
    for protocol in cells:
        record = run_eval(model, protocol, generator, data_dir=DATA_DIR, records_dir=RECORDS_DIR)
        print(f"{label} {protocol.slug()} -> {record.path}", flush=True)


def scorer(config: dict, dataset: str) -> str:
    """The primary column: the authors' scorer, else the labeled project scorer."""
    if dataset in config["normalized"]:
        return "normalized"
    spec = DATASETS[dataset]
    return "official" if spec.official is not None else "project"


def verdict(delta: dict) -> str:
    return "win" if delta["low"] > 0 else "loss" if delta["high"] < 0 else "tie"


def macro_score(scores: dict[str, float], categories: dict[str, str], macro: list[str]) -> dict:
    """The unweighted mean over categories, each the mean of its benchmarks."""
    by_category = {category: [score for dataset, score in scores.items()
                              if categories[dataset] == category] for category in macro}
    means = {category: sum(values) / len(values) for category, values in by_category.items()}
    return {"categories": means, "macro": sum(means.values()) / len(means)}


def benchmark_row(config: dict, found: dict, dataset: str, columns: list[tuple[str, str]]) -> dict:
    """One benchmark's scores, image gains and verdicts for every column."""
    variant = scorer(config, dataset)
    pixels = budget(config, dataset)

    def cell(label, prompt, mode="original"):
        return found[(label, Protocol(dataset, prompt=prompt, image_mode=mode, max_pixels=pixels))]

    def score(record):
        return 100 * record.metrics[variant]["overall"]

    cells = {}
    for label, prompt in columns:
        record = cell(label, prompt)
        short, black = cell(label, "short"), cell(label, "short", "black")
        entry = {"score": score(record), "n": record.metrics["n"],
                 "tokens": record.metrics["mean_output_tokens"],
                 "image_gain": score(short) - score(black)}
        if label != INSTRUCT:
            reference = cell(INSTRUCT, prompt)
            delta = paired_bootstrap(question_scores(reference.predictions_path(), variant),
                                     question_scores(record.predictions_path(), variant))
            entry |= {"vs_instruct": delta, "verdict": verdict(delta)}
        cells[f"{label}|{prompt}"] = entry
    return {"dataset": dataset, "display_name": DATASETS[dataset].display_name,
            "category": DATASETS[dataset].category, "scorer": variant, "cells": cells}


def summarize(config: dict) -> dict:
    found = records_for(config)
    # Instruct has one column per prompt; every other arm has its training prompt.
    columns = [(spec["label"], prompt) for spec in config["arms"] for prompt in spec["prompts"]]
    rows = [benchmark_row(config, found, dataset, columns) for dataset in config["benchmarks"]]
    categories = {row["dataset"]: row["category"] for row in rows}
    summary = {}
    for label, prompt in columns:
        column = f"{label}|{prompt}"
        scores = {row["dataset"]: row["cells"][column]["score"] for row in rows}
        counted = [row["cells"][column].get("verdict") for row in rows]
        summary[column] = {**macro_score(scores, categories, config["macro"]),
                           "wins": counted.count("win"), "ties": counted.count("tie"),
                           "losses": counted.count("loss")}
    arms = [{"column": f"{spec['label']}|{prompt}", "label": spec["label"],
             "method": spec["method"], "trained_on": spec["trained_on"], "prompt": prompt}
            for spec in config["arms"] for prompt in spec["prompts"]]
    return {"macro_categories": config["macro"], "arms": arms, "rows": rows, "summary": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("check", "commands", "run", "summarize"))
    parser.add_argument("arm", nargs="?")
    parser.add_argument("pixels", nargs="?", type=int)
    args = parser.parse_args()
    config = load_config()
    if args.action == "check":
        cells = missing(config)
        for label, protocol in cells:
            print(f"missing: {label} {protocol.slug()}")
        print(f"{len(cells)} cells missing")
    elif args.action == "commands":
        print("\n".join(commands(config)))
    elif args.action == "run":
        run(config, args.arm, args.pixels)
    else:
        OUTPUT.write_text(json.dumps(summarize(config), indent=1, ensure_ascii=False) + "\n")
        print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
