"""The report builder: every table a reader sees, generated from RunRecords.

Tables are markdown blocks between `<!-- report:NAME -->` and `<!-- /report:NAME -->`
markers in README.md and docs/; a test regenerates them and fails on drift, so no number is
transcribed by hand. The same numbers go to a JSON export for the demo. The named matrices
in experiments.toml choose each table's arms: the headline tables show the final rerun, and
the corrected campaign and the July runs appear only in labeled history tables.
"""

import json
import re
from pathlib import Path
from typing import NamedTuple

from vqar.config import MAX_PIXELS
from vqar.eval.errors import ANSWER_FORM_CHECKS, ERROR_KINDS
from vqar.eval.legacy import july_records
from vqar.eval.matrix import Arm, load_comparisons, load_matrix
from vqar.eval.record import RunRecord, record_files
from vqar.registry import DATASETS, SCORER_VARIANTS, TRAINING_ROLES
from vqar.splits import read_split

ROOT = Path(__file__).resolve().parent.parent.parent
MARKED_FILES = ("README.md", "docs/supplementary/results.md", "docs/supplementary/history.md")
BLOCK = re.compile(r"(<!-- report:(?P<name>[\w-]+) -->\n)(?P<body>.*?)(<!-- /report:(?P=name) -->)",
                   re.DOTALL)
PROMPT_ORDER = ("short", "reasoning")


class Cell(NamedTuple):
    """Where a record sits in the tables; `step` is None except for checkpoint evaluations.

    Greedy decoding is the default; sampled records (temperature, samples) get cells of
    their own.
    """

    arm: str
    dataset: str
    role: str
    tier: str
    prompt: str
    image_mode: str
    step: int | None
    temperature: float = 0
    samples: int = 1
    max_pixels: int = MAX_PIXELS


def load_cells(records_dir: Path, data_dir: Path | None = None) -> dict[Cell, RunRecord]:
    """Every stored record, keyed by its table cell.

    With `data_dir`, a record whose data no longer matches the split files there is
    superseded (for example, a test role that has since grown) and stays out of the tables;
    it remains on disk as history. Two current records for one cell are an error.
    """
    cells = {}
    for path in record_files(records_dir):
        record = RunRecord.load(path)
        if data_dir is not None and not is_current(record, data_dir):
            continue
        p = record.protocol
        key = Cell(record.model["label"], p["dataset"], p["role"], p["tier"], p["prompt"],
                   p["image_mode"], record.model.get("step"), p["temperature"], p["samples"],
                   p["max_pixels"])
        if key in cells:
            raise ValueError(f"Two records claim one table cell {key}: {cells[key].path}, {path}")
        cells[key] = record
    return cells


def is_current(record: RunRecord, data_dir: Path) -> bool:
    """Whether every subset the record read still has the fingerprint its split file records."""
    recorded = read_split(data_dir, record.protocol["dataset"])["subsets"]
    return all(name in recorded and recorded[name]["fingerprint"] == fingerprint
               for name, fingerprint in record.protocol["data"]["subsets"].items())


def ordered_datasets() -> list[str]:
    """Headline datasets, in-domain first, then OOD, so OOD columns sit together at the right.

    Optional OOD cohorts (require_split=False) have their own comparison script.
    """
    ladder = {"in-domain": 0, "near-ood": 1, "far-ood": 2}
    headline = [name for name, spec in DATASETS.items() if spec.require_split]
    return sorted(headline, key=lambda name: ladder[DATASETS[name].ladder])


def column(dataset: str) -> str:
    spec = DATASETS[dataset]
    if spec.ladder == "in-domain":
        return spec.display_name
    return f"{spec.display_name} ({spec.ladder.replace('ood', 'OOD')})"


def percent(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{100 * value:.{digits}f}"


def table(header: list[str], rows: list[list[str]], numeric_from: int) -> str:
    align = ["---" if i < numeric_from else "---:" for i in range(len(header))]
    lines = [header, align, *rows]
    return "\n".join("| " + " | ".join(line) + " |" for line in lines) + "\n"


def full_cell(cells, arm: Arm, dataset: str, prompt: str, image_mode: str = "original"):
    return cells.get(Cell(arm.label, dataset, "test", "full", prompt, image_mode, None))


def matrix_table(arms: list[Arm], cells: dict) -> str:
    """Headline: official accuracy with the normalized scorer beside it, per prompt."""
    datasets = ordered_datasets()
    rows = []
    for arm in arms:
        for prompt in PROMPT_ORDER:
            row = [arm.name, prompt]
            for dataset in datasets:
                record = full_cell(cells, arm, dataset, prompt)
                row.append("—" if record is None else
                           f"{percent(record.metrics['official']['overall'])} "
                           f"({percent(record.metrics['normalized']['overall'])})")
            rows.append(row)
    return table(["Arm", "Prompt", *map(column, datasets)], rows, 2)


def failures_table(arms: list[Arm], cells: dict, prompt: str = "reasoning") -> str:
    """Responses cut off at the token budget · responses with an empty answer."""
    datasets = ordered_datasets()
    rows = []
    for arm in arms:
        row = [arm.name]
        for dataset in datasets:
            record = full_cell(cells, arm, dataset, prompt)
            m = record.metrics if record else None
            row.append("—" if m is None else f"{percent(m['truncated_fraction'], 1)}% · "
                                             f"{percent(m['empty_answer_fraction'], 1)}%")
        rows.append(row)
    return table(["Arm", *map(column, datasets)], rows, 1)


def behavior_table(arms: list[Arm], cells: dict, prompt: str = "reasoning") -> str:
    """Retention: mean output tokens and the answer-tag rate under the reasoning prompt."""
    datasets = ordered_datasets()
    rows = []
    for arm in arms:
        row = [arm.name]
        for dataset in datasets:
            record = full_cell(cells, arm, dataset, prompt)
            m = record.metrics if record else None
            row.append("—" if m is None else
                       f"{m['mean_output_tokens']:.0f} · {percent(m['answer_tag_fraction'], 0)}%")
        rows.append(row)
    return table(["Arm", *map(column, datasets)], rows, 1)


def casing_table(arms: list[Arm], cells: dict, prompt: str = "reasoning") -> str:
    """How much of the official-normalized gap is capitalization, under one prompt."""
    rows = []
    for arm in arms:
        for dataset in ordered_datasets():
            record = full_cell(cells, arm, dataset, prompt)
            if record is None or record.metrics["official"] is None:
                continue
            m = record.metrics
            rows.append([arm.name, DATASETS[dataset].display_name,
                         percent(m["official"]["overall"]),
                         percent(m["official_on_lowercased_answers"]),
                         percent(m["normalized"]["overall"]),
                         f"{m['answers_scored_differently']:,}"])
    return table(["Arm", "Dataset", "Official", "Official, lowercased answers", "Normalized",
                  "Answers scored differently"], rows, 2)


def interventions_table(arms: list[Arm], cells: dict) -> str:
    """Image interventions: official score with original → changed images (answers changed)."""
    datasets = ordered_datasets()
    rows = []
    for arm in arms:
        for prompt in PROMPT_ORDER:
            for mode in ("black", "swapped"):
                found = {cell.dataset: record for cell, record in cells.items()
                         if (cell.arm, cell.prompt, cell.image_mode) == (arm.label, prompt, mode)
                         and cell.role == "test" and cell.step is None and cell.samples == 1}
                if not found:
                    continue
                row = [arm.name, prompt, mode]
                for dataset in datasets:
                    record = found.get(dataset)
                    if record is None:
                        row.append("—")
                        continue
                    paired = record.metrics["paired_with_original"]
                    row.append(f"{percent(paired['original']['official'], 1)} → "
                               f"{percent(record.metrics['official']['overall'], 1)} "
                               f"({percent(paired['normalized_answer_changed_fraction'], 0)}%)")
                rows.append(row)
    return table(["Arm", "Prompt", "Image", *map(column, datasets)], rows, 3)


def history_table(records: list[RunRecord]) -> str:
    """July runs, labeled history: official (normalized) where the official rescoring exists."""
    datasets = ordered_datasets()
    runs = {}
    for record in records:
        runs.setdefault(record.model["label"], {})[record.protocol["dataset"]] = record
    rows = []
    for name, by_dataset in sorted(runs.items()):
        prompt = next(iter(by_dataset.values())).protocol["prompt"]
        row = [f"`{name}`", prompt]
        for dataset in datasets:
            record = by_dataset.get(dataset)
            if record is None:
                row.append("—")
                continue
            official = record.metrics["official"]
            row.append(f"{percent(official['overall'] if official else None)} "
                       f"({percent(record.metrics['normalized']['overall'])})")
        rows.append(row)
    return table(["July run", "Prompt", *map(column, datasets)], rows, 2)


def curves_table(arms: list[Arm], cells: dict) -> str:
    """Dev-tier learning curves: each arm's training run at every checkpoint it kept, as
    official (normalized) accuracy · mean output tokens, one column per step."""
    runs = {arm.run for arm in arms if arm.run}
    curves = {}
    for cell, record in cells.items():
        run = record.model.get("run")
        if cell.role == "dev" and cell.step is not None and cell.samples == 1 and run in runs:
            curves.setdefault((run, cell.dataset, cell.prompt), {})[cell.step] = record
    if not curves:
        return "No checkpoint dev evaluations have been recorded yet.\n"

    def point(record):
        if record is None:
            return "—"
        official = record.metrics["official"]
        return (f"{percent(official['overall'] if official else None, 1)} "
                f"({percent(record.metrics['normalized']['overall'], 1)}) · "
                f"{record.metrics['mean_output_tokens']:.0f}")

    steps = sorted({step for curve in curves.values() for step in curve})
    rows = [[arm.name, DATASETS[dataset].display_name, prompt,
             *(point(curve.get(step)) for step in steps)]
            for arm in arms for dataset in ordered_datasets() for prompt in PROMPT_ORDER
            if (curve := curves.get((arm.run, dataset, prompt)))]
    return table(["Arm", "Dataset", "Prompt", *map(str, steps)], rows, 3)


def exposure_table(exposure: dict) -> str:
    """What each GRPO run actually trained on: prompt groups and reward variation."""
    rows = []
    for run, summary in exposure["runs"].items():
        overall = summary["overall"]
        rows.append([f"`{Path(run).name}`", f"{summary['completed_updates']:,}",
                     f"{overall['unique_questions']:,}", f"{overall['responses']:,}",
                     percent(overall["variable_reward_fraction"]),
                     f"{summary['updates_without_reward_variation']:,}",
                     f"{overall['mean_reward']:.3f}"])
    return table(["Run", "Updates", "Questions", "Responses", "Groups with varied rewards (%)",
                  "Updates without reward variation", "Mean reward"], rows, 1)


def diagnostics_table(exposure: dict) -> str:
    """Training signals per RL run: reward and dead-group share from every update's draws,
    with the all-right and all-wrong parts of that share (any rest agreed on a partial
    score), entropy and completion length from the trainer's logs; first → last unless
    marked."""
    def trend(rows, name, digits=2):
        values = [row[name] for row in rows or [] if name in row]
        return "—" if not values else f"{values[0]:.{digits}f} → {values[-1]:.{digits}f}"

    def mean(rows, name):
        values = [row[name] for row in rows or [] if name in row]
        return "—" if not values else f"{sum(values) / len(values):.3f}"

    def window(rows, name, size=10):
        """Mean of the first and of the last `size` updates: one update is only 4 prompts."""
        values = [row[name] for row in rows or []]
        if not values:
            return "—"
        first, last = values[:size], values[-size:]
        return f"{sum(first) / len(first):.2f} → {sum(last) / len(last):.2f}"

    rows = []
    for run, summary in exposure["runs"].items():
        diagnostics = summary["diagnostics"]
        updates, logged = diagnostics["reward_per_update"], diagnostics["logged"]
        rows.append([f"`{Path(run).name}`", window(updates, "reward_mean"),
                     mean(updates, "zero_variance_group_fraction"),
                     window(updates, "all_right_group_fraction"),
                     window(updates, "all_wrong_group_fraction"), trend(logged, "entropy"),
                     mean(logged, "clip_fraction"), trend(logged, "completion_length", 0),
                     f"{diagnostics['distinct_prompts']:,}"])
    return table(["Run", "Reward (first → last 10 updates)", "Dead groups (mean)",
                  "All right (first → last 10)", "All wrong (first → last 10)", "Entropy",
                  "Clip fraction (mean)", "Completion tokens", "Distinct prompts"], rows, 1)


def learnability_table(cells: dict) -> str:
    """Learnability pre-checks: sampled responses to RL pool prompts under the reward rule."""
    rows = []
    for cell, record in sorted(cells.items(), key=lambda item: str(item[0])):
        if cell.role not in TRAINING_ROLES or cell.samples < 2:  # greedy: a screening cell
            continue
        learnability = record.metrics["learnability"]
        histogram = learnability["correct_samples_histogram"]
        k = record.metrics["samples_per_question"]
        rows.append([f"`{cell.arm}`", DATASETS[cell.dataset].display_name, cell.prompt,
                     f"{record.metrics['n']:,}", str(k),
                     percent(learnability["fraction_between_0_and_1"], 1),
                     percent(histogram[f"0/{k}"] / record.metrics["n"], 1),
                     percent(histogram[f"{k}/{k}"] / record.metrics["n"], 1)])
    if not rows:
        return "No learnability pre-check has been recorded yet.\n"
    return table(["Model", "Dataset", "Prompt", "Prompts", "Samples each", "Learnable (%)",
                  "Never correct (%)", "Always correct (%)"], rows, 3)


def paired_deltas(results: Path) -> list[dict]:
    path = results / "paired-deltas.json"
    return json.loads(path.read_text()) if path.exists() else []


def declared(rows: list[dict], comparisons: list[tuple[str, str]],
             variants: tuple[str, ...] = SCORER_VARIANTS) -> list[dict]:
    """The arm-comparison rows of one matrix's declared pairs, under the given scorers."""
    pairs = set(comparisons)
    return [row for row in rows if "contrast" not in row and row["variant"] in variants
            and (row["reference"], row["candidate"]) in pairs]


def image_change(row: dict) -> str:
    """Megapixels before → after; the RL canvas is a padded square of the named side."""
    before = f"{row['reference_max_pixels'] / 1e6:.2f} → "
    if row["contrast"] == "canvas":
        size = row["canvas_size"]
        return before + f"{size * size / 1e6:.2f} ({size}×{size} RL canvas)"
    return before + f"{row['candidate_max_pixels'] / 1e6:.2f}"


def resolution_table(arms: list[Arm], rows: list[dict]) -> str:
    """The same arm under another image budget or on the RL canvas: the change, with its
    paired interval."""
    names = {arm.label: arm.name for arm in arms}
    lines = [[names.get(r["candidate"], r["candidate"]), DATASETS[r["dataset"]].display_name,
              r["prompt"], r["variant"], image_change(r),
              f"{r['delta']:+.2f} [{r['low']:+.2f}, {r['high']:+.2f}]"]
             for r in rows if r.get("contrast") in ("max_pixels", "canvas")]
    if not lines:
        return "No other image budget or RL canvas has been evaluated yet.\n"
    return table(["Arm", "Dataset", "Prompt", "Scorer", "Megapixels", "Change"], lines, 4)


def by_comparison(arms: list[Arm], rows: list[dict], fields: tuple[str, ...]) -> list:
    """Rows of arm comparisons grouped by (candidate, reference, prompt, *fields), each
    group mapping dataset → row, in the declared arm order and then the prompt order."""
    grouped = {}
    for row in rows:
        key = (row["candidate"], row["reference"], row["prompt"], *(row[f] for f in fields))
        grouped.setdefault(key, {})[row["dataset"]] = row
    order = {arm.label: i for i, arm in enumerate(arms)}
    return sorted(grouped.items(), key=lambda item: (order[item[0][0]], order[item[0][1]],
                                                     PROMPT_ORDER.index(item[0][2]),
                                                     *item[0][3:]))


def deltas_table(arms: list[Arm], rows: list[dict]) -> str:
    """Paired differences, candidate − reference, with 95% bootstrap intervals (points)."""
    rows = [row for row in rows if "contrast" not in row]
    if not rows:
        return "No paired comparison has been computed yet.\n"
    names = {arm.label: arm.name for arm in arms}
    datasets = ordered_datasets()
    lines = []
    for (candidate, reference, prompt, variant), found in by_comparison(arms, rows,
                                                                        ("variant",)):
        cells = [f"{r['delta']:+.2f} [{r['low']:+.2f}, {r['high']:+.2f}]" if (r := found.get(d))
                 else "—" for d in datasets]
        lines.append([f"{names[candidate]} − {names[reference]}", prompt, variant, *cells])
    return table(["Comparison", "Prompt", "Scorer", *map(column, datasets)], lines, 3)


def error_analysis(results: Path) -> dict:
    path = results / "error-analysis.json"
    return json.loads(path.read_text()) if path.exists() else {"flips": [], "cells": []}


def flips_table(arms: list[Arm], rows: list[dict], variant: str = "normalized") -> str:
    """Questions the candidate fixed / broke against its reference, paired by question."""
    rows = [row for row in rows if row["variant"] == variant]
    if not rows:
        return "No error analysis has been computed yet.\n"
    names = {arm.label: arm.name for arm in arms}
    datasets = ordered_datasets()
    lines = []
    for (candidate, reference, prompt), found in by_comparison(arms, rows, ()):
        cells = [f"{r['wrong_to_right']:,} / {r['right_to_wrong']:,}" if (r := found.get(d))
                 else "—" for d in datasets]
        lines.append([f"{names[candidate]} − {names[reference]}", prompt, *cells])
    return table(["Comparison", "Prompt", *map(column, datasets)], lines, 2)


def analyzed_cells(arms: list[Arm], rows: list[dict], prompt: str) -> list[tuple[Arm, dict]]:
    """Error-analysis cells under one prompt, in arm order and then dataset order."""
    found = {(row["arm"], row["dataset"]): row for row in rows if row["prompt"] == prompt}
    return [(arm, found[(arm.label, dataset)]) for arm in arms for dataset in ordered_datasets()
            if (arm.label, dataset) in found]


def errors_table(arms: list[Arm], rows: list[dict], prompt: str = "reasoning") -> str:
    """Share of all questions right, and wrong for each reason (vqar.eval.errors)."""
    lines = [[arm.name, DATASETS[row["dataset"]].display_name,
              *(percent(row["errors"][kind], 1) for kind in ("right", *ERROR_KINDS))]
             for arm, row in analyzed_cells(arms, rows, prompt)]
    if not lines:
        return "No error analysis has been computed yet.\n"
    return table(["Arm", "Dataset", "Right", "Scorer only", "Truncated", "No answer tag",
                  "Partial match", "Wrong content"], lines, 2)


def answer_form_table(arms: list[Arm], rows: list[dict], prompt: str = "reasoning") -> str:
    """Hedged answers · several answer tags · answers over six words, % of responses."""
    datasets = ordered_datasets()
    found = {}
    for arm, row in analyzed_cells(arms, rows, prompt):
        form = row["answer_form"]
        found.setdefault(arm, {})[row["dataset"]] = " · ".join(
            percent(form[check], 1) for check in ANSWER_FORM_CHECKS)
    if not found:
        return "No error analysis has been computed yet.\n"
    lines = [[arm.name, *(by_dataset.get(d, "—") for d in datasets)]
             for arm, by_dataset in found.items()]
    return table(["Arm", *map(column, datasets)], lines, 1)


def build(root: Path = ROOT) -> tuple[dict[str, str], dict]:
    """Every markdown block, and the JSON export for the demo."""
    toml = root / "experiments.toml"
    arms, _ = load_matrix(toml, "headline")
    recipe, _ = load_matrix(toml, "recipe")
    corrected, _ = load_matrix(toml, "corrected")
    every_arm, _ = load_matrix(toml)
    cells = load_cells(root / "results" / "records", root / "data")
    history = july_records(root / "results")
    exposure = json.loads((root / "results" / "training-exposure.json").read_text())
    corrected_exposure = json.loads(
        (root / "results" / "baselines" / "training-exposure.json").read_text())
    deltas = paired_deltas(root / "results")
    errors = error_analysis(root / "results")
    comparisons = load_comparisons(toml, "headline")
    blocks = {
        "matrix": matrix_table(arms, cells),
        "behavior": behavior_table(arms, cells),
        "failures": failures_table(arms, cells),
        "casing": casing_table(arms, cells),
        # Only the corrected campaign's arms have image-intervention records.
        "interventions": interventions_table(corrected, cells),
        "curves": curves_table(arms, cells),
        "exposure": exposure_table(exposure),
        "diagnostics": diagnostics_table(exposure),
        "learnability": learnability_table(cells),
        "deltas": deltas_table(arms, declared(deltas, comparisons)),
        "resolution": resolution_table(every_arm, deltas),
        "flips": flips_table(arms, declared(errors["flips"], comparisons)),
        "errors": errors_table(arms, errors["cells"]),
        "answer-form": answer_form_table(arms, errors["cells"]),
        "recipe": deltas_table(recipe, declared(deltas, load_comparisons(toml, "recipe"),
                                                ("official",))),
        "corrected": matrix_table(corrected, cells),
        "corrected-deltas": deltas_table(corrected, declared(
            deltas, load_comparisons(toml, "corrected"))),
        "corrected-behavior": behavior_table(corrected, cells),
        "corrected-exposure": exposure_table(corrected_exposure),
        "corrected-diagnostics": diagnostics_table(corrected_exposure),
        "history": history_table(history),
    }
    export = {
        "arms": [{"label": arm.label, "name": arm.name} for arm in arms],
        "datasets": [{"name": name, "display_name": DATASETS[name].display_name,
                      "ladder": DATASETS[name].ladder} for name in ordered_datasets()],
        "max_pixels": MAX_PIXELS,  # the image budget of the headline cells
        "cells": [
            {**cell._asdict(), "legacy": record.legacy, "metrics": record.metrics}
            for cell, record in sorted(cells.items(), key=lambda item: str(item[0]))
        ],
        "history": [
            {"run": record.model["label"], "dataset": record.protocol["dataset"],
             "prompt": record.protocol["prompt"], "metrics": record.metrics}
            for record in history
        ],
        # The per-update diagnostics stay in the exposure file; the demo needs the totals.
        "exposure": {run: {key: value for key, value in summary.items() if key != "diagnostics"}
                     for run, summary in exposure["runs"].items()},
        "same_prompt_order": exposure["same_prompt_order"],
        **historical_studies(root / "results"),
    }
    return blocks, export


def historical_studies(results: Path) -> dict:
    """July studies the dashboard's history view shows beside the July runs."""
    diagnostics = json.loads((results / "diagnostics.json").read_text())
    judge = json.loads((results / "judge_study.json").read_text())
    return {"retention_ablation": diagnostics["ablation"], "judge_study": judge}


def fill(text: str, blocks: dict[str, str]) -> str:
    """Replace every marked block's body with the generated table."""
    def replace(match):
        name = match.group("name")
        if name not in blocks:
            raise KeyError(f"Unknown report block: {name}")
        return match.group(1) + blocks[name] + match.group(4)

    return BLOCK.sub(replace, text)
