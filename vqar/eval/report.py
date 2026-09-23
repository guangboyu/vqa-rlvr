"""The report builder: every table a reader sees, generated from RunRecords.

Tables are markdown blocks between `<!-- report:NAME -->` and `<!-- /report:NAME -->`
markers in README.md and docs/; a test regenerates them and fails on drift, so no number is
transcribed by hand. The same numbers go to a JSON export for the demo. July results appear
only in the labeled history table, never beside the corrected campaign.
"""

import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from vqar.eval.legacy import july_records
from vqar.eval.record import RunRecord
from vqar.registry import DATASETS

ROOT = Path(__file__).resolve().parent.parent.parent
MARKED_FILES = ("README.md", "docs/supplementary/results.md")
BLOCK = re.compile(r"(<!-- report:(?P<name>[\w-]+) -->\n)(?P<body>.*?)(<!-- /report:(?P=name) -->)",
                   re.DOTALL)
PROMPT_ORDER = ("short", "reasoning")


@dataclass(frozen=True)
class Arm:
    label: str
    name: str
    model: str
    revision: str | None = None
    gpu_memory_utilization: float = 0.8


def load_matrix(path: Path) -> tuple[list[Arm], list[dict]]:
    """The arms and protocols declared in the experiment matrix file."""
    matrix = tomllib.loads(Path(path).read_text())
    return [Arm(**arm) for arm in matrix["arms"]], matrix.get("protocols", [])


class Cell(NamedTuple):
    """Where a record sits in the tables; `step` is None except for checkpoint evaluations."""

    arm: str
    dataset: str
    role: str
    tier: str
    prompt: str
    image_mode: str
    step: int | None


def load_cells(records_dir: Path) -> dict[Cell, RunRecord]:
    """Every stored record, keyed by its table cell."""
    cells = {}
    for path in sorted(Path(records_dir).rglob("*.json")):
        record = RunRecord.load(path)
        p = record.protocol
        key = Cell(record.model["label"], p["dataset"], p["role"], p["tier"], p["prompt"],
                   p["image_mode"], record.model.get("step"))
        if key in cells:
            raise ValueError(f"Two records claim one table cell {key}: {cells[key].path}, {path}")
        cells[key] = record
    return cells


def ordered_datasets() -> list[str]:
    """In-domain datasets first, then OOD ones, so OOD columns sit together at the right."""
    ladder = {"in-domain": 0, "near-ood": 1, "far-ood": 2}
    return sorted(DATASETS, key=lambda name: ladder[DATASETS[name].ladder])


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
                         and cell.step is None}
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


def curves_table(cells: dict) -> str:
    """Dev-tier learning curves from checkpoint-tagged records."""
    points = sorted(
        ((record.model["run"], cell.step, cell.dataset, cell.prompt), record)
        for cell, record in cells.items()
        if cell.role == "dev" and cell.step is not None
    )
    if not points:
        return "No checkpoint dev evaluations have been recorded yet.\n"
    rows = []
    for (run, step, dataset, prompt), record in points:
        official = record.metrics["official"]
        rows.append([f"`{run}`", str(step), DATASETS[dataset].display_name, prompt,
                     percent(official["overall"] if official else None),
                     percent(record.metrics["normalized"]["overall"]),
                     f"{record.metrics['mean_output_tokens']:.0f}"])
    return table(["Run", "Step", "Dataset", "Prompt", "Official", "Normalized", "Mean tokens"],
                 rows, 4)


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
    entropy and completion length from the trainer's logs; first → last unless marked."""
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
                     mean(updates, "zero_variance_group_fraction"), trend(logged, "entropy"),
                     mean(logged, "clip_fraction"), trend(logged, "completion_length", 0),
                     f"{diagnostics['distinct_prompts']:,}"])
    return table(["Run", "Reward (first → last 10 updates)", "Dead groups (mean)", "Entropy",
                  "Clip fraction (mean)", "Completion tokens", "Distinct prompts"], rows, 1)


def build(root: Path = ROOT) -> tuple[dict[str, str], dict]:
    """Every markdown block, and the JSON export for the demo."""
    arms, _ = load_matrix(root / "experiments.toml")
    cells = load_cells(root / "results" / "records")
    history = july_records(root / "results")
    exposure = json.loads((root / "results" / "baselines" / "training-exposure.json").read_text())
    blocks = {
        "matrix": matrix_table(arms, cells),
        "behavior": behavior_table(arms, cells),
        "failures": failures_table(arms, cells),
        "casing": casing_table(arms, cells),
        "interventions": interventions_table(arms, cells),
        "curves": curves_table(cells),
        "exposure": exposure_table(exposure),
        "diagnostics": diagnostics_table(exposure),
        "history": history_table(history),
    }
    export = {
        "arms": [{"label": arm.label, "name": arm.name} for arm in arms],
        "datasets": [{"name": name, "display_name": DATASETS[name].display_name,
                      "ladder": DATASETS[name].ladder} for name in ordered_datasets()],
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
    }
    return blocks, export


def fill(text: str, blocks: dict[str, str]) -> str:
    """Replace every marked block's body with the generated table."""
    def replace(match):
        name = match.group("name")
        if name not in blocks:
            raise KeyError(f"Unknown report block: {name}")
        return match.group(1) + blocks[name] + match.group(4)

    return BLOCK.sub(replace, text)
