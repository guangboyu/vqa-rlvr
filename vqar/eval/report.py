"""The report builder: every table a reader sees, generated from RunRecords.

Tables are markdown blocks between `<!-- report:NAME -->` and `<!-- /report:NAME -->`
markers in README.md and docs/; a test regenerates them and fails on drift, so no number is
transcribed by hand. The same numbers go to a JSON export for the demo. The named matrices
in experiments.toml choose each table's arms: the headline tables show the final rerun, and
the corrected campaign and the July runs appear only in labeled history tables.
"""

import json
import re
import tomllib
from pathlib import Path
from typing import NamedTuple

from vqar.config import MAX_PIXELS
from vqar.eval.errors import ANSWER_FORM_CHECKS, ERROR_KINDS
from vqar.eval.legacy import july_records
from vqar.eval.matrix import Arm, load_comparisons, load_matrix, load_pilot
from vqar.eval.record import RunRecord, record_files
from vqar.registry import DATASETS, SCORER_VARIANTS, TRAINING_ROLES
from vqar.splits import read_split

ROOT = Path(__file__).resolve().parent.parent.parent
MARKED_FILES = ("README.md", "docs/findings/reasoning-pays.md", "docs/findings/form-not-ability.md",
                "docs/findings/sft-then-rl.md", "docs/findings/what-the-gain-is.md",
                "docs/findings/one-model.md", "docs/negatives.md", "docs/matrix.md",
                "docs/ct-audit.md", "docs/harness/training-evidence.md", "docs/history.md")
BLOCK = re.compile(r"(<!-- report:(?P<name>[\w-]+) -->\n)(?P<body>.*?)(<!-- /report:(?P=name) -->)",
                   re.DOTALL)
PROMPT_ORDER = ("short", "reasoning")
OOD_COHORT_LADDER = ("near-ood", "far-ood")


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
        spec = DATASETS[record.protocol["dataset"]]
        if not spec.require_split and spec.ladder in OOD_COHORT_LADDER:
            continue  # optional OOD cohorts are reported from results/ood.json
        if not spec.public_records:
            continue  # local-only records under the dataset's licence
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


def dataset_order(name: str) -> tuple[int, int]:
    """Sort key: headline datasets in report order, then optional ones in registry order."""
    headline = ordered_datasets()
    if name in headline:
        return headline.index(name), 0
    return len(headline), list(DATASETS).index(name)


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
    return [row for row in rows if not {"contrast", "pilot", "transfer"} & row.keys()
            and row["variant"] in variants and (row["reference"], row["candidate"]) in pairs]


def image_change(row: dict) -> str:
    """Megapixels before → after; the RL canvas is a padded square of the named side, and
    upscaling resizes every image, small ones up, to the budget."""
    before = f"{row['reference_max_pixels'] / 1e6:.2f} → "
    if row["contrast"] == "canvas":
        size = row["canvas_size"]
        return before + f"{size * size / 1e6:.2f} ({size}×{size} RL canvas)"
    if row["contrast"] == "upscaled":
        after = row["candidate_max_pixels"] / 1e6
        return f"≤ {row['reference_max_pixels'] / 1e6:.2f} → {after:.2f} (upscaled)"
    return before + f"{row['candidate_max_pixels'] / 1e6:.2f}"


def resolution_table(arms: list[Arm], rows: list[dict]) -> str:
    """The same arm under another image budget, on the RL canvas or on upscaled images: the
    change, with its paired interval."""
    names = {arm.label: arm.name for arm in arms}
    lines = [[names.get(r["candidate"], r["candidate"]), DATASETS[r["dataset"]].display_name,
              r["prompt"], r["variant"], image_change(r),
              f"{r['delta']:+.2f} [{r['low']:+.2f}, {r['high']:+.2f}]"]
             for r in rows if r.get("contrast") in ("max_pixels", "canvas", "upscaled")]
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


def load_result(results: Path, name: str) -> dict | None:
    path = results / name
    return json.loads(path.read_text()) if path.exists() else None


def interval(row: dict) -> str:
    return f"{row['delta']:+.2f} [{row['low']:+.2f}, {row['high']:+.2f}]"


def passk_table(arms: list[Arm], passk: dict | None) -> str:
    """Each arm's sampled pass@k curve on the held-out questions (official scorer, %)."""
    if passk is None:
        return "Held-out pass@k has not been computed yet.\n"
    names = {arm.label: arm.name for arm in arms}
    ks = list(passk["records"][0]["pass_at_k"])
    rows = [[names[r["model"]], r["prompt"], DATASETS[r["dataset"]].display_name,
             f"{r['mean_score']:.2f}", *(f"{r['pass_at_k'][k]:.2f}" for k in ks)]
            for r in sorted(passk["records"], key=lambda r: (
                PROMPT_ORDER.index(r["prompt"]), dataset_order(r["dataset"]),
                list(names).index(r["model"])))]
    return table(["Arm", "Prompt", "Dataset", "Mean score", *(f"pass@{k}" for k in ks)],
                 rows, 3)


def passk_deltas_table(arms: list[Arm], passk: dict | None) -> str:
    """Candidate − reference for the mean score and pass@k, with image-group intervals."""
    if passk is None:
        return "Held-out pass@k has not been computed yet.\n"
    names = {arm.label: arm.name for arm in arms}
    metrics = [row["metric"] for row in passk["comparisons"][0]["deltas"]]
    shown = list(dict.fromkeys(m for m in ("mean_score", "pass@1", "pass@4", metrics[-1])
                               if m in metrics))
    rows = []
    for comparison in passk["comparisons"]:
        found = {row["metric"]: row for row in comparison["deltas"]}
        rows.append([f"{names[comparison['candidate']]} − {names[comparison['reference']]}",
                     comparison["prompt"], DATASETS[comparison["dataset"]].display_name,
                     *(interval(found[m]) for m in shown)])
    header = [m.replace("mean_score", "Mean score") for m in shown]
    return table(["Comparison", "Prompt", "Dataset", *header], rows, 3)


def cards_table(ledger: dict | None) -> str:
    """One row per declared experiment: the change, what was predicted, and what happened."""
    if ledger is None or not ledger["cards"]:
        return "No declared experiment has finished yet.\n"
    rows = []
    for card in ledger["cards"]:
        final = card.get("final")
        saw = "not finished" if final is None else (
            f"{final['baseline']:.1f} → {final['accuracy']:.1f}, "
            f"{final['baseline_tokens']:.0f} → {final['tokens']:.0f} tokens")
        held = {True: "held", False: "did not hold", None: "—"}[card.get("held")]
        hours = f"{card['train_hours']:.1f}" if card.get("train_hours") else "—"
        rows.append([card["id"], card["change"], card["predicted"].split(".")[0], saw, held, hours])
    return table(["Experiment", "Change", "Predicted", "Dev accuracy and length", "Prediction",
                  "Train GPU-h"], rows, 5)


def reach_table(reach: dict | None) -> str:
    """Instruct's pass@8 on open-answer questions against each method's greedy end point."""
    if reach is None:
        return "Reach has not been computed yet.\n"
    rows = [[row["label"], row["prompt"], f"{row['open_questions']:,}",
             f"{row['greedy']['instruct']:.1f}", f"**{row['reach']:.1f}**",
             f"{row['greedy']['rl']:.1f}", f"{row['greedy']['sft']:.1f}",
             f"{row['greedy']['sft_rl']:.1f}" if "sft_rl" in row["greedy"] else "—"]
            for row in reach["rows"]]
    return table(["Test set", "Prompt", "Open questions", "Instruct, greedy",
                  "Instruct, pass@8", "RL from Instruct", "SFT", "SFT → RL"], rows, 2)


def knobs_table(knobs: dict | None) -> str:
    """One row per training knob: its change against the baseline on the dev role."""
    if knobs is None:
        return "No knob has been run yet.\n"

    def change(row, prompt):
        return interval(row[prompt]) if prompt in row else "—"

    rows = []
    for row in knobs["knobs"]:
        reasoning = row.get("reasoning")
        tokens = ("—" if reasoning is None
                  else f"{reasoning['baseline_tokens']:.0f} → {reasoning['tokens']:.0f}")
        hours = "—" if row["train_hours"] is None else f"{row['train_hours']:.1f}"
        rows.append([row["name"], row["change"], row["expect"], change(row, "reasoning"),
                     change(row, "short"), tokens, hours, row.get("verdict", "—")])
    return table(["Knob", "Change", "Expected", "Reasoning Δ", "Short Δ", "Reasoning tokens",
                  "Train GPU-h", "Verdict"], rows, 3)


def judged_datasets(judge: dict) -> list[str]:
    return sorted({cell["dataset"] for cell in judge["cells"]}, key=dataset_order)


def judge_table(arms: list[Arm], judge: dict | None) -> str:
    """Official, judge-adjusted and audited accuracy, and the share of misses the judge accepts."""
    if judge is None:
        return "The judge has not rescored the final records yet.\n"
    datasets = judged_datasets(judge)
    found = {(c["arm"], c["prompt"], c["dataset"]): c for c in judge["cells"]}

    def text(c):
        if c is None:
            return "—"
        audited = f" / {c['audited']:.2f}" if "audited" in c else ""
        share = "" if c["rescue_rate"] is None else f" ({100 * c['rescue_rate']:.0f}%)"
        return f"{c['official']:.2f} / {c['adjusted']:.2f}{audited}{share}"

    rows = [[arm.name, prompt, *(text(found.get((arm.label, prompt, d))) for d in datasets)]
            for arm in arms for prompt in PROMPT_ORDER]
    return table(["Arm", "Prompt", *map(column, datasets)], rows, 2)


def judge_deltas_table(arms: list[Arm], judge: dict | None) -> str:
    """Each declared paired difference: official / judge-adjusted / audited, in points."""
    if judge is None:
        return "The judge has not rescored the final records yet.\n"
    names = {arm.label: arm.name for arm in arms}
    datasets = judged_datasets(judge)
    grouped = {}
    for row in judge["paired_deltas"]:
        key = (row["candidate"], row["reference"], row["prompt"])
        grouped.setdefault(key, {}).setdefault(row["dataset"], {})[row["scorer"]] = row
    rows = [[f"{names[c]} − {names[r]}", prompt, *(
                " / ".join(f"{by[d][s]['delta']:+.2f}" for s in ("official", "adjusted", "audited")
                           if s in by[d]) if d in by else "—" for d in datasets)]
            for (c, r, prompt), by in grouped.items()]
    return table(["Comparison", "Prompt", *map(column, datasets)], rows, 2)


def ood_table(arms: list[Arm], ood: dict | None) -> str:
    """Primary score on the extra OOD sets, normalized companion in parentheses."""
    if ood is None:
        return "The extra OOD sets have not been evaluated yet.\n"
    found = {(c["arm"], c["dataset"]): c for c in ood["cells"]}
    datasets = list(dict.fromkeys(c["dataset"] for c in ood["cells"]))
    rows = [[arm.name, *("—" if (c := found.get((arm.label, d))) is None else
                         f"{c['score']:.2f} ({c['normalized']:.2f})" for d in datasets)]
            for arm in arms if any((arm.label, d) in found for d in datasets)]
    header = [f"{DATASETS[d].display_name} ({found_primary(ood, d)})" for d in datasets]
    return table(["Arm", *header], rows, 1)


def found_primary(ood: dict, dataset: str) -> str:
    primary = next(c["primary"] for c in ood["cells"] if c["dataset"] == dataset)
    return {"official": "authors' evaluator", "project": "project exact match"}[primary]


def ood_deltas_table(arms: list[Arm], ood: dict | None) -> str:
    """Paired OOD differences with image-group intervals."""
    if ood is None or not ood["comparisons"]:
        return "No OOD comparison has been computed yet.\n"
    names = {arm.label: arm.name for arm in arms}
    datasets = list(dict.fromkeys(row["dataset"] for row in ood["comparisons"]))
    grouped = {}
    for row in ood["comparisons"]:
        grouped.setdefault((row["candidate"], row["reference"]), {})[row["dataset"]] = row
    rows = [[f"{names[c]} − {names[r]}", *(interval(by[d]) if d in by else "—"
                                          for d in datasets)]
            for (c, r), by in grouped.items()]
    return table(["Comparison", *(DATASETS[d].display_name for d in datasets)], rows, 1)


def beyond_rows(arms: list[Arm], beyond: dict, fields,
                datasets: tuple[str, ...] | None = None) -> list[list[str]]:
    """One row per measured model of every declared set, once each: its setting, its name, and
    `fields`. Given datasets, only the main sets on them (no named group such as "vision")."""
    names = {arm.label: arm.name for arm in arms}
    rows, seen = [], set()
    for spec in beyond["sets"]:
        if datasets is not None and (spec["dataset"] not in datasets or "group" in spec):
            continue
        setting = f"{DATASETS[spec['dataset']].display_name}, {spec['prompt']}"
        for row in spec["rows"]:
            key = (spec["dataset"], spec["prompt"], spec["tier"], row["model"])
            if row.get("missing") or key in seen or (cells := fields(row)) is None:
                continue
            seen.add(key)
            rows.append([setting, names.get(row["model"], row["model"]), *cells])
    return rows


def beyond_accuracy_table(arms: list[Arm], beyond: dict | None,
                          datasets: tuple[str, ...] | None = None) -> str:
    """Accuracy, the forward KL from the starting model on the task's questions, accuracy with a
    black image and its change from the starting model's, and the accuracy lost with a swapped
    image, original − swapped (normalized scorer, 95% intervals)."""
    if beyond is None:
        return "The beyond-accuracy measurements have not been computed yet.\n"

    def fields(row):
        kl, black = row["kl"], row.get("black_accuracy")
        return [f"{row['accuracy']:.2f}",
                "—" if kl is None else f"{kl['kl']:.2f} [{kl['low']:.2f}, {kl['high']:.2f}]",
                "—" if black is None else f"{black:.2f}",
                *("—" if row.get(key) is None else interval(row[key])
                  for key in ("black_gain", "swapped_drop"))]

    return table(["Setting", "Model", "Accuracy", "KL from start (nats)", "Black-image accuracy",
                  "Black image, change from start", "Lost, swapped image"],
                 beyond_rows(arms, beyond, fields, datasets), 2)


def calibration_table(arms: list[Arm], beyond: dict | None) -> str:
    """The majority answer of 8 samples at T = 1: its accuracy, its share of the samples as
    confidence, the expected calibration error, and accuracy on the most confident half."""
    if beyond is None:
        return "The beyond-accuracy measurements have not been computed yet.\n"

    def fields(row):
        c = row["calibration"]
        if c is None:
            return None
        return [f"{c[key]:.1f}" for key in ("majority_accuracy", "mean_confidence",
                                            "calibration_error", "selective_50",
                                            "selective_80")]

    return table(["Setting", "Model", "Majority accuracy", "Mean confidence", "ECE",
                  "Accuracy, top 50%", "Accuracy, top 80%"], beyond_rows(arms, beyond, fields), 2)


def pilot_table(arms: list[Arm], cells: dict, dataset: str) -> str:
    """A pilot's arms on its full test role: official (normalized) accuracy, each answer
    type's official accuracy, and mean output tokens, per prompt."""
    types = sorted({t for arm in arms for prompt in PROMPT_ORDER
                    if (r := full_cell(cells, arm, dataset, prompt))
                    for t in r.metrics["official"]["by_answer_type"]})
    rows = []
    for arm in arms:
        for prompt in PROMPT_ORDER:
            record = full_cell(cells, arm, dataset, prompt)
            if record is None:
                rows.append([arm.name, prompt, *["—"] * (len(types) + 2)])
                continue
            official, normalized = record.metrics["official"], record.metrics["normalized"]
            rows.append([arm.name, prompt,
                         f"{percent(official['overall'])} ({percent(normalized['overall'])})",
                         *(percent(official["by_answer_type"].get(t)) for t in types),
                         f"{record.metrics['mean_output_tokens']:.0f}"])
    return table(["Arm", "Prompt", "Accuracy", *(f"{t} answers" for t in types),
                  "Output tokens"], rows, 2)


# The README overview: (test set, setting, dataset, prompt, scorer, and the arms trained for it
# in the order Instruct, SFT, GRPO, SFT → GRPO).
OVERVIEW_ROWS = (
    ("ChartQA", "reasoning", "chartqa", "reasoning", "official",
     ("base2", "cqa-sft2", "cqa-grpo2-base-e1", "cqa-grpo2-sft-e1")),
    ("SLAKE, 2B", "short", "slake", "short", "official",
     ("base2", "slk-sft2", "slk-short2-base-e1", "slk-short2-sft-e1")),
    ("SLAKE, 8B", "short", "slake", "short", "official",
     ("base8", "slk-sft8", "slk-short8-base-e1", "slk-short8-sft-e1")),
    ("OmniMedVQA, held-out collections", "short, normalized scorer", "omnimedx", "short",
     "normalized", ("base2", "omx-sft2", "omx-short2-base", "omx-short2-sft")),
    ("PathVQA, unseen (SLAKE 2B models)", "short", "pathvqa", "short", "official",
     ("base2", "slk-sft2", "slk-short2-base-e1", "slk-short2-sft-e1")),
    ("TextVQA, unseen (VQAv2 + GQA models)", "short", "textvqa", "short", "official",
     ("base2", "sft2", "p1-f-short2-base", "p1-f-short2-sft")),
)


def overview_table(cells: dict, chartqa_judge: dict | None, ood: dict | None) -> str:
    """The README's results at a glance: Instruct and its three trained versions per test set,
    best in bold; the audited judge's ChartQA score and VizWiz from their result files."""
    def cell(arm, dataset, prompt, variant):
        record = cells.get(Cell(arm, dataset, "test", "full", prompt, "original", None))
        return None if record is None else 100 * record.metrics[variant]["overall"]

    rows = [(name, setting, [cell(arm, dataset, prompt, variant) for arm in arms])
            for name, setting, dataset, prompt, variant, arms in OVERVIEW_ROWS]
    if chartqa_judge:
        audited = {(c["arm"], c["prompt"]): c.get("audited") for c in chartqa_judge["cells"]}
        rows.insert(1, ("ChartQA", "reasoning, audited judge",
                        [audited.get((arm, "reasoning")) for arm in OVERVIEW_ROWS[0][5]]))
    if ood:
        vizwiz = {c["arm"]: c["score"] for c in ood["cells"] if c["dataset"] == "vizwiz"}
        rows.append(("VizWiz, unseen (VQAv2 + GQA models)", "short",
                     [vizwiz.get(arm) for arm in OVERVIEW_ROWS[-1][5]]))
    lines = []
    for label, setting, values in rows:
        best = max(v for v in values if v is not None)
        lines.append([label, setting, *("—" if v is None else f"**{v:.1f}**" if v == best
                                        else f"{v:.1f}" for v in values)])
    return table(["Test set", "Prompt", "Instruct", "SFT", "GRPO", "SFT → GRPO"], lines, 2)


def pilot_deltas_table(arms: list[Arm], rows: list[dict], pilot: str) -> str:
    """A pilot's paired differences, official scorer, with 95% intervals over questions."""
    names = {arm.label: arm.name for arm in arms}
    lines = [[f"{names[r['candidate']]} − {names[r['reference']]}", r["prompt"],
              f"{r['delta']:+.2f} [{r['low']:+.2f}, {r['high']:+.2f}]"]
             for r in rows if r.get("pilot") == pilot and r["variant"] == "official"]
    if not lines:
        return "No paired comparison has been computed yet.\n"
    return table(["Comparison", "Prompt", "Official"], lines, 2)


def transfer_table(arms: list[Arm], rows: list[dict], spec: dict | None, prompt: str,
                   name: str = "general") -> str:
    """Each arm minus the reference on every listed test set under one prompt, official
    scorer, with 95% intervals over questions."""
    rows = [row for row in rows if row.get("transfer_set") == name and row["prompt"] == prompt]
    if spec is None or not rows:
        return "No transfer comparison has been computed yet.\n"
    names = {arm.label: arm.name for arm in arms}
    found = {(row["candidate"], row["dataset"]): row for row in rows}
    lines = []
    for entry in spec["rows"]:
        cells = []
        for dataset in spec["datasets"]:
            row = found.get((entry["arm"], dataset))
            cells.append(f"{row['delta']:+.2f} [{row['low']:+.2f}, {row['high']:+.2f}]"
                         if row else "—")
        lines.append([names[entry["arm"]], entry["trained_on"], *cells])
    return table(["Arm", "Trained on", *map(column, spec["datasets"])], lines, 2)


def pilot_blocks(toml: Path, name: str, cells: dict, deltas: list[dict]) -> dict[str, str]:
    """A single-dataset pilot's two tables: its arms on the test role, and its comparisons."""
    arms, protocols, _ = load_pilot(toml, name)
    return {f"{name}-pilot": pilot_table(arms, cells, protocols[0].dataset),
            f"{name}-pilot-deltas": pilot_deltas_table(arms, deltas, name)}


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
    matched, _ = load_matrix(toml, "matched")
    fullpool, _ = load_matrix(toml, "fullpool")
    mixture, _ = load_matrix(toml, "mixture")
    pilot_arms, _, _ = load_pilot(toml, "chartqa")
    declared_toml = tomllib.loads(toml.read_text())
    transfers = declared_toml.get("transfers", {})
    passk = load_result(root / "results", "passk-heldout.json")
    chartqa_passk = load_result(root / "results", "passk-chartqa.json")
    knobs = load_result(root / "results", "knobs-chartqa.json")
    judge = load_result(root / "results", "judge-final.json")
    chartqa_judge = load_result(root / "results", "judge-chartqa.json")
    ood = load_result(root / "results", "ood.json")
    beyond = load_result(root / "results", "beyond-accuracy.json")
    reach = load_result(root / "results", "reach.json")
    ledger = load_result(root / "results", "ledger.json")
    blocks = {
        "cards": cards_table(ledger),
        "reach": reach_table(reach),
        "matched": matrix_table(matched, cells),
        "matched-deltas": deltas_table(matched, declared(
            deltas, load_comparisons(toml, "matched"), ("official",))),
        "fullpool": matrix_table(fullpool, cells),
        "fullpool-deltas": deltas_table(fullpool, declared(
            deltas, load_comparisons(toml, "fullpool"), ("official",))),
        "mixture-deltas": deltas_table(mixture, declared(
            deltas, load_comparisons(toml, "mixture"), ("official",))),
        "chartqa-passk": passk_table(every_arm, chartqa_passk),
        "knobs-chartqa": knobs_table(knobs),
        **{f"{'' if name == 'general' else name + '-'}transfer-{prompt}":
           transfer_table(every_arm, deltas, spec, prompt, name)
           for name, spec in transfers.items() for prompt in PROMPT_ORDER},
        **{block: table_fn for name in declared_toml.get("pilots", {})
           for block, table_fn in pilot_blocks(toml, name, cells, deltas).items()},
        "chartqa-passk-deltas": passk_deltas_table(every_arm, chartqa_passk),
        "passk": passk_table(every_arm, passk),
        "passk-deltas": passk_deltas_table(every_arm, passk),
        "judge": judge_table(arms, judge),
        "judge-deltas": judge_deltas_table(arms, judge),
        "overview": overview_table(cells, chartqa_judge, ood),
        "chartqa-judge": judge_table(pilot_arms, chartqa_judge),
        "chartqa-judge-deltas": judge_deltas_table(pilot_arms, chartqa_judge),
        "beyond-accuracy": beyond_accuracy_table(every_arm, beyond),
        "beyond-accuracy-medical": beyond_accuracy_table(every_arm, beyond,
                                                         ("slake", "omnimedx")),
        "calibration": calibration_table(every_arm, beyond),
        "ood": ood_table(every_arm, ood),
        "ood-deltas": ood_deltas_table(every_arm, ood),
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
