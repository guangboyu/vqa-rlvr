"""Render historical comparisons with an explicit scoring variant.

Official scores come from the generated diagnostics artifact; normalized scores
come from the original run JSONs. Output length is a proxy, not reasoning quality.

Usage: uv run python scripts/make_tables.py
       uv run python scripts/make_tables.py --results-dir results/baselines
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).parent.parent / "results"
DATASETS = ["vqav2", "gqa", "clevr", "textvqa"]
HEADERS = {
    "vqav2": "VQAv2-val",
    "gqa": "GQA-testdev",
    "clevr": "CLEVR-test",
    "textvqa": "TextVQA-val (OOD)",
}


def load_rows(metric="official") -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = defaultdict(dict)
    if metric == "official":
        report = json.loads((RESULTS / "diagnostics.json").read_text())
        for run_id, scores in report["official_runs"].items():
            rows[run_id] = {dataset: value / 100 for dataset, value in scores.items()}
        return rows
    for path in sorted((RESULTS / "runs").glob("*.json")):
        run = json.loads(path.read_text())
        if "eval" not in run:  # placeholder record (e.g. OOM not-run)
            continue
        dataset = run["eval"]["dataset"]
        rows[run["run_id"].removesuffix(f"-{dataset}")][dataset] = run["metrics"]["overall"]
    return rows


def cell(rows, model, dataset, baseline=None):
    value = rows.get(model, {}).get(dataset)
    if value is None:
        return "—"
    text = f"{value * 100:.1f}"
    if baseline is not None and (base := rows.get(baseline, {}).get(dataset)) is not None:
        delta = (value - base) * 100
        text += f" ({'+' if delta >= 0 else ''}{delta:.1f})"
    return text


def print_table(title, rows, entries, datasets=DATASETS, baseline=None):
    print(f"\n### {title}\n")
    print("| Model | " + " | ".join(HEADERS[d] for d in datasets) + " |")
    print("|---" * (len(datasets) + 1) + "|")
    for label, run_id in entries:
        cells = [cell(rows, run_id, d, baseline) for d in datasets]
        print(f"| {label} | " + " | ".join(cells) + " |")


def reasoning_retention(run_id: str) -> str:
    path = RESULTS / "preds" / f"{run_id}-gqa.jsonl"
    if not path.exists():
        return "—"
    lengths = [len(json.loads(line)["raw"]) for line in path.read_text().splitlines()]
    reasoned = sum(length > 100 for length in lengths)
    return f"{100 * reasoned / len(lengths):.0f}% (median {statistics.median(lengths):.0f} ch)"


def load_matched_results(directory):
    """Compare fresh runs only when questions, scoring, and decoding match."""
    protocols, runs = {}, defaultdict(dict)
    for path in sorted(directory.rglob("runs/*.json")):
        record = json.loads(path.read_text())
        evaluation = record["eval"]
        dataset, template = evaluation["dataset"], evaluation["prompt_template"]
        key = (template, dataset, evaluation["image_mode"])
        identity = {
            "questions": record["qids_sha256"],
            "fingerprint": record["dataset_fingerprint"],
            "evaluation": evaluation,
            "generation": record["gen"],
            "source": record["source_sha256"],
            "versions": record["versions"],
        }
        if key in protocols and protocols[key] != identity:
            raise ValueError(f"Evaluation protocol differs: {path}")
        protocols[key] = identity
        run = record["run_id"].removesuffix(f"-{dataset}")
        if dataset in runs[run]:
            raise ValueError(f"Duplicate run/dataset: {path}")
        runs[run][dataset] = record
    if not runs:
        raise ValueError(f"No evaluation results in {directory}")
    return runs


def print_matched_results(directory):
    runs = load_matched_results(directory)
    print("Scores on the materialized held-out subsets; accuracy in percent.\n")
    print("| Run | VQAv2 | GQA | CLEVR counting | TextVQA |")
    print("|---|---:|---:|---:|---:|")
    for run, datasets in sorted(runs.items()):
        values = [f"{100 * datasets[d]['metrics']['overall']:.2f}" if d in datasets else "—"
                  for d in DATASETS]
        print(f"| {run} | " + " | ".join(values) + " |")
    print("\n| Run | Dataset | Questions | Mean output tokens | Truncated | Answer tags |")
    print("|---|---|---:|---:|---:|---:|")
    for run, datasets in sorted(runs.items()):
        for dataset, record in sorted(datasets.items()):
            metrics = record["metrics"]
            print(f"| {run} | {dataset} | {record['eval']['n']} | "
                  f"{metrics['mean_output_tokens']:.1f} | "
                  f"{100 * metrics['truncated_fraction']:.2f}% | "
                  f"{100 * metrics['answer_tag_fraction']:.2f}% |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", choices=("official", "normalized"), default="official")
    parser.add_argument("--results-dir", type=Path,
                        help="render fresh matched evaluation folders instead of historical tables")
    args = parser.parse_args()
    if args.results_dir:
        print_matched_results(args.results_dir)
        return
    rows = load_rows(args.metric)
    print(f"Scoring: {args.metric}. Historical training runs; see README errata and data overlap.")

    print_table(
        "Historical results — short template (direct answer)",
        rows,
        [
            ("Qwen3-VL-2B zero-shot", "zero_shot_2b"),
            ("2B + SFT", "sft_2b"),
            ("2B + SFT + GRPO", "grpo_2b_main_sft"),
            ("2B + GRPO only", "grpo_2b_main_base"),
            ("Qwen3-VL-8B zero-shot", "zero_shot_8b"),
            ("8B + SFT (QLoRA)", "sft_8b"),
        ],
    )

    print_table(
        "Reasoning template (chain-of-thought mode; Δ vs 2B zero-shot)",
        rows,
        [
            ("2B zero-shot", "zero_shot_2b_reasoning"),
            ("2B + SFT", "sft_2b_reasoning"),
            ("2B + GRPO only", "grpo_2b_main_base_reasoning"),
            ("2B + SFT + GRPO", "grpo_2b_main_sft_reasoning"),
            ("8B zero-shot", "zero_shot_8b_reasoning"),
            ("8B + SFT", "sft_8b_reasoning"),
        ],
        baseline="zero_shot_2b_reasoning",
    )

    print_table(
        "Transfer matrix — GRPO train set (rows) x eval set (cols); reasoning mode, Δ vs zero-shot",
        rows,
        [
            ("RL on CLEVR only", "grpo_2b_clevr_base_reasoning"),
            ("RL on VQAv2 only", "grpo_2b_vqav2only_reasoning"),
            ("RL on GQA only", "grpo_2b_gqaonly_reasoning"),
            ("RL on VQAv2+GQA mix", "grpo_2b_main_base_reasoning"),
        ],
        baseline="zero_shot_2b_reasoning",
    )

    print("\n### Reward-design ablation (base policy, VQAv2+GQA mix)\n")
    print("| Format weight | Steps | Outputs >100 characters (GQA) | VQAv2 | GQA | CLEVR |")
    print("|---|---|---|---|---|---|")
    for label, run_id, steps in [
        ("0.0 (correctness only)", "grpo_2b_fmt0_reasoning", 300),
        ("0.2", "grpo_2b_main_base_reasoning", 500),
        ("0.5", "grpo_2b_fmt05_reasoning", 300),
    ]:
        print(
            f"| {label} | {steps} | {reasoning_retention(run_id)} | "
            f"{cell(rows, run_id, 'vqav2')} | {cell(rows, run_id, 'gqa')} | "
            f"{cell(rows, run_id, 'clevr')} |"
        )

    print("\n### Learning-signal comparison — GRPO vs distillation family (reasoning protocol)\n")
    print("| Arm | Steps | Outputs >100 characters (GQA) | VQAv2 | GQA | CLEVR | TextVQA |")
    print("|---|---|---|---|---|---|---|")
    for label, run_id, steps in [
        ("2B zero-shot (reasoning)", "zero_shot_2b_reasoning", 0),
        ("GRPO, EM reward (fixedreward)", "grpo_2b_fmt0_fixedreward_reasoning", 300),
        ("OPD, reverse-KL to 8B", "opd_2b_main_reasoning", 300),
        ("Self-distill, gold-conditioned 2B", "opd_self_2b_main_reasoning", 300),
        ("Self-distill, seed 43 (replication)", "opd_self_2b_main_s43_reasoning", 300),
        ("8B teacher (reference: untrained zero-shot)", "zero_shot_8b_reasoning", 0),
        ("Self-distill, 8B student (L40S)", "opd_self_8b_main_reasoning", 300),
        ("GRPO 8B, EM reward (RTX6000)", "grpo_8b_fmt0_reasoning", 300),
    ]:
        print(
            f"| {label} | {steps} | {reasoning_retention(run_id)} | "
            f"{cell(rows, run_id, 'vqav2')} | {cell(rows, run_id, 'gqa')} | "
            f"{cell(rows, run_id, 'clevr')} | {cell(rows, run_id, 'textvqa')} |"
        )

    print_table(
        "OPD reference — short template (direct answer)",
        rows,
        [
            ("2B zero-shot", "zero_shot_2b"),
            ("GRPO, EM reward (fixedreward)", "grpo_2b_fmt0_fixedreward"),
            ("OPD, reverse-KL to 8B", "opd_2b_main"),
            ("Self-distill, gold-conditioned 2B", "opd_self_2b_main"),
        ],
        datasets=["vqav2", "gqa"],
    )

    study_path = RESULTS / "judge_study.json"
    if study_path.exists():
        print("\n### LLM-judge study (Haiku; EM-miss rescue analysis)\n")
        study = json.loads(study_path.read_text())
        print("| Run | EM acc | Judge rescue rate | Historical judge-adjusted score | "
              "Agreement on EM hits |")
        print("|---|---|---|---|---|")
        for r in study["results"]:
            corrected = r["judge_corrected_accuracy"] * 100
            agreement = r["control_agreement_on_em_hits"] * 100
            print(
                f"| {r['run_id']} | {r['em_accuracy']*100:.1f} | "
                f"{r['judge_rescue_rate']*100:.1f}% | {corrected:.1f} | {agreement:.0f}% |"
            )
        print(f"\nTotal judge spend: ${study['judge_spend_usd']:.2f}")


if __name__ == "__main__":
    main()
