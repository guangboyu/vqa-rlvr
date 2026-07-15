"""Render the README results tables from results/runs/*.json (+ preds for reasoning
retention). README numbers are pasted from this output — never hand-typed.

Usage: uv run python scripts/make_tables.py
"""

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


def load_rows() -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = defaultdict(dict)
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


def main() -> None:
    rows = load_rows()

    print_table(
        "Main results — short template (direct answer)",
        rows,
        [
            ("nanoVLM-460M zero-shot", "zero_shot_nanovlm"),
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
    print("| Format weight | Steps | CoT retention (GQA) | VQAv2 | GQA | CLEVR |")
    print("|---|---|---|---|---|---|")
    for label, run_id, steps in [
        ("0.0 (correctness only)", "grpo_2b_fmt0_reasoning", 300),
        ("0.2", "grpo_2b_main_base_reasoning", 500),
        ("0.5", "grpo_2b_fmt05_reasoning", 300),
    ]:
        r = rows.get(run_id, {})
        print(
            f"| {label} | {steps} | {reasoning_retention(run_id)} | "
            f"{cell(rows, run_id, 'vqav2')} | {cell(rows, run_id, 'gqa')} | "
            f"{cell(rows, run_id, 'clevr')} |"
        )

    study_path = RESULTS / "judge_study.json"
    if study_path.exists():
        print("\n### LLM-judge study (Haiku; EM-miss rescue analysis)\n")
        study = json.loads(study_path.read_text())
        print("| Run | EM acc | Judge rescue rate | Judge-corrected acc | Agreement on EM hits |")
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
