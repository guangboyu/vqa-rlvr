"""Evaluate a model on the materialized eval subsets.

Writes one results JSON per dataset to results/runs/ (committed) and per-example
predictions to results/preds/ (git-ignored, for error analysis). All README tables
are generated from these JSONs by scripts/make_tables.py — never hand-typed.

Usage:
    uv run python evaluate.py --preset zero_shot_2b --limit 200      # smoke run
    uv run python evaluate.py --preset zero_shot_8b
    uv run python evaluate.py --preset zero_shot_2b --model checkpoints/merged/sft_2b \
        --run-id sft_2b                                              # eval a merged ckpt
"""

import argparse
import json
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from datasets import Image, load_from_disk

from vqar import config
from vqar.data import PROMPT_REASONING, PROMPT_SHORT
from vqar.inference import build_messages, extract_answer, generate_vllm, load_vllm
from vqar.metrics import score

ROOT = Path(__file__).parent
EVAL_SUBSETS = {
    "vqav2": "vqav2_eval",
    "gqa": "gqa_eval",
    "clevr": "clevr_test",
    "textvqa": "textvqa_eval",
}
TEMPLATES = {"short": PROMPT_SHORT, "reasoning": PROMPT_REASONING}


def git_sha() -> str:
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return out.stdout.strip() or "unknown"


def evaluate_dataset(llm, preset, dataset: str, limit: int | None) -> tuple[dict, list[dict]]:
    ds = load_from_disk(str(ROOT / "data" / EVAL_SUBSETS[dataset]))
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    ds = ds.cast_column("image", Image(decode=False))  # raw bytes for data URIs

    template = TEMPLATES[preset.template]
    conversations = [
        build_messages(row["question"], row["image"]["bytes"], template) for row in ds
    ]
    texts = generate_vllm(llm, conversations, preset.max_tokens)

    records, by_type = [], defaultdict(list)
    for row, text in zip(ds, texts, strict=True):
        pred = extract_answer(text)
        s = score(dataset, pred, row["answers"])
        records.append(
            {"qid": row["qid"], "prediction": pred, "raw": text, "score": s,
             "answers": row["answers"]}
        )
        by_type[row["answer_type"] or "all"].append(s)

    n = len(records)
    metrics = {
        "overall": sum(r["score"] for r in records) / n,
        "by_answer_type": {t: sum(v) / len(v) for t, v in sorted(by_type.items())},
        "n": n,
    }
    return metrics, records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", required=True, choices=sorted(config.EVAL_PRESETS))
    parser.add_argument("--model", default=None, help="override model path (merged checkpoint)")
    parser.add_argument("--datasets", nargs="+", default=None, choices=sorted(EVAL_SUBSETS))
    parser.add_argument("--limit", type=int, default=None, help="cap examples per dataset")
    parser.add_argument("--run-id", default=None, help="results file prefix (default: preset)")
    args = parser.parse_args()

    preset = config.EVAL_PRESETS[args.preset]
    model = args.model or preset.model
    run_id = args.run_id or preset.name
    datasets = args.datasets or list(preset.datasets)

    llm = load_vllm(model, preset.max_model_len, preset.gpu_memory_utilization)

    runs_dir = ROOT / "results" / "runs"
    preds_dir = ROOT / "results" / "preds"
    runs_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        metrics, records = evaluate_dataset(llm, preset, dataset, args.limit)
        result = {
            "run_id": f"{run_id}-{dataset}",
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "git_sha": git_sha(),
            "model": {"base": model, "preset": preset.name},
            "eval": {
                "dataset": dataset,
                "subset": EVAL_SUBSETS[dataset],
                "n": metrics["n"],
                "limit": args.limit,
                "prompt_template": preset.template,
            },
            "gen": {
                "backend": preset.backend,
                "temperature": 0.0,
                "max_tokens": preset.max_tokens,
                "max_model_len": preset.max_model_len,
            },
            "metrics": {k: v for k, v in metrics.items() if k != "n"},
            "config": config.dump(preset),
        }
        out = runs_dir / f"{result['run_id']}.json"
        out.write_text(json.dumps(result, indent=2))
        with (preds_dir / f"{result['run_id']}.jsonl").open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"{dataset}: overall={metrics['overall']:.4f} n={metrics['n']} -> {out}")


if __name__ == "__main__":
    main()
