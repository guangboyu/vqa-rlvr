"""Evaluate a model on the materialized eval subsets.

Writes one results JSON per dataset to results/runs/ (committed) and per-example
predictions to results/preds/ for error analysis. All README tables
are generated from these JSONs by scripts/make_tables.py — never hand-typed.

Usage:
    uv run python evaluate.py --preset zero_shot_2b --limit 200      # smoke run
    uv run python evaluate.py --preset zero_shot_8b
    uv run python evaluate.py --preset zero_shot_2b --model checkpoints/merged/sft_2b \
        --run-id sft_2b                                              # eval a merged ckpt
"""

import argparse
import hashlib
import importlib.metadata
import io
import json
import subprocess
from collections import defaultdict
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from datasets import Image, load_from_disk
from PIL import Image as PILImage

from vqar import config
from vqar.data import PROMPT_REASONING, PROMPT_SHORT
from vqar.inference import build_messages, extract_answer, generate_vllm_records, load_vllm
from vqar.metrics import evaluation_score
from vqar.splits import image_keys, load_training_subsets, split_dev, split_provenance

ROOT = Path(__file__).parent
EVAL_SUBSETS = {
    "vqav2": "vqav2_eval",
    "gqa": "gqa_eval",
    "clevr": "clevr_test",
    "textvqa": "textvqa_eval",
}
TEMPLATES = {"short": PROMPT_SHORT, "reasoning": PROMPT_REASONING}
ANSWER_INSTRUCTION = (
    "Then write the final answer as a single word or phrase inside <answer> </answer> tags."
)
TEMPLATES.update({
    "direct": "{question}\nOutput an empty <think></think> block. " + ANSWER_INSTRUCTION,
    "brief": "{question}\nGive at most one sentence of reasoning inside <think> </think> tags. "
             + ANSWER_INSTRUCTION,
    "stepwise": "{question}\nThink step by step inside <think> </think> tags. "
                + ANSWER_INSTRUCTION,
})


def load_subset(dataset: str, split: str, data_dir: Path):
    """Use frozen dev images for prompt and checkpoint selection."""
    if split == "eval":
        return load_from_disk(str(data_dir / EVAL_SUBSETS[dataset])), {}
    if dataset not in ("vqav2", "gqa"):
        raise ValueError("The frozen dev split contains only VQAv2 and GQA.")
    manifest = json.loads((data_dir / "split_manifest.json").read_text())
    spec = manifest["sft_dev"]
    _, dev = split_dev(load_training_subsets(data_dir, spec["subsets"], spec["seed"]))
    selected = dev.select([i for i, name in enumerate(dev["dataset"]) if name == dataset])
    return selected, split_provenance(data_dir, legacy=False)


def git_sha() -> str:
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return out.stdout.strip() or "unknown"


def atomic_json(path: Path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_progress(directory: Path, identity: dict, qids, resume: bool):
    """Accept only complete, contiguous batches from this exact evaluation."""
    state = directory / "identity.json"
    if state.exists():
        if not resume:
            raise FileExistsError(f"{directory} exists; use --resume or a new --run-id.")
        if json.loads(state.read_text()) != identity:
            raise ValueError(f"Evaluation settings or inputs changed: {directory}")
    else:
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(state, identity)
    records = []
    for path in sorted(directory.glob("batch-*.json")):
        if path.name != f"batch-{len(records):08d}.json":
            raise ValueError(f"Missing or overlapping evaluation batch: {path}")
        batch = json.loads(path.read_text())
        expected = list(qids[len(records):len(records) + len(batch)])
        if not batch or [r["qid"] for r in batch] != expected:
            raise ValueError(f"Prediction IDs do not match the evaluation subset: {path}")
        records.extend(batch)
    return records


def image_payload(row):
    payload = row["image"]
    return payload["bytes"] if payload["bytes"] is not None else Path(payload["path"]).read_bytes()


def swap_indices(keys):
    """Choose a deterministic different image, keeping questions and labels fixed."""
    if len(set(keys)) < 2:
        raise ValueError("Swapped-image evaluation needs at least two different images.")
    donors = []
    for i, key in enumerate(keys):
        j = (i + 1) % len(keys)
        while keys[j] == key:
            j = (j + 1) % len(keys)
        donors.append(j)
    return donors


def evaluate_dataset(llm, preset, dataset, ds, records, progress, batch_size, metric, image_mode):
    template = TEMPLATES[preset.template]
    donors = swap_indices(image_keys(ds)) if image_mode == "swapped" else None
    for start in range(len(records), len(ds), batch_size):
        rows = list(ds.select(range(start, min(start + batch_size, len(ds)))))
        conversations, image_qids = [], []
        for offset, row in enumerate(rows):
            donor = ds[donors[start + offset]] if donors is not None else row
            payload = image_payload(donor)
            if image_mode == "black":
                with PILImage.open(io.BytesIO(payload)) as original:
                    blank = PILImage.new("RGB", original.size)
                buffer = io.BytesIO()
                blank.save(buffer, format="PNG")
                payload = buffer.getvalue()
            conversations.append(build_messages(row["question"], payload, template))
            image_qids.append(donor["qid"])
        outputs = generate_vllm_records(llm, conversations, preset.max_tokens)
        batch = []
        for row, output, image_qid in zip(rows, outputs, image_qids, strict=True):
            text = output.pop("text")
            pred = extract_answer(text)
            batch.append({
                "qid": row["qid"], "question": row["question"], "prediction": pred,
                "raw": text, "score": evaluation_score(dataset, pred, row["answers"], metric),
                "answers": row["answers"], "answer_type": row["answer_type"],
                "image_qid": image_qid, **output,
            })
        atomic_json(progress / f"batch-{start:08d}.json", batch)
        records.extend(batch)
        print(f"{dataset}: saved {len(records)}/{len(ds)}", flush=True)
    return summarize(records)


def summarize(records):
    if not records:
        raise ValueError("Evaluation subset is empty.")
    by_type = defaultdict(list)
    for record in records:
        by_type[record["answer_type"] or "all"].append(record["score"])
    n = len(records)
    return {
        "overall": sum(r["score"] for r in records) / n,
        "by_answer_type": {t: sum(v) / len(v) for t, v in sorted(by_type.items())},
        "mean_output_tokens": sum(r["output_tokens"] for r in records) / n,
        "truncated_fraction": sum(r["finish_reason"] == "length" for r in records) / n,
        "answer_tag_fraction": sum("</answer>" in r["raw"] for r in records) / n,
        "empty_answer_fraction": sum(not r["prediction"] for r in records) / n,
        "n": n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", required=True, choices=sorted(config.EVAL_PRESETS))
    parser.add_argument("--model", default=None, help="override model path (merged checkpoint)")
    parser.add_argument("--datasets", nargs="+", default=None, choices=sorted(EVAL_SUBSETS))
    parser.add_argument("--limit", type=int, default=None, help="cap examples per dataset")
    parser.add_argument("--run-id", default=None, help="results file prefix (default: preset)")
    parser.add_argument("--metric", choices=("official", "normalized"), default="official")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--resume", action="store_true", help="resume matching saved batches")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-mode", choices=("original", "black", "swapped"),
                        default="original")
    parser.add_argument("--split", choices=("eval", "dev"), default="eval")
    parser.add_argument("--template", choices=sorted(TEMPLATES), default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="check data and settings without a GPU")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("batch-size and limit must be positive")
    if args.max_tokens is not None and args.max_tokens < 1:
        parser.error("max-tokens must be positive")

    preset = config.EVAL_PRESETS[args.preset]
    preset = replace(preset, template=args.template or preset.template,
                     max_tokens=args.max_tokens or preset.max_tokens)
    model = args.model or preset.model
    suffix = f"_{args.split}_{preset.template}" if args.split != "eval" or args.template else ""
    run_id = args.run_id or f"{preset.name}_{args.metric}{suffix}"
    datasets = args.datasets or (["vqav2", "gqa"] if args.split == "dev" else list(preset.datasets))
    if args.split == "dev" and set(datasets) - {"vqav2", "gqa"}:
        parser.error("the frozen dev split contains only VQAv2 and GQA")
    if args.split == "dev" and "vqav2" in datasets and args.metric == "official":
        parser.error("VQAv2 dev has single-reference answers; use --metric normalized")

    runs_dir = args.results_dir / "runs"
    preds_dir = args.results_dir / "preds"
    source_hashes = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ("evaluate.py", "vqar/inference.py", "vqar/metrics.py", "vqar/config.py",
                     "vqar/data.py", "vqar/splits.py", "uv.lock")
    }
    prepared = []
    for dataset in datasets:
        out = runs_dir / f"{run_id}-{dataset}.json"
        predictions = preds_dir / f"{run_id}-{dataset}.jsonl"
        if not args.resume and (out.exists() or predictions.exists()):
            raise FileExistsError(f"{out} or its predictions exist; choose a new --run-id.")
        ds, split_info = load_subset(dataset, args.split, ROOT / "data")
        fingerprint = ds._fingerprint
        if args.limit:
            ds = ds.select(range(min(args.limit, len(ds))))
        ds = ds.cast_column("image", Image(decode=False))
        qids = list(ds["qid"])
        identity = {
            "model": model, "preset": config.dump(preset), "dataset": dataset,
            "fingerprint": fingerprint,
            "qids_sha256": hashlib.sha256(json.dumps(qids).encode()).hexdigest(),
            "metric": args.metric, "image_mode": args.image_mode, "batch_size": args.batch_size,
            "source_sha256": source_hashes,
            "split": args.split, "split_provenance": split_info,
            "prompt_text": TEMPLATES[preset.template],
        }
        # JSON round-trip also normalizes tuples in preset fields for strict resume checks.
        identity = json.loads(json.dumps(identity))
        if args.dry_run:
            print(json.dumps({"dataset": dataset, "n": len(ds), **identity}, indent=2))
            continue
        progress = args.results_dir / "progress" / f"{run_id}-{dataset}"
        if (args.resume and (out.exists() or predictions.exists())
                and not (progress / "identity.json").exists()):
            raise ValueError(f"{out} has no resumable provenance; choose a new --run-id.")
        records = load_progress(progress, identity, qids, args.resume)
        prepared.append((dataset, ds, identity, progress, records, out, predictions))
    if args.dry_run:
        return
    llm = None
    runs_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)
    for dataset, ds, identity, progress, records, out, predictions in prepared:
        if len(records) < len(ds) and llm is None:
            llm = load_vllm(model, preset.max_model_len, preset.gpu_memory_utilization)
        metrics = evaluate_dataset(llm, preset, dataset, ds, records, progress,
                                   args.batch_size, args.metric, args.image_mode)
        result = {
            "run_id": f"{run_id}-{dataset}",
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "git_sha": git_sha(),
            "source_sha256": source_hashes,
            "qids_sha256": identity["qids_sha256"],
            "dataset_fingerprint": identity["fingerprint"],
            "versions": {name: importlib.metadata.version(name)
                         for name in ("torch", "transformers", "vllm", "datasets")},
            "model": {"base": model, "preset": preset.name},
            "eval": {
                "dataset": dataset,
                "subset": EVAL_SUBSETS[dataset] if args.split == "eval" else "sft_dev",
                "split": args.split,
                "split_provenance": identity["split_provenance"],
                "n": metrics["n"],
                "limit": args.limit,
                "prompt_template": preset.template,
                "prompt_text": TEMPLATES[preset.template],
                "metric_variant": args.metric,
                "answer_extraction": "last_answer_tag_or_full_text",
                "image_mode": args.image_mode,
            },
            "gen": {
                "backend": "vllm",
                "temperature": 0.0,
                "max_tokens": preset.max_tokens,
                "max_model_len": preset.max_model_len,
            },
            "metrics": {k: v for k, v in metrics.items() if k != "n"},
            "config": config.dump(preset),
        }
        temporary = predictions.with_suffix(".tmp")
        with temporary.open("w") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary.replace(predictions)
        atomic_json(out, result)
        print(f"{dataset}: overall={metrics['overall']:.4f} n={metrics['n']} -> {out}")


if __name__ == "__main__":
    main()
