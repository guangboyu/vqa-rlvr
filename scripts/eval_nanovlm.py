"""Zero-shot eval of the nanoVLM-460M continuity baseline row.

nanoVLM is not an HF AutoModel, so this script drives the upstream nanoVLM code
directly (huggingface/nanoVLM or a fork). Point NANOVLM_REPO at a checkout; results
land in results/runs/ using the same schema as evaluate.py.

Usage:
    NANOVLM_REPO=~/codes/VLM-medical uv run python scripts/eval_nanovlm.py \
        --datasets clevr --limit 200
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent
NANOVLM_REPO = Path(os.environ.get("NANOVLM_REPO", Path.home() / "codes/VLM-medical"))
sys.path.insert(0, str(NANOVLM_REPO))

from datasets import load_from_disk  # noqa: E402

from vqar.data import PROMPT_SHORT  # noqa: E402
from vqar.inference import extract_answer  # noqa: E402
from vqar.metrics import score  # noqa: E402

MODEL_ID = "lusxvr/nanoVLM-460M-8k"
EVAL_SUBSETS = {
    "vqav2": "vqav2_eval",
    "gqa": "gqa_eval",
    "clevr": "clevr_test",
    "textvqa": "textvqa_eval",
}


def load_nanovlm(device):
    from data.processors import get_image_processor, get_tokenizer
    from models.vision_language_model import VisionLanguageModel

    model = VisionLanguageModel.from_pretrained(MODEL_ID).to(device).eval()
    cfg = model.cfg
    tokenizer = get_tokenizer(cfg.lm_tokenizer, cfg.vlm_extra_tokens, cfg.lm_chat_template)
    image_processor = get_image_processor(cfg.max_img_size, cfg.vit_img_size)
    return model, cfg, tokenizer, image_processor


def generate_batch(model, cfg, tokenizer, image_processor, batch, device, max_new_tokens=48):
    from data.processors import get_image_string

    images, prompts = [], []
    for row in batch:
        processed, ratio = image_processor(row["image"].convert("RGB"))
        if (
            not hasattr(tokenizer, "global_image_token")
            and ratio[0] * ratio[1] == len(processed) - 1
        ):
            processed = processed[1:]
        images.append([processed])
        image_string = get_image_string(tokenizer, [ratio], cfg.mp_image_token_length)
        content = image_string + PROMPT_SHORT.format(question=row["question"])
        prompts.append([{"role": "user", "content": content}])

    texts = tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(texts, return_tensors="pt", padding="longest", padding_side="left")
    generated = model.generate(
        inputs["input_ids"].to(device),
        images,
        inputs["attention_mask"].to(device),
        max_new_tokens=max_new_tokens,
        greedy=True,
    )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(EVAL_SUBSETS), choices=EVAL_SUBSETS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, tokenizer, image_processor = load_nanovlm(device)

    git_sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    runs_dir = ROOT / "results" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        ds = load_from_disk(str(ROOT / "data" / EVAL_SUBSETS[dataset]))
        if args.limit:
            ds = ds.select(range(min(args.limit, len(ds))))
        scores = []
        for start in range(0, len(ds), args.batch_size):
            batch = [ds[i] for i in range(start, min(start + args.batch_size, len(ds)))]
            texts = generate_batch(model, cfg, tokenizer, image_processor, batch, device)
            for row, text in zip(batch, texts, strict=True):
                scores.append(score(dataset, extract_answer(text), row["answers"]))
        result = {
            "run_id": f"zero_shot_nanovlm-{dataset}",
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "git_sha": git_sha,
            "model": {"base": MODEL_ID, "preset": "zero_shot_nanovlm"},
            "eval": {
                "dataset": dataset,
                "subset": EVAL_SUBSETS[dataset],
                "n": len(scores),
                "limit": args.limit,
                "prompt_template": "short",
            },
            "gen": {"backend": "nanovlm", "temperature": 0.0, "max_tokens": 48},
            "metrics": {"overall": sum(scores) / len(scores)},
        }
        out = runs_dir / f"{result['run_id']}.json"
        out.write_text(json.dumps(result, indent=2))
        print(f"{dataset}: overall={result['metrics']['overall']:.4f} n={len(scores)} -> {out}")


if __name__ == "__main__":
    main()
