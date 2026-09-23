"""Evaluate a model on registered datasets through the harness in `vqar.eval`.

Each (model, dataset, prompt) cell writes one RunRecord under results/records/ with both
scorer variants and a predictions sidecar. A cell that already has a record is skipped, and
an interrupted cell resumes from its saved batches.

Usage:
    uv run python evaluate.py --model Qwen/Qwen3-VL-2B-Instruct --label base2 --tier smoke
    uv run python evaluate.py --model checkpoints/merged/sharedpool_20260913_sft_2b \\
        --label sft2 --prompts short reasoning
    uv run python evaluate.py --model Qwen/Qwen3-VL-2B-Instruct --role dev --dry-run
"""

import argparse
from pathlib import Path

from vqar.eval import ModelRef, Protocol, find_cell, run_eval
from vqar.eval.protocol import IMAGE_MODES
from vqar.eval.runner import DATA_DIR, RECORDS_DIR
from vqar.inference import VLLMGenerator
from vqar.registry import DATASETS, PROMPTS


def protocols(args) -> list[Protocol]:
    datasets = args.datasets or [name for name in DATASETS if has_role(name, args.role)]
    return [
        Protocol(dataset, prompt=prompt, role=args.role, tier=args.tier,
                 image_mode=args.image_mode, max_tokens=args.max_tokens,
                 temperature=args.temperature, samples=args.samples, seed=args.seed)
        for dataset in datasets
        for prompt in args.prompts
    ]


def has_role(dataset: str, role: str) -> bool:
    try:
        Protocol(dataset, role=role)
    except ValueError:
        return False
    return True


def describe(record) -> str:
    official, normalized = record.metrics["official"], record.metrics["normalized"]
    scores = [f"normalized {normalized['overall']:.4f}"]
    if official is not None:
        scores.insert(0, f"official {official['overall']:.4f}")
    return f"{', '.join(scores)}, n={record.metrics['n']}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Hub id or local merged checkpoint")
    parser.add_argument("--revision", default=None, help="Hub revision to pin")
    parser.add_argument("--label", default=None, help="records folder name (default: basename)")
    parser.add_argument("--run", default=None, help="training run that produced the model")
    parser.add_argument("--step", type=int, default=None, help="training step of the model")
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=None,
                        help="default: every dataset with the chosen role")
    parser.add_argument("--prompts", nargs="+", choices=sorted(PROMPTS), default=["short"])
    parser.add_argument("--role", choices=("test", "dev"), default="test")
    parser.add_argument("--tier", choices=tuple(Protocol.TIERS), default="full")
    parser.add_argument("--image-mode", choices=IMAGE_MODES, default="original")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="default: the prompt template's budget")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8,
                        help="0.88 leaves an 8B model enough KV cache on one card")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--records-dir", type=Path, default=RECORDS_DIR)
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve every cell and report cached ones without a GPU")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    try:
        cells = protocols(args)
    except ValueError as error:
        parser.error(str(error))

    model = ModelRef(args.model, args.revision, args.label, args.run, args.step)
    # Pin a Hub model to the commit its records name, so generation loads exactly that.
    revision = model.identity()["weights"].get("revision")
    model = ModelRef(args.model, revision, args.label, args.run, args.step)
    generator = VLLMGenerator(args.model, revision, args.max_model_len,
                              args.gpu_memory_utilization)
    for protocol in cells:
        cell = find_cell(model, protocol, data_dir=args.data_dir, records_dir=args.records_dir)
        if args.dry_run:
            state = "cached" if cell.cached else "missing"
            print(f"{protocol.slug()}: {state}, {cell.questions} questions, "
                  f"protocol {cell.protocol_id[:10]} -> {cell.path}")
            continue
        cached = cell.cached
        record = run_eval(model, protocol, generator, data_dir=args.data_dir,
                          records_dir=args.records_dir, batch_size=args.batch_size)
        # A legacy record is the same cell, migrated from an earlier revision of the code.
        note = (" (cached, legacy)" if record.legacy else " (cached)") if cached else ""
        print(f"{protocol.slug()}: {describe(record)}{note} -> {record.path}")


if __name__ == "__main__":
    main()
