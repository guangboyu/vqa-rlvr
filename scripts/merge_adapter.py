"""Merge a LoRA/QLoRA adapter into its bf16 base model for vLLM serving.

vLLM's dynamic LoRA loading is broken for Qwen3-VL (vllm#28640), so every
checkpoint is merged before evaluation. QLoRA adapters merge into the *bf16*
base — standard practice, with a small quantization mismatch we accept
consistently across all checkpoints.

Usage:
    uv run python scripts/merge_adapter.py checkpoints/sft_2b
    # -> checkpoints/merged/sft_2b
"""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path(__file__).parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter", help="adapter checkpoint dir (e.g. checkpoints/sft_2b)")
    parser.add_argument("--out", default=None, help="default: checkpoints/merged/<name>")
    args = parser.parse_args()

    adapter_dir = Path(args.adapter)
    run_config = json.loads((adapter_dir / "run_config.json").read_text())
    # GRPO checkpoints record the resolved policy base; SFT ones use the preset model.
    base_id = run_config.get("policy_model") or run_config["preset"]["model"]
    out = Path(args.out) if args.out else ROOT / "checkpoints" / "merged" / adapter_dir.name

    base = AutoModelForImageTextToText.from_pretrained(base_id, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.save_pretrained(out)
    AutoProcessor.from_pretrained(base_id).save_pretrained(out)
    (out / "run_config.json").write_text(json.dumps(run_config, indent=2))
    print(f"merged -> {out}")


if __name__ == "__main__":
    main()
