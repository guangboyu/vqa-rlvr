"""Push adapter checkpoints to the Hugging Face Hub with generated model cards.

Usage:
    uv run python scripts/push_to_hub.py sft_2b grpo_2b_main_base ...
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).parent.parent
GITHUB = "https://github.com/guangboyu/vqa-rlvr"

DESCRIPTIONS = {
    "sft_2b": "LoRA SFT adapter (r=16, LM-only) for Qwen3-VL-2B on 40k VQAv2+GQA",
    "sft_8b": "QLoRA SFT adapter (NF4, r=16) for Qwen3-VL-8B on 40k VQAv2+GQA",
    "grpo_2b_main_base": "GRPO RLVR adapter for Qwen3-VL-2B (RL from base, VQAv2+GQA)",
    "grpo_2b_main_sft": "GRPO RLVR adapter for Qwen3-VL-2B (SFT-then-RL, VQAv2+GQA)",
}
RESULTS = {
    "sft_2b": ("79.9 / 62.3 / 99.8 / 79.6", "71.8 / 57.9 / 99.8 / 71.3"),
    "sft_8b": ("83.0 / 63.6 / 92.8 / 85.0", "70.2 / 50.8 / 96.2 / 75.4"),
    "grpo_2b_main_base": ("77.8 / 59.7 / 98.5 / 80.7", "69.5 / 53.9 / 83.9 / 72.7"),
    "grpo_2b_main_sft": ("79.8 / 62.3 / 99.8 / 79.6", "69.1 / 54.3 / 87.1 / 72.3"),
}
BASES = {
    "sft_2b": ("Qwen/Qwen3-VL-2B-Instruct", None),
    "sft_8b": ("Qwen/Qwen3-VL-8B-Instruct", None),
    "grpo_2b_main_base": ("Qwen/Qwen3-VL-2B-Instruct", None),
    # RL adapter trained on top of the MERGED SFT model — must be reconstructed first.
    "grpo_2b_main_sft": (
        "Qwen/Qwen3-VL-2B-Instruct",
        "This adapter applies on top of the base **with the `vqa-rlvr-sft-2b` adapter "
        "merged in first** (load base → apply sft adapter → `merge_and_unload()` → "
        "apply this adapter).",
    ),
}


def model_card(name: str) -> str:
    short, reasoning = RESULTS[name]
    base, stack_note = BASES[name]
    return f"""---
base_model: {base}
library_name: peft
license: apache-2.0
tags: [vqa, multimodal, lora, {"grpo, rlvr, reinforcement-learning" if "grpo" in name else "sft"}]
---

# {name} — {DESCRIPTIONS[name]}

Part of [vqa-rlvr]({GITHUB}): post-training Qwen3-VL for visual question answering
on a single RTX 4090 (QLoRA SFT + GRPO with verifiable rewards).

## Results (full eval sets; VQAv2 / GQA / CLEVR / TextVQA)

- **Short template:** {short}
- **Reasoning template:** {reasoning}

Metrics: official VQA accuracy (VQAv2/TextVQA), normalized EM (GQA/CLEVR); harness
cross-checked against lmms-eval. Full tables, configs, and per-run JSONs: [{GITHUB}]({GITHUB}).

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForImageTextToText

base = AutoModelForImageTextToText.from_pretrained("{base}", dtype="bfloat16")
model = PeftModel.from_pretrained(base, "REPO_ID").merge_and_unload()
```

{stack_note or ""}
Training config is in `run_config.json` in this repo.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", choices=sorted(DESCRIPTIONS))
    args = parser.parse_args()

    api = HfApi()
    user = api.whoami()["name"]
    for name in args.checkpoints:
        ckpt = ROOT / "checkpoints" / name
        repo_id = f"{user}/vqa-rlvr-{name.replace('_', '-')}"
        api.create_repo(repo_id, exist_ok=True, repo_type="model")
        card = model_card(name)
        api.upload_folder(
            folder_path=str(ckpt),
            repo_id=repo_id,
            # README excluded: PEFT's auto-card can carry a local base_model path
            # that Hub metadata validation rejects; our generated card replaces it.
            ignore_patterns=["checkpoint-*", "*.pth", "optimizer*", "README.md"],
        )
        api.upload_file(
            path_or_fileobj=card.replace("REPO_ID", repo_id).encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
        )
        print(f"pushed -> https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
