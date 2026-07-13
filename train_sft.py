"""Supervised fine-tuning: TRL SFTTrainer + PEFT LoRA (2B, bf16) / QLoRA (8B, NF4).

Trains on the materialized SFT subsets (VQAv2 + GQA mixture) with the fixed `short`
prompt template. The vision tower stays frozen; adapters target the LM only.

Usage:
    uv run python train_sft.py --preset sft_2b
    uv run python train_sft.py --preset sft_8b
    uv run python train_sft.py --preset sft_2b --max-steps 20   # sanity run
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

import torch
from datasets import concatenate_datasets, load_from_disk
from peft import LoraConfig
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from vqar import config
from vqar.data import PROMPT_SHORT, cap_pixels

os.environ.setdefault("WANDB_PROJECT", "vqa-rlvr")

ROOT = Path(__file__).parent
DEV_SLICE = 500  # held-out examples for eval-loss monitoring


def to_sft_columns(batch: dict) -> dict:
    """Map unified examples to TRL's VLM prompt-completion contract.

    Prompt-completion (not messages) on purpose: TRL masks the entire prompt —
    including the image pad tokens — so loss lands only on the answer. In messages
    mode the collator trained on image pads for this model (loss ~15, random-level).
    TRL injects the image before the first user turn and applies the chat template.
    """
    return {
        "prompt": [
            [{"role": "user", "content": PROMPT_SHORT.format(question=q)}]
            for q in batch["question"]
        ],
        "completion": [
            [{"role": "assistant", "content": answers[0]}] for answers in batch["answers"]
        ],
        "image": [cap_pixels(img) for img in batch["image"]],
    }


def load_datasets(preset: config.SFTPreset):
    ds = concatenate_datasets(
        [load_from_disk(str(ROOT / "data" / name)) for name in preset.subsets]
    ).shuffle(seed=preset.seed)
    ds = ds.select_columns(["image", "question", "answers"])
    train = ds.select(range(DEV_SLICE, len(ds))).with_transform(to_sft_columns)
    dev = ds.select(range(DEV_SLICE)).with_transform(to_sft_columns)
    return train, dev


def build_model(preset: config.SFTPreset):
    quantization = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        if preset.quantize_4bit
        else None
    )
    model = AutoModelForImageTextToText.from_pretrained(
        preset.model, dtype=torch.bfloat16, quantization_config=quantization
    )
    processor = AutoProcessor.from_pretrained(preset.model)
    processor.image_processor.size["longest_edge"] = config.MAX_PIXELS  # pixel-area cap
    return model, processor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", required=True, choices=sorted(config.SFT_PRESETS))
    parser.add_argument("--max-steps", type=int, default=-1, help="cap steps (sanity runs)")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    preset = config.SFT_PRESETS[args.preset]
    train_ds, dev_ds = load_datasets(preset)
    model, processor = build_model(preset)

    output_dir = ROOT / "checkpoints" / preset.name
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=preset.per_device_batch_size,
        per_device_eval_batch_size=preset.per_device_batch_size,
        gradient_accumulation_steps=preset.gradient_accumulation,
        learning_rate=preset.learning_rate,
        num_train_epochs=preset.epochs,
        max_steps=args.max_steps,
        lr_scheduler_type="cosine",
        warmup_ratio=preset.warmup_ratio,
        max_length=preset.max_length,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit" if preset.quantize_4bit else "adamw_torch",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=250,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=2,
        dataloader_num_workers=2,  # WSL2: keep worker memory low
        dataloader_pin_memory=False,
        # The messages/image columns are produced on the fly by with_transform from
        # question/answers/image; the Trainer must not strip its inputs beforehand.
        remove_unused_columns=False,
        seed=preset.seed,
        report_to=[] if args.no_wandb else ["wandb"],
        run_name=preset.name,
    )
    peft_config = LoraConfig(
        r=preset.lora_r,
        lora_alpha=preset.lora_alpha,
        lora_dropout=preset.lora_dropout,
        target_modules=list(preset.lora_targets),
        task_type="CAUSAL_LM",
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=processor,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))

    git_sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    (output_dir / "run_config.json").write_text(
        json.dumps({"preset": config.dump(preset), "git_sha": git_sha,
                    "max_steps": args.max_steps}, indent=2)
    )
    print(f"adapter saved -> {output_dir}")


if __name__ == "__main__":
    main()
