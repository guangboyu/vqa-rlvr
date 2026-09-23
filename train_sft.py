"""Supervised fine-tuning: TRL SFTTrainer + PEFT LoRA (2B, bf16) / QLoRA (8B, NF4).

Trains on the sft role of VQAv2 and GQA (their split files hold dev and test images out)
with the fixed `short` prompt template. The vision tower stays frozen; adapters target the LM only.

Usage:
    uv run python train_sft.py --preset sft_2b
    uv run python train_sft.py --preset sft_8b
    uv run python train_sft.py --preset sft_2b --max-steps 20   # sanity run
"""

import argparse
import os
from dataclasses import replace
from pathlib import Path

import torch
from peft import LoraConfig
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from vqar import config
from vqar.data import cap_pixels
from vqar.registry import PROMPTS
from vqar.splits import load_role, provenance
from vqar.training import last_complete_checkpoint, write_run_config, write_training_summary

os.environ.setdefault("WANDB_PROJECT", "vqa-rlvr")

ROOT = Path(__file__).parent


def to_sft_columns(batch: dict) -> dict:
    """Map unified examples to TRL's VLM prompt-completion contract.

    Prompt-completion (not messages) on purpose: TRL masks the entire prompt —
    including the image pad tokens — so loss lands only on the answer. In messages
    mode the collator trained on image pads for this model (loss ~15, random-level).
    TRL injects the image before the first user turn and applies the chat template.
    """
    return {
        "prompt": [
            [{"role": "user", "content": PROMPTS["short"].text.format(question=q)}]
            for q in batch["question"]
        ],
        "completion": [
            [{"role": "assistant", "content": answers[0]}] for answers in batch["answers"]
        ],
        "image": [cap_pixels(img) for img in batch["image"]],
    }


def load_datasets(preset: config.SFTPreset):
    """The sft role for training and the dev role for eval-loss monitoring."""
    train = load_role(ROOT / "data", preset.datasets, "sft", preset.seed)
    dev = load_role(ROOT / "data", preset.datasets, "dev", preset.seed)
    columns = ["image", "question", "answers"]
    return (train.select_columns(columns).with_transform(to_sft_columns),
            dev.select_columns(columns).with_transform(to_sft_columns))


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
    # Pixel budget enforced in the data (cap_pixels), not processor config — keeps
    # every consumer's preprocessing identical (see train_grpo.py).
    processor = AutoProcessor.from_pretrained(preset.model)
    return model, processor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", required=True, choices=sorted(config.SFT_PRESETS))
    parser.add_argument("--model", default=None, help="override base model or pinned snapshot")
    parser.add_argument("--hardware", choices=config.HARDWARE_PROFILES, default="legacy")
    parser.add_argument("--max-steps", type=int, default=-1, help="cap steps (sanity runs)")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--save-total-limit", type=int, default=2,
                        help="keep this many saved checkpoints; 0 keeps all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    if args.save_total_limit < 0:
        parser.error("--save-total-limit must be nonnegative")

    preset = config.SFT_PRESETS[args.preset]
    if args.model:
        preset = replace(preset, model=args.model)
    preset = config.for_hardware(preset, args.hardware)
    train_ds, dev_ds = load_datasets(preset)
    if args.dry_run:
        print(f"hardware={args.hardware}; train={len(train_ds)}, dev={len(dev_ds)}")
        print(f"completion: {train_ds[0]['completion']}")
        return

    run_name = args.run_name or preset.name
    output_dir = args.run_dir / "checkpoints" if args.run_dir else ROOT / "checkpoints" / run_name
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
        completion_only_loss=True,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit" if preset.quantize_4bit else "adamw_torch",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=250,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit or None,
        dataloader_num_workers=2,  # WSL2: keep worker memory low
        dataloader_pin_memory=False,
        # The messages/image columns are produced on the fly by with_transform from
        # question/answers/image; the Trainer must not strip its inputs beforehand.
        remove_unused_columns=False,
        seed=preset.seed,
        report_to=[] if args.no_wandb else ["wandb"],
        run_name=run_name,
    )
    peft_config = LoraConfig(
        r=preset.lora_r,
        lora_alpha=preset.lora_alpha,
        lora_dropout=preset.lora_dropout,
        target_modules=list(preset.lora_targets),
        task_type="CAUSAL_LM",
    )
    write_run_config(output_dir, {
        "preset": config.dump(preset), "policy_model": preset.model,
        "overrides": {"hardware": args.hardware, "max_steps": args.max_steps},
        "data": provenance(ROOT / "data", preset.datasets),
        "samples": {"train": len(train_ds), "dev": len(dev_ds)},
    }, sft_config)
    model, processor = build_model(preset)
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=processor,
        peft_config=peft_config,
    )

    from scripts.hf_preempt_callback import JsonlMetrics, PreemptCallback, install_signal_flag

    install_signal_flag()
    trainer.add_callback(PreemptCallback())
    if args.run_dir:
        trainer.add_callback(JsonlMetrics(str(args.run_dir / "metrics.jsonl")))
    trainer.train(resume_from_checkpoint=last_complete_checkpoint(output_dir))
    trainer.save_model(str(output_dir))
    write_training_summary(trainer, output_dir)

    print(f"adapter saved -> {output_dir}")


if __name__ == "__main__":
    main()
