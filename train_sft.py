"""Supervised fine-tuning: TRL SFTTrainer + PEFT LoRA (2B, bf16) / QLoRA (8B, NF4).

Trains on the sft role of VQAv2 and GQA (their split files hold dev and test images out)
with the fixed `short` prompt template. The vision tower stays frozen; adapters target the LM only.

Usage:
    uv run python train_sft.py --preset sft_2b
    uv run python train_sft.py --preset sft_8b
    uv run python train_sft.py --preset sft_2b --max-steps 20   # sanity run
"""

import argparse

from vqar import config
from vqar.data import cap_pixels
from vqar.pools import load_pool
from vqar.pools import provenance as pool_provenance
from vqar.registry import PROMPTS
from vqar.splits import load_role, provenance
from vqar.training import (
    ROOT,
    add_common_arguments,
    checkpoint_dir,
    evaluate_checkpoints_next,
    load_kwargs,
    lora_config,
    parse_arguments,
    resolve_preset,
    run_training,
    wandb_reporting,
    write_run_config,
)


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


def load_datasets(preset: config.SFTPreset, pool: str | None = None):
    """SFT rows, or declared RL-pool questions supervised with their original labels."""
    train = (load_pool(ROOT / "data", preset.datasets, preset.seed, pool)
             if pool else load_role(ROOT / "data", preset.datasets, "sft", preset.seed))
    dev = load_role(ROOT / "data", preset.datasets, "dev", preset.seed)
    columns = ["image", "question", "answers"]
    return (train.select_columns(columns).with_transform(to_sft_columns),
            dev.select_columns(columns).with_transform(to_sft_columns))


def build_model(preset: config.SFTPreset):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(
        preset.model, **load_kwargs(preset.quantize_4bit)
    )
    # Pixel budget enforced in the data (cap_pixels), not processor config — keeps
    # every consumer's preprocessing identical (see vqar.rl).
    processor = AutoProcessor.from_pretrained(preset.model)
    return model, processor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, config.SFT_PRESETS, save_steps=250)
    parser.add_argument("--max-steps", type=int, default=-1, help="cap steps (sanity runs)")
    parser.add_argument("--pool", default=None,
                        help="supervise the exact questions in a declared RL pool")
    args = parse_arguments(parser)

    preset = resolve_preset(config.SFT_PRESETS, args)
    train_ds, dev_ds = load_datasets(preset, args.pool)
    if args.dry_run:
        print(f"hardware={args.hardware}; train={len(train_ds)}, dev={len(dev_ds)}")
        if args.pool:
            print(f"pool: {pool_provenance(ROOT / 'data', args.pool)}")
        print(f"completion: {train_ds[0]['completion']}")
        return

    from transformers import set_seed
    from trl import SFTConfig, SFTTrainer

    run_name = args.run_name or preset.name
    output = checkpoint_dir(args.run_dir, run_name)
    sft_config = SFTConfig(
        output_dir=str(output),
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
        report_to=wandb_reporting(args.no_wandb),
        run_name=run_name,
    )
    peft_config = lora_config(preset)
    write_run_config(output, {
        "preset": config.dump(preset), "policy_model": preset.model,
        "overrides": {"hardware": args.hardware, "max_steps": args.max_steps},
        "data": provenance(ROOT / "data", preset.datasets),
        "samples": {"train": len(train_ds), "dev": len(dev_ds)},
        **({"pool": pool_provenance(ROOT / "data", args.pool)} if args.pool else {}),
    }, sft_config)
    # TRL constructs PEFT before Trainer sets its seed, so seed the adapter initialization too.
    set_seed(preset.seed)
    model, processor = build_model(preset)
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=processor,
        peft_config=peft_config,
    )
    completed = run_training(trainer, output, args.run_dir)
    if completed and not args.no_dev_eval:
        evaluate_checkpoints_next(output)


if __name__ == "__main__":
    main()
