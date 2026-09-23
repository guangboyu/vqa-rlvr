"""Distill student rollouts using token scores from a frozen teacher.

The external teacher sees the student's prompt. The self-teacher is the frozen
base model with the reference answer added to its prompt. Both score student
completion tokens; detached teacher_logp - rollout_logp advantages replace GRPO's
group-relative reward. The clipped token-ratio loss and rollout path are shared.
Teacher capacity and conditioning must be considered when comparing these arms.
Correctness is logged for diagnosis and does not drive the OPD gradient.

Usage:
    uv run python train_opd.py --preset opd_2b_main --dry-run   # no GPU, no model
    uv run python train_opd.py --preset opd_2b_main --max-steps 20 --run-name opd_sanity
    uv run python train_opd.py --preset opd_2b_main
"""

import argparse
import copy
import math
import os
from dataclasses import replace
from pathlib import Path

from train_grpo import (
    dry_run,
    load_rl_dataset,
    make_chat_rollout,
    resolve_model,
)
from vqar import config
from vqar.splits import provenance
from vqar.training import (
    check_rl_precision,
    last_complete_checkpoint,
    log_prompt_draws,
    write_run_config,
    write_training_summary,
)

os.environ.setdefault("WANDB_PROJECT", "vqa-rlvr")

ROOT = Path(__file__).parent


def check_teacher_compat(student_id: str, teacher_id: str, teacher_config) -> None:
    """The teacher consumes input_ids and pixel_values produced by the student's
    processor, so vocab and vision patch geometry must match exactly."""
    from transformers import AutoConfig, AutoProcessor

    s = AutoConfig.from_pretrained(student_id)
    t = teacher_config
    pairs = [
        ("text_config.vocab_size", s.text_config.vocab_size, t.text_config.vocab_size),
        ("image_token_id", s.image_token_id, t.image_token_id),
        ("vision_config.patch_size", s.vision_config.patch_size, t.vision_config.patch_size),
        (
            "vision_config.spatial_merge_size",
            s.vision_config.spatial_merge_size,
            t.vision_config.spatial_merge_size,
        ),
    ]
    for name, sv, tv in pairs:
        if sv != tv:
            raise ValueError(f"student/teacher mismatch on {name}: {sv} vs {tv}")
    student = AutoProcessor.from_pretrained(student_id)
    teacher = AutoProcessor.from_pretrained(teacher_id)
    if student.tokenizer.get_vocab() != teacher.tokenizer.get_vocab():
        raise ValueError("Student and teacher token IDs do not share the same vocabulary.")
    for name in ("image_mean", "image_std", "rescale_factor", "patch_size", "temporal_patch_size"):
        if getattr(student.image_processor, name) != getattr(teacher.image_processor, name):
            raise ValueError(f"Student and teacher image processors differ on {name}.")


def with_reference_answer(prompt, answer: str):
    """Add gold context to the teacher's copy, never to the student's prompt."""
    messages = copy.deepcopy(prompt)
    reference = f"\nReference answer: {answer}"
    if isinstance(messages[-1]["content"], str):
        messages[-1]["content"] += reference
    else:
        messages[-1]["content"].append({"type": "text", "text": reference})
    return messages


def make_opd_trainer_cls():
    """Deferred import wrapper so --dry-run works without torch/trl installed."""
    import torch
    from trl import GRPOTrainer

    class OPDTrainer(GRPOTrainer):
        """GRPO subclass whose learning signal is per-token reverse KL to a teacher.

        Overrides only rollout post-processing: after the parent generates and
        scores completions, the teacher computes per-token logprobs on the student's
        completion tokens and the (B, T) advantage A_t = teacher_logp - rollout_logp
        replaces the parent's scalar one. Everything else (vLLM colocate, sleep
        mode, LoRA, clipped-ratio loss) is inherited unchanged.

        Two teacher modes:
        - external (teacher_model set): a frozen larger model scores the identical
          (input_ids, pixel_values) batch — valid because student and teacher share
          tokenizer and patch geometry (asserted at startup).
        - self (teacher_model None): gold-conditioned self-teacher — the
          BASE policy (adapters disabled, so no self-amplification loop) scores the
          same completions under a prompt augmented with the reference answer.
          The teacher gets extra information; this is not a pure capacity comparison.
        """

        def __init__(self, *args, teacher_model=None, teacher_chunk_size=2, **kwargs):
            super().__init__(*args, **kwargs)
            self.teacher_model = (
                teacher_model.eval().requires_grad_(False) if teacher_model is not None else None
            )
            if self.teacher_model is not None:
                # Keep teacher log-softmax in fp32, matching the student's scoring path.
                self.teacher_model.get_output_embeddings().register_forward_hook(
                    lambda _module, _inputs, output: output.float()
                )
            self.teacher_chunk_size = teacher_chunk_size
            if teacher_chunk_size < 1:
                raise ValueError("teacher_chunk_size must be positive")

        def _self_teacher_logps(self, inputs, out):
            """Score completions with the base policy under a gold-augmented prompt."""
            from trl.data_utils import prepare_multimodal_messages

            convs = []
            for x in inputs:
                msgs = with_reference_answer(x["prompt"], x["answers"][0])
                convs.append(prepare_multimodal_messages(msgs, images=[x["image"]]))
            tokenizer = self.processing_class.tokenizer
            old_side = tokenizer.padding_side
            tokenizer.padding_side = "left"  # completions must align at the boundary
            try:
                batch = self.processing_class.apply_chat_template(
                    convs,
                    add_generation_prompt=True,
                    tokenize=True,
                    padding=True,
                    return_dict=True,
                    return_tensors="pt",
                )
            finally:
                tokenizer.padding_side = old_side

            device = self.accelerator.device
            completion_ids, completion_mask = out["completion_ids"], out["completion_mask"]
            input_ids = torch.cat([batch["input_ids"].to(device), completion_ids], dim=1)
            attention_mask = torch.cat([batch["attention_mask"].to(device), completion_mask], dim=1)
            mm_token_type_ids = batch.get("mm_token_type_ids")
            if mm_token_type_ids is not None:
                mm_token_type_ids = torch.cat(
                    [mm_token_type_ids.to(device), torch.zeros_like(completion_ids)], dim=1
                )
            image_grid_thw = batch.get("image_grid_thw")
            model = self.accelerator.unwrap_model(self.model)
            with torch.no_grad(), model.disable_adapter():
                logps, _, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    input_ids,
                    attention_mask,
                    completion_ids.size(1),
                    # TRL 1.8's image-grid chunker requires complete chunks.
                    batch_size=math.gcd(self.teacher_chunk_size, len(convs)),
                    pixel_values=batch["pixel_values"].to(device),
                    image_grid_thw=None if image_grid_thw is None else image_grid_thw.to(device),
                    num_images=[1] * len(convs),
                    mm_token_type_ids=mm_token_type_ids,
                )
            return logps

        def _generate_and_score_completions(self, inputs):
            out = super()._generate_and_score_completions(inputs)
            if "old_per_token_logps" not in out:
                raise RuntimeError(
                    "OPD needs rollout-time policy logps; run with vLLM importance "
                    "sampling correction enabled (the default)."
                )
            if self.teacher_model is None:
                teacher_logps = self._self_teacher_logps(inputs, out)
            else:
                input_ids = torch.cat([out["prompt_ids"], out["completion_ids"]], dim=1)
                attention_mask = torch.cat([out["prompt_mask"], out["completion_mask"]], dim=1)
                logits_to_keep = out["completion_ids"].size(1)
                with torch.no_grad():
                    teacher_logps, _, _ = self._get_per_token_logps_and_entropies(
                        self.teacher_model,
                        input_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=math.gcd(self.teacher_chunk_size, len(inputs)),
                        pixel_values=out.get("pixel_values"),
                        image_grid_thw=out.get("image_grid_thw"),
                        num_images=out.get("num_images"),
                        token_type_ids=out.get("token_type_ids"),
                        mm_token_type_ids=out.get("mm_token_type_ids"),
                    )
            mask = out["completion_mask"].bool()
            out["advantages"] = (teacher_logps - out["old_per_token_logps"]).detach() * mask

            mode = "train" if self.model.training else "eval"
            reverse_kl = (out["old_per_token_logps"] - teacher_logps)[mask]
            self._metrics[mode]["opd/reverse_kl_per_token"].append(reverse_kl.mean().item())
            self._metrics[mode]["opd/teacher_logp_per_token"].append(
                teacher_logps[mask].mean().item()
            )
            return out

    return OPDTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", required=True, choices=sorted(config.OPD_PRESETS))
    parser.add_argument("--model", default=None, help="override student model/checkpoint")
    parser.add_argument("--hardware", choices=config.HARDWARE_PROFILES, default="legacy")
    parser.add_argument("--teacher", default=None, help="override teacher model/checkpoint")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="seed replication arms")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--run-dir", default=None,
                        help="cluster mode: checkpoints + metrics.jsonl under this dir "
                        "(auto-resumes from its last checkpoint; see hpc-env/AGENTS.md)")
    parser.add_argument("--save-steps", type=int, default=100,
                        help="checkpoint cadence; on free partitions size to <=15 min")
    parser.add_argument("--save-total-limit", type=int, default=2,
                        help="keep this many saved checkpoints; 0 keeps all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    if args.save_total_limit < 0:
        parser.error("--save-total-limit must be nonnegative")

    preset = config.OPD_PRESETS[args.preset]
    if args.model:
        preset = replace(preset, model=args.model)
    preset = config.for_hardware(preset, args.hardware)
    if args.seed is not None:
        preset = replace(preset, seed=args.seed)

    if args.dry_run:
        print(f"hardware={args.hardware}; recipe={config.RL_RECIPE}")
        dry_run(preset, rewards=("correctness",))
        return

    check_rl_precision(preset)

    import torch
    from peft import LoraConfig
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        BitsAndBytesConfig,
    )
    from trl import GRPOConfig


    run_name = args.run_name or preset.name
    student_path = resolve_model(args.model or preset.model)
    teacher_ref = args.teacher or preset.teacher
    teacher_path = teacher_ref if teacher_ref == "self" else resolve_model(teacher_ref)
    max_steps = args.max_steps or preset.max_steps

    processor = AutoProcessor.from_pretrained(student_path, padding_side="left")

    if teacher_ref == "self":
        teacher = None  # gold-conditioned base policy scores its own rollouts
    else:
        teacher_kwargs = dict(dtype=torch.bfloat16, device_map={"": 0})
        if preset.teacher_quantize_4bit:
            teacher_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        teacher = AutoModelForImageTextToText.from_pretrained(teacher_path, **teacher_kwargs)
        check_teacher_compat(student_path, teacher_path, teacher.config)

    from vqar.rewards import REWARDS

    student_kwargs = dict(dtype=torch.bfloat16)
    if preset.quantize_4bit:
        student_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    local_out = ROOT / "checkpoints" / run_name
    output_dir = Path(args.run_dir) / "checkpoints" if args.run_dir else local_out
    opd_config = GRPOConfig(
        **config.RL_RECIPE,
        output_dir=str(output_dir),
        run_name=run_name,
        model_init_kwargs=student_kwargs,
        num_generations=preset.num_generations,
        per_device_train_batch_size=preset.per_device_batch_size,
        gradient_accumulation_steps=preset.gradient_accumulation,
        learning_rate=preset.learning_rate,
        max_steps=max_steps,
        max_completion_length=preset.max_completion_length,
        temperature=1.0,
        beta=0.0,
        reward_weights=[1.0],  # logging only; advantages are replaced by OPD
        use_vllm=True,
        vllm_mode="colocate",
        vllm_enable_sleep_mode=True,
        vllm_gpu_memory_utilization=preset.vllm_gpu_memory_utilization,
        vllm_max_model_length=2048,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit" if preset.quantize_4bit else "adamw_torch",
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit or None,
        dataloader_num_workers=2,
        dataloader_pin_memory=False,
        seed=preset.seed,
        report_to=[] if args.no_wandb else ["wandb"],
        log_completions=True,
        num_completions_to_print=2,
    )
    peft_config = LoraConfig(
        r=preset.lora_r,
        lora_alpha=preset.lora_alpha,
        target_modules=list(preset.lora_targets),
        task_type="CAUSAL_LM",
    )
    write_run_config(output_dir, {
        "preset": config.dump(preset), "policy_model": student_path,
        "teacher_model": teacher_path,
        "teacher_log_softmax_dtype": "float32",
        "data": provenance(ROOT / "data", preset.datasets),
        "overrides": {"hardware": args.hardware, "max_steps": max_steps},
    }, opd_config)
    trainer_cls = make_opd_trainer_cls()
    trainer = trainer_cls(
        model=student_path,
        reward_funcs=[REWARDS["correctness"]],
        args=opd_config,
        train_dataset=load_rl_dataset(preset),
        processing_class=processor,
        peft_config=peft_config,
        rollout_func=make_chat_rollout(preset),
        teacher_model=teacher,
        teacher_chunk_size=preset.teacher_chunk_size,
    )

    # Preemption contract (free HPC3 partitions CANCEL with no warning): periodic
    # checkpoints above + signal-triggered save + metrics.jsonl + auto-resume.
    # Harmless locally — a fresh output_dir has no checkpoint to resume from.

    from scripts.hf_preempt_callback import JsonlMetrics, PreemptCallback, install_signal_flag

    install_signal_flag()
    trainer.add_callback(PreemptCallback())
    if args.run_dir:
        trainer.add_callback(JsonlMetrics(str(Path(args.run_dir) / "metrics.jsonl")))
        log_prompt_draws(trainer, Path(args.run_dir) / "prompt_draws.jsonl")
    last = last_complete_checkpoint(output_dir) if output_dir.is_dir() else None
    if last:
        print(f"resuming from {last}")
    trainer.train(resume_from_checkpoint=last)
    trainer.save_model(str(output_dir))
    write_training_summary(trainer, output_dir)

    print(f"adapter saved -> {output_dir}")


if __name__ == "__main__":
    main()
