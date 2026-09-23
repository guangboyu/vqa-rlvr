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
from dataclasses import replace

from vqar import config
from vqar.rl import dry_run, grpo_config, load_rl_dataset, make_chat_rollout, resolve_model
from vqar.splits import provenance
from vqar.training import (
    ROOT,
    add_common_arguments,
    check_rl_precision,
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
    add_common_arguments(parser, config.OPD_PRESETS, save_steps=100)
    parser.add_argument("--teacher", default=None, help="override teacher model/checkpoint")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="seed replication arms")
    args = parse_arguments(parser)

    preset = resolve_preset(config.OPD_PRESETS, args)
    if args.seed is not None:
        preset = replace(preset, seed=args.seed)

    if args.dry_run:
        print(f"hardware={args.hardware}; recipe={config.RL_RECIPE}")
        dry_run(preset, rewards=("correctness",))
        return

    check_rl_precision(preset)

    from transformers import AutoModelForImageTextToText, AutoProcessor

    from vqar.rewards import REWARDS

    run_name = args.run_name or preset.name
    student_path = resolve_model(args.model or preset.model)
    teacher_ref = args.teacher or preset.teacher
    teacher_path = teacher_ref if teacher_ref == "self" else resolve_model(teacher_ref)
    max_steps = args.max_steps or preset.max_steps

    processor = AutoProcessor.from_pretrained(student_path, padding_side="left")

    if teacher_ref == "self":
        teacher = None  # gold-conditioned base policy scores its own rollouts
    else:
        teacher = AutoModelForImageTextToText.from_pretrained(
            teacher_path, device_map={"": 0}, **load_kwargs(preset.teacher_quantize_4bit)
        )
        check_teacher_compat(student_path, teacher_path, teacher.config)

    output = checkpoint_dir(args.run_dir, run_name)
    training_args = grpo_config(
        preset, output, run_name, load_kwargs(preset.quantize_4bit), max_steps=max_steps,
        beta=0.0,
        reward_weights=[1.0],  # logging only; advantages are replaced by OPD
        batch_size=preset.per_device_batch_size, grad_accum=preset.gradient_accumulation,
        save_steps=args.save_steps, save_total_limit=args.save_total_limit,
        report_to=wandb_reporting(args.no_wandb),
    )
    peft_config = lora_config(preset)
    write_run_config(output, {
        "preset": config.dump(preset), "policy_model": student_path,
        "teacher_model": teacher_path,
        "teacher_log_softmax_dtype": "float32",
        "data": provenance(ROOT / "data", preset.datasets),
        "overrides": {"hardware": args.hardware, "max_steps": max_steps},
    }, training_args)
    trainer_cls = make_opd_trainer_cls()
    trainer = trainer_cls(
        model=student_path,
        reward_funcs=[REWARDS["correctness"]],
        args=training_args,
        train_dataset=load_rl_dataset(preset),
        processing_class=processor,
        peft_config=peft_config,
        rollout_func=make_chat_rollout(),
        teacher_model=teacher,
        teacher_chunk_size=preset.teacher_chunk_size,
    )
    completed = run_training(trainer, output, args.run_dir, log_draws=True)
    if completed and not args.no_dev_eval:
        evaluate_checkpoints_next(output)


if __name__ == "__main__":
    main()
