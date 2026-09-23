# Hardware profiles

The 24 GB path retains the existing small-model recipes and 8B QLoRA SFT. Larger GPUs
use separate profiles selected with `--hardware`; omitting it retains the legacy
memory settings. Short numerical checks are complete on the actual RTX 4090
24 GB, RTX PRO 6000 96 GB, and A100 80 GB cards.

| Profile | SFT | GRPO / self-distillation | Rollout engine memory |
|---|---|---|---|
| `24gb` | 2B bf16 LoRA; 8B NF4 QLoRA | 2B bf16 | Existing preset allocation |
| `80gb` | 2B or 8B bf16 LoRA | 2B or 8B bf16 | About 12 GB / 29 GB |
| `96gb` | 2B or 8B bf16 LoRA | 2B or 8B bf16 | About 12 GB / 29 GB |

Profiles preserve learning rates, batch sizes, gradient accumulation, group size,
and completion limits. Changing NF4 to bf16 also changes the optimizer from
paged 8-bit AdamW to regular AdamW. Treat that as a distinct precision configuration.
Select a preset that matches the model size. Memory profiles follow the preset's
`2b`/`8b` name, so custom checkpoint paths and snapshot hashes cannot change the allocation.
The old NF4 **student** GRPO presets are retained for historical reference but
rejected at launch: their TRL/vLLM weight synchronization produced corrupted rollouts.
An NF4 external teacher does not use that synchronization path.

## Measured local checks

These workloads completed on an RTX 4090 with the locked environment. Peaks are
PyTorch allocations, including the numerical probes; they are not total device
memory measurements. The external-teacher path leaves little spare memory.

| Workload | Peak PyTorch allocation | Result |
|---|---:|---|
| 2B SFT, bf16 | 9.73 GiB | 2 updates completed |
| 8B SFT, NF4 | 15.76 GiB | 2 updates completed |
| 2B GRPO, bf16 | 16.76 GiB | 2 updates completed |
| 2B gold-conditioned self-teacher | 18.79 GiB | 2 updates completed |
| 2B OPD with 8B NF4 teacher | 21.67 GiB | 2 updates completed |

Each RL update used 32 responses in four groups of eight, with the existing
640-token completion limit. Small checks do not establish full-run convergence.

## Measured 80/96 GB checks

All four paths completed two updates on both an RTX PRO 6000 Blackwell and an
A100 80GB PCIe. SFT loss and gradients, GRPO prompt IDs and group advantages,
and both teachers' token log-probabilities matched independent checks. SFT also
completed its dev evaluation. Source hashes, probes, and Slurm job IDs are in
[`results/validation.json`](../results/validation.json).

| Workload | RTX PRO 6000 peak | A100 peak |
|---|---:|---:|
| 8B SFT, bf16 | 28.60 GiB | 28.59 GiB |
| 8B GRPO, bf16 | 52.54 GiB | 50.98 GiB |
| 8B gold-conditioned self-teacher | 52.70 GiB | 51.26 GiB |
| 2B OPD with 8B NF4 teacher | 45.14 GiB | 46.17 GiB |

These are peak PyTorch allocations, which exclude some non-PyTorch memory.
The tested workloads fit on the actual cards; longer completions or different
batch sizes require new measurements. Shared-storage waits dominated job wall
time, so these runs are not throughput benchmarks.

## CPU checks

Use Python 3.12 and the committed `uv.lock`: `uv sync --frozen`. Materialize the
datasets with `uv run --frozen python prepare_data.py`, then run:

```bash
uv run --frozen python -m scripts.check_data
uv run --frozen python train_sft.py --preset sft_2b --hardware 24gb --dry-run --no-wandb
uv run --frozen python train_grpo.py --preset grpo_2b_main --hardware 24gb --dry-run
uv run --frozen python train_opd.py --preset opd_self_8b_main --hardware 80gb --dry-run
uv run --frozen pytest
```

For the current materialized data, expect 39,458 SFT training examples, 506 dev
examples, and 55,438 mixed RL prompts. Both RL starting points reuse the SFT
training examples alongside the original RL sources. Dev and evaluation images
are excluded from both stages. `check_data` validates the versioned split files in
`data/splits/` (one per dataset) and leaves the source datasets intact; a newly
registered dataset gets its own file. Historical runs that predate the held-out
splits reproduce from the archived July code, not from this tree.

## Small GPU checks

Use a fresh run directory. These commands are validation runs, not benchmark results.

```bash
uv run --frozen python train_sft.py --preset sft_2b --hardware 24gb \
  --max-steps 2 --save-steps 1 --run-dir runs/sft-smoke --no-wandb
uv run --frozen python train_grpo.py --preset grpo_2b_main --hardware 24gb \
  --model Qwen/Qwen3-VL-2B-Instruct --max-steps 2 \
  --save-steps 1 --run-dir runs/grpo-smoke --no-wandb
```

For an A100 80 GB or RTX 6000 96 GB, select `80gb` or `96gb` with `sft_8b`,
`grpo_8b_fmt0`, or `opd_self_8b_main`. An RTX 6000 profile does not simulate a
24 GB memory ceiling: fit claims require measurement on the actual card.

Every training entrypoint writes resolved TRL arguments, dependency versions,
data-split identity, and source hashes before training. RL runs with `--run-dir`
also save actual question IDs and scores in `prompt_draws.jsonl`; use the recorded
step to distinguish replayed draws after interruption from completed updates.
A compatible interrupted
run resumes from its latest checkpoint. Changed settings require a new directory.
Short smoke checks do not establish robustness to an interruption during checkpoint writing.

A local abrupt-exit drill stopped GRPO immediately after a completed checkpoint
at update 1. The normal resume path restored the adapter and training state and
completed update 2; prompt IDs and group advantages still matched. The vLLM
engine's sampling state is not included in HF checkpoints, so a resumed run is
not promised to generate bitwise-identical responses. Completion markers protect
against incomplete saves, not disk failure. Multi-hour Slurm runs still need the
bounded successor mechanism in the site's operating manual.

Evaluation writes atomic batches under `results/progress/`. If interrupted, rerun
its original command with `--resume`; model, data, scoring, decoding, and source
identity must match. Completed batches are reused. Final prediction files retain
output-token counts and stop reasons. For a small paired image-sensitivity check,
use the same `--limit` and separate run IDs with `--image-mode original`, `black`,
or `swapped`. A swap uses a different image from that fixed evaluation subset;
this measures prediction sensitivity, not causal isolation of the image encoder.
