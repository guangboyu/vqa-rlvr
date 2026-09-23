# Method: the corrected campaign and how to rerun it

The campaign asks two questions with one fixed evaluation. What does RL with a verifiable
reward do to a model's accuracy under a short-answer prompt and under a reasoning prompt?
And does short-answer SFT before the same RL change the result? Every arm is evaluated on
the same test questions with both prompt templates and both scorer variants, so a change
in one view can be checked in the other.

## Arms

| Arm | Starting model | Supervision | What it tests |
|---|---|---|---|
| Instruct 2B, Instruct 8B | released Qwen3-VL Instruct | none (upstream post-training only) | the reference point for every arm |
| SFT | Instruct 2B | one epoch of short answers: answer and end-of-turn tokens | direct answer supervision |
| Instruct → GRPO | Instruct 2B / 8B | 300 GRPO updates, correctness reward | RL with a verifiable reward alone |
| SFT → GRPO | the SFT model | the same 300 GRPO updates | whether SFT first changes what RL adds |
| Instruct → gold-conditioned OPD | Instruct 2B | 300 on-policy distillation updates; the frozen Instruct model, shown the reference answer in its prompt, scores each student token | a dense, token-level learning signal on the same sampled responses |

"Instruct" is the released checkpoint before this project's fine-tuning; it has already
received upstream post-training. The SFT data are short answers, not reasoning
demonstrations. The gold-conditioned teacher has information the student lacks (the
reference answer), so its comparison with GRPO changes the signal and the conditioning
together.

All RL arms draw from the same pool with the same seed and the same update budget: 300
updates, each of four prompt groups of eight responses, for 1,200 prompt draws and 9,600
sampled responses. The recorded runs drew exactly the same 1,200 questions in the same
order. SFT → GRPO adds the SFT epoch on top, so its total exposure and compute are larger;
the comparison measures the value of adding SFT before the same RL, not a match at equal
compute. Every arm is one training seed.

## Data and model identity

Each dataset's split file in `data/splits/` lists the exact rows of every role. The
training roles hold 39,458 SFT questions (VQAv2 and GQA) and a 55,438-question RL pool
that includes the SFT questions alongside the original RL sources; the dev role holds 506
VQAv2 and GQA questions on whole held-out images, plus CLEVR's validation partition.
`python -m scripts.check_data` fails if any dev or test image appears in any training
role of any dataset. Images are compared by a hash of their decoded pixels and
dimensions, so re-encoded copies across sources are caught; near-duplicates and upstream
pretraining exposure are outside what an exact hash can check.

SFT and RL share training questions by design. SFT fits a supplied answer; RL samples new
responses to the same question. Both remain training data, and dev and test images stay
out of both stages. [SimpleRL-Zoo, Section 4](https://arxiv.org/html/2503.18892v3#S4)
uses the same pattern (SFT and RL data from the same problem sources); here the split
files and each run's prompt-draw log make the reuse exact. Of the 1,200 questions the RL
runs drew, 841 are also SFT training questions and 359 come from the RL-only sources.

`prepare_data.py` pins the five source revisions, and a fresh materialization from them
reproduced all ten recorded subsets byte for byte: the same question IDs, text, and image
bytes. `check_data` compares a new materialization with the split files row by row, so
any difference is reported by subset. Five duplicate
image-question pairs within training, one with conflicting count labels, were recorded and
kept in the frozen pool.

The Instruct models are pinned to Hub revisions; every evaluation record stores the
revision or, for a local merge, a hash of its weights.

```bash
uv run --frozen hf download Qwen/Qwen3-VL-2B-Instruct \
  --revision 89644892e4d85e24eaac8bacfd4f463576704203 \
  --local-dir checkpoints/base/Qwen3-VL-2B-Instruct
BASE_2B=checkpoints/base/Qwen3-VL-2B-Instruct
```

## Training recipe

**SFT.** One epoch, learning rate `1e-4`, effective batch 32, LoRA rank 16.

**GRPO.** Learning rate `1e-5`, eight responses per prompt at temperature 1, batch size 8
with accumulation 4, at most 640 completion tokens, no KL penalty. The pinned TRL 1.8
configuration uses DAPO token-level loss reduction, group reward scaling,
weighted-sum-then-normalize reward aggregation, token-level policy ratios clipped
symmetrically at 0.2, and vLLM sampling correction with `sequence_mask` and an upper
ratio of 3. Dynamic resampling of zero-variance groups is off, so this is not the
complete DAPO recipe. The settings live in [`vqar/config.py`](../../vqar/config.py) and are
saved with every run.

**Reward.** The correctness reward requires a complete `<answer>...</answer>` tag, then
scores the tagged answer with the normalized scorer, which falls back to exact match when
a question has fewer than two references. There is no separate format bonus in these
runs. Evaluation also accepts untagged answers and reports the dataset authors' scorer
beside the normalized one.

**OPD.** The same rollout geometry as GRPO. The learning signal is the teacher's token
log-probability minus the student's rollout-time log-probability. The correctness reward
is still logged, for diagnosis only; in the saved completion samples TRL's `advantage`
column is that logged diagnostic, and the teacher statistics are under `opd/*`.

**Hardware.** The recorded SFT run used an RTX PRO 6000 96 GB, both 2B GRPO arms A100 80
GB cards, and the OPD arm an RTX 6000. The Instruct → GRPO run reserved 36% of the A100 for
vLLM and the SFT → GRPO run 15%: a path-based size check read `8b` inside the Instruct
model's snapshot hash. The profiles now size memory from the preset name and reserve about
12 GB for either 2B arm (15% of an 80 GB card, 12.5% of a 96 GB card); the recorded
allocations stay in each run's metadata. The `24gb` profile runs the same 2B recipe with a
smaller rollout allocation.

## Train the arms

Run from the repository root with a new run directory for each run; an existing
compatible directory resumes from its last complete checkpoint. For a two-update check of
the whole path, see the [tutorial](../tutorial.md).

```bash
uv run --frozen python train_sft.py --preset sft_2b --hardware 96gb \
  --model "$BASE_2B" --save-steps 50 --run-dir runs/sft2 --no-wandb
uv run --frozen python scripts/merge_adapter.py runs/sft2/checkpoints \
  --base "$BASE_2B" --out checkpoints/merged/sharedpool_20260913_sft_2b

uv run --frozen python train_grpo.py --preset grpo_2b_main --hardware 80gb \
  --model "$BASE_2B" --max-steps 300 --save-steps 25 --run-dir runs/grpo2-base --no-wandb
uv run --frozen python scripts/merge_adapter.py runs/grpo2-base/checkpoints \
  --base "$BASE_2B" --out checkpoints/merged/sharedpool_20260913_grpo_2b_base

uv run --frozen python train_grpo.py --preset grpo_2b_main --hardware 80gb \
  --model checkpoints/merged/sharedpool_20260913_sft_2b --max-steps 300 --save-steps 25 \
  --run-dir runs/grpo2-sft --no-wandb
uv run --frozen python scripts/merge_adapter.py runs/grpo2-sft/checkpoints \
  --base checkpoints/merged/sharedpool_20260913_sft_2b \
  --out checkpoints/merged/sharedpool_20260913_grpo_2b_sft

uv run --frozen python train_opd.py --preset opd_self_2b_main --hardware 96gb \
  --model "$BASE_2B" --max-steps 300 --save-steps 25 --save-total-limit 0 \
  --run-dir runs/opd2-self --no-wandb
uv run --frozen python scripts/merge_adapter.py runs/opd2-self/checkpoints \
  --base "$BASE_2B" --out checkpoints/merged/sharedpool_20260913_opd_2b_self
```

`grpo_2b_main` trains with the correctness reward alone. The 8B arm uses the bf16
`grpo_8b_fmt0` preset on an 80 or 96 GB card, with the same learning rate, group size,
batch geometry, and 300-update budget as the 2B arms:

```bash
uv run --frozen hf download Qwen/Qwen3-VL-8B-Instruct \
  --revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --local-dir checkpoints/base/Qwen3-VL-8B-Instruct
BASE_8B=checkpoints/base/Qwen3-VL-8B-Instruct
uv run --frozen python train_grpo.py --preset grpo_8b_fmt0 --hardware 96gb \
  --model "$BASE_8B" --max-steps 300 --save-steps 25 --run-dir runs/grpo8-base --no-wandb
uv run --frozen python scripts/merge_adapter.py runs/grpo8-base/checkpoints \
  --base "$BASE_8B" --out checkpoints/merged/sharedpool_20260913_grpo_8b_base
```

The CPU merge of an 8B model needs about 64 GB of host RAM. When a run completes, it
evaluates every saved adapter on the dev tier with both prompts; those records are the
run's learning curve.

## Evaluate

`experiments.toml` declares the arms and the protocols each is evaluated under: every
dataset's full test role, with the short and the reasoning prompt, greedy decoding, and
the prompt template's token budget (64 and 512). Once the merged models are on disk,

```bash
uv run --frozen python -m scripts.report --check   # declared cells without a record
uv run --frozen python -m scripts.report --run     # evaluate them, then the tables
```

evaluates whatever is missing and regenerates every table. One cell by hand:

```bash
uv run --frozen python evaluate.py --model checkpoints/merged/sharedpool_20260913_sft_2b \
  --label sft2 --prompts short reasoning
```

For image sensitivity, evaluate the original-image cell first, then repeat it with
`--image-mode black` or `--image-mode swapped` at the same tier; each changed-image record
pairs every answer with the original one. The references stay those of the original
image, so a changed-image score measures agreement with them, not accuracy on the new
image. The recorded interventions used the first 128 test questions of each dataset.

## Inspect the saved evidence without a GPU

```bash
uv run --frozen python -m scripts.report
uv run --frozen python -m scripts.summarize_training \
  results/baselines/training/grpo2-base results/baselines/training/grpo2-sft \
  results/baselines/training/grpo8-base results/baselines/training/opd2-self \
  --output results/baselines/training-exposure.json
uv run --frozen python -m scripts.make_gallery
```

The report reads the stored records; each names its protocol, so cells evaluated under
different questions, prompts, scorers, or decoding never share a column. The campaign's
records were written from its saved outputs by `python -m scripts.migrate_campaign`. With
the saved predictions present, rerunning it verifies the stored records, and
`pytest tests/test_legacy.py` recomputes every campaign number from them. The training summary checks the actual prompt order, counts reward variation
per update, and embeds each run's diagnostics; regenerating it needs the materialized
training data, and its output is already committed.

## Checkpoint storage

The trainers keep the latest two full checkpoints by default; `--save-total-limit 0`
keeps every one, with optimizer and RNG state. The adapter of every scheduled save is
kept either way, under `checkpoints/adapters/`, for the learning curve. After a run,
archive useful intermediate checkpoints with their run configuration, metrics, and
provenance, and verify the copy before deleting a working one. A final adapter alone
cannot resume training.
