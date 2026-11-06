# Method: the final recipe and how to rerun it

The study asks three questions with one fixed evaluation. What does RL with a verifiable
reward do to a model's accuracy under a short-answer prompt and under a reasoning prompt?
Does short-answer SFT before the same RL change the result? And which standard changes to
the RL recipe matter here? Every arm is evaluated on the same test questions with both
prompt templates and both scorer variants, so a change in one view can be checked in the
other. The recipe below was frozen after one-change experiments against a fixed baseline
([recipe experiments](../negatives.md#recipe-experiments)); every headline number comes from arms trained
under it.

## Arms

| Arm | Starting model | Supervision | What it tests |
|---|---|---|---|
| Instruct 2B, Instruct 8B | released Qwen3-VL Instruct | none (upstream post-training only) | the reference point for every arm |
| SFT 2B, SFT 8B | Instruct 2B / 8B | one epoch of short answers: answer and end-of-turn tokens | direct answer supervision |
| Instruct → GRPO | Instruct 2B / 8B | 1,000 GRPO updates under the reasoning prompt, correctness reward | RL with a verifiable reward alone |
| SFT → GRPO | the SFT model | the same 1,000 updates | whether SFT first changes what RL adds |
| Instruct → short-prompt GRPO, SFT → short-prompt GRPO | Instruct / SFT, 2B and 8B | the same 1,000 updates under the short prompt, rewarding the whole trimmed answer | RL at the short-answer ceiling, without reasoning |
| Instruct → gold-conditioned OPD | Instruct 2B | 1,000 on-policy distillation updates; the frozen Instruct model, shown the reference answer in its prompt, scores each student token | a dense, token-level learning signal on the same sampled responses |

"Instruct" is the released checkpoint before this project's fine-tuning; it has already
received upstream post-training. The SFT data are short answers, not reasoning
demonstrations. The gold-conditioned teacher has information the student lacks (the
reference answer), so its comparison with GRPO changes the signal and the conditioning
together.

All RL arms draw from the same pool with the same seed and the same update budget: 1,000
updates, each of four prompt groups of eight responses, for 4,000 prompt draws and 32,000
sampled responses. The recorded runs drew exactly the same questions in the same order
(`results/training-exposure.json` records one prompt-order hash for all nine). SFT → GRPO
adds the SFT epoch on top, so its total exposure and compute are larger; the comparison
measures the value of adding SFT before the same RL, not a match at equal compute. Every
arm is one training seed. The 2B SFT model is the corrected campaign's: its recipe did not
change.

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
files and each run's prompt-draw log make the reuse exact. Of the 4,000 questions the RL
runs drew, 2,836 are also SFT training questions and 1,164 come from the RL-only sources.

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

**SFT.** One epoch, LoRA rank 16, effective batch 32; learning rate `1e-4` at 2B and
`5e-5` at 8B (bf16 LoRA on an 80 or 96 GB card; `1e-4` damaged the 8B's counting and
instruction following in the July runs).

**GRPO.** Learning rate `1e-5` with linear decay to 0 over the 1,000 updates, eight
responses per prompt at temperature 1, batch size 8 with accumulation 4 (four questions
per update), at most 640 completion tokens (64 for short-prompt GRPO), no KL penalty,
LoRA rank 16 on the language model's attention and MLP projections. The pinned TRL 1.8
configuration uses DAPO token-level loss reduction, group reward scaling,
weighted-sum-then-normalize reward aggregation, token-level policy ratios clipped
symmetrically at 0.2, and vLLM sampling correction with `sequence_mask` and an upper
ratio of 3. With one gradient pass per generated batch the policy ratio is exactly 1 when
the loss is computed, so the PPO clip never binds: the logged clip fraction stays at bf16
noise (below 0.001) in every run, and clip-higher variants would change nothing here.
Dynamic resampling of zero-variance groups is off (see refilling dead groups below), so
this is not the complete DAPO recipe. Rollouts are served by vLLM from the unmerged adapter
(`--rollout-lora`, next section), and on 80/96 GB cards vLLM stays awake between rollouts
(`--no-vllm-sleep`), which changes speed only. The settings live in
[`vqar/config.py`](../../vqar/config.py) and are saved with every run.

**Reward.** The correctness reward requires a complete `<answer>...</answer>` tag, then
scores the tagged answer with the normalized scorer, which falls back to exact match when
a question has fewer than two references. There is no separate format bonus in these
runs. Evaluation also accepts untagged answers and reports the dataset authors' scorer
beside the normalized one. The tag requirement is the reasoning template's answer rule
(`vqar/answers.py`): `train_grpo.py --rollout-prompt short` trains under the short template
instead, whose rule scores the whole trimmed response. `--pool <name>` draws prompts from a
declared pool (`scripts/build_pool.py`, ADR 0003)
rather than the whole rl roles; the final runs use the whole rl roles.

**Refilling dead groups (opt-in).** `train_grpo.py --refill-rounds N` turns on DAPO's
dynamic sampling. A group is dead when all eight of its answers get the same total reward,
so it gives GRPO no gradient. After scoring an update's four groups, the trainer drops the
dead ones, draws as many replacement questions from the same pool, samples and scores
them, and repeats, for at most `N` extra rounds per update. The update then trains on the
live groups. If the cap is reached or the pool runs out, the earliest dead groups (the
update's own questions first) fill the batch, so it always holds four groups. Replacements
come from a separate stream for each update, seeded by the run seed and the update number
([`vqar/pools.py`](../../vqar/pools.py)). The stream skips the update's own questions. A
resumed update draws the same replacements, so checkpoints carry no extra state. Each extra
round syncs the policy weights into vLLM again before sampling. Logging:

- `prompt_draws.jsonl` gets one record per round, with a `round` number and a `kept` flag
  for each answer.
- The trainer logs `refill/attempted_groups`, `refill/live_groups`, and `refill/rounds`.
- The per-update diagnostics report the reward over the kept answers, plus attempted
  groups, attempted rollouts, and rounds.

TRL's reward, zero-variance share, and advantages describe the batch that was trained on.
Completion-length and token counts include every attempted rollout. Refilling changes how
many rollouts each update generates, so compute per update is no longer fixed. Compare a
refilling run by attempted rollouts and GPU hours, not only by updates. Every preset
leaves this option off (`0`).

### Sampler and trained policy

TRL's colocate sync merges the LoRA adapter into the bf16 base weights before each update
and loads them into vLLM. At learning rate `1e-5` the adapter's change to most weights is
far below one bf16 step, so rounding the merged weights removes part of what the trainer
learned: vLLM samples from a rounded, lagging copy of the policy. `scripts/check_logprob_gap.py`
measured this on a GRPO adapter after 300 updates (64 questions, eight samples each). vLLM
with the merged weights and the trainer (bf16 base plus the unmerged fp32 adapter) differ by
a mean |Δ log p| of 0.132 per sampled token; the same adapter merged in bf16 on both sides
gives 0.011, and the Instruct model without an adapter 0.028 on longer reasoning text. The
gap sits on near-tied tokens: the space before `</answer>` has a median probability of 0.905
under vLLM and 0.378 under the trainer, so the sampler writes `<answer> x </answer>` more
often than the trained policy would. Prompt token IDs matched exactly, so this is not a
tokenization or image-processing difference. In training it shows as
`sampling/sampling_logp_difference/mean` rising from 0.03 to about 0.2 and the
importance-sampling ratio dropping to 0.6–0.8 once responses become short.

`train_grpo.py --rollout-lora` keeps the base weights vLLM loaded at start-up and serves the
adapter unmerged, as the dev evaluation does ([`vqar/adapter_rollouts.py`](../../vqar/adapter_rollouts.py)).
At an update's first generation the trainer's adapter is written to a node-local temporary
directory under a new adapter ID and passed with every rollout request; refill rounds of the
same update reuse it. With it, the same check gives 0.008, and the training logs of every
final run keep the gap at 0.004–0.03 with an importance-sampling ratio near 1.00. Accuracy
did not change: the 2B baseline trained both ways differs by at most 0.52 points on any
test cell. Greedy evaluation is similarly insensitive: the merged and the unmerged form of
one trained adapter score within 0.42 points on every cell. The fix therefore matters for
what the update optimizes (on-policy samples) rather than for these accuracies. Cost: when
vLLM sleeps between rollouts it must sleep at level 1, which keeps the base weights in host
memory (about 4 GB for the 2B model, 17 GB for the 8B); on the 96 GB profile
`--no-vllm-sleep` keeps the engine awake instead. With level-1 sleep a 2B update took 16.7 s instead of
about 5 s; with the engine awake, a final 2B GRPO run took 2.7–3.2 GPU hours on an A100 80
GB and an 8B run 2.9–3.7 on an RTX PRO 6000, each including the dev evaluation of its ten
saved adapters.

**OPD.** The same rollout geometry as GRPO. The learning signal is the teacher's token
log-probability minus the student's rollout-time log-probability. The correctness reward
is still logged, for diagnosis only; in the saved completion samples TRL's `advantage`
column is that logged diagnostic, and the teacher statistics are under `opd/*`.

**Hardware.** The final rerun trained the 2B GRPO arms and OPD on A100 80 GB cards (`80gb`
profile), the 2B short-prompt arms on an RTX 4090 (`24gb`), and the 8B SFT and GRPO arms
on RTX PRO 6000 96 GB cards (`96gb`); the profiles change memory placement, not the recipe.
Each run records its card, profile, and resolved settings in `run_config.json`.

## Train the arms

Run from the repository root with a new run directory for each run; an existing
compatible directory resumes from its last complete checkpoint. For a two-update check of
the whole path, see the [tutorial](../tutorial.md). Every GRPO arm below shares these flags:

```bash
RECIPE="--max-steps 1000 --save-steps 100 --save-total-limit 0 --rollout-lora --no-vllm-sleep"
SHORT="--rollout-prompt short --max-completion-length 64"
```

2B (the SFT model is `sharedpool_20260913_sft_2b`, trained with the unchanged SFT recipe):

```bash
uv run --frozen python train_sft.py --preset sft_2b --hardware 96gb \
  --model "$BASE_2B" --save-steps 50 --run-dir runs/sft2 --no-wandb
uv run --frozen python scripts/merge_adapter.py runs/sft2/checkpoints \
  --base "$BASE_2B" --out checkpoints/merged/sharedpool_20260913_sft_2b
SFT_2B=checkpoints/merged/sharedpool_20260913_sft_2b

uv run --frozen python train_grpo.py --preset grpo_2b_main --hardware 80gb $RECIPE \
  --model "$BASE_2B" --run-dir runs/grpo2-base --no-wandb
uv run --frozen python train_grpo.py --preset grpo_2b_main --hardware 80gb $RECIPE \
  --model "$SFT_2B" --run-dir runs/grpo2-sft --no-wandb
uv run --frozen python train_grpo.py --preset grpo_2b_main --hardware 80gb $RECIPE $SHORT \
  --model "$BASE_2B" --run-dir runs/short2-base --no-wandb
uv run --frozen python train_grpo.py --preset grpo_2b_main --hardware 80gb $RECIPE $SHORT \
  --model "$SFT_2B" --run-dir runs/short2-sft --no-wandb
uv run --frozen python train_opd.py --preset opd_self_2b_main --hardware 80gb \
  --model "$BASE_2B" --max-steps 1000 --save-steps 100 --save-total-limit 0 \
  --run-dir runs/opd2 --no-wandb
```

8B, on an 80 or 96 GB card (`--no-vllm-sleep` needs the 96 GB card at 8B; drop it on 80 GB):

```bash
uv run --frozen hf download Qwen/Qwen3-VL-8B-Instruct \
  --revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --local-dir checkpoints/base/Qwen3-VL-8B-Instruct
BASE_8B=checkpoints/base/Qwen3-VL-8B-Instruct
uv run --frozen python train_sft.py --preset sft_8b --hardware 96gb \
  --model "$BASE_8B" --save-steps 100 --run-dir runs/sft8 --no-wandb
uv run --frozen python scripts/merge_adapter.py runs/sft8/checkpoints \
  --base "$BASE_8B" --out checkpoints/merged/p1_sft8
SFT_8B=checkpoints/merged/p1_sft8
for start in "$BASE_8B" "$SFT_8B"; do
  uv run --frozen python train_grpo.py --preset grpo_8b_fmt0 --hardware 96gb $RECIPE \
    --model "$start" --run-dir "runs/grpo8-$(basename $start)" --no-wandb
  uv run --frozen python train_grpo.py --preset grpo_8b_fmt0 --hardware 96gb $RECIPE $SHORT \
    --model "$start" --run-dir "runs/short8-$(basename $start)" --no-wandb
done
```

Merge each run's final adapter with `scripts/merge_adapter.py <run>/checkpoints --base
<its starting model> --out checkpoints/merged/<name>`; the CPU merge of an 8B model needs
about 64 GB of host RAM. `grpo_2b_main` and `grpo_8b_fmt0` train with the correctness
reward alone and share the batch geometry. When a run completes, it evaluates every saved
adapter on the dev tier with both prompts; those records are the run's learning curve.

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

RL shows every prompt image resized to fit and centered on a black 608×608 canvas (361
vision tokens), while evaluation resizes to at most 768 vision tokens. `--image-mode canvas`
applies the RL transform (`vqar.data.to_uniform_canvas`) before generation, so an arm can be
evaluated at the geometry it trained on. It is its own protocol (`-canvas` in the record
name) and needs no original-image cell; `python -m scripts.paired_deltas` pairs it with the
same arm's original-image cell, and the report lists the change beside the image-budget
contrasts.

## Inspect the saved evidence without a GPU

```bash
uv run --frozen python -m scripts.report
uv run --frozen python -m scripts.summarize_training runs/{grpo2-base,grpo2-sft,short2-base,short2-sft,opd2} \
  runs/{grpo8,short8}-{Qwen3-VL-8B-Instruct,p1_sft8} --output results/training-exposure.json
uv run --frozen python -m scripts.make_gallery
uv run --frozen python -m scripts.paired_deltas   # needs the local predictions
uv run --frozen python -m scripts.error_analysis  # flips, error kinds, answer forms; same
```

For the checks on the [validation](validation.md#parity-with-lmms-eval) page:
`evaluate.py --max-pixels 16777216 --max-model-len 20480` evaluates a cell at the model's own
image budget, and `python -m scripts.compare_lmms_eval` pairs an lmms-eval run's logged
responses with this harness's records.

The report reads the stored records; each names its protocol, so cells evaluated under
different questions, prompts, scorers, or decoding never share a column. The corrected
campaign's records were written from its saved outputs by `python -m scripts.migrate_campaign`. With
the saved predictions present, rerunning it verifies the stored records, and
`pytest tests/test_legacy.py` recomputes every campaign number from them. The training summary checks the actual prompt order, counts reward variation
per update, splits the groups without variation into all right, all wrong, and all at one
partial score (per update, and averaged over the first and the last ten updates), and embeds
each run's diagnostics; regenerating it needs the materialized training data and each run's
`prompt_draws.jsonl`, and its output is already committed (the corrected campaign's is in
`results/baselines/training-exposure.json`).

## Checkpoint storage

The trainers keep the latest two full checkpoints by default; `--save-total-limit 0`
keeps every one, with optimizer and RNG state. The adapter of every scheduled save is
kept either way, under `checkpoints/adapters/`, for the learning curve. After a run,
archive useful intermediate checkpoints with their run configuration, metrics, and
provenance, and verify the copy before deleting a working one. A final adapter alone
cannot resume training.
