# Reproduce the corrected comparison

The comparison asks whether answer-only SFT helps the same subsequent GRPO
training. It includes released Instruct, SFT, Instruct→GRPO, and SFT→GRPO.
Both RL arms use the same seed, prompt order, reward, and update budget. SFT
adds training exposure and compute; this is not an equal-total-compute comparison.

## Recorded 2B results

Dataset-author accuracy (%), seed 42, greedy decoding on the fixed materialized
subsets. The short and reasoning prompts have output limits of 64 and 512 tokens.

| Model | Prompt | VQAv2 | GQA | CLEVR counting | TextVQA |
|---|---|---:|---:|---:|---:|
| Instruct | short | 77.87 | 59.45 | 97.67 | 80.55 |
| SFT | short | 79.87 | 62.35 | 99.40 | 79.83 |
| Instruct → GRPO | short | 78.02 | 59.55 | 98.27 | 80.38 |
| SFT → GRPO | short | 79.85 | 62.44 | 99.40 | 79.82 |
| Instruct | reasoning | 55.06 | 36.13 | 87.13 | 69.60 |
| SFT | reasoning | 67.99 | 56.61 | 92.10 | 68.78 |
| Instruct → GRPO | reasoning | 61.79 | 44.90 | 86.10 | 74.23 |
| SFT → GRPO | reasoning | 69.18 | 53.70 | 97.80 | 74.28 |

All four models score higher with the short prompt on every dataset. Its scores
change little after GRPO. For SFT→GRPO, reasoning-prompt scores improve on three
datasets but decline on GQA. The decline remains under normalized matching
(57.00% → 54.86%); it is not explained away by capitalization.
Under official scoring, VQAv2 yes/no accuracy rises from 82.88% to 88.08%, while
the "other" category falls from 57.40% to 55.96%. GQA query questions fall from
41.75% to 36.68%. Aggregate improvement does not mean every question type improves.

Normalized matching on the **same reasoning responses** gives:

| Model | VQAv2 | GQA | CLEVR counting | TextVQA |
|---|---:|---:|---:|---:|
| Instruct | 64.68 | 51.95 | 87.13 | 69.61 |
| SFT | 68.17 | 57.00 | 92.10 | 68.79 |
| Instruct → GRPO | 69.01 | 54.09 | 86.10 | 74.22 |
| SFT → GRPO | 70.80 | 54.86 | 97.80 | 74.28 |

The official/normalized gap is especially large for the original Instruct model.
Lowercasing alone accounts for 479 of 481 affected VQAv2 responses and 1,845 of
1,989 affected GQA responses. The report retains the original text and both scores;
normalization is a diagnostic, not a replacement for the declared primary metric.

SFT→GRPO changes mean reasoning-prompt output length from 20.6 to 101.7 tokens on
VQAv2 and 7.8 to 78.7 on GQA. Counting goes the other way: 61.6 to 23.7 tokens,
with higher accuracy. These observations do not support a general rule that more
visible reasoning improves answering. Selected successes and regressions are in
the recorded gallery.

### What the RL run actually consumed

Both runs completed 300 updates using exactly the same ordered 1,200 distinct
questions and 9,600 responses. Of those questions, 841 also belong to SFT training
and 359 come from the additional RL sources. This is about 2.2% of the available
55,438-question pool: these runs do not test what happens after exhausting it.

| Initialization | Groups with varied rewards | Updates without reward variation |
|---|---:|---:|
| Instruct | 464 / 1,200 (38.67%) | 43 / 300 |
| SFT | 588 / 1,200 (49.00%) | 16 / 300 |

Uniform rewards give zero group advantages. Adam momentum can still move weights,
so an update without reward variation does not necessarily mean no parameter change.
These counts motivate testing how to obtain useful feedback, but do not establish
that dynamic sampling or another normalization will improve held-out accuracy.

The fixed image checks replace images with black images or other images for the
first 128 questions in each dataset. Scores against the original references fall
substantially for every model. This establishes sensitivity under these interventions;
it does not isolate the encoder or validate the explanations.

All values are derived from [`comparison.json`](../results/baselines/comparison.json)
and [`training-exposure.json`](../results/baselines/training-exposure.json).
These are single-seed observations; small point differences do not establish a
reliable method ranking across training runs.

## Data and model identity

The `heldout-images-v2` split contains 39,458 SFT training questions, 506 dev
questions, and 55,438 RL questions. SFT training questions are included in the
RL pool. Exact decoded-RGB images from dev and evaluation are excluded from both
training stages, including matches across data sources. This check does not
establish absence of near-duplicates or upstream model pretraining exposure.

Reusing a training question across SFT and RL is allowed by this experimental
design. SFT fits a supplied answer; RL samples new responses to that question.
The question still overlaps, but it remains training data in both stages.
Academic precedent includes [SimpleRL-Zoo, Section 4](https://arxiv.org/html/2503.18892v3#S4),
which uses GSM8K/MATH-derived SFT data and RL from the same problem sources.
That report does not give an exact shared-question count. Our own manifest and
draw logs make the reuse explicit.

Use the [materialized data bundle](../artifacts/README.md) for the exact recorded
question subsets. It is prepared locally; public hosting is pending. A fresh
`prepare_data.py` download records source revisions but is not guaranteed to
recreate the earlier materialization. Five duplicate image/question pairs remain
within training, including one conflicting count label. They were recorded and
kept in the frozen pool.

```bash
uv run --frozen python -m scripts.check_data
uv run --frozen hf download Qwen/Qwen3-VL-2B-Instruct \
  --revision 89644892e4d85e24eaac8bacfd4f463576704203 \
  --local-dir checkpoints/base/Qwen3-VL-2B-Instruct
BASE_2B=checkpoints/base/Qwen3-VL-2B-Instruct
```

All commands below run from the repository root. Use new run directories;
an existing compatible training directory resumes its last complete checkpoint.
For a small execution check, use the two-update commands in the
[hardware guide](hardware.md) instead of this full comparison.

## Train the three adapters

SFT uses one epoch, learning rate `1e-4`, and an effective batch of 32 examples.
The recorded SFT run used an RTX PRO 6000 96 GB. Both recorded 2B GRPO arms
used A100 80 GB cards. The `24gb` profile keeps a smaller rollout allocation
for the same 2B training recipe.

The recorded base-to-GRPO run reserved 36% of A100 memory for vLLM; SFT-to-GRPO
reserved 15%. A path-based size check misread `8b` inside the base model's snapshot
hash. All other resolved trainer arguments match, apart from output paths.
The current profile uses the named recipe and reserves 15% for either 2B arm.
The historical allocation difference remains in the run metadata; it should be
considered when interpreting memory use and runtime.

```bash
uv run --frozen python train_sft.py --preset sft_2b --hardware 96gb \
  --model "$BASE_2B" --save-steps 50 --run-dir runs/sft2 --no-wandb
uv run --frozen python scripts/merge_adapter.py runs/sft2/checkpoints \
  --base "$BASE_2B" --out checkpoints/merged/sharedpool_20260913_sft_2b

uv run --frozen python train_grpo.py --preset grpo_2b_main --hardware 80gb \
  --model "$BASE_2B" --rewards correctness --reward-weights 1 \
  --max-steps 300 --save-steps 25 --run-dir runs/grpo2-base --no-wandb
uv run --frozen python scripts/merge_adapter.py runs/grpo2-base/checkpoints \
  --base "$BASE_2B" --out checkpoints/merged/sharedpool_20260913_grpo_2b_base

uv run --frozen python train_grpo.py --preset grpo_2b_main --hardware 80gb \
  --model checkpoints/merged/sharedpool_20260913_sft_2b \
  --rewards correctness --reward-weights 1 --max-steps 300 --save-steps 25 \
  --run-dir runs/grpo2-sft --no-wandb
uv run --frozen python scripts/merge_adapter.py runs/grpo2-sft/checkpoints \
  --base checkpoints/merged/sharedpool_20260913_sft_2b \
  --out checkpoints/merged/sharedpool_20260913_grpo_2b_sft
```

The two GRPO arms use learning rate `1e-5`, eight responses per prompt, batch
size 8, accumulation 4, and at most 640 completion tokens. Each update consumes
four prompt groups. Three hundred updates therefore sample 9,600 responses from
1,200 prompt draws. Dynamic resampling is off. Uniform rewards produce zero
group advantages; the number of useful groups must be measured separately.

The correctness reward requires a complete answer tag and then uses normalized
answer matching. No format bonus is added. See the [README](../README.md) for the
explicit loss and sampling-correction settings.

## Evaluate both prompts

Use greedy decoding and the dataset-author scorers on all four materialized
evaluation subsets. The short prompt allows 64 output tokens; the reasoning
prompt allows 512. Both prompts matter: a stronger short-answer baseline can
change the interpretation of an apparent reasoning-prompt gain.

```bash
evaluate_arm() (
  set -e
  ARM=$1
  MODEL=$2
  SIZE=$3
  for TEMPLATE in short reasoning; do
    PRESET="zero_shot_$SIZE"
    if [ "$TEMPLATE" = reasoning ]; then PRESET="${PRESET}_reasoning"; fi
    uv run --frozen python evaluate.py --preset "$PRESET" --model "$MODEL" \
      --run-id "$ARM-$TEMPLATE" --metric official --batch-size 128 --resume \
      --results-dir "results/baselines/evaluation/$ARM"
  done
)
evaluate_arm base2 "$BASE_2B" 2b
evaluate_arm sft2 checkpoints/merged/sharedpool_20260913_sft_2b 2b
evaluate_arm grpo2-base checkpoints/merged/sharedpool_20260913_grpo_2b_base 2b
evaluate_arm grpo2-sft checkpoints/merged/sharedpool_20260913_grpo_2b_sft 2b
```

For image sensitivity, repeat those evaluations with `--limit 128 --batch-size 128`
and `--image-mode black` or `swapped`. Use run IDs `$ARM-$TEMPLATE-$MODE` and
`--results-dir results/baselines/interventions/$ARM`. Original-image scores come
from the first matching batch of the full evaluation. These interventions keep
the original references; their scores are not ground-truth accuracy on altered images.

## Inspect saved evidence without a GPU

```bash
uv run --frozen python -m scripts.make_tables \
  --results-dir results/baselines/evaluation
uv run --frozen python -m scripts.summarize_comparison results/baselines
uv run --frozen python -m scripts.summarize_training \
  results/baselines/training/grpo2-base results/baselines/training/grpo2-sft \
  results/baselines/training/grpo8-base --output results/baselines/training-exposure.json
uv run --frozen python -m scripts.make_diagnostics --gallery-only
```

Tables reject mismatched question cohorts, evaluation code, scorer settings, or
decoding. The comparison report adds normalized scoring, a lowercase diagnostic,
response length, truncation, answer-tag rates, and paired image interventions.
The training report checks the actual prompt order and measures reward variation.
Regenerating it also needs the materialized training-data bundle; its saved JSON
is already included. Saved samples of training completions are included under
`results/baselines/training/<arm>/checkpoints/completions/` in the evidence bundle.
No additional model inference is needed for these summaries.

## 8B confirmation

The same questions, prompts, scoring, and decoding settings were used before and
after 300 GRPO updates. Both evaluations are complete.

| Model | Prompt | VQAv2 | GQA | CLEVR counting | TextVQA |
|---|---|---:|---:|---:|---:|
| Instruct | short | 81.18 | 61.66 | 99.67 | 83.98 |
| Instruct → GRPO | short | 81.12 | 61.78 | 99.57 | 84.07 |
| Instruct | reasoning | 63.67 | 46.84 | 96.97 | 75.31 |
| Instruct → GRPO | reasoning | 77.37 | 59.43 | 97.30 | 80.78 |

The largest absolute short-prompt change is 0.13 percentage points. Reasoning-prompt
scores rise on all four datasets, while remaining below each model's short-prompt
scores. VQAv2 mean output length falls from 118.2 to 9.0 tokens, and GQA from
104.0 to 9.2. Counting responses remain longer (192.7 → 83.4 tokens).
The gains are not solely capitalization: normalized VQAv2 accuracy rises from
68.75% to 78.63%, and GQA from 52.27% to 59.82%. These runs show improvement under
a particular prompt, without evidence of better accuracy than the strong
short-answer baseline. They do not show that longer visible reasoning is necessary.

The 8B GRPO run completed 300 updates with exactly the same 1,200 ordered
questions as both 2B GRPO runs. Only 280/1,200 groups (23.33%) had varied rewards;
110/300 updates had none. Its mean sampled completion length fell from 133.3
tokens in the first logged window to 12.8 in the last. These training observations
are separate from the held-out comparison and do not establish better reasoning.

Use the bf16 `grpo_8b_fmt0` preset with `--hardware 96gb` or `80gb`. It has the
same learning rate `1e-5`, group size, batch geometry, and 300-update budget as
the 2B GRPO controls. The older quantized `grpo_8b_main` is a different recipe.

```bash
uv run --frozen hf download Qwen/Qwen3-VL-8B-Instruct \
  --revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --local-dir checkpoints/base/Qwen3-VL-8B-Instruct
BASE_8B=checkpoints/base/Qwen3-VL-8B-Instruct
uv run --frozen python train_grpo.py --preset grpo_8b_fmt0 --hardware 96gb \
  --model "$BASE_8B" --rewards correctness --reward-weights 1 \
  --max-steps 300 --save-steps 25 --run-dir runs/grpo8-base --no-wandb
uv run --frozen python scripts/merge_adapter.py runs/grpo8-base/checkpoints \
  --base "$BASE_8B" --out checkpoints/merged/sharedpool_20260913_grpo_8b_base
evaluate_arm base8 "$BASE_8B" 8b
evaluate_arm grpo8-base checkpoints/merged/sharedpool_20260913_grpo_8b_base 8b
```

The `evaluate_arm` function defined above selects the matching short/reasoning
presets. Keep the dataset subsets, batch size, and scoring unchanged. The CPU merge of an
8B model needs substantially more host RAM than the 2B merge; reserve 64 GB of
host RAM for this step.

## Selected OPD comparison

Training completed all 300 updates on the same 1,200 ordered questions and 9,600
student responses as the GRPO controls. Both held-out prompt evaluations are complete.
The earlier numerical OPD checks are described in [validation notes](validation.md).

The focused OPD arm asks whether a gold-conditioned frozen teacher supplies useful
token feedback on the same student-sampled questions. It uses the original 2B
Instruct model as both the starting student and the frozen teacher. Only the
teacher sees the reference answer; correctness is logged but does not drive its
loss. This comparison changes the supervision signal and conditioning together.
It cannot isolate an advantage caused solely by denser feedback.


| Model | Prompt | VQAv2 | GQA | CLEVR counting | TextVQA |
|---|---|---:|---:|---:|---:|
| Instruct → gold-conditioned OPD | short | 77.79 | 59.72 | 97.73 | 80.36 |
| Instruct → gold-conditioned OPD | reasoning | 68.61 | 51.35 | 87.10 | 72.65 |

Against the original Instruct model, short-prompt changes range from −0.20 to
+0.26 percentage points. Reasoning-prompt official scores rise on VQAv2, GQA,
and TextVQA, while counting is essentially unchanged. Every OPD reasoning score
remains below the original Instruct short-answer baseline.

GQA shows why both scoring views matter. Official reasoning accuracy rises from
36.13% to 51.35%, but normalized accuracy changes from 51.95% to 51.58%.
Lowercasing the responses before official scoring reduces the before/after gap
to 50.80% → 51.43%. Thus much of the official gain comes from answer casing.
OPD also ranks above base→GRPO on official GQA accuracy (51.35% vs. 44.90%),
but below it under normalized matching (51.58% vs. 54.09%). The official result
remains the primary score; the diagnostic changes what improvement we can infer.

There are gains beyond casing on other datasets: normalized VQAv2 reasoning
accuracy rises from 64.68% to 69.23%, and TextVQA from 69.61% to 72.66%.
Mean reasoning output length stays close to the base model on VQAv2, GQA, and
TextVQA; counting grows from 185.7 to 206.0 tokens without an accuracy gain.
These results do not establish a general advantage for OPD or longer responses.
The recorded OPD training used an RTX 6000 and the 2B GRPO controls used A100s;
this is not a matched hardware/runtime comparison.

```bash
uv run --frozen python train_opd.py --preset opd_self_2b_main --hardware 96gb \
  --model "$BASE_2B" --max-steps 300 --save-steps 25 --save-total-limit 0 \
  --run-dir runs/opd2-self --no-wandb
uv run --frozen python scripts/merge_adapter.py runs/opd2-self/checkpoints \
  --base "$BASE_2B" --out checkpoints/merged/sharedpool_20260913_opd_2b_self
evaluate_arm opd2-self checkpoints/merged/sharedpool_20260913_opd_2b_self 2b
```

The same `evaluate_arm` function evaluates both 2B prompts. This is one selected
comparison, not an external-teacher or model-size grid. Report its token-level
teacher objective separately from the GRPO reward-variation counts. In the saved
completion samples, TRL's `advantage` column is still the parent logger's
correctness-derived group diagnostic. OPD replaces that signal for optimization;
its actual token feedback is teacher log-probability minus rollout-time student log-probability,
with teacher statistics recorded under `opd/*`.

Repeated training seeds are deferred. Single-seed results and selected examples
support lessons about these runs, not claims of universal method superiority.

## Checkpoint storage

All three trainers keep the latest two saved checkpoints by default. Add
`--save-total-limit 0` to preserve every save, as in the OPD command above.
Keep active runs and current final models nearby; archive useful intermediate
and historical checkpoints to a separate disk after the run. Preserve the full
checkpoint directory (including optimizer, scheduler, RNG, and completion marker),
run configuration, metrics, and model provenance. Verify the copied files before
deleting a redundant working copy. A final adapter alone cannot resume training.
