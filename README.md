# Visual QA post-training, with the failures included

A hands-on study of short-answer SFT, GRPO, and on-policy distillation for
Qwen vision-language models. The aim is to make each training choice understandable:
what receives a gradient, what a score measures, and when an apparent improvement
fails a closer check.

Start with the **2B model on a 24 GB GPU**, or use the separate **80/96 GB profiles**
for existing 8B recipes. The recorded demo runs without a GPU.

## Explore without training

```bash
uv sync --frozen --group demo
cd demo/frontend
npm ci
npm run build
cd ../..
uv run --frozen uvicorn demo.backend.app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. The gallery shows recorded successes and counterexamples;
the dashboard distinguishes official scoring from historical normalized scoring.
See [demo setup](demo/README.md) for optional live inference.

![Recorded VQA examples](demo/assets/gallery.png)

## Run the small pipeline

Use Python 3.12 and the committed lockfile. Downloads require network access and
substantial disk space. Preparing data is separate from training:

```bash
uv run --frozen python prepare_data.py
uv run --frozen python -m scripts.check_data
uv run --frozen python train_sft.py --preset sft_2b --hardware 24gb \
  --max-steps 2 --save-steps 1 --run-dir runs/sft-smoke --no-wandb
uv run --frozen python train_grpo.py --preset grpo_2b_main --hardware 24gb \
  --model Qwen/Qwen3-VL-2B-Instruct --max-steps 2 \
  --save-steps 1 --run-dir runs/grpo-smoke --no-wandb
```

These two-step runs check execution, not model quality. They train separate SFT and
base-to-GRPO controls. [Hardware instructions](docs/hardware.md) cover larger models,
resolved run metadata, and checkpoint resumption. [Validation notes](docs/validation.md)
explain the verified behavior and remaining limits.
The [full comparison walkthrough](docs/baselines.md) covers shared SFT/RL training
data, the four 2B arms, the 8B confirmation, the selected OPD comparison, and
matched evaluation under both prompts.

## Corrected 2B results

Dataset-author accuracy (%), greedy short answers, one training seed. Both GRPO
arms use the same 1,200 training questions in the same order for 300 updates.

| Starting point and training | VQAv2 | GQA | CLEVR counting | TextVQA |
|---|---:|---:|---:|---:|
| Instruct | 77.87 | 59.45 | 97.67 | 80.55 |
| SFT | 79.87 | 62.35 | 99.40 | 79.83 |
| Instruct → GRPO | 78.02 | 59.55 | 98.27 | 80.38 |
| SFT → GRPO | 79.85 | 62.44 | 99.40 | 79.82 |

SFT improves three datasets; adding this short GRPO run changes its short-answer
scores by less than 0.1 percentage points each. Under the reasoning prompt,
SFT→GRPO improves VQAv2, counting, and TextVQA but loses GQA accuracy. Every arm
scores higher with the short prompt on every dataset. Output length and formatting
explain part of the behavior; a longer answer alone is not evidence of better reasoning.

The [8B confirmation](docs/baselines.md#8b-confirmation) shows the same caution:
GRPO improves reasoning-prompt scores with much shorter responses, but changes
short-prompt accuracy by at most 0.13 percentage points.

The [selected OPD comparison](docs/baselines.md#selected-opd-comparison) also shows
how scoring changes the interpretation: official GQA reasoning accuracy rises
from 36.13% to 51.35%, largely through answer casing, while normalized accuracy
changes from 51.95% to 51.58%. Its short-answer scores remain close to Instruct.

The dashboard includes both prompts, two scoring variants, response lengths, and
fixed image interventions. [Results and interpretation](docs/baselines.md#recorded-2b-results)
include the complete tables and actual training exposure. Raw predictions, adapters,
and frozen source snapshots are kept in a local evidence bundle.

## What the methods teach

| Arm | Supervision | Question |
|---|---|---|
| Short-answer SFT | Answer and end-of-turn tokens | How does direct answer supervision change visible responses? |
| GRPO | Dataset correctness, optionally format | Can sampled responses improve with a verifiable outcome reward? |
| External OPD | Frozen teacher log-probabilities on student responses | Is trajectory-level teacher supervision useful here? |
| Gold-conditioned self-teacher | Frozen base model, with the reference answer added only to its prompt | Does this additional information provide a useful learning signal? |

SFT does not supervise reasoning demonstrations. Shorter visible outputs do not prove
that internal reasoning was erased. Likewise, a long `<think>` block does not prove
correct reasoning. The gold-conditioned teacher has extra information; comparisons
with an external teacher do not isolate teacher size.

Here, "base" means the released Qwen3-VL **Instruct** checkpoint before our VQA
fine-tuning. It already received upstream post-training.

The pinned GRPO implementation uses DAPO token loss reduction, group reward scaling,
weighted-sum-then-normalize rewards, token policy ratios, symmetric clipping at 0.2,
and default KL coefficient zero. vLLM sampling correction uses `sequence_mask` with
an upper ratio bound of 3. **Dynamic resampling of zero-variance groups is not enabled.**
This is not the complete DAPO recipe. Settings are explicit in
[`vqar/config.py`](vqar/config.py) and saved with each new run.

Even the correctness-only reward requires a complete `<answer>...</answer>` tag;
it has no additional format bonus. Within the tag, training uses normalized answer
matching, with single-reference handling. Evaluation also accepts untagged answers
and reports the dataset-author metric separately from normalized matching.

At batch size 8, accumulation 4, and group size 8, one update uses 32 responses from
four prompt groups. Three hundred updates mean 9,600 sampled responses and 1,200
prompt draws from the current 55,438-question RL pool. Equal update counts do not
imply equal token counts or GPU time.

SFT and RL share training examples. The manifest reserves 39,458 SFT training
examples and 506 development examples, then includes the SFT training examples
in both RL arms' shared pool. Development and evaluation images are excluded
from both stages. Comparing base-to-RL with SFT-to-RL measures the benefit of
adding SFT before the same RL training, including SFT's additional exposure and compute.

## Evaluation and reproducibility

| Dataset | Materialized evaluation questions | Primary scoring |
|---|---:|---|
| VQAv2 | 5,000 | Original VQA ten-annotator accuracy and normalization |
| GQA | 12,578 | Original exact answer match |
| CLEVR counting subset | 3,000 test; 3,000 validation | Exact counting answer |
| TextVQA | 2,000 | MMF/EvalAI answer processing and ten-annotator accuracy |

These are the repository's materialized subsets, not submissions to every official
benchmark server. CLEVR uses a local partition of a counting subset. GQA's broader
consistency, validity, plausibility, distribution, and attention-based grounding
suite is not reproduced by an accuracy-only score.

`evaluate.py` scores every prediction with both variants: `official` (the dataset
authors' evaluator, the primary column) and `normalized` (the historical rule, shown
beside it). Training rewards retain their separate, documented rules. Existing prediction and result files are preserved. To rescore them and
refresh the dashboard:

```bash
uv run --frozen python -m scripts.rescore --metric official
uv run --frozen python -m scripts.make_diagnostics
```

The new scorers matched the authors' evaluator code on **626,074 saved predictions
across 89 runs**, with zero per-question mismatches. For GQA this validates the
author's accuracy expression, not its full evaluation suite. Source versions and
reproduction steps are in [validation notes](docs/validation.md).

[`results/diagnostics.json`](results/diagnostics.json) contains generated official
and historical scores with source hashes. Original `results/*.json` and
`results/preds/` remain historical artifacts. Do not combine their scores across
metric variants without labeling the change.

## Lessons from the existing experiments

- **A reward can run successfully while providing no useful signal.** Before the
  July 16 VQAv2 single-reference fix, the old leave-one-out reward returned zero
  for single-answer training examples. Affected runs cannot establish the value
  of a working VQAv2 correctness reward. Rescoring predictions cannot repair training.
- **Question-disjoint is not image-disjoint.** The materialized training pools
  overlap VQAv2 evaluation on 47 exact images, covering 53 evaluation questions.
  The old SFT dev set also shares 317 of its 487 unique images with training.
  This establishes pool overlap; individual checkpoint exposure depends on the
  actual samples consumed. New runs use filtered pools and a whole-image dev split.
- **Padding affects multimodal meaning.** A right-padded processor paired with
  left-padded GRPO rollout tokens can shift image markers. The previous fallback
  could replace image positions with 1D text positions. The fix aligns padding
  and retains the native Qwen position computation.
- **Precision is part of the implementation.** NF4 student weights corrupted the
  historical TRL/vLLM synchronization path. Those RL configurations now fail
  clearly at launch. The 24 GB 2B bf16 route and 8B QLoRA SFT remain available.
- **Interesting hypotheses can fail.** The proposed metric ranking reversal did
  not replicate. The corrected reasoning-deletion study did not establish tight
  equivalence. The corrected static-predictor pilot remains inconclusive.

The public-checkpoint image interventions found no detected grounding degradation
under that protocol. Image sensitivity alone neither establishes correct grounding
nor isolates the image encoder's causal contribution. The learned verifier still
needs real held-out human calibration before becoming a trusted training reward.

## Existing adapters

These published adapters correspond to historical runs. The reward, data-overlap,
and position-alignment limitations above apply; they have not been silently replaced.

| Adapter | Download |
|---|---|
| 2B SFT | [vqa-rlvr-sft-2b](https://huggingface.co/omnifish123/vqa-rlvr-sft-2b) |
| 8B SFT | [vqa-rlvr-sft-8b](https://huggingface.co/omnifish123/vqa-rlvr-sft-8b) |
| 2B GRPO from base | [vqa-rlvr-grpo-2b-main-base](https://huggingface.co/omnifish123/vqa-rlvr-grpo-2b-main-base) |
| 2B SFT then GRPO | [vqa-rlvr-grpo-2b-main-sft](https://huggingface.co/omnifish123/vqa-rlvr-grpo-2b-main-sft) |

To evaluate a newly trained adapter, merge it into its recorded base, then evaluate
it. Each (model, dataset, prompt) cell writes one record under `results/records/`;
a cell that already has a record is skipped:

```bash
uv run --frozen python scripts/merge_adapter.py runs/grpo-smoke/checkpoints \
  --out checkpoints/merged/grpo-smoke
uv run --frozen python evaluate.py --model checkpoints/merged/grpo-smoke \
  --label grpo-smoke --prompts short reasoning --tier smoke
```

## Repository map

- `train_sft.py`, `train_grpo.py`, `train_opd.py`: training entrypoints.
- `vqar/`: configurations, data, split selection, rewards, and scoring.
- `evaluate.py`, `scripts/`: evaluation and artifact-derived diagnostics.
- `demo/`: recorded examples, results dashboard, and optional live comparison.
- `tests/`: CPU checks; GitHub Actions runs them without installing PyTorch.

Repeated training seeds and new medical-domain training are outside this validation
milestone. Existing replications remain usable when identified as such.

## Acknowledgments

Built with Qwen, Hugging Face Transformers, TRL, PEFT, and vLLM. Dataset and model
licenses apply to their respective assets. See [LICENSE](LICENSE) for this code.
