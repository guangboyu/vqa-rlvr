# What the current implementation establishes

Validation date: 2026-09-13. These implementation checks preceded the
[corrected baseline campaign](baselines.md), whose four 2B arms are now complete.
Historical predictions remain intact. New split selection, official scoring,
and training fixes have separate provenance.

## Starting branch

Continue from `research/v2-topconf`. Its SFT, GRPO, and evaluation core is largely
shared with `main`; it also contains the OPD, verifier, image-intervention, and
corrected pilot evidence needed for useful lessons. `main` is smaller and easier
to browse, but omits much of this material. The research branch also missed the
README reward erratum present on `main`; it has now been restored.

For research, the strongest next step is a well-defined image-dependent task and
calibrated evaluation. For public usefulness, prioritize a runnable 24 GB example,
recorded failures, precise metric explanations, and reproducible tables. Extra
functions and larger models alone offer no reliable route to more stars.

## Training checks

| Path | Verified behavior | Interpretation limit |
|---|---|---|
| SFT | 2B bf16 and 8B NF4 real-batch loss, gradients, and two-update checks on RTX 4090; image-disjoint dev selection | Short-answer supervision does not train explicit reasoning demonstrations. |
| GRPO | Explicit loss defaults; prompt/group/reward alignment; native image-position reproduction; actual 24 GB two-update check | Tiny checks establish execution and local numerical consistency, not final accuracy. |
| External OPD | 2B student / 8B NF4 teacher passed two updates on RTX 4090; manual token log-probabilities, frozen teacher, and detached advantages match | Teacher capacity and added information require separate controls. |
| Gold-conditioned self-teacher | 2B two-update check on RTX 4090; manual token log-probabilities and detached advantages match; no prompt mutation | This is a frozen-base variant, not canonical feedback-conditioned SDPO. |

The same numerical probes also passed for 8B bf16 SFT, GRPO, the gold-conditioned
self-teacher, and 2B external OPD on an actual RTX PRO 6000 96 GB. Each completed
two updates. All four paths also passed on an A100 80GB PCIe. Shared-storage
delays interrupted result retrieval, but the job finished successfully and all
required results were subsequently retrieved. These are short numerical checks;
the separate baseline campaign measures behavior after one SFT epoch and 300
GRPO updates, without claiming convergence.

The actual GPU check inventory and numbers are recorded in
[`results/validation.json`](../results/validation.json). A path without a completed
GPU check there must not be described as GPU-validated.

### The multimodal padding defect

TRL pads rollout prompts on the left. When given a processor explicitly, it does
not necessarily change that processor's default padding side. Our processor
right-padded the image-token type markers. Mixed-length prompts therefore combined
left-padded tokens with right-padded modality markers. A two-prompt CPU reproduction
showed 16 misplaced markers and a native M-RoPE shape error; a mixed-batch GPU run
also reproduced the failure.

Both RL entrypoints now construct the processor with `padding_side="left"`. Native
batched positions match an independent unpadded, per-example reference. The former
shape-error handler, including its 1D position fallback, was removed. This is a
correctness fix, not a new training objective. The historical fallback's occurrence
was not logged, so successful legacy training cannot establish that it never ran.

### The effective GRPO recipe

The locked TRL 1.8.0 configuration uses DAPO token loss reduction, group reward
standardization, and weighted-sum-then-normalize aggregation. Policy ratios are
per-token, clipped symmetrically at 0.2. Default KL penalty is zero. The vLLM
sequence-mask correction has an upper ratio limit of 3. Dynamic resampling,
clip-higher, adaptive entropy, and truncated-response masking are absent.

The correctness-only arm still requires an answer tag before assigning reward.
It removes the separate format bonus, not the extraction requirement. Training
normalizes answers; author-code evaluation preserves its dataset-specific rules
and accepts raw text when no answer tag exists. Report these differences explicitly.

Zero-variance groups yield no group-relative correctness advantage. Measure their
frequency before selecting dynamic sampling or another normalization. Do not add
all newer recipe options by default. Length, token budget, and runtime matter
when comparing methods at the same update count.

In the 8B GRPO hardware check, five of eight prompt groups had zero reward
variance. That small sample illustrates the issue but is insufficient to choose
a new sampling recipe.

External-teacher logits are now cast to fp32 before log-softmax, matching the
student scoring precision. The old external path returned bf16 log-probabilities;
the effect of that precision difference on final accuracy was not measured.

The historical NF4 student failure is a separate synchronization issue. That path
is blocked at launch. Larger-GPU profiles use bf16 students; the 24 GB 2B route
remains available. An external NF4 teacher is not synchronized into vLLM.

## Data identity and task coverage

The split check hashes decoded RGB pixels and dimensions, so different lossless
encodings of the same image agree. It checks exact identity, not near duplicates,
pretraining exposure, or patient identity.

- VQAv2 evaluation overlaps the combined historical training pools on 47 images
  and 53 questions. Pool membership alone does not prove exposure in every run.
- The historical SFT dev slice has 487 unique images, of which 317 also occur in
  its training slice. Question IDs alone did not protect this split.
- New filtering retains 20,000 VQAv2 SFT and 19,964 GQA SFT examples. Splitting whole
  image groups gives 39,458 training and 506 dev examples.
- The initial `image-disjoint-v1` checks used 7,911 VQAv2 and 7,966 GQA RL examples.
  That policy also excluded SFT training images; this is optional experimental
  separation, not a requirement for valid held-out evaluation.
- The current `heldout-images-v2` policy reuses SFT training examples in RL: 27,553
  VQAv2 and 27,885 GQA examples, totaling 55,438. The same 506-example SFT dev set
  is fixed in the manifest and excluded from all RL sources, including shared
  images under other dataset names. Evaluation images remain excluded everywhere.
  SFT and RL may reuse a question; GRPO generates fresh on-policy responses.
  Both RL arms use the same pool and seed. SFT adds prior exposure and compute,
  which are reported separately. The completed GPU smoke checks used v1; their
  numerical conclusions are separate from the completed four-arm v2 comparison.
- CLEVR's 8,000 training, 3,000 validation, and 3,000 test examples are locally
  image-disjoint. They are a counting-subset partition, not a general CLEVR split.

Run `python -m scripts.check_data` after materialization. It writes
`data/split_manifest.json` without editing the Arrow datasets. Training verifies
question IDs and dataset fingerprints against that manifest. Historical source
revisions were not pinned sufficiently to reconstruct every upstream download;
new manifests establish local identity, not retroactive upstream provenance. New
materializations now pin the five source repositories and record those revisions.
`prepare_data.py` refuses to overwrite existing subsets; use a new `--out-dir`
for another materialization.

VQAv2 is useful for answering, formatting, and metric behavior, but it is not by
itself evidence for improved multi-step reasoning. GQA offers compositional
questions; the current CLEVR materialization only covers counting. Increase
training data only after checking actual prompt exposure, difficulty, and reward
variation. A single focused reasoning task is a better next decision than an
undirected dataset expansion. Medical VQA should be a separate, domain-specific
choice with patient/study separation and appropriate answer evaluation.

## What remains after short-answer SFT

A small fixed-set check used ten VQAv2 and ten GQA questions, each on a distinct
image absent from the historical training pools. Both arms used the reasoning
prompt, greedy decoding plus eight temperature-1 samples, and a 512-token limit.
The samples are random draws, not the eight most probable complete sequences.

| Arm and decoding | Responses | Mean tokens | Valid answer tags | Long-explanation proxy |
|---|---:|---:|---:|---:|
| Base, greedy | 20 | 134.7 | 19 | 18 |
| SFT, greedy | 20 | 9.7 | 5 | 1 |
| Base, eight samples | 160 | 127.5 | 155 | 147 |
| SFT, eight samples | 160 | 13.6 | 55 | 9 |

The proxy means more than 100 characters before an opening answer tag. It does
not establish substantive or correct reasoning. SFT produced such samples for
six of twenty questions; two of the nine samples received full normalized answer
credit, and one lacked a closing answer tag. Some explanatory behavior remains,
but this does not establish reliable recovery through RL. Base-to-GRPO remains
the straightforward control. A reasoning-oriented SFT mixture is a later,
separately evaluated objective choice.

The fixed set is too small for general model rankings. In particular, mean
normalized answer credit was 61.0% for base greedy and 49.5% for SFT greedy, while
original scoring gave 41.0% and 49.5%. This illustrates why score definitions and
sampling protocols must accompany a result. One base greedy response hit the
512-token limit. No repeated training seeds were added.

[All 360 responses](../results/sft-samples/predictions.jsonl),
[summary counts](../results/sft-samples/analysis.json), and
[run and merge metadata](../results/sft-samples/summary.json) are included.
To reproduce with the existing 2B SFT adapter and materialized data:

```bash
uv run --frozen python -m scripts.compare_sft --output-dir runs/sft-comparison
```

The measured comparison took 215 seconds after loading on an RTX 4090. An fp32
LoRA merge check on one fixed next-token distribution had maximum logit difference
0.0000315 and the same greedy next token. This checks merge arithmetic on that
input; it is not a guarantee of bitwise-identical bf16 generation.

## Official scorer comparison

The primary scorers were compared against dataset-author implementations on
626,074 saved predictions from 89 runs, with zero mismatches at tolerance 1e-10.
Answer extraction was held identical. The original outputs were never overwritten.

- VQAv2: [original evaluator](https://github.com/GT-Vision-Lab/VQA/blob/a013f0043c1e2cdc995922dfe257f7149aa9af06/PythonEvaluationTools/vqaEvaluation/vqaEval.py).
  Its Python 2 print statements are converted for execution; scoring is unchanged.
- TextVQA: [MMF evaluator](https://github.com/facebookresearch/mmf/blob/b0ffc7fba39f210f833c127362146aec01a2f259/mmf/utils/m4c_evaluators.py).
  The two answer-processing/evaluation classes are executed directly.
- GQA: `eval.py` from the [official evaluation archive](https://downloads.cs.stanford.edu/nlp/data/gqa/eval.zip).
  The comparison executes its accuracy expression. The broader original suite
  needs additional annotations and, for grounding, attention maps.

For independent reproduction, save these files as `vqav2.py`, `textvqa.py`, and
`eval.py` in one directory, then run:

```bash
uv run --frozen python -m scripts.check_metrics --sources-dir path/to/evaluators \
  --output results/metric-parity-reproduction.json
uv run --frozen python -m scripts.rescore --metric official
uv run --frozen python -m scripts.make_diagnostics
uv run --frozen python -m scripts.make_tables --metric official
```

Original VQA normalization is conditional: unanimous answers can bypass
normalization. GQA uses strict answer equality. TextVQA adds EvalAI tokenization.
These details can change scores for capitalization, punctuation, and articles.
The older normalized scorer remains available for historical comparisons and
training rewards; it is not silently relabeled official.

## Reading the research evidence

The corrected P1 result does not support a metric ranking reversal. P2's 143 paired
examples show an empty-reasoning change of −2.32 percentage points, with a 90%
interval of [−5.5, +0.8]; this does not establish equivalence within one, two, or
three points. Swapped reasoning remains load-bearing in that intervention.
P3 is inconclusive under the corrected published retention proxy.

The image audit evaluated 2,300 probes on ten models, not all 4,300 prepared battery
items. No detected degradation under its short-prompt protocol does not establish
correct grounding or isolate the vision encoder. The verifier's high internal dev
accuracy comes from derived labels and is not a substitute for human calibration.

Repeated training seeds are deferred. Reusing a saved replication or resampling
saved predictions is allowed; prediction bootstrap does not estimate training-seed
variability. The recorded gallery deliberately includes successes and counterexamples
and should not be treated as a random sample.

## Lessons to run without another training campaign

- Reproduce the single-reference reward edge case with
  `uv run --frozen pytest tests/test_metrics.py tests/test_rewards.py`.
- Compare official and normalized scores on the same saved predictions with
  `scripts.rescore` and `scripts.make_tables`; inspect an example in the gallery.
- Inspect `run_config.json` and `run_summary.json` after a two-update GRPO check.
  Count prompt groups, response tokens, zero-variance rewards, and actual updates.
- Read `train_opd.py` alongside the token-level checks in `results/validation.json`.
  Follow which prompt gets the reference answer and where advantages are detached.
- Use the Arena's original/black-image toggle, or the existing
  `idea-stage/battery/run_battery.py --help`, to study image sensitivity. Keep
  attention-based grounding and counterfactual sensitivity as separate measures.

Human verifier calibration and a new medical task remain later decisions.
The corrected comparison supplies the next set of measured implementation lessons.
