# What the current implementation establishes

Validation date: 2026-09-13. These implementation checks preceded the
[corrected campaign](method.md) and check that its training and evaluation paths
compute what they are meant to. Historical predictions remain intact; the split selection,
official scoring, and training fixes each have their own provenance.

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
[`results/validation.json`](../../results/validation.json). A path without a completed
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
  numerical conclusions are separate from the corrected campaign's four 2B arms.
- CLEVR's 8,000 training, 3,000 validation, and 3,000 test examples are locally
  image-disjoint. They are a counting-subset partition, not a general CLEVR split.

Run `python -m scripts.check_data` after materialization. It validates the split
files in `data/splits/` (one per dataset, versioned in the repository) without editing
the Arrow datasets, fails if any dev or test image appears in any training pool, and
writes `data/splits/overlap-report.json`. The `heldout-images-v2` manifest was migrated
into these files with every role's rows unchanged. Training and evaluation verify each
subset against them: the dataset fingerprint is a fast path, and when it differs the
question IDs, question and answer text, and decoded images are compared instead. On
2026-09-23 a fresh materialization from the pinned revisions reproduced all ten subsets
byte for byte; the three CLEVR subsets came back under new fingerprints, which the row
comparison accepts. Historical source
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
| Instruct, greedy | 20 | 134.7 | 19 | 18 |
| SFT, greedy | 20 | 9.7 | 5 | 1 |
| Instruct, eight samples | 160 | 127.5 | 155 | 147 |
| SFT, eight samples | 160 | 13.6 | 55 | 9 |

The proxy means more than 100 characters before an opening answer tag. It does
not establish substantive or correct reasoning. SFT produced such samples for
six of twenty questions; two of the nine samples received full normalized answer
credit, and one lacked a closing answer tag. Some explanatory behavior remains,
but this does not establish reliable recovery through RL. Instruct → GRPO remains
the straightforward control. A reasoning-oriented SFT mixture is a later,
separately evaluated objective choice.

The fixed set is too small for general model rankings. In particular, mean
normalized answer credit was 61.0% for Instruct greedy and 49.5% for SFT greedy, while
original scoring gave 41.0% and 49.5%. This illustrates why score definitions and
sampling protocols must accompany a result. One Instruct greedy response hit the
512-token limit. No repeated training seeds were added.

[All 360 responses](../../results/sft-samples/predictions.jsonl),
[summary counts](../../results/sft-samples/analysis.json), and
[run and merge metadata](../../results/sft-samples/summary.json) are included.
The sampling script that produced them is kept in the research archive, not in this tree.

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
uv run --frozen python -m scripts.report
```

Original VQA normalization is conditional: unanimous answers can bypass
normalization. GQA uses strict answer equality. TextVQA adds EvalAI tokenization.
These details can change scores for capitalization, punctuation, and articles.
The older normalized scorer remains available for historical comparisons and
training rewards; it is not silently relabeled official.

## Parity with lmms-eval

The same Instruct 2B model was evaluated with lmms-eval 0.7.3 (its vLLM backend, the same
0.79 MP image budget) on the full GQA testdev-balanced and TextVQA validation splits, and
`python -m scripts.compare_lmms_eval` paired its logged responses with this harness's
short-prompt records question by question
([results/parity-lmms-eval.json](../../results/parity-lmms-eval.json)).

| | lmms-eval | This harness (official) | lmms-eval's responses, this harness's scorer | Identical responses |
|---|---:|---:|---:|---:|
| GQA | 59.37 | 59.45 | 59.36 (normalized: 59.37) | 12,352 of 12,578 (98.2%) |
| TextVQA | 79.38 | 80.53 | 79.38 | 3,579 of 5,000 (71.6%) |

The scorers agree exactly: lmms-eval's own responses score the same under this harness's
TextVQA scorer and GQA normalized scorer as under lmms-eval (its GQA match ignores case,
like the normalized scorer). Any gap is therefore in the responses. On GQA it is 0.09
points. On TextVQA it is 1.15 points, and it has one cause: lmms-eval passes each TextVQA
question through Python's `str.capitalize()`, while the source questions are lowercase.
Asking this harness's engine the capitalized questions reproduced lmms-eval's response on
4,852 of 5,000 questions (97.0%) and scored 79.37 against lmms-eval's 79.38. The harness
keeps the source questions as released.

## Reproducing generations

The current harness reproduces the corrected campaign's saved short-prompt outputs exactly:
on the first 32 test questions of each of the four datasets, all 32 raw responses match.
Reasoning-prompt outputs, greedy but up to 512 tokens long, do not repeat run to run on
the local RTX 4090. Two runs of the same code on the same 32 questions share 8 to
12 identical responses per dataset, the earlier evaluation code behaves the same way, and
`VLLM_BATCH_INVARIANT=1` does not remove the difference. Over five runs, smoke-tier VQAv2
official accuracy under the reasoning prompt ranged from 0.503 to 0.553 on those 32
questions. The regression test therefore rescores saved predictions rather than generating
again, and a reasoning-prompt difference smaller than this spread is not evidence of a
change.

## Reading the research evidence

The corrected P1 result does not support a metric ranking reversal. In P2, replacing the
reasoning with an empty one changes accuracy by −2.32 percentage points (90% interval
[−5.48, +0.84], n = 487), which does not establish equivalence within one, two, or three
points. In the 143 pairs without answer leakage, swapping in a different reasoning text
changes 63.6% of answers, so in that intervention the answers depend on the reasoning. P3 is
inconclusive under the corrected published retention proxy.

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
  `uv run --frozen pytest tests/test_metrics.py -k SingleReference`.
- Compare official and normalized scores on the same saved predictions in the
  [results](results.md); inspect an example in the gallery.
- Inspect `run_config.json` and `run_summary.json` after a two-update GRPO check.
  Count prompt groups, response tokens, zero-variance rewards, and actual updates.
- Read `train_opd.py` alongside the token-level checks in `results/validation.json`.
  Follow which prompt gets the reference answer and where advantages are detached.
- Use the Arena's original/black-image toggle to study image sensitivity. Keep
  attention-based grounding and counterfactual sensitivity as separate measures.

Human verifier calibration and a new medical task remain later decisions.
The corrected comparison supplies the next set of measured implementation lessons.

## Part 1 checks (2026-09-25)

These checks preceded the recipe experiments and the final rerun.

- **Resumption.** A 2B GRPO run with a save after every update was killed with `SIGKILL`
  after its second checkpoint; rerunning the same command resumed from that checkpoint and
  completed its planned updates. The same path ran with dead-group refilling and with
  unmerged-adapter rollouts (RTX 4090, 2–10 updates each); refill rounds scored like first
  rounds and every update trained on four groups.
- **Sampler against trainer.** `scripts/check_logprob_gap.py` scored 512 sampled responses
  of one GRPO adapter under vLLM and under the trainer's forward: prompt token IDs matched
  in every response, and the per-token gap was 0.132 with merged bf16 weights in vLLM,
  0.011 with the same merged weights on both sides, and 0.008 with the adapter served
  unmerged ([method](method.md#sampler-and-trained-policy)).
- **Decoding noise.** Three repeats of Instruct 2B under the reasoning prompt on the full
  VQAv2 and GQA test subsets span 0.42 and 0.43 official points; 76–80% of responses were
  identical between any two runs. Records: `results/noise-floor/`.
- **Merged against unmerged evaluation.** One trained adapter evaluated both ways on the
  full test matrix differs by at most 0.42 points in every cell.
- **Screening battery.** `scripts/screen_datasets.py` on the four general datasets, with
  Instruct 2B and 8B, finds a negative reasoning payoff on all four (−3 to −13 points,
  intervals excluding zero), rejects CLEVR counting as saturated and TextVQA for having no
  training pool, as expected ([evaluation](evaluation.md)).


## Part 2 checks (2026-10-08)

These checks ran on ChartQA before any training knob was promoted to a full run.

- **Overfitting check.** Can the frozen GRPO recipe fit its own training data? The set is
  32 ChartQA pool questions that Instruct 2B answers right 1–7 times in 8 (pool
  `overfit32-chartqa-base2`); 200 updates of 4 questions make 25 passes over it. The pass
  rule was fixed before the run: mean training reward over the last five passes at least
  0.90, and at least 0.25 above the first pass. The first run rose from 0.38 to 0.70 and
  failed. Its completions showed two defects outside the learning algorithm. The reward
  marked right answers wrong when the label is a bracketed list or uses a Unicode dash
  (`[Gambia, Niger]` against `Gambia, Niger`; 22 of the 1,250 test labels are bracketed).
  And the 608-pixel training canvas blurred chart values the model read correctly at full
  resolution (next item). With a list-aware reward (`--rewards correctness_lists`) and an
  896-pixel canvas the rerun rose from 0.51 to 0.80, still short of 0.90: 15 questions
  ended solved (last-five mean at least 0.95), 12 between 0.5 and 0.95, 4 below 0.5 and 1
  at zero. The rising reward, the solved questions and a normal sampler–trainer gap show
  the update path working. A proposed explanation for the rest, not tested by its own run:
  the conservative step size (learning rate 1e-5 on LoRA r16) meets groups that score
  all alike and so give no gradient, 54% of groups over the last five passes. Scripts:
  `runs/overfit-20261008/` (local).
- **Training image geometry.** RL pads every image onto a 608 × 608 canvas (361 vision
  tokens); evaluation keeps each image's shape within 0.79 megapixels. On ChartQA's dev role
  Instruct 2B scores 69.52 under the reasoning prompt with original images and 65.24 on the
  canvas (−4.28 [−6.78, −1.77]); numeric answers lose 7.6 points (68.1 → 60.5) and text
  answers none (72.4 → 75.0). Photos lose at most 1 point on the same canvas
  ([image resolution](supplement.md#image-resolution)). `train_grpo.py --image-fit budget`
  trains on the evaluation geometry instead (opt-in; recorded runs keep the canvas). As a
  300-update ChartQA knob it reads 0 against the canvas-trained baseline, +0.1 points on dev
  ([knobs](supplement.md#training-knobs-chartqa)): the trained models are evaluated on original
  images either way.
- **Sampler against trainer, evaluation geometry.** Images of varying shape could make vLLM
  and the trainer's processor expand image tokens differently. `scripts/check_logprob_gap.py
  --datasets chartqa --questions 32 --image-fit budget` (Instruct 2B, 256 responses) found
  prompt token IDs identical in every response and a mean per-token gap of 0.0116, against
  0.0120 on the canvas; importance-sampling ratios averaged 1.023 and 1.022.
