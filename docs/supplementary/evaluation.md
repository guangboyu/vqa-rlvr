# Evaluation: how the harness works, in common terms

This page describes the evaluation harness (`vqar/eval/`) in the terms public harnesses
such as lmms-eval and VLMEvalKit use, and states where it follows their practice and where
it differs. The [validation](validation.md) page has the checks behind it.

## What one evaluation is

An evaluation is one **cell**: one model under one **protocol**. The protocol declares
everything that makes two numbers comparable:

| Field | Here | Common name |
|---|---|---|
| dataset and role | e.g. VQAv2 `test` | benchmark and split |
| tier | `smoke` (32), `dev` (500), `full`, or `first-N` | subset size |
| prompt template | `short` ("Answer the question using a single word or phrase.") or `reasoning` | post-prompt / instruction |
| decoding | greedy (temperature 0), or k samples at a temperature | generation config |
| token budget | 64 (short) or 512 (reasoning) | max new tokens |
| image mode | original, black, swapped, or canvas (the RL training geometry) | ablation / counterfactual input |
| extraction | last `<answer>` tag, else the whole response | answer parsing |
| scorers | the authors' evaluator and a normalized matcher, with versions | metric |
| data | the materialized subsets' fingerprints and a hash of the rows selected | — |

The protocol's identity is a hash of these declared fields, never of source files. The
model's identity is its Hub revision or a hash of its weights. A **record** stores both
identities, both scores overall and per answer type, behavior metrics, and every prediction.
Repeating a cell is a cache hit; an interrupted cell resumes from its saved batches. Every
table is generated from records, and a test fails when a committed table drifts from them.

## Scoring

- **VQAv2**: the official VQA accuracy, min(matching annotators / 3, 1), averaged over the
  ten leave-one-out subsets of the ten annotators, after the official answer processing
  (which skips normalization when all annotators agree, so casing can matter).
- **TextVQA**: the same accuracy with the MMF/EvalAI answer processing.
- **GQA** and **CLEVR counting**: exact match; GQA's official match is case-sensitive.
- **Normalized** (companion column, and the RL reward): lowercase, strip punctuation and
  articles, map number words, then the same accuracy rule. It shows how much of a gap is
  answer form rather than content ([ADR 0002](../adr/0002-official-scorer-primary-normalized-companion.md)).

The official implementations were checked against the authors' code on 626,074 saved
predictions with no per-question mismatch.

## Comparing arms

- **Paired design.** Every arm answers the same questions with the same protocol, so
  differences are paired per question.
- **Uncertainty.** `scripts/paired_deltas.py` reports each difference with a 95% percentile
  bootstrap interval over questions (10,000 resamples). With one training seed per arm, the
  interval covers question sampling only, not training-seed variation.
- **Sampled protocols.** For k samples at a temperature, records report avg@k (mean over
  samples) and pass@k (at least one fully correct sample). `scripts/pass_at_k.py` turns the
  saved samples into the unbiased estimate 1 − C(n−c, k) / C(n, k) for every k up to n, with
  paired intervals that resample images ([held-out curves](supplement.md#held-out-passk-before-and-after-rl)).
- **Learnability.** The share of prompts whose k samples are neither all right nor all wrong
  under the training reward: the prompts that give GRPO a gradient.

## Dataset screening

`scripts/screen_datasets.py` chooses training datasets by measurement. For each registered
dataset and model it runs one fixed battery on the same questions from the dataset's
training pool (the `rl` role, else `sft`; the `dev` tier, up to 500 questions in the
pool's seeded order), never on dev or test rows:

| Measure | Protocols | Definition |
|---|---|---|
| Reasoning payoff | short and reasoning prompts, greedy | accuracy(reasoning) − accuracy(short), paired 95% interval |
| Matched payoff | direct and stepwise prompts, greedy | accuracy(stepwise) − accuracy(direct): both ask for the same tagged final answer, so answer format cannot explain the difference |
| RL headroom | reasoning prompt, 8 samples at T = 1, 640-token budget | pass@8 − pass@1, paired 95% interval |
| Learnable share | the same samples | questions with 1–7 of 8 samples correct |
| Image dependence | short prompt, original vs black and vs swapped images | accuracy(image) − accuracy(black), accuracy(image) − accuracy(swapped) |

Accuracies use the normalized scorer, the only one every role can score. The sampled cell is
exactly the learnability pre-check of `scripts.learnability --preset grpo_2b_main`, so
either script reuses the other's record, and "correct" there means the GRPO correctness
reward: pass@1 is the mean share of correct samples per question, pass@8 the share of
questions with at least one. The greedy cells read a training role through the protocol's
`screening` flag, which is not part of the protocol identity.

Hard gates, fixed in `vqar/eval/screening.py` before any candidate is measured. The last
three were added on 2026-10-06, before the reasoning candidates were measured, because the
general datasets had passed the first five although reasoning hurt them:

| Gate | Rule |
|---|---|
| training questions | at least 2,000 rows in the training pool role |
| split separable | declared per dataset, with the unit held out (image or patient) |
| license | declared usable per dataset |
| parse rate | at least 95% of the sampled reasoning responses give the reward's answer rule a non-empty answer (a last `<answer>` tag) |
| not saturated | short-prompt accuracy below 90% |
| reasoning not worse | the matched payoff's upper 95% bound is at least 0 |
| image needed | a black and a swapped image each cost at least 10 points (lower 95% bound) |
| learnable | at least 25% of questions have 1–7 of 8 samples right |

Since 2026-10-10 the gates are split by what they protect, because RL counts as a win when
it beats the starting model or SFT, with or without reasoning. Training questions, a separable
split, a usable license, not saturated and image needed hold for any RL. The other three
decide a route: reasoning-prompt RL needs the parse rate, reasoning no worse than a direct
answer, and the learnable share of sampled reasoning responses; short-answer RL needs the
learnable share of 8 short-prompt samples at T = 1, a ninth cell added to the battery. A
candidate passes when the common gates hold and at least one route is viable.
Survivors are ranked by matched payoff, then RL headroom. Every candidate gets a row,
rejects included with the gates they failed, in `results/screening/screening.{json,md}`.
A dataset without a training pool (TextVQA) is rejected; `--validate-on-test` screens it on
its test role instead, labelled validation only, to check the module against known results.

```bash
python -m scripts.screen_datasets --arms base2 --validate-on-test --dry-run  # the battery
python -m scripts.screen_datasets --arms base2 --validate-on-test            # missing cells
python -m scripts.screen_datasets --arms base2 base8 --validate-on-test --from-records
```

With Instruct 2B (2026-10-06, [table](../../results/screening/screening.md)), every general
dataset fails the reasoning gate: stepwise answers are 3–9 points below direct ones. The two
reasoning candidates split. CLEVR compositional is nearly saturated (88.4% short-prompt
accuracy), reasoning costs it 5.6 points under matched format, and 5.8% of its sampled
reasoning responses lack a readable answer, so it is rejected. ChartQA's human-written
questions pass every gate: reasoning neither helps nor hurts overall (reasoning − short
+2.0 [−2.0, +6.2], stepwise − direct 0.0 [−3.2, +3.2]), but it splits by answer type, +5.6
points on numeric questions and −5.5 on text ones under the reasoning prompt; 41% of its
questions give GRPO a gradient and pass@8 reaches 88.6% against 71.0% pass@1. Under the
direct and stepwise prompts the model often leaves out the answer tag (98% and 36% of
responses), and an untagged stepwise response is scored as a whole, so the matched payoff
understates reasoning. MedThinkVQA (radiology cases with five candidate diagnoses; up to eight
images per case are shown, and multi-image datasets add first-image-only and no-image cells) is
rejected as well: Instruct 2B answers from the clinical history and options alone. A black, a
swapped, a single or no image each leave its 37.2% short-prompt accuracy within 2 points, and
90% of its reasoning responses run past the 512-token budget. Instruct 8B (2026-10-08,
[table](../../results/screening/medthink-base8.md)) confirms the rejection: it scores 49.6%
with the images and 45.8% with none; a swapped image costs 6.2 points [+2.8, +9.6] and a
black one 2.6 [−1.0, +6.2], both short of the 10-point gate, and only 27% of its sampled
reasoning responses end in a readable answer. Of its 30 points over chance (20% with five
options), 26 remain without any image, so RL on these questions would mostly train text
reasoning, not visual question answering.

The medical candidates (2026-10-10, Instruct 2B, with the short-answer route) split three
ways. SLAKE and OmniMedVQA pass: the image matters (a black image costs 29 and 23 points, a
swapped one 20 and 29), both routes give a learning signal (learnable shares of 49% and 53%
under the reasoning prompt, 30% and 26% under the short one), and reasoning is no worse than a
direct answer under matched format (−1.4 [−4.0, +1.2] and −2.6 [−5.6, +0.4]) while costing
11 and 4 points against the short prompt, so short-answer RL is their natural route.
Kvasir-VQA-x1 and PathVQA fail the image gate: images are worth 7–10 points, with lower bounds
of 3–6. For Kvasir that is partly a floor: Instruct scores 28.4% because it does not know the
dataset's label vocabulary ("paris iia", "z-line"), so a black image has little to take away;
SFT on the labels first might make it screenable, which the gates here do not test. PathVQA,
the planned negative control, fails as its documented text shortcuts predicted. Under the
split gates VQAv2 and GQA now pass on the short-answer route, which matches what training
showed: short-answer RL there beats Instruct by 2–3 points but not SFT, so passing the screen
means RL can learn, not that it beats SFT.

ReXVQA (chest X-rays, 2026-10-11; the first of ten image archive parts, about a tenth of each
split's patients, one frontal view per study) fails the image gate as well. Its patient-level
split is the cleanest of the medical candidates, but Instruct 2B scores 52.6% with the X-ray
and 54.2% with a black image (−1.6 [−5.0, +1.8]); a swapped X-ray costs 4.4 points [+1.2,
+7.8]. No task type depends on the image: presence, negation, location and differential
diagnosis questions all score within 4 points with a black image. Its questions were written
from radiology reports, and with four options the model answers most of them from the wording
(chance is 25%), so RL on them would mostly train text cues.

## Held-out hygiene

Split files list the exact rows of every role. Images are compared by a hash of their
decoded pixels, so re-encoded copies across sources match. No dev or test image may appear
in any training pool of any dataset; this is checked before any training data loads.
Model selection and learning curves use the `dev` role; `test` numbers are reported only.

## Where this follows common practice

- Greedy decoding and the model's own chat template.
- The standard short-answer post-prompt used by lmms-eval for VQA-style benchmarks.
- The benchmark authors' evaluators as the primary metric.
- Per-sample predictions stored, so any number can be rescored or audited.
- Decontamination of training data against evaluation images.

A parity run against lmms-eval 0.7.3 on the same model matched its scores to 0.09 points on
GQA; on TextVQA the 1.15-point gap comes entirely from lmms-eval capitalizing each question
([validation](validation.md#parity-with-lmms-eval)).

## Where it differs, and why

| Difference | Why | Effect |
|---|---|---|
| VQAv2 is a seeded 5,000-question sample of validation, not test-dev | test-dev answers are hidden behind a submission server; full validation (214k) costs about 43 times as much | about ±0.6 points of sampling error; compare arms, not leaderboard numbers |
| CLEVR counting is a local held-out slice of the R1-V training source | the official CLEVR test has no public answers | no official number to compare with |
| Images are capped at about 0.79 megapixels | memory and speed on a 24 GB card | no effect on VQAv2, GQA, or CLEVR images; 826 of 5,000 TextVQA images (16.5%) are downscaled, and Instruct 2B at the model's own 16.8 MP budget scores −0.1 (short) and −0.7 (reasoning) points, intervals including zero |
| RL trains on images padded onto a 608×608 canvas (361 vision tokens); evaluation resizes to at most 768 tokens | the historical RL geometry, kept for comparable runs | evaluated on the canvas, Instruct 2B loses 5.6–6.9 points on TextVQA and 4.3 on ChartQA dev (photos within 1–2); training on the evaluation geometry instead (`train_grpo.py --image-fit budget`) reads 0 at 300 ChartQA updates, because trained models are evaluated on original images either way ([supplement](supplement.md#image-resolution)) |
| One training seed per arm | compute; seeds are deferred | intervals measure question sampling only |
| No model-based judge in the headline | exact scorers are reproducible and cheap | open-ended paraphrases count as wrong under both scorers; an LLM judge, audited by an image-aware model, rescores the misses on the [results](supplement.md#exact-match-misses-rescored-by-an-llm-judge) page |
| Reasoning-prompt generations are not bitwise repeatable | batched vLLM kernels vary with batch composition | differences smaller than the run-to-run spread are not evidence ([validation](validation.md)) |
