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
| image mode | original, black, or swapped | ablation / counterfactual input |
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
  samples) and pass@k (at least one fully correct sample).
- **Learnability.** The share of prompts whose k samples are neither all right nor all wrong
  under the training reward: the prompts that give GRPO a gradient.

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

## Where it differs, and why

| Difference | Why | Effect |
|---|---|---|
| VQAv2 is a seeded 5,000-question sample of validation, not test-dev | test-dev answers are hidden behind a submission server; full validation (214k) costs about 43 times as much | about ±0.6 points of sampling error; compare arms, not leaderboard numbers |
| CLEVR counting is a local held-out slice of the R1-V training source | the official CLEVR test has no public answers | no official number to compare with |
| Images are capped at about 0.79 megapixels | memory and speed on a 24 GB card | no effect on VQAv2, GQA, or CLEVR images; 826 of 5,000 TextVQA images (16.5%) are downscaled |
| One training seed per arm | compute; seeds are deferred | intervals measure question sampling only |
| No model-based judge in the headline | exact scorers are reproducible and cheap | open-ended paraphrases count as wrong under both scorers |
| Reasoning-prompt generations are not bitwise repeatable | batched vLLM kernels vary with batch composition | differences smaller than the run-to-run spread are not evidence ([validation](validation.md)) |
