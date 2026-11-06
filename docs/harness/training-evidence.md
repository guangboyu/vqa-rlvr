# What each RL run trained on

Exposure, learnability and learning curves for the RL arms, generated from their logs and records.

## What each RL run trained on

All nine RL runs completed 1,000 updates on exactly the same ordered 4,000 questions, 7.2%
of the 55,438-question pool, with 32,000 sampled responses each. A group of eight responses
with identical rewards has zero advantage, so it contributes no correctness gradient;
optimizer momentum can still move the weights on such an update. The OPD run logs the same
correctness reward for diagnosis; its learning signal is the teacher's token scores, so its
reward columns do not drive its updates.

<!-- report:exposure -->
| Run | Updates | Questions | Responses | Groups with varied rewards (%) | Updates without reward variation | Mean reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `vqa-p1-f-grpo2-base` | 1,000 | 4,000 | 32,000 | 19.05 | 451 | 0.682 |
| `vqa-p1-f-grpo2-sft` | 1,000 | 4,000 | 32,000 | 30.88 | 257 | 0.682 |
| `vqa-p1-f-short2-base` | 1,000 | 4,000 | 32,000 | 21.12 | 389 | 0.697 |
| `vqa-p1-f-short2-sft` | 1,000 | 4,000 | 32,000 | 24.55 | 346 | 0.726 |
| `vqa-p1-f-opd2` | 1,000 | 4,000 | 32,000 | 48.30 | 82 | 0.613 |
| `vqa-p1-f-grpo8-base` | 1,000 | 4,000 | 32,000 | 15.65 | 535 | 0.725 |
| `vqa-p1-f-grpo8-sft` | 1,000 | 4,000 | 32,000 | 19.18 | 477 | 0.753 |
| `vqa-p1-f-short8-base` | 1,000 | 4,000 | 32,000 | 15.35 | 518 | 0.743 |
| `vqa-p1-f-short8-sft` | 1,000 | 4,000 | 32,000 | 20.70 | 428 | 0.765 |
<!-- /report:exposure -->

The next table follows each run's training signal. The reward (mean of the first and of
the last ten updates) and the share of prompt groups whose samples all scored the same (no
gradient) come from every update's saved draws; so does the split of those groups into
all right (every sample full credit) and all wrong (every sample 0), as means of the first
and of the last ten updates. Entropy and completion length are the trainer's logs (every
ten updates), first → last. KL is absent because these runs used no KL penalty.

No group in these runs agreed on a partial score: the RL roles keep one reference answer,
so the reward is 0 or 1. Most of each GRPO run trained on few live groups: only 15–31% of
groups had varied rewards over the whole run, and in 26–54% of updates none did. The dead
share grows through all-right groups as the model improves (2B from Instruct 0.38 → 0.62,
8B 0.42 → 0.70); from the SFT start, all-wrong groups fall (2B 0.45 → 0.12) while all-right
groups rise from 0.03 to 0.65 as the answer tag is relearned. Refilling dead groups
targets exactly this and did not change the test results
([recipe experiments](../negatives.md#recipe-experiments)).

<!-- report:diagnostics -->
| Run | Reward (first → last 10 updates) | Dead groups (mean) | All right (first → last 10) | All wrong (first → last 10) | Entropy | Clip fraction (mean) | Completion tokens | Distinct prompts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `vqa-p1-f-grpo2-base` | 0.63 → 0.71 | 0.809 | 0.38 → 0.62 | 0.17 → 0.25 | 0.73 → 0.06 | 0.000 | 152 → 8 | 4,000 |
| `vqa-p1-f-grpo2-sft` | 0.26 → 0.75 | 0.691 | 0.03 → 0.65 | 0.45 → 0.12 | 1.21 → 0.07 | 0.000 | 33 → 10 | 4,000 |
| `vqa-p1-f-short2-base` | 0.72 → 0.74 | 0.789 | 0.55 → 0.65 | 0.12 → 0.20 | 0.33 → 0.11 | 0.000 | 2 → 2 | 4,000 |
| `vqa-p1-f-short2-sft` | 0.68 → 0.77 | 0.754 | 0.45 → 0.68 | 0.12 → 0.12 | 0.41 → 0.15 | 0.000 | 2 → 2 | 4,000 |
| `vqa-p1-f-opd2` | 0.62 → 0.61 | 0.517 | 0.33 → 0.20 | 0.17 → 0.23 | 0.73 → 0.71 | 0.000 | 150 → 127 | 4,000 |
| `vqa-p1-f-grpo8-base` | 0.63 → 0.76 | 0.844 | 0.42 → 0.70 | 0.17 → 0.17 | 0.39 → 0.04 | 0.000 | 129 → 8 | 4,000 |
| `vqa-p1-f-grpo8-sft` | 0.54 → 0.77 | 0.808 | 0.10 → 0.75 | 0.20 → 0.23 | 0.84 → 0.02 | 0.000 | 66 → 8 | 4,000 |
| `vqa-p1-f-short8-base` | 0.74 → 0.71 | 0.847 | 0.65 → 0.68 | 0.20 → 0.23 | 0.14 → 0.06 | 0.000 | 2 → 2 | 4,000 |
| `vqa-p1-f-short8-sft` | 0.70 → 0.78 | 0.793 | 0.47 → 0.70 | 0.07 → 0.12 | 0.34 → 0.10 | 0.000 | 2 → 2 | 4,000 |
<!-- /report:diagnostics -->

## Learnability of the RL pool

Before an RL run, `python -m scripts.learnability` takes a seeded random sample of each
dataset's RL pool (500 questions, or the first N with `--tier first-N`), draws a group of
eight responses per prompt from the starting model with the run's rollout settings, and
scores them with the GRPO correctness reward. Only prompts whose responses are neither all
right nor all wrong give GRPO a gradient. The pre-check shows images at the evaluation
resolution rather than the RL canvas, so it approximates the rate a run sees. The 5,000-
question reasoning-prompt records of Instruct 2B define the filtered pool of the recipe
experiments (1–7 of 8 correct: 4,609 questions); the short-prompt records checked that
short answers sampled at temperature 1 still disagree often enough to train on. They do:
36–40% learnable for Instruct 2B, 53–54% for SFT 2B, and 24–26% for Instruct 8B, whose
short answers are right on all eight samples for 57–60% of questions. The learnable share
at the start overstates what a run keeps: the exposure table above shows 15–31% of groups
with varied rewards over 1,000 updates.

<!-- report:learnability -->
| Model | Dataset | Prompt | Prompts | Samples each | Learnable (%) | Never correct (%) | Always correct (%) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `base2` | ChartQA (human) | reasoning | 500 | 8 | 41.2 | 11.4 | 47.4 |
| `base2` | ChartQA (human) | short | 500 | 8 | 30.0 | 20.2 | 49.8 |
| `base2` | CLEVR counting | reasoning | 500 | 8 | 48.4 | 0.0 | 51.6 |
| `base2` | CLEVR counting | short | 500 | 8 | 4.4 | 0.8 | 94.8 |
| `base2` | CLEVR compositional | reasoning | 500 | 8 | 62.0 | 5.2 | 32.8 |
| `base2` | CLEVR compositional | short | 500 | 8 | 14.8 | 6.2 | 79.0 |
| `base2` | DeepTumorVQA (whole-volume montage) | reasoning | 500 | 8 | 55.6 | 29.6 | 14.8 |
| `base2` | DeepTumorVQA (whole-volume montage) | short | 500 | 8 | 61.6 | 23.6 | 14.8 |
| `base2` | DeepTumorVQA (debiased options) | reasoning | 500 | 8 | 57.2 | 30.6 | 12.2 |
| `base2` | DeepTumorVQA (debiased options) | short | 500 | 8 | 64.4 | 24.4 | 11.2 |
| `base2` | GQA | reasoning | 500 | 8 | 42.0 | 26.6 | 31.4 |
| `base2` | GQA | short | 500 | 8 | 33.4 | 19.2 | 47.4 |
| `base2` | GQA | reasoning | 2,500 | 8 | 44.5 | 26.0 | 29.5 |
| `base2` | GQA | short | 2,500 | 8 | 35.5 | 19.1 | 45.4 |
| `base2` | GQA | reasoning | 5,000 | 8 | 45.0 | 24.3 | 30.7 |
| `base2` | Kvasir-VQA-x1 (atomic) | reasoning | 500 | 8 | 36.8 | 53.8 | 9.4 |
| `base2` | Kvasir-VQA-x1 (atomic) | short | 500 | 8 | 23.0 | 63.2 | 13.8 |
| `base2` | MedThinkVQA | reasoning | 500 | 8 | 47.4 | 52.0 | 0.6 |
| `base2` | MedThinkVQA | short | 500 | 8 | 33.2 | 43.8 | 23.0 |
| `base2` | OmniMedVQA (open access) | reasoning | 500 | 8 | 52.8 | 8.2 | 39.0 |
| `base2` | OmniMedVQA (open access) | short | 500 | 8 | 25.6 | 15.8 | 58.6 |
| `base2` | OmniMedVQA (held-out collections) | reasoning | 500 | 8 | 53.6 | 10.0 | 36.4 |
| `base2` | OmniMedVQA (held-out collections) | short | 500 | 8 | 25.6 | 20.0 | 54.4 |
| `base2` | PathVQA | reasoning | 467 | 8 | 38.8 | 54.2 | 6.6 |
| `base2` | PathVQA | short | 467 | 8 | 33.8 | 52.5 | 12.8 |
| `base2` | SLAKE (English) | reasoning | 500 | 8 | 48.8 | 35.0 | 16.2 |
| `base2` | SLAKE (English) | short | 500 | 8 | 30.4 | 34.6 | 35.0 |
| `base2` | VQAv2 | reasoning | 500 | 8 | 47.8 | 19.8 | 32.4 |
| `base2` | VQAv2 | short | 500 | 8 | 40.6 | 12.4 | 47.0 |
| `base2` | VQAv2 | reasoning | 2,500 | 8 | 48.9 | 21.3 | 29.8 |
| `base2` | VQAv2 | short | 2,500 | 8 | 40.4 | 15.1 | 44.5 |
| `base2` | VQAv2 | reasoning | 5,000 | 8 | 47.1 | 22.6 | 30.3 |
| `base8` | CLEVR counting | reasoning | 500 | 8 | 15.2 | 0.2 | 84.6 |
| `base8` | GQA | reasoning | 500 | 8 | 31.2 | 25.4 | 43.4 |
| `base8` | GQA | short | 2,500 | 8 | 23.6 | 19.2 | 57.2 |
| `base8` | GQA | reasoning | 5,000 | 8 | 34.7 | 23.9 | 41.4 |
| `base8` | MedThinkVQA | reasoning | 500 | 8 | 39.2 | 58.6 | 2.2 |
| `base8` | VQAv2 | reasoning | 500 | 8 | 31.8 | 21.0 | 47.2 |
| `base8` | VQAv2 | short | 2,500 | 8 | 25.9 | 13.8 | 60.2 |
| `base8` | VQAv2 | reasoning | 5,000 | 8 | 33.1 | 22.9 | 43.9 |
| `sft2` | GQA | short | 2,500 | 8 | 54.2 | 10.4 | 35.3 |
| `sft2` | VQAv2 | short | 2,500 | 8 | 53.1 | 11.5 | 35.4 |
<!-- /report:learnability -->

## Learning curves

Each point evaluates a saved adapter on the dev tier through vLLM's LoRA serving rather
than a merged copy; on the first 500 VQAv2 test questions (short prompt) the two gave the
same answer 496 times ([check](../../results/lora-serving-check.json)). Columns are
training updates; each cell is official (normalized) dev accuracy · mean output tokens.
The corrected campaign's runs predate this step.
The dev role is small (447 VQAv2 and 59 GQA questions, so one question moves GQA by 1.7
points); read the curves for their shape. Under the reasoning prompt, both GRPO runs from
Instruct reach their plateau by update 200–300, when the responses have shrunk to 8
tokens, and stay there to update 1,000. From SFT the 2B run climbs more slowly while it
still writes 40–60 tokens, then shortens after update 600. Short-prompt dev accuracy moves
by about 1 point or less in every run, and OPD's reasoning-prompt curve stays flat at 63–65
on VQAv2 with its responses near 140 tokens.

<!-- report:curves -->
| Arm | Dataset | Prompt | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 1000 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Instruct → GRPO | VQAv2 | short | — (71.6) · 2 | — (72.3) · 2 | — (71.8) · 2 | — (71.4) · 2 | — (71.6) · 2 | — (71.6) · 2 | — (71.6) · 2 | — (72.0) · 2 | — (71.8) · 2 | — (72.0) · 2 |
| Instruct → GRPO | VQAv2 | reasoning | — (64.4) · 113 | — (74.9) · 11 | — (74.0) · 8 | — (74.0) · 8 | — (74.5) · 8 | — (74.3) · 8 | — (74.9) · 8 | — (74.5) · 8 | — (74.9) · 8 | — (75.2) · 8 |
| Instruct → GRPO | GQA | short | 66.1 (66.1) · 2 | 66.1 (66.1) · 2 | 66.1 (66.1) · 2 | 66.1 (66.1) · 2 | 66.1 (66.1) · 2 | 66.1 (66.1) · 2 | 67.8 (67.8) · 2 | 66.1 (66.1) · 2 | 66.1 (66.1) · 2 | 66.1 (66.1) · 2 |
| Instruct → GRPO | GQA | reasoning | 49.2 (57.6) · 99 | 62.7 (62.7) · 8 | 61.0 (61.0) · 8 | 71.2 (71.2) · 8 | 69.5 (69.5) · 8 | 69.5 (69.5) · 8 | 69.5 (69.5) · 8 | 67.8 (67.8) · 8 | 67.8 (67.8) · 8 | 66.1 (66.1) · 8 |
| SFT → GRPO | VQAv2 | short | — (74.3) · 2 | — (73.4) · 2 | — (73.6) · 2 | — (73.8) · 2 | — (74.3) · 2 | — (74.3) · 2 | — (74.5) · 2 | — (74.0) · 2 | — (74.5) · 2 | — (74.7) · 2 |
| SFT → GRPO | VQAv2 | reasoning | — (64.4) · 65 | — (69.1) · 41 | — (70.5) · 51 | — (71.8) · 59 | — (71.1) · 50 | — (71.8) · 42 | — (72.7) · 20 | — (72.0) · 11 | — (72.9) · 11 | — (72.0) · 11 |
| SFT → GRPO | GQA | short | 66.1 (66.1) · 2 | 66.1 (66.1) · 2 | 66.1 (66.1) · 2 | 67.8 (67.8) · 2 | 67.8 (67.8) · 2 | 67.8 (67.8) · 2 | 67.8 (67.8) · 2 | 66.1 (66.1) · 2 | 69.5 (69.5) · 2 | 67.8 (67.8) · 2 |
| SFT → GRPO | GQA | reasoning | 54.2 (54.2) · 73 | 55.9 (55.9) · 26 | 64.4 (64.4) · 33 | 66.1 (66.1) · 37 | 67.8 (67.8) · 33 | 66.1 (66.1) · 29 | 64.4 (64.4) · 14 | 62.7 (62.7) · 11 | 66.1 (66.1) · 11 | 66.1 (66.1) · 11 |
| Instruct → short-prompt GRPO | VQAv2 | short | — (70.9) · 2 | — (72.0) · 2 | — (71.8) · 2 | — (71.1) · 2 | — (72.3) · 2 | — (71.6) · 2 | — (71.4) · 2 | — (71.1) · 2 | — (71.6) · 2 | — (71.6) · 2 |
| Instruct → short-prompt GRPO | VQAv2 | reasoning | — (57.7) · 145 | — (61.3) · 143 | — (63.3) · 139 | — (66.0) · 137 | — (64.7) · 136 | — (63.3) · 135 | — (65.3) · 131 | — (63.3) · 132 | — (64.4) · 132 | — (64.9) · 132 |
| Instruct → short-prompt GRPO | GQA | short | 67.8 (67.8) · 2 | 67.8 (67.8) · 2 | 67.8 (67.8) · 2 | 67.8 (67.8) · 2 | 71.2 (71.2) · 2 | 71.2 (71.2) · 2 | 69.5 (69.5) · 2 | 71.2 (71.2) · 2 | 69.5 (69.5) · 2 | 71.2 (71.2) · 2 |
| Instruct → short-prompt GRPO | GQA | reasoning | 37.3 (50.8) · 113 | 39.0 (52.5) · 104 | 44.1 (55.9) · 105 | 39.0 (49.2) · 107 | 45.8 (57.6) · 105 | 39.0 (52.5) · 112 | 47.5 (59.3) · 106 | 44.1 (55.9) · 106 | 42.4 (52.5) · 106 | 44.1 (55.9) · 105 |
| SFT → short-prompt GRPO | VQAv2 | short | — (74.9) · 2 | — (74.7) · 2 | — (73.4) · 2 | — (72.9) · 2 | — (73.2) · 2 | — (73.4) · 2 | — (73.4) · 2 | — (73.4) · 2 | — (73.8) · 2 | — (73.6) · 2 |
| SFT → short-prompt GRPO | VQAv2 | reasoning | — (64.4) · 22 | — (64.9) · 19 | — (63.5) · 24 | — (63.3) · 28 | — (63.1) · 27 | — (62.0) · 26 | — (62.0) · 27 | — (61.5) · 28 | — (61.1) · 29 | — (61.3) · 29 |
| SFT → short-prompt GRPO | GQA | short | 67.8 (67.8) · 2 | 69.5 (69.5) · 2 | 67.8 (67.8) · 2 | 69.5 (69.5) · 2 | 67.8 (67.8) · 2 | 69.5 (69.5) · 2 | 69.5 (69.5) · 2 | 67.8 (67.8) · 2 | 69.5 (69.5) · 2 | 67.8 (67.8) · 2 |
| SFT → short-prompt GRPO | GQA | reasoning | 57.6 (59.3) · 5 | 62.7 (62.7) · 8 | 62.7 (62.7) · 13 | 64.4 (64.4) · 15 | 64.4 (64.4) · 14 | 62.7 (62.7) · 16 | 62.7 (62.7) · 17 | 61.0 (61.0) · 15 | 61.0 (61.0) · 15 | 62.7 (62.7) · 16 |
| Instruct → gold-conditioned OPD | VQAv2 | short | — (71.6) · 2 | — (72.3) · 2 | — (71.8) · 2 | — (71.6) · 2 | — (71.6) · 2 | — (71.6) · 2 | — (71.8) · 2 | — (71.4) · 2 | — (71.6) · 2 | — (72.0) · 2 |
| Instruct → gold-conditioned OPD | VQAv2 | reasoning | — (62.9) · 132 | — (63.5) · 139 | — (63.8) · 140 | — (64.7) · 138 | — (63.1) · 143 | — (65.1) · 142 | — (62.9) · 145 | — (64.7) · 142 | — (64.9) · 140 | — (64.0) · 145 |
| Instruct → gold-conditioned OPD | GQA | short | 66.1 (66.1) · 2 | 66.1 (66.1) · 2 | 66.1 (66.1) · 2 | 67.8 (67.8) · 2 | 67.8 (67.8) · 2 | 67.8 (67.8) · 2 | 67.8 (67.8) · 2 | 67.8 (67.8) · 2 | 67.8 (67.8) · 2 | 67.8 (67.8) · 2 |
| Instruct → gold-conditioned OPD | GQA | reasoning | 55.9 (55.9) · 106 | 54.2 (54.2) · 102 | 52.5 (52.5) · 98 | 54.2 (54.2) · 106 | 55.9 (55.9) · 105 | 54.2 (54.2) · 104 | 50.8 (50.8) · 115 | 50.8 (50.8) · 109 | 52.5 (52.5) · 115 | 55.9 (55.9) · 106 |
| Instruct 8B → GRPO | VQAv2 | short | — (75.2) · 2 | — (74.7) · 2 | — (75.2) · 2 | — (74.7) · 2 | — (74.7) · 2 | — (75.4) · 2 | — (75.4) · 2 | — (75.4) · 2 | — (75.6) · 2 | — (75.4) · 2 |
| Instruct 8B → GRPO | VQAv2 | reasoning | — (71.1) · 10 | — (74.0) · 8 | — (77.0) · 8 | — (75.4) · 8 | — (76.5) · 8 | — (77.6) · 8 | — (77.2) · 8 | — (78.1) · 8 | — (78.3) · 8 | — (78.3) · 8 |
| Instruct 8B → GRPO | GQA | short | 72.9 (72.9) · 2 | 72.9 (72.9) · 2 | 74.6 (74.6) · 2 | 74.6 (74.6) · 2 | 74.6 (74.6) · 2 | 74.6 (74.6) · 2 | 74.6 (74.6) · 2 | 74.6 (74.6) · 2 | 74.6 (74.6) · 2 | 74.6 (74.6) · 2 |
| Instruct 8B → GRPO | GQA | reasoning | 61.0 (61.0) · 10 | 66.1 (66.1) · 8 | 66.1 (66.1) · 8 | 69.5 (69.5) · 8 | 71.2 (71.2) · 8 | 72.9 (72.9) · 8 | 71.2 (71.2) · 8 | 72.9 (72.9) · 8 | 72.9 (72.9) · 8 | 72.9 (72.9) · 8 |
| SFT 8B → GRPO | VQAv2 | short | — (78.3) · 2 | — (78.3) · 2 | — (78.5) · 2 | — (78.1) · 2 | — (79.0) · 2 | — (78.3) · 2 | — (78.5) · 2 | — (78.7) · 2 | — (79.0) · 2 | — (78.7) · 2 |
| SFT 8B → GRPO | VQAv2 | reasoning | — (74.3) · 8 | — (78.5) · 8 | — (76.7) · 8 | — (78.3) · 8 | — (77.6) · 8 | — (77.9) · 8 | — (78.3) · 8 | — (79.0) · 8 | — (78.3) · 8 | — (78.1) · 8 |
| SFT 8B → GRPO | GQA | short | 76.3 (76.3) · 2 | 76.3 (76.3) · 2 | 78.0 (78.0) · 2 | 78.0 (78.0) · 2 | 78.0 (78.0) · 2 | 78.0 (78.0) · 2 | 78.0 (78.0) · 2 | 78.0 (78.0) · 2 | 78.0 (78.0) · 2 | 78.0 (78.0) · 2 |
| SFT 8B → GRPO | GQA | reasoning | 69.5 (69.5) · 8 | 71.2 (71.2) · 8 | 69.5 (69.5) · 8 | 71.2 (71.2) · 8 | 72.9 (72.9) · 8 | 72.9 (72.9) · 8 | 72.9 (72.9) · 8 | 72.9 (72.9) · 8 | 72.9 (72.9) · 8 | 72.9 (72.9) · 8 |
| Instruct 8B → short-prompt GRPO | VQAv2 | short | — (75.4) · 2 | — (75.2) · 2 | — (74.3) · 2 | — (74.0) · 2 | — (75.8) · 2 | — (74.3) · 2 | — (75.8) · 2 | — (75.8) · 2 | — (75.8) · 2 | — (76.5) · 2 |
| Instruct 8B → short-prompt GRPO | VQAv2 | reasoning | — (60.9) · 112 | — (62.0) · 111 | — (62.9) · 110 | — (63.8) · 111 | — (63.8) · 110 | — (63.5) · 110 | — (64.2) · 111 | — (64.0) · 108 | — (63.5) · 110 | — (65.5) · 108 |
| Instruct 8B → short-prompt GRPO | GQA | short | 74.6 (74.6) · 2 | 72.9 (72.9) · 2 | 74.6 (74.6) · 2 | 74.6 (74.6) · 2 | 74.6 (74.6) · 2 | 74.6 (74.6) · 2 | 74.6 (74.6) · 2 | 74.6 (74.6) · 2 | 74.6 (74.6) · 2 | 74.6 (74.6) · 2 |
| Instruct 8B → short-prompt GRPO | GQA | reasoning | 52.5 (59.3) · 115 | 52.5 (59.3) · 111 | 52.5 (59.3) · 111 | 52.5 (59.3) · 101 | 54.2 (59.3) · 104 | 54.2 (61.0) · 100 | 54.2 (61.0) · 101 | 55.9 (61.0) · 100 | 55.9 (61.0) · 107 | 55.9 (61.0) · 105 |
| SFT 8B → short-prompt GRPO | VQAv2 | short | — (79.0) · 2 | — (77.6) · 2 | — (78.3) · 2 | — (78.3) · 2 | — (78.5) · 2 | — (77.9) · 2 | — (78.1) · 2 | — (77.9) · 2 | — (77.9) · 2 | — (77.9) · 2 |
| SFT 8B → short-prompt GRPO | VQAv2 | reasoning | — (63.3) · 71 | — (64.9) · 73 | — (65.1) · 77 | — (67.1) · 73 | — (63.5) · 78 | — (65.3) · 72 | — (65.5) · 75 | — (63.1) · 74 | — (64.7) · 74 | — (64.2) · 74 |
| SFT 8B → short-prompt GRPO | GQA | short | 76.3 (76.3) · 2 | 76.3 (76.3) · 2 | 78.0 (78.0) · 2 | 76.3 (76.3) · 2 | 76.3 (76.3) · 2 | 76.3 (76.3) · 2 | 78.0 (78.0) · 2 | 78.0 (78.0) · 2 | 76.3 (76.3) · 2 | 78.0 (78.0) · 2 |
| SFT 8B → short-prompt GRPO | GQA | reasoning | 52.5 (54.2) · 111 | 45.8 (45.8) · 107 | 52.5 (52.5) · 103 | 47.5 (49.2) · 110 | 52.5 (52.5) · 113 | 50.8 (50.8) · 111 | 47.5 (49.2) · 126 | 47.5 (47.5) · 119 | 45.8 (47.5) · 107 | 47.5 (47.5) · 118 |
<!-- /report:curves -->
