# Results

Every table on this page is generated from the stored evaluation records by
`python -m scripts.report`; a test fails if a committed copy drifts from the records.
Scores are percentages on the materialized test subsets with greedy decoding and one
training seed per arm. The official scorer is the dataset authors' evaluator and is the
primary number; the normalized scorer, which also defines the training reward, appears
beside it. The [method](method.md) describes the arms and how to rerun them.

## Prompt template × scorer variant

Official accuracy, with the normalized scorer in parentheses. The short prompt asks for "a
single word or phrase" within 64 tokens; the reasoning prompt asks the model to think step
by step inside `<think>` tags and then give only the final answer inside `<answer>` tags,
within 512 tokens.

<!-- report:matrix -->
| Arm | Prompt | VQAv2 | GQA | CLEVR counting | TextVQA (far-OOD) |
| --- | --- | ---: | ---: | ---: | ---: |
| Instruct 2B | short | 77.87 (77.91) | 59.45 (59.46) | 97.67 (97.67) | 80.53 (80.55) |
| Instruct 2B | reasoning | 55.06 (64.68) | 36.13 (51.95) | 87.13 (87.13) | 69.74 (69.75) |
| SFT | short | 79.87 (79.89) | 62.35 (62.35) | 99.40 (99.40) | 80.08 (80.10) |
| SFT | reasoning | 67.99 (68.17) | 56.61 (57.00) | 92.10 (92.10) | 68.86 (68.87) |
| Instruct → GRPO | short | 78.02 (78.06) | 59.55 (59.55) | 98.27 (98.27) | 80.56 (80.57) |
| Instruct → GRPO | reasoning | 61.79 (69.01) | 44.90 (54.09) | 86.10 (86.10) | 74.06 (74.06) |
| SFT → GRPO | short | 79.85 (79.87) | 62.44 (62.44) | 99.40 (99.40) | 80.10 (80.12) |
| SFT → GRPO | reasoning | 69.18 (70.80) | 53.70 (54.86) | 97.80 (97.80) | 74.58 (74.59) |
| Instruct → gold-conditioned OPD | short | 77.79 (77.83) | 59.72 (59.72) | 97.73 (97.73) | 80.59 (80.60) |
| Instruct → gold-conditioned OPD | reasoning | 68.61 (69.23) | 51.35 (51.58) | 87.10 (87.10) | 72.87 (72.87) |
| Instruct 8B | short | 81.18 (81.22) | 61.66 (61.66) | 99.67 (99.67) | 83.49 (83.51) |
| Instruct 8B | reasoning | 63.67 (68.75) | 46.84 (52.27) | 96.97 (96.97) | 74.00 (74.00) |
| Instruct 8B → GRPO | short | 81.12 (81.16) | 61.78 (61.78) | 99.57 (99.57) | 83.67 (83.69) |
| Instruct 8B → GRPO | reasoning | 77.37 (78.63) | 59.43 (59.82) | 97.30 (97.30) | 80.46 (80.47) |
<!-- /report:matrix -->

**RL recovers reasoning-prompt accuracy.** Under the reasoning prompt, 300 GRPO updates
from Instruct 2B raise official accuracy by 6.7 points on VQAv2, 8.8 on GQA, and 4.3 on
far-OOD TextVQA; the normalized scorer gives +4.3, +2.1, and +4.3, so the gains go beyond
answer formatting. At 8B the same recipe gives +13.7, +12.6, and +6.5 official (+9.9,
+7.6, +6.5 normalized) and brings reasoning-prompt accuracy within 3.8 points of the short
prompt on VQAv2 and 2.4 on GQA, from 17.5 and 14.8. CLEVR counting, which is not in the
RL mix, moves by −1.0 points at 2B and +0.3 at 8B. Gold-conditioned OPD, trained on the same
sampled questions, raises official reasoning-prompt accuracy by 13.6 points on VQAv2 and
3.1 on TextVQA (+4.6 and +3.1 normalized); its 15.2-point official gain on GQA is almost
entirely answer casing (−0.4 normalized).

**No arm passes its short-answer ceiling.** Every arm scores higher with the short prompt
than with the reasoning prompt on every dataset, under both scorers: 28 of 28 arm and
dataset pairs, with the smallest gap 1.6 points (SFT → GRPO on CLEVR counting, official).
RL leaves the ceiling where it was. After GRPO or OPD, short-prompt official accuracy
moves by at most 0.26 points on VQAv2 and GQA and at most 0.6 on any dataset.

**SFT moves the ceiling.** One epoch of short-answer SFT raises short-prompt official
accuracy by 2.0 points on VQAv2 and 2.9 on GQA, the two datasets it trains on, and by 1.7
on CLEVR counting; far-OOD TextVQA moves by −0.5. GRPO after SFT keeps that ceiling
(within 0.1 points) and adds reasoning-prompt accuracy over SFT on CLEVR counting (+5.7)
and TextVQA (+5.7). On GQA it gives up 2.9 points, and the normalized scorer shows the
same decline (57.00 → 54.86), so casing does not explain it. Aggregates hide opposite
movements by question type: under official scoring, VQAv2 yes/no rises from 82.88 to
88.08 while "other" falls from 57.40 to 55.96, and GQA query questions fall from 41.75 to
36.68.

## Paired differences

Each candidate minus its reference on the same test questions, in accuracy points, with a
95% bootstrap interval over questions (10,000 resamples). The interval reflects which
questions were asked; with one training seed per arm it does not measure seed variation.

<!-- report:deltas -->
| Comparison | Prompt | Scorer | VQAv2 | GQA | CLEVR counting | TextVQA (far-OOD) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| SFT − Instruct 2B | short | normalized | +1.98 [+1.33, +2.65] | +2.89 [+2.33, +3.46] | +1.73 [+1.27, +2.20] | -0.45 [-1.11, +0.21] |
| SFT − Instruct 2B | short | official | +2.00 [+1.36, +2.67] | +2.89 [+2.34, +3.47] | +1.73 [+1.27, +2.20] | -0.45 [-1.11, +0.22] |
| SFT − Instruct 2B | reasoning | normalized | +3.49 [+2.32, +4.65] | +5.06 [+4.26, +5.84] | +4.97 [+3.57, +6.30] | -0.89 [-2.05, +0.29] |
| SFT − Instruct 2B | reasoning | official | +12.93 [+11.59, +14.28] | +20.47 [+19.56, +21.36] | +4.97 [+3.57, +6.30] | -0.89 [-2.05, +0.29] |
| Instruct → GRPO − Instruct 2B | short | normalized | +0.15 [-0.04, +0.35] | +0.09 [-0.06, +0.23] | +0.60 [+0.33, +0.90] | +0.02 [-0.20, +0.26] |
| Instruct → GRPO − Instruct 2B | short | official | +0.15 [-0.04, +0.35] | +0.10 [-0.05, +0.24] | +0.60 [+0.33, +0.90] | +0.03 [-0.20, +0.26] |
| Instruct → GRPO − Instruct 2B | reasoning | normalized | +4.33 [+3.43, +5.22] | +2.15 [+1.55, +2.74] | -1.03 [-2.30, +0.20] | +4.31 [+3.47, +5.17] |
| Instruct → GRPO − Instruct 2B | reasoning | official | +6.73 [+5.77, +7.72] | +8.76 [+8.06, +9.44] | -1.03 [-2.30, +0.20] | +4.31 [+3.48, +5.18] |
| SFT → GRPO − Instruct 2B | short | normalized | +1.96 [+1.32, +2.61] | +2.98 [+2.42, +3.55] | +1.73 [+1.27, +2.20] | -0.43 [-1.09, +0.24] |
| SFT → GRPO − Instruct 2B | short | official | +1.98 [+1.34, +2.63] | +2.99 [+2.43, +3.55] | +1.73 [+1.27, +2.20] | -0.42 [-1.08, +0.24] |
| SFT → GRPO − Instruct 2B | reasoning | normalized | +6.13 [+5.15, +7.13] | +2.91 [+2.23, +3.59] | +10.67 [+9.40, +11.93] | +4.84 [+3.85, +5.82] |
| SFT → GRPO − Instruct 2B | reasoning | official | +14.13 [+12.95, +15.33] | +17.57 [+16.73, +18.37] | +10.67 [+9.40, +11.93] | +4.84 [+3.85, +5.82] |
| SFT → GRPO − SFT | short | normalized | -0.02 [-0.21, +0.16] | +0.10 [-0.07, +0.25] | +0.00 [+0.00, +0.00] | +0.02 [-0.21, +0.26] |
| SFT → GRPO − SFT | short | official | -0.02 [-0.21, +0.16] | +0.10 [-0.07, +0.25] | +0.00 [+0.00, +0.00] | +0.02 [-0.21, +0.26] |
| SFT → GRPO − SFT | reasoning | normalized | +2.63 [+1.57, +3.72] | -2.15 [-2.88, -1.42] | +5.70 [+4.73, +6.67] | +5.73 [+4.68, +6.76] |
| SFT → GRPO − SFT | reasoning | official | +1.19 [+0.10, +2.30] | -2.90 [-3.63, -2.17] | +5.70 [+4.73, +6.67] | +5.73 [+4.69, +6.76] |
| Instruct → gold-conditioned OPD − Instruct 2B | short | normalized | -0.08 [-0.37, +0.21] | +0.25 [+0.06, +0.46] | +0.07 [-0.07, +0.20] | +0.05 [-0.29, +0.39] |
| Instruct → gold-conditioned OPD − Instruct 2B | short | official | -0.08 [-0.37, +0.21] | +0.26 [+0.06, +0.46] | +0.07 [-0.07, +0.20] | +0.06 [-0.28, +0.40] |
| Instruct → gold-conditioned OPD − Instruct 2B | reasoning | normalized | +4.55 [+3.53, +5.59] | -0.37 [-1.03, +0.30] | -0.03 [-1.20, +1.13] | +3.12 [+2.18, +4.06] |
| Instruct → gold-conditioned OPD − Instruct 2B | reasoning | official | +13.55 [+12.31, +14.78] | +15.22 [+14.41, +16.03] | -0.03 [-1.20, +1.13] | +3.13 [+2.19, +4.06] |
| Instruct 8B → GRPO − Instruct 8B | short | normalized | -0.06 [-0.22, +0.10] | +0.13 [-0.02, +0.28] | -0.10 [-0.23, +0.00] | +0.18 [-0.01, +0.38] |
| Instruct 8B → GRPO − Instruct 8B | short | official | -0.06 [-0.22, +0.10] | +0.13 [-0.02, +0.28] | -0.10 [-0.23, +0.00] | +0.18 [-0.01, +0.38] |
| Instruct 8B → GRPO − Instruct 8B | reasoning | normalized | +9.88 [+8.96, +10.79] | +7.55 [+6.84, +8.28] | +0.33 [-0.27, +0.93] | +6.46 [+5.60, +7.36] |
| Instruct 8B → GRPO − Instruct 8B | reasoning | official | +13.70 [+12.67, +14.73] | +12.59 [+11.82, +13.36] | +0.33 [-0.27, +0.93] | +6.45 [+5.59, +7.35] |
<!-- /report:deltas -->

## Image resolution

The headline cells resize images to at most 0.79 megapixels, which no VQAv2, GQA, or CLEVR
test image exceeds but 826 of the 5,000 TextVQA images do. The same arm evaluated at the
model's own budget (16.8 megapixels) shows what the cap costs, with a paired 95% interval.
For Instruct 2B it costs nothing measurable: the full budget changes short-prompt TextVQA by
−0.1 points and reasoning-prompt TextVQA by −0.7, both intervals including zero.

<!-- report:resolution -->
| Arm | Dataset | Prompt | Scorer | Megapixels | Change |
| --- | --- | --- | --- | ---: | ---: |
| Instruct 2B | TextVQA | reasoning | official | 0.79 → 16.78 | -0.65 [-1.34, +0.05] |
| Instruct 2B | TextVQA | reasoning | normalized | 0.79 → 16.78 | -0.65 [-1.34, +0.05] |
| Instruct 2B | TextVQA | short | official | 0.79 → 16.78 | -0.09 [-0.41, +0.22] |
| Instruct 2B | TextVQA | short | normalized | 0.79 → 16.78 | -0.10 [-0.41, +0.21] |
<!-- /report:resolution -->

## Response behavior under the reasoning prompt

Mean output tokens · share of responses that close an `<answer>` tag.

<!-- report:behavior -->
| Arm | VQAv2 | GQA | CLEVR counting | TextVQA (far-OOD) |
| --- | ---: | ---: | ---: | ---: |
| Instruct 2B | 148 · 95% | 123 · 97% | 186 · 98% | 107 · 95% |
| SFT | 21 · 40% | 8 · 20% | 62 · 98% | 34 · 47% |
| Instruct → GRPO | 141 · 96% | 119 · 98% | 197 · 97% | 100 · 99% |
| SFT → GRPO | 102 · 97% | 79 · 98% | 24 · 100% | 61 · 99% |
| Instruct → gold-conditioned OPD | 141 · 93% | 119 · 93% | 206 · 98% | 106 · 96% |
| Instruct 8B | 118 · 99% | 104 · 99% | 193 · 99% | 111 · 99% |
| Instruct 8B → GRPO | 9 · 100% | 9 · 100% | 83 · 99% | 11 · 100% |
<!-- /report:behavior -->

SFT on short answers shortens reasoning-prompt responses (VQAv2 148 → 21 tokens) and most
of them lose the answer tag (95% → 40%). GRPO after SFT restores both: 102 tokens and 97%
tags on VQAv2, with CLEVR counting going the other way (62 → 24 tokens) at higher
accuracy. GRPO from Instruct keeps the length at 2B (148 → 141 tokens) and removes almost
all of it at 8B (118 → 9 tokens) while accuracy rises. Visible reasoning length follows
what the reward pays for; it is not by itself a measure of better answers.

Responses cut off at the token budget · responses with an empty answer, same prompt.

<!-- report:failures -->
| Arm | VQAv2 | GQA | CLEVR counting | TextVQA (far-OOD) |
| --- | ---: | ---: | ---: | ---: |
| Instruct 2B | 3.7% · 0.0% | 2.3% · 0.0% | 1.8% · 0.0% | 1.3% · 0.0% |
| SFT | 1.8% · 0.8% | 0.5% · 0.3% | 1.3% · 0.0% | 2.1% · 0.2% |
| Instruct → GRPO | 3.7% · 0.0% | 1.8% · 0.0% | 2.7% · 0.0% | 1.2% · 0.0% |
| SFT → GRPO | 2.8% · 0.0% | 1.9% · 0.0% | 0.2% · 0.0% | 1.1% · 0.0% |
| Instruct → gold-conditioned OPD | 4.3% · 0.0% | 2.8% · 0.0% | 1.5% · 0.0% | 1.5% · 0.0% |
| Instruct 8B | 0.9% · 0.0% | 0.4% · 0.0% | 0.7% · 0.0% | 0.5% · 0.0% |
| Instruct 8B → GRPO | 0.0% · 0.0% | 0.0% · 0.0% | 0.9% · 0.0% | 0.0% · 0.0% |
<!-- /report:failures -->

## Capitalization under the reasoning prompt

The official evaluators compare some answers case-sensitively (VQAv2 when all annotators
agree, GQA always), so a capitalized answer can miss.
Lowercasing the answers before official scoring shows how much of the gap to the normalized
scorer is casing. For Instruct 2B it is nearly all of it: lowercasing closes 9.6 of the
9.6-point VQAv2 gap and 14.7 of the 15.8-point GQA gap.

<!-- report:casing -->
| Arm | Dataset | Official | Official, lowercased answers | Normalized | Answers scored differently |
| --- | --- | ---: | ---: | ---: | ---: |
| Instruct 2B | VQAv2 | 55.06 | 64.64 | 64.68 | 481 |
| Instruct 2B | GQA | 36.13 | 50.80 | 51.95 | 1,989 |
| Instruct 2B | CLEVR counting | 87.13 | 87.13 | 87.13 | 0 |
| Instruct 2B | TextVQA | 69.74 | 69.74 | 69.75 | 2 |
| SFT | VQAv2 | 67.99 | 68.15 | 68.17 | 9 |
| SFT | GQA | 56.61 | 56.68 | 57.00 | 50 |
| SFT | CLEVR counting | 92.10 | 92.10 | 92.10 | 0 |
| SFT | TextVQA | 68.86 | 68.86 | 68.87 | 3 |
| Instruct → GRPO | VQAv2 | 61.79 | 68.99 | 69.01 | 361 |
| Instruct → GRPO | GQA | 44.90 | 53.43 | 54.09 | 1,157 |
| Instruct → GRPO | CLEVR counting | 86.10 | 86.10 | 86.10 | 0 |
| Instruct → GRPO | TextVQA | 74.06 | 74.06 | 74.06 | 1 |
| SFT → GRPO | VQAv2 | 69.18 | 70.76 | 70.80 | 81 |
| SFT → GRPO | GQA | 53.70 | 54.58 | 54.86 | 145 |
| SFT → GRPO | CLEVR counting | 97.80 | 97.80 | 97.80 | 0 |
| SFT → GRPO | TextVQA | 74.58 | 74.58 | 74.59 | 2 |
| Instruct → gold-conditioned OPD | VQAv2 | 68.61 | 69.21 | 69.23 | 31 |
| Instruct → gold-conditioned OPD | GQA | 51.35 | 51.43 | 51.58 | 29 |
| Instruct → gold-conditioned OPD | CLEVR counting | 87.10 | 87.10 | 87.10 | 0 |
| Instruct → gold-conditioned OPD | TextVQA | 72.87 | 72.87 | 72.87 | 1 |
| Instruct 8B | VQAv2 | 63.67 | 68.71 | 68.75 | 254 |
| Instruct 8B | GQA | 46.84 | 51.28 | 52.27 | 682 |
| Instruct 8B | CLEVR counting | 96.97 | 96.97 | 96.97 | 0 |
| Instruct 8B | TextVQA | 74.00 | 74.00 | 74.00 | 1 |
| Instruct 8B → GRPO | VQAv2 | 77.37 | 78.61 | 78.63 | 63 |
| Instruct 8B → GRPO | GQA | 59.43 | 59.64 | 59.82 | 49 |
| Instruct 8B → GRPO | CLEVR counting | 97.30 | 97.30 | 97.30 | 0 |
| Instruct 8B → GRPO | TextVQA | 80.46 | 80.46 | 80.47 | 4 |
<!-- /report:casing -->

## Image interventions

Official score with the original image → with a black or swapped image, on the first 128
test questions, and the share of normalized answers that changed. The references stay those
of the original image, so the changed-image score measures agreement with them, not
accuracy on the new image. Every arm depends on the image: a black image changes 60–100% of
answers and lowers scores by 18 to 98 points.

<!-- report:interventions -->
| Arm | Prompt | Image | VQAv2 | GQA | CLEVR counting | TextVQA (far-OOD) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Instruct 2B | short | black | 68.8 → 31.2 (74%) | 57.8 → 32.8 (61%) | 97.7 → 0.0 (100%) | 84.5 → 4.8 (97%) |
| Instruct 2B | short | swapped | 68.8 → 32.1 (75%) | 57.8 → 32.8 (63%) | 97.7 → 17.2 (84%) | 84.5 → 2.0 (99%) |
| Instruct 2B | reasoning | black | 47.3 → 10.5 (92%) | 40.6 → 8.6 (89%) | 83.6 → 0.0 (100%) | 75.0 → 1.2 (99%) |
| Instruct 2B | reasoning | swapped | 47.3 → 23.1 (80%) | 40.6 → 18.8 (77%) | 83.6 → 17.2 (84%) | 75.0 → 1.8 (99%) |
| SFT | short | black | 73.4 → 35.5 (74%) | 57.0 → 39.1 (64%) | 98.4 → 0.0 (100%) | 83.5 → 4.8 (97%) |
| SFT | short | swapped | 73.4 → 33.0 (74%) | 57.0 → 29.7 (71%) | 98.4 → 17.2 (83%) | 83.5 → 2.7 (98%) |
| SFT | reasoning | black | 58.7 → 26.0 (78%) | 57.0 → 24.2 (69%) | 89.8 → 0.0 (100%) | 74.4 → 2.7 (98%) |
| SFT | reasoning | swapped | 58.7 → 26.3 (77%) | 57.0 → 25.8 (67%) | 89.8 → 15.6 (85%) | 74.4 → 0.0 (99%) |
| Instruct → GRPO | short | black | 69.0 → 31.1 (74%) | 57.8 → 34.4 (60%) | 98.4 → 0.0 (100%) | 83.7 → 4.8 (97%) |
| Instruct → GRPO | short | swapped | 69.0 → 31.6 (74%) | 57.8 → 32.0 (64%) | 98.4 → 17.2 (83%) | 83.7 → 2.0 (99%) |
| Instruct → GRPO | reasoning | black | 57.6 → 17.7 (84%) | 47.7 → 13.3 (83%) | 85.9 → 0.0 (100%) | 80.0 → 2.0 (98%) |
| Instruct → GRPO | reasoning | swapped | 57.6 → 26.5 (77%) | 47.7 → 22.7 (73%) | 85.9 → 14.1 (87%) | 80.0 → 1.8 (98%) |
| SFT → GRPO | short | black | 74.1 → 35.1 (75%) | 57.0 → 37.5 (62%) | 98.4 → 0.0 (100%) | 83.0 → 4.8 (97%) |
| SFT → GRPO | short | swapped | 74.1 → 32.3 (75%) | 57.0 → 30.5 (72%) | 98.4 → 17.2 (83%) | 83.0 → 2.5 (98%) |
| SFT → GRPO | reasoning | black | 57.5 → 26.1 (80%) | 53.9 → 28.1 (65%) | 97.7 → 0.0 (100%) | 78.2 → 2.0 (98%) |
| SFT → GRPO | reasoning | swapped | 57.5 → 27.4 (80%) | 53.9 → 21.1 (76%) | 97.7 → 16.4 (84%) | 78.2 → 1.0 (99%) |
<!-- /report:interventions -->

## What each RL run trained on

All four RL runs completed 300 updates on exactly the same ordered 1,200 questions, 2.2%
of the 55,438-question pool, with 9,600 sampled responses each. A group of eight responses
with identical rewards has zero advantage, so it contributes no correctness gradient;
optimizer momentum can still move the weights on such an update. The OPD run logs the same
correctness reward for diagnosis; its learning signal is the teacher's token scores, so its
reward columns do not drive its updates.

<!-- report:exposure -->
| Run | Updates | Questions | Responses | Groups with varied rewards (%) | Updates without reward variation | Mean reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `grpo2-base` | 300 | 1,200 | 9,600 | 38.67 | 43 | 0.599 |
| `grpo2-sft` | 300 | 1,200 | 9,600 | 49.00 | 16 | 0.581 |
| `grpo8-base` | 300 | 1,200 | 9,600 | 23.33 | 110 | 0.646 |
| `opd2-self` | 300 | 1,200 | 9,600 | 49.25 | 21 | 0.588 |
<!-- /report:exposure -->

The next table follows each run's training signal. The reward (mean of the first and of
the last ten updates) and the share of prompt groups whose samples all scored the same (no
gradient) come from every update's saved draws; entropy and completion length are the
trainer's logs (every ten updates), first → last. KL is absent because these runs used no
KL penalty.

<!-- report:diagnostics -->
| Run | Reward (first → last 10 updates) | Dead groups (mean) | Entropy | Clip fraction (mean) | Completion tokens | Distinct prompts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `grpo2-base` | 0.62 → 0.61 | 0.613 | 0.73 → 0.63 | 0.000 | 154 → 139 | 1,200 |
| `grpo2-sft` | 0.27 → 0.63 | 0.510 | 1.08 → 0.90 | 0.000 | 30 → 110 | 1,200 |
| `grpo8-base` | 0.62 → 0.67 | 0.767 | 0.40 → 0.12 | 0.000 | 133 → 13 | 1,200 |
| `opd2-self` | 0.64 → 0.61 | 0.507 | 0.75 → 0.68 | 0.000 | 150 → 132 | 1,200 |
<!-- /report:diagnostics -->

## Learnability of the RL pool

Before an RL run, `python -m scripts.learnability` takes a seeded random sample of each
dataset's RL pool, draws a group of eight responses per prompt from the starting model with
the run's rollout settings, and scores them with the GRPO correctness reward. Only prompts
whose responses are neither all right nor all wrong give GRPO a gradient. The pre-check
shows images at the evaluation resolution rather than the RL canvas, so it approximates the
rate a run sees.

<!-- report:learnability -->
| Model | Dataset | Prompts | Samples each | Learnable (%) | Never correct (%) | Always correct (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `base2` | CLEVR counting | 500 | 8 | 48.4 | 0.0 | 51.6 |
| `base2` | GQA | 500 | 8 | 42.0 | 26.6 | 31.4 |
| `base2` | VQAv2 | 500 | 8 | 47.8 | 19.8 | 32.4 |
<!-- /report:learnability -->

## Learning curves

Each point evaluates a saved adapter on the dev tier through vLLM's LoRA serving rather
than a merged copy; on the first 500 VQAv2 test questions (short prompt) the two gave the
same answer 496 times ([check](../../results/lora-serving-check.json)). The
corrected campaign's runs predate this step; new runs add their points here once their
dev records are copied into `results/records/`.

<!-- report:curves -->
No checkpoint dev evaluations have been recorded yet.
<!-- /report:curves -->

The July runs, which used a defective VQAv2 reward, are tabulated with the erratum in
[history](history.md).
