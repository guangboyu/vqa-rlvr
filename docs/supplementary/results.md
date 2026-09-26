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
| SFT 2B | short | 79.87 (79.89) | 62.35 (62.35) | 99.40 (99.40) | 80.08 (80.10) |
| SFT 2B | reasoning | 67.99 (68.17) | 56.61 (57.00) | 92.10 (92.10) | 68.86 (68.87) |
| Instruct → GRPO | short | 77.94 (77.98) | 60.06 (60.06) | 99.47 (99.47) | 81.02 (81.03) |
| Instruct → GRPO | reasoning | 78.27 (79.01) | 60.23 (60.30) | 99.27 (99.27) | 79.99 (80.00) |
| SFT → GRPO | short | — | — | — | — |
| SFT → GRPO | reasoning | — | — | — | — |
| Instruct → short-prompt GRPO | short | 78.62 (78.66) | 60.34 (60.35) | 99.50 (99.50) | 81.01 (81.02) |
| Instruct → short-prompt GRPO | reasoning | 61.85 (69.81) | 42.89 (54.01) | 86.50 (86.50) | 72.34 (72.35) |
| SFT → short-prompt GRPO | short | 79.83 (79.85) | 62.50 (62.50) | 99.50 (99.50) | 80.89 (80.90) |
| SFT → short-prompt GRPO | reasoning | 68.05 (68.31) | 56.88 (57.26) | 88.97 (88.97) | 70.72 (70.73) |
| Instruct → gold-conditioned OPD | short | — | — | — | — |
| Instruct → gold-conditioned OPD | reasoning | — | — | — | — |
| Instruct 8B | short | 81.18 (81.22) | 61.66 (61.66) | 99.67 (99.67) | 83.49 (83.51) |
| Instruct 8B | reasoning | 63.67 (68.75) | 46.84 (52.27) | 96.97 (96.97) | 74.00 (74.00) |
| SFT 8B | short | 83.44 (83.46) | 63.86 (63.86) | 88.83 (88.83) | 84.73 (84.76) |
| SFT 8B | reasoning | 68.36 (69.26) | 48.71 (49.18) | 96.70 (96.70) | 75.16 (75.18) |
| Instruct 8B → GRPO | short | — | — | — | — |
| Instruct 8B → GRPO | reasoning | — | — | — | — |
| SFT 8B → GRPO | short | — | — | — | — |
| SFT 8B → GRPO | reasoning | — | — | — | — |
| Instruct 8B → short-prompt GRPO | short | — | — | — | — |
| Instruct 8B → short-prompt GRPO | reasoning | — | — | — | — |
| SFT 8B → short-prompt GRPO | short | — | — | — | — |
| SFT 8B → short-prompt GRPO | reasoning | — | — | — | — |
<!-- /report:matrix -->

<!-- TODO(final rerun): the three paragraphs below describe the corrected campaign's arms; rewrite them for the final-rerun arms in the table above. -->
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
| SFT 2B − Instruct 2B | short | normalized | +1.98 [+1.33, +2.65] | +2.89 [+2.33, +3.46] | +1.73 [+1.27, +2.20] | -0.45 [-1.11, +0.21] |
| SFT 2B − Instruct 2B | short | official | +2.00 [+1.36, +2.67] | +2.89 [+2.34, +3.47] | +1.73 [+1.27, +2.20] | -0.45 [-1.11, +0.22] |
| SFT 2B − Instruct 2B | reasoning | normalized | +3.49 [+2.32, +4.65] | +5.06 [+4.26, +5.84] | +4.97 [+3.57, +6.30] | -0.89 [-2.05, +0.29] |
| SFT 2B − Instruct 2B | reasoning | official | +12.93 [+11.59, +14.28] | +20.47 [+19.56, +21.36] | +4.97 [+3.57, +6.30] | -0.89 [-2.05, +0.29] |
| Instruct → GRPO − Instruct 2B | short | normalized | +0.07 [-0.32, +0.47] | +0.60 [+0.29, +0.90] | +1.80 [+1.30, +2.33] | +0.48 [+0.03, +0.92] |
| Instruct → GRPO − Instruct 2B | short | official | +0.07 [-0.32, +0.47] | +0.60 [+0.29, +0.91] | +1.80 [+1.30, +2.33] | +0.49 [+0.05, +0.94] |
| Instruct → GRPO − Instruct 2B | reasoning | normalized | +14.34 [+13.25, +15.43] | +8.36 [+7.57, +9.11] | +12.13 [+10.90, +13.37] | +10.25 [+9.17, +11.29] |
| Instruct → GRPO − Instruct 2B | reasoning | official | +23.22 [+21.95, +24.47] | +24.10 [+23.17, +24.98] | +12.13 [+10.90, +13.37] | +10.24 [+9.17, +11.29] |
| Instruct → short-prompt GRPO − Instruct 2B | short | normalized | +0.75 [+0.27, +1.23] | +0.89 [+0.48, +1.30] | +1.83 [+1.33, +2.37] | +0.47 [-0.01, +0.96] |
| Instruct → short-prompt GRPO − Instruct 2B | short | official | +0.75 [+0.27, +1.23] | +0.89 [+0.48, +1.30] | +1.83 [+1.33, +2.37] | +0.48 [-0.00, +0.97] |
| Instruct → short-prompt GRPO − Instruct 2B | reasoning | normalized | +5.14 [+4.23, +6.07] | +2.06 [+1.45, +2.67] | -0.63 [-1.90, +0.60] | +2.60 [+1.75, +3.45] |
| Instruct → short-prompt GRPO − Instruct 2B | reasoning | official | +6.80 [+5.85, +7.78] | +6.76 [+6.07, +7.43] | -0.63 [-1.90, +0.60] | +2.60 [+1.75, +3.45] |
| Instruct → short-prompt GRPO − SFT 2B | short | normalized | -1.23 [-1.87, -0.60] | -2.00 [-2.54, -1.47] | +0.10 [-0.10, +0.33] | +0.92 [+0.24, +1.60] |
| Instruct → short-prompt GRPO − SFT 2B | short | official | -1.25 [-1.89, -0.62] | -2.00 [-2.54, -1.48] | +0.10 [-0.10, +0.33] | +0.93 [+0.24, +1.60] |
| Instruct → short-prompt GRPO − SFT 2B | reasoning | normalized | +1.64 [+0.56, +2.74] | -3.00 [-3.78, -2.23] | -5.60 [-7.00, -4.27] | +3.49 [+2.35, +4.68] |
| Instruct → short-prompt GRPO − SFT 2B | reasoning | official | -6.14 [-7.41, -4.85] | -13.71 [-14.59, -12.84] | -5.60 [-7.00, -4.27] | +3.49 [+2.35, +4.68] |
| Instruct → short-prompt GRPO − Instruct → GRPO | short | normalized | +0.68 [+0.24, +1.12] | +0.29 [-0.10, +0.68] | +0.03 [-0.10, +0.20] | -0.01 [-0.37, +0.36] |
| Instruct → short-prompt GRPO − Instruct → GRPO | short | official | +0.68 [+0.24, +1.12] | +0.29 [-0.10, +0.68] | +0.03 [-0.10, +0.20] | -0.01 [-0.37, +0.36] |
| Instruct → short-prompt GRPO − Instruct → GRPO | reasoning | normalized | -9.20 [-10.17, -8.23] | -6.30 [-7.04, -5.56] | -12.77 [-14.03, -11.53] | -7.65 [-8.68, -6.64] |
| Instruct → short-prompt GRPO − Instruct → GRPO | reasoning | official | -16.42 [-17.59, -15.25] | -17.34 [-18.21, -16.47] | -12.77 [-14.03, -11.53] | -7.64 [-8.68, -6.64] |
| SFT → short-prompt GRPO − Instruct 2B | short | normalized | +1.94 [+1.28, +2.61] | +3.04 [+2.50, +3.60] | +1.83 [+1.37, +2.33] | +0.35 [-0.31, +1.00] |
| SFT → short-prompt GRPO − Instruct 2B | short | official | +1.96 [+1.30, +2.63] | +3.04 [+2.50, +3.61] | +1.83 [+1.37, +2.33] | +0.36 [-0.29, +1.01] |
| SFT → short-prompt GRPO − Instruct 2B | reasoning | normalized | +3.64 [+2.46, +4.81] | +5.31 [+4.54, +6.08] | +1.83 [+0.50, +3.17] | +0.98 [-0.11, +2.11] |
| SFT → short-prompt GRPO − Instruct 2B | reasoning | official | +13.00 [+11.64, +14.34] | +20.74 [+19.84, +21.63] | +1.83 [+0.50, +3.17] | +0.98 [-0.11, +2.11] |
| SFT → short-prompt GRPO − SFT 2B | short | normalized | -0.04 [-0.50, +0.41] | +0.15 [-0.23, +0.52] | +0.10 [-0.07, +0.27] | +0.80 [+0.35, +1.26] |
| SFT → short-prompt GRPO − SFT 2B | short | official | -0.04 [-0.50, +0.41] | +0.15 [-0.23, +0.52] | +0.10 [-0.07, +0.27] | +0.81 [+0.36, +1.26] |
| SFT → short-prompt GRPO − SFT 2B | reasoning | normalized | +0.14 [-0.70, +0.98] | +0.25 [-0.22, +0.74] | -3.13 [-4.40, -1.83] | +1.86 [+1.00, +2.72] |
| SFT → short-prompt GRPO − SFT 2B | reasoning | official | +0.06 [-0.80, +0.91] | +0.27 [-0.21, +0.76] | -3.13 [-4.40, -1.83] | +1.86 [+1.00, +2.73] |
| SFT 8B − Instruct 8B | short | normalized | +2.24 [+1.66, +2.83] | +2.20 [+1.69, +2.71] | -10.83 [-11.97, -9.73] | +1.25 [+0.64, +1.85] |
| SFT 8B − Instruct 8B | short | official | +2.26 [+1.68, +2.85] | +2.20 [+1.69, +2.71] | -10.83 [-11.97, -9.73] | +1.25 [+0.64, +1.85] |
| SFT 8B − Instruct 8B | reasoning | normalized | +0.51 [-0.56, +1.60] | -3.08 [-3.88, -2.31] | -0.27 [-1.07, +0.50] | +1.17 [+0.18, +2.19] |
| SFT 8B − Instruct 8B | reasoning | official | +4.69 [+3.53, +5.88] | +1.87 [+1.03, +2.71] | -0.27 [-1.07, +0.50] | +1.16 [+0.17, +2.18] |
<!-- /report:deltas -->

## Image resolution

The headline cells resize images to at most 0.79 megapixels, which no VQAv2, GQA, or CLEVR
test image exceeds but 826 of the 5,000 TextVQA images do. The same arm evaluated at the
model's own budget (16.8 megapixels) shows what the cap costs, with a paired 95% interval.
For Instruct 2B it costs nothing measurable: the full budget changes short-prompt TextVQA by
−0.1 points and reasoning-prompt TextVQA by −0.7, both intervals including zero. An arm
evaluated on the 608×608 canvas RL trains on appears here too, paired with its
original-image cell.

<!-- report:resolution -->
| Arm | Dataset | Prompt | Scorer | Megapixels | Change |
| --- | --- | --- | --- | ---: | ---: |
| Instruct 2B | CLEVR counting | reasoning | official | 0.79 → 0.37 (608×608 RL canvas) | -0.97 [-2.37, +0.43] |
| Instruct 2B | CLEVR counting | reasoning | normalized | 0.79 → 0.37 (608×608 RL canvas) | -0.97 [-2.37, +0.43] |
| Instruct 2B | CLEVR counting | short | official | 0.79 → 0.37 (608×608 RL canvas) | +2.13 [+1.60, +2.70] |
| Instruct 2B | CLEVR counting | short | normalized | 0.79 → 0.37 (608×608 RL canvas) | +2.13 [+1.60, +2.70] |
| Instruct 2B | GQA | reasoning | official | 0.79 → 0.37 (608×608 RL canvas) | +0.30 [-0.33, +0.91] |
| Instruct 2B | GQA | reasoning | normalized | 0.79 → 0.37 (608×608 RL canvas) | +0.16 [-0.47, +0.79] |
| Instruct 2B | GQA | short | official | 0.79 → 0.37 (608×608 RL canvas) | +0.02 [-0.42, +0.46] |
| Instruct 2B | GQA | short | normalized | 0.79 → 0.37 (608×608 RL canvas) | +0.01 [-0.43, +0.45] |
| Instruct 2B | TextVQA | reasoning | official | 0.79 → 0.37 (608×608 RL canvas) | -5.64 [-6.78, -4.51] |
| Instruct 2B | TextVQA | reasoning | normalized | 0.79 → 0.37 (608×608 RL canvas) | -5.65 [-6.78, -4.52] |
| Instruct 2B | TextVQA | reasoning | official | 0.79 → 16.78 | -0.65 [-1.34, +0.05] |
| Instruct 2B | TextVQA | reasoning | normalized | 0.79 → 16.78 | -0.65 [-1.34, +0.05] |
| Instruct 2B | TextVQA | short | official | 0.79 → 0.37 (608×608 RL canvas) | -6.89 [-7.83, -5.96] |
| Instruct 2B | TextVQA | short | normalized | 0.79 → 0.37 (608×608 RL canvas) | -6.90 [-7.84, -5.97] |
| Instruct 2B | TextVQA | short | official | 0.79 → 16.78 | -0.09 [-0.41, +0.22] |
| Instruct 2B | TextVQA | short | normalized | 0.79 → 16.78 | -0.10 [-0.41, +0.21] |
| Instruct 2B | VQAv2 | reasoning | official | 0.79 → 0.37 (608×608 RL canvas) | +0.88 [-0.10, +1.89] |
| Instruct 2B | VQAv2 | reasoning | normalized | 0.79 → 0.37 (608×608 RL canvas) | +0.78 [-0.19, +1.75] |
| Instruct 2B | VQAv2 | short | official | 0.79 → 0.37 (608×608 RL canvas) | -0.14 [-0.72, +0.44] |
| Instruct 2B | VQAv2 | short | normalized | 0.79 → 0.37 (608×608 RL canvas) | -0.16 [-0.74, +0.42] |
| Instruct → GRPO (Sep 13) | CLEVR counting | reasoning | official | 0.79 → 0.37 (608×608 RL canvas) | -1.03 [-2.47, +0.37] |
| Instruct → GRPO (Sep 13) | CLEVR counting | reasoning | normalized | 0.79 → 0.37 (608×608 RL canvas) | -1.03 [-2.47, +0.37] |
| Instruct → GRPO (Sep 13) | CLEVR counting | short | official | 0.79 → 0.37 (608×608 RL canvas) | +1.53 [+1.07, +2.03] |
| Instruct → GRPO (Sep 13) | CLEVR counting | short | normalized | 0.79 → 0.37 (608×608 RL canvas) | +1.53 [+1.07, +2.03] |
| Instruct → GRPO (Sep 13) | GQA | reasoning | official | 0.79 → 0.37 (608×608 RL canvas) | +0.15 [-0.49, +0.80] |
| Instruct → GRPO (Sep 13) | GQA | reasoning | normalized | 0.79 → 0.37 (608×608 RL canvas) | +0.23 [-0.37, +0.83] |
| Instruct → GRPO (Sep 13) | GQA | short | official | 0.79 → 0.37 (608×608 RL canvas) | -0.15 [-0.59, +0.29] |
| Instruct → GRPO (Sep 13) | GQA | short | normalized | 0.79 → 0.37 (608×608 RL canvas) | -0.15 [-0.59, +0.29] |
| Instruct → GRPO (Sep 13) | TextVQA | reasoning | official | 0.79 → 0.37 (608×608 RL canvas) | -6.65 [-7.69, -5.62] |
| Instruct → GRPO (Sep 13) | TextVQA | reasoning | normalized | 0.79 → 0.37 (608×608 RL canvas) | -6.65 [-7.69, -5.62] |
| Instruct → GRPO (Sep 13) | TextVQA | short | official | 0.79 → 0.37 (608×608 RL canvas) | -6.78 [-7.73, -5.87] |
| Instruct → GRPO (Sep 13) | TextVQA | short | normalized | 0.79 → 0.37 (608×608 RL canvas) | -6.79 [-7.73, -5.88] |
| Instruct → GRPO (Sep 13) | VQAv2 | reasoning | official | 0.79 → 0.37 (608×608 RL canvas) | +0.95 [+0.07, +1.80] |
| Instruct → GRPO (Sep 13) | VQAv2 | reasoning | normalized | 0.79 → 0.37 (608×608 RL canvas) | +1.21 [+0.36, +2.05] |
| Instruct → GRPO (Sep 13) | VQAv2 | short | official | 0.79 → 0.37 (608×608 RL canvas) | -0.05 [-0.61, +0.53] |
| Instruct → GRPO (Sep 13) | VQAv2 | short | normalized | 0.79 → 0.37 (608×608 RL canvas) | -0.07 [-0.63, +0.51] |
| SFT → GRPO (Sep 13) | CLEVR counting | reasoning | official | 0.79 → 0.37 (608×608 RL canvas) | -12.20 [-13.50, -10.90] |
| SFT → GRPO (Sep 13) | CLEVR counting | reasoning | normalized | 0.79 → 0.37 (608×608 RL canvas) | -12.20 [-13.50, -10.90] |
| SFT → GRPO (Sep 13) | CLEVR counting | short | official | 0.79 → 0.37 (608×608 RL canvas) | +0.37 [+0.07, +0.70] |
| SFT → GRPO (Sep 13) | CLEVR counting | short | normalized | 0.79 → 0.37 (608×608 RL canvas) | +0.37 [+0.07, +0.70] |
| SFT → GRPO (Sep 13) | GQA | reasoning | official | 0.79 → 0.37 (608×608 RL canvas) | +0.02 [-0.56, +0.60] |
| SFT → GRPO (Sep 13) | GQA | reasoning | normalized | 0.79 → 0.37 (608×608 RL canvas) | +0.14 [-0.44, +0.70] |
| SFT → GRPO (Sep 13) | GQA | short | official | 0.79 → 0.37 (608×608 RL canvas) | -0.13 [-0.55, +0.29] |
| SFT → GRPO (Sep 13) | GQA | short | normalized | 0.79 → 0.37 (608×608 RL canvas) | -0.13 [-0.55, +0.29] |
| SFT → GRPO (Sep 13) | TextVQA | reasoning | official | 0.79 → 0.37 (608×608 RL canvas) | -6.09 [-7.16, -5.04] |
| SFT → GRPO (Sep 13) | TextVQA | reasoning | normalized | 0.79 → 0.37 (608×608 RL canvas) | -6.10 [-7.16, -5.04] |
| SFT → GRPO (Sep 13) | TextVQA | short | official | 0.79 → 0.37 (608×608 RL canvas) | -6.34 [-7.26, -5.42] |
| SFT → GRPO (Sep 13) | TextVQA | short | normalized | 0.79 → 0.37 (608×608 RL canvas) | -6.35 [-7.27, -5.43] |
| SFT → GRPO (Sep 13) | VQAv2 | reasoning | official | 0.79 → 0.37 (608×608 RL canvas) | -0.21 [-1.10, +0.67] |
| SFT → GRPO (Sep 13) | VQAv2 | reasoning | normalized | 0.79 → 0.37 (608×608 RL canvas) | -0.35 [-1.20, +0.48] |
| SFT → GRPO (Sep 13) | VQAv2 | short | official | 0.79 → 0.37 (608×608 RL canvas) | +0.13 [-0.39, +0.63] |
| SFT → GRPO (Sep 13) | VQAv2 | short | normalized | 0.79 → 0.37 (608×608 RL canvas) | +0.13 [-0.39, +0.63] |
| SFT 2B | CLEVR counting | reasoning | official | 0.79 → 0.37 (608×608 RL canvas) | -6.73 [-8.07, -5.43] |
| SFT 2B | CLEVR counting | reasoning | normalized | 0.79 → 0.37 (608×608 RL canvas) | -6.73 [-8.07, -5.43] |
| SFT 2B | CLEVR counting | short | official | 0.79 → 0.37 (608×608 RL canvas) | +0.37 [+0.07, +0.70] |
| SFT 2B | CLEVR counting | short | normalized | 0.79 → 0.37 (608×608 RL canvas) | +0.37 [+0.07, +0.70] |
| SFT 2B | GQA | reasoning | official | 0.79 → 0.37 (608×608 RL canvas) | -0.80 [-1.28, -0.34] |
| SFT 2B | GQA | reasoning | normalized | 0.79 → 0.37 (608×608 RL canvas) | -0.68 [-1.15, -0.21] |
| SFT 2B | GQA | short | official | 0.79 → 0.37 (608×608 RL canvas) | -0.08 [-0.50, +0.33] |
| SFT 2B | GQA | short | normalized | 0.79 → 0.37 (608×608 RL canvas) | -0.08 [-0.50, +0.33] |
| SFT 2B | TextVQA | reasoning | official | 0.79 → 0.37 (608×608 RL canvas) | -5.25 [-6.40, -4.14] |
| SFT 2B | TextVQA | reasoning | normalized | 0.79 → 0.37 (608×608 RL canvas) | -5.26 [-6.41, -4.15] |
| SFT 2B | TextVQA | short | official | 0.79 → 0.37 (608×608 RL canvas) | -6.31 [-7.24, -5.38] |
| SFT 2B | TextVQA | short | normalized | 0.79 → 0.37 (608×608 RL canvas) | -6.32 [-7.25, -5.39] |
| SFT 2B | VQAv2 | reasoning | official | 0.79 → 0.37 (608×608 RL canvas) | +0.06 [-0.73, +0.86] |
| SFT 2B | VQAv2 | reasoning | normalized | 0.79 → 0.37 (608×608 RL canvas) | +0.08 [-0.71, +0.87] |
| SFT 2B | VQAv2 | short | official | 0.79 → 0.37 (608×608 RL canvas) | -0.03 [-0.55, +0.50] |
| SFT 2B | VQAv2 | short | normalized | 0.79 → 0.37 (608×608 RL canvas) | -0.03 [-0.55, +0.50] |
<!-- /report:resolution -->

## Response behavior under the reasoning prompt

Mean output tokens · share of responses that close an `<answer>` tag.

<!-- report:behavior -->
| Arm | VQAv2 | GQA | CLEVR counting | TextVQA (far-OOD) |
| --- | ---: | ---: | ---: | ---: |
| Instruct 2B | 148 · 95% | 123 · 97% | 186 · 98% | 107 · 95% |
| SFT 2B | 21 · 40% | 8 · 20% | 62 · 98% | 34 · 47% |
| Instruct → GRPO | 8 · 100% | 8 · 100% | 8 · 100% | 10 · 100% |
| SFT → GRPO | — | — | — | — |
| Instruct → short-prompt GRPO | 132 · 98% | 116 · 98% | 169 · 99% | 101 · 96% |
| SFT → short-prompt GRPO | 27 · 50% | 11 · 28% | 146 · 99% | 40 · 52% |
| Instruct → gold-conditioned OPD | — | — | — | — |
| Instruct 8B | 118 · 99% | 104 · 99% | 193 · 99% | 111 · 99% |
| SFT 8B | 69 · 78% | 72 · 82% | 96 · 82% | 46 · 71% |
| Instruct 8B → GRPO | — | — | — | — |
| SFT 8B → GRPO | — | — | — | — |
| Instruct 8B → short-prompt GRPO | — | — | — | — |
| SFT 8B → short-prompt GRPO | — | — | — | — |
<!-- /report:behavior -->

<!-- TODO(final rerun): this paragraph describes the corrected campaign's arms (see history.md for their table); rewrite for the final rerun. -->
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
| SFT 2B | 1.8% · 0.8% | 0.5% · 0.3% | 1.3% · 0.0% | 2.1% · 0.2% |
| Instruct → GRPO | 0.0% · 0.0% | 0.0% · 0.0% | 0.0% · 0.0% | 0.0% · 0.0% |
| SFT → GRPO | — | — | — | — |
| Instruct → short-prompt GRPO | 1.5% · 0.0% | 0.9% · 0.0% | 0.9% · 0.0% | 0.7% · 0.0% |
| SFT → short-prompt GRPO | 1.0% · 0.1% | 0.3% · 0.1% | 0.7% · 0.0% | 0.7% · 0.1% |
| Instruct → gold-conditioned OPD | — | — | — | — |
| Instruct 8B | 0.9% · 0.0% | 0.4% · 0.0% | 0.7% · 0.0% | 0.5% · 0.0% |
| SFT 8B | 3.5% · 0.0% | 5.0% · 0.0% | 0.3% · 0.0% | 0.7% · 0.0% |
| Instruct 8B → GRPO | — | — | — | — |
| SFT 8B → GRPO | — | — | — | — |
| Instruct 8B → short-prompt GRPO | — | — | — | — |
| SFT 8B → short-prompt GRPO | — | — | — | — |
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
| SFT 2B | VQAv2 | 67.99 | 68.15 | 68.17 | 9 |
| SFT 2B | GQA | 56.61 | 56.68 | 57.00 | 50 |
| SFT 2B | CLEVR counting | 92.10 | 92.10 | 92.10 | 0 |
| SFT 2B | TextVQA | 68.86 | 68.86 | 68.87 | 3 |
| Instruct → GRPO | VQAv2 | 78.27 | 78.99 | 79.01 | 37 |
| Instruct → GRPO | GQA | 60.23 | 60.30 | 60.30 | 9 |
| Instruct → GRPO | CLEVR counting | 99.27 | 99.27 | 99.27 | 0 |
| Instruct → GRPO | TextVQA | 79.99 | 79.99 | 80.00 | 2 |
| Instruct → short-prompt GRPO | VQAv2 | 61.85 | 69.71 | 69.81 | 398 |
| Instruct → short-prompt GRPO | GQA | 42.89 | 53.12 | 54.01 | 1,398 |
| Instruct → short-prompt GRPO | CLEVR counting | 86.50 | 86.50 | 86.50 | 0 |
| Instruct → short-prompt GRPO | TextVQA | 72.34 | 72.34 | 72.35 | 2 |
| SFT → short-prompt GRPO | VQAv2 | 68.05 | 68.27 | 68.31 | 13 |
| SFT → short-prompt GRPO | GQA | 56.88 | 56.95 | 57.26 | 48 |
| SFT → short-prompt GRPO | CLEVR counting | 88.97 | 88.97 | 88.97 | 0 |
| SFT → short-prompt GRPO | TextVQA | 70.72 | 70.72 | 70.73 | 2 |
| Instruct 8B | VQAv2 | 63.67 | 68.71 | 68.75 | 254 |
| Instruct 8B | GQA | 46.84 | 51.28 | 52.27 | 682 |
| Instruct 8B | CLEVR counting | 96.97 | 96.97 | 96.97 | 0 |
| Instruct 8B | TextVQA | 74.00 | 74.00 | 74.00 | 1 |
| SFT 8B | VQAv2 | 68.36 | 69.22 | 69.26 | 45 |
| SFT 8B | GQA | 48.71 | 49.00 | 49.18 | 59 |
| SFT 8B | CLEVR counting | 96.70 | 96.70 | 96.70 | 0 |
| SFT 8B | TextVQA | 75.16 | 75.16 | 75.18 | 3 |
<!-- /report:casing -->

## Error analysis

`python -m scripts.error_analysis` reads every declared arm's saved predictions and writes
[`results/error-analysis.json`](../../results/error-analysis.json); the rules are defined
in `vqar/eval/errors.py` and copied into that file. A question counts as right when its
score is at least 0.5: a match, or at least two of the ten annotators on VQAv2 and TextVQA.

A net change hides how many answers moved. For each declared comparison, the questions the
candidate fixed / broke against its reference (normalized scorer, paired by question):

<!-- report:flips -->
| Comparison | Prompt | VQAv2 | GQA | CLEVR counting | TextVQA (far-OOD) |
| --- | --- | ---: | ---: | ---: | ---: |
| SFT 2B − Instruct 2B | short | 239 / 135 | 841 / 478 | 52 / 0 | 166 / 187 |
| SFT 2B − Instruct 2B | reasoning | 598 / 421 | 1,600 / 964 | 290 / 141 | 487 / 531 |
| Instruct → GRPO − Instruct 2B | short | 69 / 66 | 235 / 160 | 57 / 3 | 99 / 69 |
| Instruct → GRPO − Instruct 2B | reasoning | 894 / 148 | 1,791 / 740 | 380 / 16 | 713 / 184 |
| Instruct → short-prompt GRPO − Instruct 2B | short | 127 / 84 | 399 / 287 | 59 / 4 | 112 / 89 |
| Instruct → short-prompt GRPO − Instruct 2B | reasoning | 450 / 188 | 902 / 643 | 174 / 193 | 338 / 196 |
| Instruct → short-prompt GRPO − SFT 2B | short | 142 / 203 | 469 / 720 | 7 / 4 | 200 / 156 |
| Instruct → short-prompt GRPO − SFT 2B | reasoning | 497 / 412 | 1,021 / 1,398 | 140 / 308 | 589 / 403 |
| Instruct → short-prompt GRPO − Instruct → GRPO | short | 108 / 68 | 328 / 291 | 3 / 2 | 52 / 59 |
| Instruct → short-prompt GRPO − Instruct → GRPO | reasoning | 156 / 640 | 761 / 1,553 | 14 / 397 | 195 / 582 |
| SFT → short-prompt GRPO − Instruct 2B | short | 250 / 146 | 836 / 454 | 56 / 1 | 183 / 169 |
| SFT → short-prompt GRPO − Instruct 2B | reasoning | 599 / 412 | 1,589 / 921 | 234 / 179 | 470 / 421 |
| SFT → short-prompt GRPO − SFT 2B | short | 88 / 88 | 299 / 280 | 5 / 2 | 97 / 62 |
| SFT → short-prompt GRPO − SFT 2B | reasoning | 268 / 258 | 494 / 462 | 146 / 240 | 309 / 216 |
| SFT 8B − Instruct 8B | short | 220 / 92 | 669 / 392 | 2 / 327 | 182 / 120 |
| SFT 8B − Instruct 8B | reasoning | 454 / 430 | 1,068 / 1,456 | 68 / 76 | 405 / 351 |
<!-- /report:flips -->

<!-- TODO(final rerun): flips narrative quotes corrected-campaign counts; rewrite for the final rerun. -->
Instruct → GRPO under the reasoning prompt fixes 854 GQA questions and breaks 584, and
SFT → GRPO breaks more GQA questions than it fixes. Under the short prompt, RL changes few
verdicts (21 / 12 on VQAv2 from Instruct 2B).

Why reasoning-prompt answers are wrong, as a share of all test questions (official scorer).
The first matching reason wins, in the column order: official wrong but normalized right,
cut off at the token budget, no answer tag, the answer and a reference contain one another
(a minority annotator's answer, "tan" for "light tan"), anything else. The rules are
heuristics for triage; a synonym, for example, counts as wrong content.

<!-- report:errors -->
| Arm | Dataset | Right | Scorer only | Truncated | No answer tag | Partial match | Wrong content |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Instruct 2B | VQAv2 | 55.2 | 9.6 | 3.7 | 1.4 | 18.2 | 11.9 |
| Instruct 2B | GQA | 36.1 | 15.8 | 2.3 | 0.9 | 7.8 | 37.0 |
| Instruct 2B | CLEVR counting | 87.1 | 0.0 | 1.8 | 0.0 | 0.0 | 11.1 |
| Instruct 2B | TextVQA | 70.3 | 0.0 | 1.3 | 3.3 | 13.8 | 11.3 |
| SFT 2B | VQAv2 | 68.1 | 0.2 | 1.8 | 21.0 | 4.5 | 4.4 |
| SFT 2B | GQA | 56.6 | 0.4 | 0.5 | 34.6 | 1.0 | 6.9 |
| SFT 2B | CLEVR counting | 92.1 | 0.0 | 1.3 | 0.6 | 0.0 | 6.0 |
| SFT 2B | TextVQA | 69.4 | 0.0 | 2.1 | 16.0 | 6.7 | 5.8 |
| Instruct → GRPO | VQAv2 | 79.0 | 0.7 | 0.0 | 0.0 | 10.2 | 10.1 |
| Instruct → GRPO | GQA | 60.2 | 0.1 | 0.0 | 0.0 | 4.3 | 35.4 |
| Instruct → GRPO | CLEVR counting | 99.3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.7 |
| Instruct → GRPO | TextVQA | 80.8 | 0.0 | 0.0 | 0.0 | 9.4 | 9.7 |
| Instruct → short-prompt GRPO | VQAv2 | 62.1 | 8.0 | 1.5 | 0.8 | 15.3 | 12.4 |
| Instruct → short-prompt GRPO | GQA | 42.9 | 11.1 | 0.9 | 0.7 | 6.3 | 38.2 |
| Instruct → short-prompt GRPO | CLEVR counting | 86.5 | 0.0 | 0.9 | 0.0 | 0.0 | 12.6 |
| Instruct → short-prompt GRPO | TextVQA | 73.1 | 0.0 | 0.7 | 2.9 | 11.4 | 11.9 |
| SFT → short-prompt GRPO | VQAv2 | 68.3 | 0.3 | 1.0 | 18.6 | 6.0 | 5.9 |
| SFT → short-prompt GRPO | GQA | 56.9 | 0.4 | 0.3 | 30.5 | 1.6 | 10.3 |
| SFT → short-prompt GRPO | CLEVR counting | 89.0 | 0.0 | 0.7 | 0.0 | 0.0 | 10.4 |
| SFT → short-prompt GRPO | TextVQA | 71.2 | 0.0 | 0.7 | 14.3 | 6.8 | 6.9 |
| Instruct 8B | VQAv2 | 64.0 | 5.1 | 0.9 | 0.3 | 17.8 | 11.9 |
| Instruct 8B | GQA | 46.8 | 5.4 | 0.4 | 0.7 | 8.2 | 38.5 |
| Instruct 8B | CLEVR counting | 97.0 | 0.0 | 0.7 | 0.0 | 0.0 | 2.4 |
| Instruct 8B | TextVQA | 74.8 | 0.0 | 0.5 | 0.6 | 14.0 | 10.1 |
| SFT 8B | VQAv2 | 68.7 | 0.9 | 3.5 | 8.1 | 10.9 | 7.9 |
| SFT 8B | GQA | 48.7 | 0.5 | 5.0 | 11.5 | 5.0 | 29.3 |
| SFT 8B | CLEVR counting | 96.7 | 0.0 | 0.3 | 1.1 | 0.0 | 1.9 |
| SFT 8B | TextVQA | 75.8 | 0.0 | 0.7 | 8.1 | 8.7 | 6.7 |
<!-- /report:errors -->

<!-- TODO(final rerun): error-kind and answer-form narrative quotes corrected-campaign numbers; rewrite for the final rerun. -->
SFT's reasoning-prompt errors are mostly missing tags (34.6% of GQA questions). From
Instruct 2B, GRPO shrinks casing misses (VQAv2 9.6 → 7.2%) and missing tags (1.4 → 0.0%)
while wrong content does not shrink (11.9 → 12.4%).

Answer forms that could game an answer checker, as a share of reasoning-prompt responses:
hedged answers ("red or blue", "red/blue") · several `<answer>` tags · answers over six
words. The checks read the tagged answer the RL reward reads, so an untagged response is
never flagged. From Instruct, none of the three rises after GRPO at either size. From SFT,
GRPO raises hedged and long answers on VQAv2 (0.2 → 0.8% and 0.5 → 2.4%) while the share of
responses with an answer tag rises from 40% to 97%, to below the Instruct 2B level.

<!-- report:answer-form -->
| Arm | VQAv2 | GQA | CLEVR counting | TextVQA (far-OOD) |
| --- | ---: | ---: | ---: | ---: |
| Instruct 2B | 2.1 · 0.0 · 5.7 | 0.8 · 0.0 · 3.9 | 0.0 · 0.0 · 0.0 | 0.8 · 0.0 · 2.4 |
| SFT 2B | 0.2 · 0.0 · 0.5 | 0.0 · 0.0 · 0.2 | 0.0 · 0.0 · 0.0 | 0.5 · 0.0 · 1.3 |
| Instruct → GRPO | 0.1 · 0.0 · 0.2 | 0.0 · 0.0 · 0.0 | 0.0 · 0.0 · 0.0 | 0.5 · 0.0 · 0.8 |
| Instruct → short-prompt GRPO | 0.9 · 0.0 · 3.5 | 0.3 · 0.0 · 2.1 | 0.0 · 0.0 · 0.0 | 0.8 · 0.0 · 2.0 |
| SFT → short-prompt GRPO | 0.4 · 0.0 · 0.6 | 0.0 · 0.0 · 0.3 | 0.0 · 0.0 · 0.0 | 0.4 · 0.0 · 1.2 |
| Instruct 8B | 2.6 · 0.0 · 5.4 | 1.5 · 0.0 · 4.6 | 0.0 · 0.0 · 0.0 | 0.9 · 0.0 · 2.6 |
| SFT 8B | 0.6 · 0.0 · 1.4 | 0.2 · 0.0 · 0.8 | 0.0 · 0.0 · 0.0 | 0.5 · 0.0 · 1.3 |
<!-- /report:answer-form -->

## Image interventions

Official score with the original image → with a black or swapped image, on the first 128
test questions, and the share of normalized answers that changed. The references stay those
of the original image, so the changed-image score measures agreement with them, not
accuracy on the new image. Every arm depends on the image: a black image changes 60–100% of
answers and lowers scores by 18 to 98 points.
These records exist only for the corrected campaign's arms (September 13 recipe; see
[history](history.md#the-corrected-campaign-september-13-recipe)); the final-rerun arms
have none.

<!-- report:interventions -->
| Arm | Prompt | Image | VQAv2 | GQA | CLEVR counting | TextVQA (far-OOD) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Instruct 2B | short | black | 68.8 → 31.2 (74%) | 57.8 → 32.8 (61%) | 97.7 → 0.0 (100%) | 84.5 → 4.8 (97%) |
| Instruct 2B | short | swapped | 68.8 → 32.1 (75%) | 57.8 → 32.8 (63%) | 97.7 → 17.2 (84%) | 84.5 → 2.0 (99%) |
| Instruct 2B | reasoning | black | 47.3 → 10.5 (92%) | 40.6 → 8.6 (89%) | 83.6 → 0.0 (100%) | 75.0 → 1.2 (99%) |
| Instruct 2B | reasoning | swapped | 47.3 → 23.1 (80%) | 40.6 → 18.8 (77%) | 83.6 → 17.2 (84%) | 75.0 → 1.8 (99%) |
| SFT 2B | short | black | 73.4 → 35.5 (74%) | 57.0 → 39.1 (64%) | 98.4 → 0.0 (100%) | 83.5 → 4.8 (97%) |
| SFT 2B | short | swapped | 73.4 → 33.0 (74%) | 57.0 → 29.7 (71%) | 98.4 → 17.2 (83%) | 83.5 → 2.7 (98%) |
| SFT 2B | reasoning | black | 58.7 → 26.0 (78%) | 57.0 → 24.2 (69%) | 89.8 → 0.0 (100%) | 74.4 → 2.7 (98%) |
| SFT 2B | reasoning | swapped | 58.7 → 26.3 (77%) | 57.0 → 25.8 (67%) | 89.8 → 15.6 (85%) | 74.4 → 0.0 (99%) |
| Instruct → GRPO (Sep 13) | short | black | 69.0 → 31.1 (74%) | 57.8 → 34.4 (60%) | 98.4 → 0.0 (100%) | 83.7 → 4.8 (97%) |
| Instruct → GRPO (Sep 13) | short | swapped | 69.0 → 31.6 (74%) | 57.8 → 32.0 (64%) | 98.4 → 17.2 (83%) | 83.7 → 2.0 (99%) |
| Instruct → GRPO (Sep 13) | reasoning | black | 57.6 → 17.7 (84%) | 47.7 → 13.3 (83%) | 85.9 → 0.0 (100%) | 80.0 → 2.0 (98%) |
| Instruct → GRPO (Sep 13) | reasoning | swapped | 57.6 → 26.5 (77%) | 47.7 → 22.7 (73%) | 85.9 → 14.1 (87%) | 80.0 → 1.8 (98%) |
| SFT → GRPO (Sep 13) | short | black | 74.1 → 35.1 (75%) | 57.0 → 37.5 (62%) | 98.4 → 0.0 (100%) | 83.0 → 4.8 (97%) |
| SFT → GRPO (Sep 13) | short | swapped | 74.1 → 32.3 (75%) | 57.0 → 30.5 (72%) | 98.4 → 17.2 (83%) | 83.0 → 2.5 (98%) |
| SFT → GRPO (Sep 13) | reasoning | black | 57.5 → 26.1 (80%) | 53.9 → 28.1 (65%) | 97.7 → 0.0 (100%) | 78.2 → 2.0 (98%) |
| SFT → GRPO (Sep 13) | reasoning | swapped | 57.5 → 27.4 (80%) | 53.9 → 21.1 (76%) | 97.7 → 16.4 (84%) | 78.2 → 1.0 (99%) |
<!-- /report:interventions -->

## What each RL run trained on

<!-- TODO(final rerun): exposure and diagnostics narrative describes the corrected campaign (now in history.md); rewrite for the final rerun's 1,000-update runs.
results/training-exposure.json was written from the completed final runs (OPD 2B and SFT 8B → short-prompt GRPO not yet finished); rerun with every final run, in the headline order:
uv run --frozen python -m scripts.summarize_training runs/part1-general-20260925/hpc/vqa-p1-f-{grpo2-base,grpo2-sft,short2-base,short2-sft,opd2,grpo8-base,grpo8-sft,short8-base,short8-sft} --output results/training-exposure.json -->
All four RL runs completed 300 updates on exactly the same ordered 1,200 questions, 2.2%
of the 55,438-question pool, with 9,600 sampled responses each. A group of eight responses
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
| `vqa-p1-f-grpo8-base` | 1,000 | 4,000 | 32,000 | 15.65 | 535 | 0.725 |
| `vqa-p1-f-grpo8-sft` | 1,000 | 4,000 | 32,000 | 19.18 | 477 | 0.753 |
| `vqa-p1-f-short8-base` | 1,000 | 4,000 | 32,000 | 15.35 | 518 | 0.743 |
<!-- /report:exposure -->

The next table follows each run's training signal. The reward (mean of the first and of
the last ten updates) and the share of prompt groups whose samples all scored the same (no
gradient) come from every update's saved draws; so does the split of those groups into
all right (every sample full credit) and all wrong (every sample 0), as means of the first
and of the last ten updates. Entropy and completion length are the trainer's logs (every
ten updates), first → last. KL is absent because these runs used no KL penalty.

No group in these runs agreed on a partial score: the RL roles keep one reference answer,
so the reward is 0 or 1. The 8B run's dead share grows through all-right groups (0.42 →
0.57); from the SFT start, all-wrong groups halve (0.30 → 0.15) while all-right groups
rise from 0.03 to 0.35.

<!-- report:diagnostics -->
| Run | Reward (first → last 10 updates) | Dead groups (mean) | All right (first → last 10) | All wrong (first → last 10) | Entropy | Clip fraction (mean) | Completion tokens | Distinct prompts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `vqa-p1-f-grpo2-base` | 0.63 → 0.71 | 0.809 | 0.38 → 0.62 | 0.17 → 0.25 | 0.73 → 0.06 | 0.000 | 152 → 8 | 4,000 |
| `vqa-p1-f-grpo2-sft` | 0.26 → 0.75 | 0.691 | 0.03 → 0.65 | 0.45 → 0.12 | 1.21 → 0.07 | 0.000 | 33 → 10 | 4,000 |
| `vqa-p1-f-short2-base` | 0.72 → 0.74 | 0.789 | 0.55 → 0.65 | 0.12 → 0.20 | 0.33 → 0.11 | 0.000 | 2 → 2 | 4,000 |
| `vqa-p1-f-short2-sft` | 0.68 → 0.77 | 0.754 | 0.45 → 0.68 | 0.12 → 0.12 | 0.41 → 0.15 | 0.000 | 2 → 2 | 4,000 |
| `vqa-p1-f-grpo8-base` | 0.63 → 0.76 | 0.844 | 0.42 → 0.70 | 0.17 → 0.17 | 0.39 → 0.04 | 0.000 | 129 → 8 | 4,000 |
| `vqa-p1-f-grpo8-sft` | 0.54 → 0.77 | 0.808 | 0.10 → 0.75 | 0.20 → 0.23 | 0.84 → 0.02 | 0.000 | 66 → 8 | 4,000 |
| `vqa-p1-f-short8-base` | 0.74 → 0.71 | 0.847 | 0.65 → 0.68 | 0.20 → 0.23 | 0.14 → 0.06 | 0.000 | 2 → 2 | 4,000 |
<!-- /report:diagnostics -->

## Learnability of the RL pool

<!-- TODO(final rerun): the table now also lists the Part 1 pool scoring (first-N tiers) and short-prompt pre-checks; update this description. -->
Before an RL run, `python -m scripts.learnability` takes a seeded random sample of each
dataset's RL pool, draws a group of eight responses per prompt from the starting model with
the run's rollout settings, and scores them with the GRPO correctness reward. Only prompts
whose responses are neither all right nor all wrong give GRPO a gradient. The pre-check
shows images at the evaluation resolution rather than the RL canvas, so it approximates the
rate a run sees.

<!-- report:learnability -->
| Model | Dataset | Prompt | Prompts | Samples each | Learnable (%) | Never correct (%) | Always correct (%) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `base2` | CLEVR counting | reasoning | 500 | 8 | 48.4 | 0.0 | 51.6 |
| `base2` | GQA | reasoning | 500 | 8 | 42.0 | 26.6 | 31.4 |
| `base2` | GQA | reasoning | 2,500 | 8 | 44.5 | 26.0 | 29.5 |
| `base2` | GQA | short | 2,500 | 8 | 35.5 | 19.1 | 45.4 |
| `base2` | GQA | reasoning | 5,000 | 8 | 45.0 | 24.3 | 30.7 |
| `base2` | VQAv2 | reasoning | 500 | 8 | 47.8 | 19.8 | 32.4 |
| `base2` | VQAv2 | reasoning | 2,500 | 8 | 48.9 | 21.3 | 29.8 |
| `base2` | VQAv2 | short | 2,500 | 8 | 40.4 | 15.1 | 44.5 |
| `base2` | VQAv2 | reasoning | 5,000 | 8 | 47.1 | 22.6 | 30.3 |
| `base8` | GQA | short | 2,500 | 8 | 23.6 | 19.2 | 57.2 |
| `base8` | GQA | reasoning | 5,000 | 8 | 34.7 | 23.9 | 41.4 |
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
<!-- TODO(final rerun): describe the curves. -->

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
<!-- /report:curves -->

## Recipe experiments

Before the final rerun, one change at a time was tested against a baseline run at 2B,
evaluated on the full test cells. Each candidate minus its reference, official scorer,
with a 95% bootstrap interval over questions (10,000 resamples); one training seed per arm.

| Arm | Change against its reference |
|---|---|
| Recipe baseline | Instruct 2B → GRPO: whole rl roles of VQAv2 and GQA in random order, 4 questions × 8 answers per update, learning rate 1e-5, correctness reward, 1,000 updates, merged-weight sampler |
| Filtered pool | pool limited to questions Instruct 2B answers right on 1–7 of 8 samples (4,609 questions from the first 5,000 rl questions of each dataset) |
| Short rollout prompt | rollouts under the short prompt with 64-token completions |
| Short rollout prompt, from SFT | the same, starting from SFT 2B; compared with the baseline and with SFT 2B |
| Refill dead groups | up to 3 extra sampling rounds replace groups whose rewards are all equal |
| 16 questions per update | 16 questions × 8 answers per update for 250 updates (the same 4,000 questions) |
| Learning rate 3e-5, 1e-4 | the learning rate |
| Unmerged-adapter rollouts | the sampler serves the LoRA adapter unmerged instead of bf16 merged weights |
| 8B, correctness + 0.2 format | Instruct 8B with a 0.2 format reward added, 300 updates; compared with Instruct 8B → GRPO (Sep 13) |

<!-- report:recipe -->
| Comparison | Prompt | Scorer | VQAv2 | GQA | CLEVR counting | TextVQA (far-OOD) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Filtered pool − Recipe baseline | short | official | +0.11 [-0.23, +0.45] | -0.22 [-0.49, +0.05] | +0.00 [+0.00, +0.00] | -0.14 [-0.49, +0.20] |
| Filtered pool − Recipe baseline | reasoning | official | -1.75 [-2.31, -1.19] | -0.78 [-1.21, -0.34] | -18.10 [-19.53, -16.70] | -1.93 [-2.51, -1.37] |
| Short rollout prompt − Recipe baseline | short | official | +0.53 [+0.11, +0.94] | +0.56 [+0.21, +0.90] | -0.07 [-0.17, +0.00] | +0.13 [-0.23, +0.49] |
| Short rollout prompt − Recipe baseline | reasoning | official | -16.11 [-17.26, -14.94] | -17.14 [-17.99, -16.29] | -12.20 [-13.47, -10.97] | -8.10 [-9.10, -7.07] |
| Short rollout prompt, from SFT − Recipe baseline | short | official | +1.84 [+1.19, +2.50] | +2.15 [+1.63, +2.67] | -0.03 [-0.17, +0.07] | -0.17 [-0.80, +0.46] |
| Short rollout prompt, from SFT − Recipe baseline | reasoning | official | -7.61 [-8.57, -6.67] | -1.88 [-2.54, -1.25] | -13.83 [-15.13, -12.57] | -7.19 [-8.16, -6.24] |
| Short rollout prompt, from SFT − SFT 2B | short | official | +0.06 [-0.40, +0.53] | -0.09 [-0.48, +0.29] | +0.07 [-0.10, +0.23] | +0.57 [+0.10, +1.04] |
| Short rollout prompt, from SFT − SFT 2B | reasoning | official | +2.75 [+1.94, +3.55] | +1.72 [+1.22, +2.21] | -6.80 [-8.20, -5.43] | +3.55 [+2.67, +4.46] |
| Refill dead groups − Recipe baseline | short | official | -0.25 [-0.57, +0.07] | +0.10 [-0.18, +0.40] | +0.20 [-0.07, +0.47] | -0.06 [-0.38, +0.24] |
| Refill dead groups − Recipe baseline | reasoning | official | +0.24 [-0.19, +0.66] | +0.25 [-0.08, +0.56] | -1.97 [-2.47, -1.50] | +0.33 [-0.12, +0.77] |
| 16 questions per update − Recipe baseline | short | official | -0.12 [-0.40, +0.16] | -0.34 [-0.60, -0.08] | -0.83 [-1.17, -0.50] | -0.23 [-0.52, +0.06] |
| 16 questions per update − Recipe baseline | reasoning | official | -1.28 [-1.77, -0.80] | -2.27 [-2.70, -1.83] | +0.57 [+0.30, +0.87] | -0.81 [-1.24, -0.38] |
| Learning rate 3e-5 − Recipe baseline | short | official | +0.27 [-0.16, +0.70] | -0.06 [-0.42, +0.31] | -0.10 [-0.27, +0.03] | -0.07 [-0.49, +0.34] |
| Learning rate 3e-5 − Recipe baseline | reasoning | official | -0.71 [-1.32, -0.11] | -0.56 [-1.07, -0.06] | +0.53 [+0.27, +0.83] | +0.86 [+0.27, +1.44] |
| Learning rate 1e-4 − Recipe baseline | short | official | -0.74 [-1.35, -0.10] | -1.64 [-2.16, -1.12] | -11.07 [-12.20, -9.97] | -2.24 [-2.88, -1.61] |
| Learning rate 1e-4 − Recipe baseline | reasoning | official | -5.56 [-6.47, -4.64] | -4.37 [-5.07, -3.70] | -1.20 [-1.77, -0.67] | -3.79 [-4.76, -2.82] |
| Unmerged-adapter rollouts − Recipe baseline | short | official | -0.19 [-0.48, +0.10] | +0.04 [-0.23, +0.31] | -0.17 [-0.33, -0.03] | +0.19 [-0.08, +0.46] |
| Unmerged-adapter rollouts − Recipe baseline | reasoning | official | +0.25 [-0.14, +0.64] | -0.21 [-0.54, +0.13] | +0.43 [+0.20, +0.67] | +0.52 [+0.10, +0.92] |
| 8B, correctness + 0.2 format − Instruct 8B → GRPO (Sep 13) | short | official | +0.00 [-0.18, +0.19] | -0.02 [-0.17, +0.14] | +0.03 [+0.00, +0.10] | +0.00 [-0.23, +0.24] |
| 8B, correctness + 0.2 format − Instruct 8B → GRPO (Sep 13) | reasoning | official | -9.65 [-10.63, -8.67] | -9.60 [-10.33, -8.86] | -0.70 [-1.33, -0.10] | -5.55 [-6.42, -4.70] |
<!-- /report:recipe -->

<!-- TODO(final rerun): describe the recipe results. -->

The July runs, which used a defective VQAv2 reward, are tabulated with the erratum in
[history](history.md).
