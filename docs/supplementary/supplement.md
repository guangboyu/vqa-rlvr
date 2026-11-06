# Supplementary results

Negative and null results, controls, and diagnostics behind the [main results](results.md):
the full test matrix of every Part 1 arm, what RL does to reasoning on VQAv2 and GQA and how
much of its exact-match gain is answer form, short-answer RL on the full pool, held-out
pass@k, the mixture of both dataset families, the training knobs that did not move results,
image resolution, and the per-run diagnostics. Conventions as on the main page: generated
tables, greedy decoding, one training seed per arm, official scorer first.

## Prompt template × scorer variant

The full test matrix of the Part 1 study: every arm under both prompts on the four
datasets.

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
| SFT → GRPO | short | 79.76 (79.78) | 62.82 (62.82) | 99.70 (99.70) | 80.19 (80.20) |
| SFT → GRPO | reasoning | 78.76 (78.88) | 61.52 (61.54) | 86.90 (86.90) | 79.02 (79.02) |
| Instruct → short-prompt GRPO | short | 78.62 (78.66) | 60.34 (60.35) | 99.50 (99.50) | 81.01 (81.02) |
| Instruct → short-prompt GRPO | reasoning | 61.85 (69.81) | 42.89 (54.01) | 86.50 (86.50) | 72.34 (72.35) |
| SFT → short-prompt GRPO | short | 79.83 (79.85) | 62.50 (62.50) | 99.50 (99.50) | 80.89 (80.90) |
| SFT → short-prompt GRPO | reasoning | 68.05 (68.31) | 56.88 (57.26) | 88.97 (88.97) | 70.72 (70.73) |
| Instruct → gold-conditioned OPD | short | 77.96 (78.00) | 59.80 (59.80) | 97.93 (97.93) | 80.52 (80.53) |
| Instruct → gold-conditioned OPD | reasoning | 70.41 (70.79) | 52.53 (52.73) | 86.50 (86.50) | 72.75 (72.76) |
| Instruct 8B | short | 81.18 (81.22) | 61.66 (61.66) | 99.67 (99.67) | 83.49 (83.51) |
| Instruct 8B | reasoning | 63.67 (68.75) | 46.84 (52.27) | 96.97 (96.97) | 74.00 (74.00) |
| SFT 8B | short | 83.44 (83.46) | 63.86 (63.86) | 88.83 (88.83) | 84.73 (84.76) |
| SFT 8B | reasoning | 68.36 (69.26) | 48.71 (49.18) | 96.70 (96.70) | 75.16 (75.18) |
| Instruct 8B → GRPO | short | 81.24 (81.26) | 62.32 (62.32) | 99.63 (99.63) | 84.35 (84.37) |
| Instruct 8B → GRPO | reasoning | 81.89 (82.19) | 62.66 (62.72) | 98.20 (98.20) | 83.28 (83.30) |
| SFT 8B → GRPO | short | 83.23 (83.25) | 63.90 (63.90) | 97.00 (97.00) | 84.67 (84.70) |
| SFT 8B → GRPO | reasoning | 82.03 (82.29) | 63.21 (63.32) | 98.33 (98.33) | 83.83 (83.85) |
| Instruct 8B → short-prompt GRPO | short | 81.89 (81.93) | 62.41 (62.41) | 98.80 (98.80) | 84.29 (84.31) |
| Instruct 8B → short-prompt GRPO | reasoning | 66.97 (70.31) | 50.05 (53.59) | 96.80 (96.80) | 74.29 (74.30) |
| SFT 8B → short-prompt GRPO | short | 83.30 (83.32) | 63.82 (63.82) | 82.13 (82.13) | 84.73 (84.75) |
| SFT 8B → short-prompt GRPO | reasoning | 70.23 (70.87) | 51.94 (52.36) | 96.43 (96.43) | 75.37 (75.38) |
<!-- /report:matrix -->

**RL turns the reasoning prompt into an answer.** Under the reasoning prompt, 1,000 GRPO
updates from Instruct 2B raise official accuracy by 23.2 points on VQAv2, 24.1 on GQA, and
10.2 on far-OOD TextVQA; the normalized scorer gives +14.3, +8.4, and +10.3, so most of the
gain is not answer casing. At 8B the gains are 18.2, 15.8, and 9.3 official (+13.4, +10.5,
+9.3 normalized). CLEVR counting, which is not in the RL mix, gains 12.1 points at 2B: the
reasoning that was removed was what miscounted. Responses shrink to 8–10 tokens and all of
them close the answer tag (behavior table below), so the reasoning prompt ends level with
the same model's short prompt on VQAv2 and GQA and about 1 point below it on TextVQA
(within-arm differences below). Gold-conditioned OPD, trained on the same sampled
questions, keeps the reasoning (141 tokens on VQAv2) and raises official reasoning-prompt
accuracy by 15.4 points on VQAv2 and 3.0 on TextVQA (+6.1 and +3.0 normalized); its
16.4-point official gain on GQA is mostly answer casing (+0.8 normalized).

**The short-answer ceiling moves a little under RL and more under SFT.** GRPO trained under
the short prompt from Instruct raises short-prompt official accuracy by 0.75 points on
VQAv2 and 0.89 on GQA at 2B, and by 0.71 and 0.76 at 8B; GRPO trained under the reasoning
prompt moves it by 0.07 and 0.60 at 2B and 0.06 and 0.66 at 8B. One epoch of short-answer
SFT raises it by 2.0 and 2.9 at 2B and by 2.3 and 2.2 at 8B. On far-OOD TextVQA at 2B the
order reverses: short-prompt GRPO adds 0.5 points and SFT −0.5, so short-prompt GRPO leads
SFT by 0.9 (95% interval [+0.2, +1.6]); at 8B both add about 1 point.

**SFT then RL.** GRPO after SFT keeps SFT's short-prompt accuracy (within 0.5 points on
VQAv2 and GQA at both sizes) and adds reasoning-prompt accuracy over SFT: +10.8 on VQAv2,
+4.9 on GQA, and +10.2 on TextVQA at 2B; +13.7, +14.5, and +8.7 at 8B. The 2B SFT start
keeps slightly longer answers under the reasoning prompt (11–17 tokens) and stays 1.0–1.3
points below its short prompt in-domain and 12.8 below on CLEVR counting. At 8B, where SFT
costs 10.8 points of CLEVR counting under the short prompt, GRPO with the reasoning prompt
wins back 8.2 of them and GRPO with the short prompt loses 6.7 more.

Within-arm differences, reasoning prompt minus short prompt, official, paired 95% intervals:

| Arm | VQAv2 | GQA | CLEVR counting | TextVQA |
|---|---:|---:|---:|---:|
| Instruct → GRPO | +0.33 [−0.34, +1.00] | +0.17 [−0.31, +0.66] | −0.20 [−0.57, +0.17] | −1.03 [−1.71, −0.34] |
| SFT → GRPO | −1.00 [−1.60, −0.40] | −1.30 [−1.76, −0.83] | −12.80 [−13.97, −11.63] | −1.17 [−1.92, −0.43] |
| Instruct 8B → GRPO | +0.65 [+0.09, +1.21] | +0.35 [−0.06, +0.77] | −1.43 [−1.90, −1.00] | −1.06 [−1.62, −0.50] |

Repeated greedy runs of the same model under the reasoning prompt vary by about 0.4 points
(three repeats of Instruct 2B on VQAv2 and GQA: 54.91–55.33 and 35.70–36.13 official, with
22–24% of responses differing between runs), so the 8B VQAv2 difference is not read as a
gain.

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
| SFT → GRPO − Instruct 2B | short | normalized | +1.87 [+1.22, +2.50] | +3.36 [+2.79, +3.91] | +2.03 [+1.53, +2.57] | -0.35 [-1.00, +0.32] |
| SFT → GRPO − Instruct 2B | short | official | +1.89 [+1.24, +2.52] | +3.36 [+2.80, +3.92] | +2.03 [+1.53, +2.57] | -0.33 [-0.99, +0.33] |
| SFT → GRPO − Instruct 2B | reasoning | normalized | +14.21 [+13.11, +15.31] | +9.60 [+8.82, +10.35] | -0.23 [-1.73, +1.20] | +9.27 [+8.22, +10.33] |
| SFT → GRPO − Instruct 2B | reasoning | official | +23.71 [+22.42, +24.97] | +25.39 [+24.48, +26.28] | -0.23 [-1.73, +1.20] | +9.28 [+8.23, +10.34] |
| SFT → GRPO − SFT 2B | short | normalized | -0.11 [-0.46, +0.22] | +0.47 [+0.17, +0.76] | +0.30 [+0.07, +0.57] | +0.11 [-0.21, +0.43] |
| SFT → GRPO − SFT 2B | short | official | -0.11 [-0.46, +0.22] | +0.47 [+0.17, +0.76] | +0.30 [+0.07, +0.57] | +0.11 [-0.20, +0.44] |
| SFT → GRPO − SFT 2B | reasoning | normalized | +10.71 [+9.77, +11.68] | +4.54 [+3.94, +5.14] | -5.20 [-6.47, -3.93] | +10.15 [+9.14, +11.18] |
| SFT → GRPO − SFT 2B | reasoning | official | +10.77 [+9.82, +11.75] | +4.91 [+4.31, +5.52] | -5.20 [-6.47, -3.93] | +10.16 [+9.15, +11.20] |
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
| SFT → short-prompt GRPO − SFT → GRPO | short | normalized | +0.07 [-0.37, +0.50] | -0.32 [-0.68, +0.03] | -0.20 [-0.40, -0.03] | +0.70 [+0.28, +1.12] |
| SFT → short-prompt GRPO − SFT → GRPO | short | official | +0.07 [-0.37, +0.50] | -0.32 [-0.68, +0.03] | -0.20 [-0.40, -0.03] | +0.70 [+0.28, +1.12] |
| SFT → short-prompt GRPO − SFT → GRPO | reasoning | normalized | -10.57 [-11.54, -9.59] | -4.29 [-4.87, -3.69] | +2.07 [+0.63, +3.53] | -8.29 [-9.25, -7.33] |
| SFT → short-prompt GRPO − SFT → GRPO | reasoning | official | -10.71 [-11.68, -9.72] | -4.64 [-5.23, -4.04] | +2.07 [+0.63, +3.53] | -8.30 [-9.26, -7.34] |
| Instruct → gold-conditioned OPD − Instruct 2B | short | normalized | +0.09 [-0.23, +0.41] | +0.34 [+0.09, +0.60] | +0.27 [+0.10, +0.47] | -0.02 [-0.40, +0.36] |
| Instruct → gold-conditioned OPD − Instruct 2B | short | official | +0.09 [-0.23, +0.41] | +0.35 [+0.10, +0.60] | +0.27 [+0.10, +0.47] | -0.01 [-0.39, +0.36] |
| Instruct → gold-conditioned OPD − Instruct 2B | reasoning | normalized | +6.12 [+5.05, +7.17] | +0.78 [+0.08, +1.49] | -0.63 [-1.80, +0.53] | +3.01 [+2.03, +4.01] |
| Instruct → gold-conditioned OPD − Instruct 2B | reasoning | official | +15.36 [+14.11, +16.61] | +16.39 [+15.54, +17.23] | -0.63 [-1.80, +0.53] | +3.01 [+2.03, +4.01] |
| SFT 8B − Instruct 8B | short | normalized | +2.24 [+1.66, +2.83] | +2.20 [+1.69, +2.71] | -10.83 [-11.97, -9.73] | +1.25 [+0.64, +1.85] |
| SFT 8B − Instruct 8B | short | official | +2.26 [+1.68, +2.85] | +2.20 [+1.69, +2.71] | -10.83 [-11.97, -9.73] | +1.25 [+0.64, +1.85] |
| SFT 8B − Instruct 8B | reasoning | normalized | +0.51 [-0.56, +1.60] | -3.08 [-3.88, -2.31] | -0.27 [-1.07, +0.50] | +1.17 [+0.18, +2.19] |
| SFT 8B − Instruct 8B | reasoning | official | +4.69 [+3.53, +5.88] | +1.87 [+1.03, +2.71] | -0.27 [-1.07, +0.50] | +1.16 [+0.17, +2.18] |
| Instruct 8B → GRPO − Instruct 8B | short | normalized | +0.04 [-0.27, +0.34] | +0.66 [+0.41, +0.92] | -0.03 [-0.17, +0.07] | +0.86 [+0.49, +1.23] |
| Instruct 8B → GRPO − Instruct 8B | short | official | +0.06 [-0.25, +0.37] | +0.66 [+0.41, +0.92] | -0.03 [-0.17, +0.07] | +0.86 [+0.49, +1.23] |
| Instruct 8B → GRPO − Instruct 8B | reasoning | normalized | +13.44 [+12.45, +14.44] | +10.45 [+9.71, +11.22] | +1.23 [+0.47, +1.97] | +9.30 [+8.36, +10.25] |
| Instruct 8B → GRPO − Instruct 8B | reasoning | official | +18.22 [+17.14, +19.34] | +15.82 [+15.00, +16.63] | +1.23 [+0.47, +1.97] | +9.28 [+8.35, +10.23] |
| SFT 8B → GRPO − Instruct 8B | short | normalized | +2.04 [+1.43, +2.64] | +2.24 [+1.74, +2.75] | -2.67 [-3.27, -2.07] | +1.19 [+0.58, +1.80] |
| SFT 8B → GRPO − Instruct 8B | short | official | +2.06 [+1.45, +2.66] | +2.24 [+1.74, +2.75] | -2.67 [-3.27, -2.07] | +1.19 [+0.58, +1.80] |
| SFT 8B → GRPO − Instruct 8B | reasoning | normalized | +13.54 [+12.54, +14.53] | +11.06 [+10.31, +11.81] | +1.37 [+0.63, +2.07] | +9.84 [+8.84, +10.85] |
| SFT 8B → GRPO − Instruct 8B | reasoning | official | +18.36 [+17.26, +19.47] | +16.37 [+15.55, +17.18] | +1.37 [+0.63, +2.07] | +9.83 [+8.83, +10.84] |
| SFT 8B → GRPO − SFT 8B | short | normalized | -0.21 [-0.47, +0.06] | +0.04 [-0.20, +0.28] | +8.17 [+7.20, +9.17] | -0.06 [-0.33, +0.21] |
| SFT 8B → GRPO − SFT 8B | short | official | -0.21 [-0.47, +0.06] | +0.04 [-0.20, +0.28] | +8.17 [+7.20, +9.17] | -0.06 [-0.33, +0.21] |
| SFT 8B → GRPO − SFT 8B | reasoning | normalized | +13.02 [+12.02, +14.05] | +14.14 [+13.36, +14.93] | +1.63 [+0.93, +2.33] | +8.67 [+7.72, +9.60] |
| SFT 8B → GRPO − SFT 8B | reasoning | official | +13.66 [+12.65, +14.70] | +14.50 [+13.71, +15.30] | +1.63 [+0.93, +2.33] | +8.67 [+7.72, +9.61] |
| Instruct 8B → short-prompt GRPO − Instruct 8B | short | normalized | +0.71 [+0.35, +1.07] | +0.76 [+0.45, +1.07] | -0.87 [-1.23, -0.53] | +0.81 [+0.45, +1.16] |
| Instruct 8B → short-prompt GRPO − Instruct 8B | short | official | +0.71 [+0.35, +1.07] | +0.76 [+0.45, +1.07] | -0.87 [-1.23, -0.53] | +0.81 [+0.45, +1.16] |
| Instruct 8B → short-prompt GRPO − Instruct 8B | reasoning | normalized | +1.56 [+0.93, +2.20] | +1.32 [+0.88, +1.75] | -0.17 [-0.73, +0.40] | +0.30 [-0.29, +0.90] |
| Instruct 8B → short-prompt GRPO − Instruct 8B | reasoning | official | +3.30 [+2.56, +4.05] | +3.20 [+2.70, +3.70] | -0.17 [-0.73, +0.40] | +0.29 [-0.29, +0.89] |
| Instruct 8B → short-prompt GRPO − SFT 8B | short | normalized | -1.53 [-2.08, -0.98] | -1.45 [-1.94, -0.95] | +9.97 [+8.90, +11.07] | -0.44 [-1.00, +0.12] |
| Instruct 8B → short-prompt GRPO − SFT 8B | short | official | -1.55 [-2.09, -1.00] | -1.45 [-1.94, -0.95] | +9.97 [+8.90, +11.07] | -0.44 [-1.00, +0.12] |
| Instruct 8B → short-prompt GRPO − SFT 8B | reasoning | normalized | +1.05 [+0.01, +2.10] | +4.40 [+3.62, +5.19] | +0.10 [-0.70, +0.90] | -0.87 [-1.87, +0.12] |
| Instruct 8B → short-prompt GRPO − SFT 8B | reasoning | official | -1.39 [-2.51, -0.27] | +1.34 [+0.52, +2.15] | +0.10 [-0.70, +0.90] | -0.87 [-1.86, +0.13] |
| Instruct 8B → short-prompt GRPO − Instruct 8B → GRPO | short | normalized | +0.67 [+0.38, +0.96] | +0.10 [-0.17, +0.37] | -0.83 [-1.20, -0.50] | -0.05 [-0.28, +0.18] |
| Instruct 8B → short-prompt GRPO − Instruct 8B → GRPO | short | official | +0.65 [+0.36, +0.94] | +0.10 [-0.17, +0.37] | -0.83 [-1.20, -0.50] | -0.05 [-0.28, +0.18] |
| Instruct 8B → short-prompt GRPO − Instruct 8B → GRPO | reasoning | normalized | -11.87 [-12.83, -10.92] | -9.13 [-9.87, -8.40] | -1.40 [-2.17, -0.67] | -9.00 [-9.93, -8.09] |
| Instruct 8B → short-prompt GRPO − Instruct 8B → GRPO | reasoning | official | -14.91 [-15.96, -13.87] | -12.62 [-13.39, -11.85] | -1.40 [-2.17, -0.67] | -8.99 [-9.92, -8.08] |
| SFT 8B → short-prompt GRPO − Instruct 8B | short | normalized | +2.10 [+1.50, +2.70] | +2.16 [+1.66, +2.68] | -17.53 [-18.90, -16.17] | +1.24 [+0.62, +1.87] |
| SFT 8B → short-prompt GRPO − Instruct 8B | short | official | +2.12 [+1.52, +2.72] | +2.16 [+1.66, +2.68] | -17.53 [-18.90, -16.17] | +1.24 [+0.62, +1.87] |
| SFT 8B → short-prompt GRPO − Instruct 8B | reasoning | normalized | +2.12 [+1.09, +3.16] | +0.10 [-0.65, +0.83] | -0.53 [-1.30, +0.27] | +1.38 [+0.38, +2.37] |
| SFT 8B → short-prompt GRPO − Instruct 8B | reasoning | official | +6.56 [+5.44, +7.71] | +5.10 [+4.29, +5.89] | -0.53 [-1.30, +0.27] | +1.37 [+0.37, +2.36] |
| SFT 8B → short-prompt GRPO − SFT 8B | short | normalized | -0.14 [-0.45, +0.17] | -0.04 [-0.32, +0.25] | -6.70 [-7.60, -5.80] | -0.01 [-0.34, +0.33] |
| SFT 8B → short-prompt GRPO − SFT 8B | short | official | -0.14 [-0.45, +0.17] | -0.04 [-0.32, +0.25] | -6.70 [-7.60, -5.80] | -0.01 [-0.34, +0.33] |
| SFT 8B → short-prompt GRPO − SFT 8B | reasoning | normalized | +1.61 [+0.77, +2.44] | +3.18 [+2.60, +3.75] | -0.27 [-0.83, +0.27] | +0.21 [-0.50, +0.92] |
| SFT 8B → short-prompt GRPO − SFT 8B | reasoning | official | +1.87 [+1.03, +2.73] | +3.23 [+2.65, +3.80] | -0.27 [-0.83, +0.27] | +0.21 [-0.50, +0.92] |
| SFT 8B → short-prompt GRPO − SFT 8B → GRPO | short | normalized | +0.06 [-0.24, +0.36] | -0.08 [-0.34, +0.18] | -14.87 [-16.13, -13.57] | +0.05 [-0.20, +0.30] |
| SFT 8B → short-prompt GRPO − SFT 8B → GRPO | short | official | +0.06 [-0.24, +0.36] | -0.08 [-0.34, +0.18] | -14.87 [-16.13, -13.57] | +0.05 [-0.20, +0.30] |
| SFT 8B → short-prompt GRPO − SFT 8B → GRPO | reasoning | normalized | -11.41 [-12.36, -10.46] | -10.96 [-11.71, -10.22] | -1.90 [-2.63, -1.17] | -8.46 [-9.37, -7.53] |
| SFT 8B → short-prompt GRPO − SFT 8B → GRPO | reasoning | official | -11.79 [-12.75, -10.83] | -11.27 [-12.02, -10.53] | -1.90 [-2.63, -1.17] | -8.46 [-9.37, -7.53] |
<!-- /report:deltas -->

## Exact-match misses rescored by an LLM judge

Exact-match scorers miss answers that mean the same as a reference ("grey" for "gray",
"2 dogs" for "2"). Every headline cell of the final rerun was rescored: each non-empty answer the
official scorer gives 0 went to Anthropic's Haiku 4.5 (pinned in `vqar/judge.py`, temperature 0),
which saw the question, the references and the extracted answer, never the reasoning text or the
image, and answered whether the answer means the same as a reference (`vqar/judge.py`,
`python -m scripts.judge_records`). A rescued miss gets full credit, so the adjusted accuracy is
exact for GQA and CLEVR counting and an upper bound for VQAv2 and TextVQA; answers with partial
annotator credit are not judged. As a control, 100 exact hits per cell went to the judge as well.
Each cell shows official / judge-adjusted / audited accuracy, and in parentheses the share of
misses the judge accepted; the audit is explained below the tables.

<!-- report:judge -->
| Arm | Prompt | VQAv2 | GQA | CLEVR counting | TextVQA (far-OOD) |
| --- | --- | ---: | ---: | ---: | ---: |
| Instruct 2B | short | 77.87 / 81.83 / 81.54 (27%) | 59.45 / 66.83 / 66.54 (18%) | 97.67 / 97.67 (0%) | 80.53 / 84.21 / 83.52 (25%) |
| Instruct 2B | reasoning | 55.06 / 79.48 / 77.66 (61%) | 36.13 / 63.15 / 61.79 (42%) | 87.13 / 87.63 (4%) | 69.74 / 82.38 / 80.09 (50%) |
| SFT 2B | short | 79.87 / 82.91 / 82.69 (24%) | 62.35 / 68.30 / 68.06 (16%) | 99.40 / 99.40 (0%) | 80.08 / 83.42 / 82.76 (23%) |
| SFT 2B | reasoning | 67.99 / 80.49 / 79.60 (49%) | 56.61 / 65.48 / 64.98 (21%) | 92.10 / 92.50 (5%) | 68.86 / 81.38 / 79.18 (48%) |
| Instruct → GRPO | short | 77.94 / 81.70 / 81.42 (26%) | 60.06 / 67.35 / 67.06 (18%) | 99.47 / 99.47 (0%) | 81.02 / 84.34 / 83.67 (24%) |
| Instruct → GRPO | reasoning | 78.27 / 82.37 / 82.08 (28%) | 60.23 / 66.84 / 66.57 (17%) | 99.27 / 99.27 (0%) | 79.99 / 83.95 / 83.15 (26%) |
| SFT → GRPO | short | 79.76 / 82.78 / 82.56 (24%) | 62.82 / 68.79 / 68.55 (16%) | 99.70 / 99.70 (0%) | 80.19 / 83.51 / 82.84 (22%) |
| SFT → GRPO | reasoning | 78.76 / 82.62 / 82.35 (27%) | 61.52 / 67.73 / 67.48 (16%) | 86.90 / 86.90 (0%) | 79.02 / 83.60 / 82.75 (29%) |
| Instruct → short-prompt GRPO | short | 78.62 / 82.02 / 81.77 (24%) | 60.34 / 67.16 / 66.89 (17%) | 99.50 / 99.50 (0%) | 81.01 / 84.27 / 83.60 (23%) |
| Instruct → short-prompt GRPO | reasoning | 61.85 / 80.83 / 79.43 (58%) | 42.89 / 63.15 / 62.21 (35%) | 86.50 / 86.60 (1%) | 72.34 / 82.28 / 80.39 (44%) |
| SFT → short-prompt GRPO | short | 79.83 / 82.71 / 82.50 (23%) | 62.50 / 68.57 / 68.33 (16%) | 99.50 / 99.50 (0%) | 80.89 / 84.05 / 83.40 (22%) |
| SFT → short-prompt GRPO | reasoning | 68.05 / 81.91 / 80.91 (53%) | 56.88 / 66.27 / 65.68 (22%) | 88.97 / 89.20 (2%) | 70.72 / 82.32 / 80.29 (48%) |
| Instruct → gold-conditioned OPD | short | 77.96 / 82.00 / 81.70 (28%) | 59.80 / 67.21 / 66.92 (18%) | 97.93 / 97.93 (0%) | 80.52 / 84.02 / 83.32 (24%) |
| Instruct → gold-conditioned OPD | reasoning | 70.41 / 78.93 / 78.31 (36%) | 52.53 / 63.32 / 62.52 (23%) | 86.50 / 86.60 (1%) | 72.75 / 82.01 / 80.12 (41%) |
| Instruct 8B | short | 81.18 / 85.02 / 84.74 (33%) | 61.66 / 69.14 / 68.84 (20%) | 99.67 / 99.67 (0%) | 83.49 / 86.69 / 86.14 (28%) |
| Instruct 8B | reasoning | 63.67 / 82.41 / 81.07 (60%) | 46.84 / 63.10 / 62.25 (31%) | 96.97 / 97.33 (12%) | 74.00 / 84.28 / 82.50 (50%) |
| SFT 8B | short | 83.44 / 85.98 / 85.79 (27%) | 63.86 / 70.73 / 70.45 (19%) | 88.83 / 88.83 (0%) | 84.73 / 87.25 / 86.75 (24%) |
| SFT 8B | reasoning | 68.36 / 82.88 / 81.84 (56%) | 48.71 / 65.04 / 63.54 (32%) | 96.70 / 97.17 (14%) | 75.16 / 85.02 / 83.23 (51%) |
| Instruct 8B → GRPO | short | 81.24 / 84.88 / 84.61 (32%) | 62.32 / 69.54 / 69.25 (19%) | 99.63 / 99.63 (0%) | 84.35 / 87.03 / 86.55 (25%) |
| Instruct 8B → GRPO | reasoning | 81.89 / 85.53 / 85.28 (32%) | 62.66 / 69.65 / 69.37 (19%) | 98.20 / 98.20 (0%) | 83.28 / 86.98 / 86.29 (31%) |
| SFT 8B → GRPO | short | 83.23 / 85.81 / 85.63 (27%) | 63.90 / 70.54 / 70.27 (18%) | 97.00 / 97.00 (0%) | 84.67 / 87.27 / 86.76 (25%) |
| SFT 8B → GRPO | reasoning | 82.03 / 85.45 / 85.20 (31%) | 63.21 / 70.22 / 69.94 (19%) | 98.33 / 98.33 (0%) | 83.83 / 86.85 / 86.28 (27%) |
| Instruct 8B → short-prompt GRPO | short | 81.89 / 85.29 / 85.04 (31%) | 62.41 / 69.55 / 69.26 (19%) | 98.80 / 98.80 (0%) | 84.29 / 86.85 / 86.39 (24%) |
| Instruct 8B → short-prompt GRPO | reasoning | 66.97 / 83.07 / 81.93 (58%) | 50.05 / 63.71 / 63.00 (27%) | 96.80 / 97.40 (19%) | 74.29 / 84.25 / 82.55 (48%) |
| SFT 8B → short-prompt GRPO | short | 83.30 / 85.88 / 85.69 (27%) | 63.82 / 70.58 / 70.31 (19%) | 82.13 / 82.13 (0%) | 84.73 / 87.21 / 86.72 (24%) |
| SFT 8B → short-prompt GRPO | reasoning | 70.23 / 83.23 / 82.31 (54%) | 51.94 / 65.79 / 64.58 (29%) | 96.43 / 96.83 (11%) | 75.37 / 85.05 / 83.29 (50%) |
<!-- /report:judge -->

Each declared paired difference in points, official / judge-adjusted / audited:

<!-- report:judge-deltas -->
| Comparison | Prompt | VQAv2 | GQA | CLEVR counting | TextVQA (far-OOD) |
| --- | --- | ---: | ---: | ---: | ---: |
| SFT 2B − Instruct 2B | short | +2.00 / +1.08 / +1.15 | +2.89 / +1.47 / +1.53 | +1.73 / +1.73 | -0.45 / -0.79 / -0.76 |
| SFT 2B − Instruct 2B | reasoning | +12.93 / +1.01 / +1.94 | +20.47 / +2.33 / +3.19 | +4.97 / +4.87 | -0.89 / -1.01 / -0.91 |
| Instruct → GRPO − Instruct 2B | short | +0.07 / -0.13 / -0.12 | +0.60 / +0.52 / +0.52 | +1.80 / +1.80 | +0.49 / +0.13 / +0.15 |
| Instruct → GRPO − Instruct 2B | reasoning | +23.22 / +2.90 / +4.42 | +24.10 / +3.69 / +4.78 | +12.13 / +11.63 | +10.24 / +1.56 / +3.06 |
| SFT → GRPO − Instruct 2B | short | +1.89 / +0.95 / +1.02 | +3.36 / +1.96 / +2.01 | +2.03 / +2.03 | -0.33 / -0.69 / -0.68 |
| SFT → GRPO − Instruct 2B | reasoning | +23.71 / +3.15 / +4.69 | +25.39 / +4.58 / +5.68 | -0.23 / -0.73 | +9.28 / +1.22 / +2.67 |
| Instruct → short-prompt GRPO − Instruct 2B | short | +0.75 / +0.19 / +0.23 | +0.89 / +0.33 / +0.36 | +1.83 / +1.83 | +0.48 / +0.06 / +0.08 |
| Instruct → short-prompt GRPO − Instruct 2B | reasoning | +6.80 / +1.36 / +1.77 | +6.76 / +0.00 / +0.41 | -0.63 / -1.03 | +2.60 / -0.10 / +0.30 |
| SFT → short-prompt GRPO − Instruct 2B | short | +1.96 / +0.88 / +0.95 | +3.04 / +1.74 / +1.79 | +1.83 / +1.83 | +0.36 / -0.16 / -0.11 |
| SFT → short-prompt GRPO − Instruct 2B | reasoning | +13.00 / +2.44 / +3.25 | +20.74 / +3.12 / +3.89 | +1.83 / +1.57 | +0.98 / -0.06 / +0.20 |
| Instruct → gold-conditioned OPD − Instruct 2B | short | +0.09 / +0.17 / +0.16 | +0.35 / +0.38 / +0.38 | +0.27 / +0.27 | -0.01 / -0.19 / -0.20 |
| Instruct → gold-conditioned OPD − Instruct 2B | reasoning | +15.36 / -0.54 / +0.65 | +16.39 / +0.17 / +0.73 | -0.63 / -1.03 | +3.01 / -0.37 / +0.04 |
| SFT → GRPO − SFT 2B | short | -0.11 / -0.13 / -0.13 | +0.47 / +0.48 / +0.48 | +0.30 / +0.30 | +0.11 / +0.09 / +0.08 |
| SFT → GRPO − SFT 2B | reasoning | +10.77 / +2.13 / +2.74 | +4.91 / +2.25 / +2.49 | -5.20 / -5.60 | +10.16 / +2.22 / +3.57 |
| Instruct → short-prompt GRPO − SFT 2B | short | -1.25 / -0.89 / -0.92 | -2.00 / -1.14 / -1.17 | +0.10 / +0.10 | +0.93 / +0.85 / +0.84 |
| Instruct → short-prompt GRPO − SFT 2B | reasoning | -6.14 / +0.34 / -0.17 | -13.71 / -2.33 / -2.78 | -5.60 / -5.90 | +3.49 / +0.91 / +1.21 |
| SFT → short-prompt GRPO − SFT 2B | short | -0.04 / -0.20 / -0.20 | +0.15 / +0.27 / +0.27 | +0.10 / +0.10 | +0.81 / +0.63 / +0.64 |
| SFT → short-prompt GRPO − SFT 2B | reasoning | +0.06 / +1.42 / +1.31 | +0.27 / +0.79 / +0.70 | -3.13 / -3.30 | +1.86 / +0.94 / +1.11 |
| SFT 8B − Instruct 8B | short | +2.26 / +0.96 / +1.05 | +2.20 / +1.59 / +1.61 | -10.83 / -10.83 | +1.25 / +0.57 / +0.62 |
| SFT 8B − Instruct 8B | reasoning | +4.69 / +0.47 / +0.77 | +1.87 / +1.94 / +1.29 | -0.27 / -0.17 | +1.16 / +0.74 / +0.73 |
| Instruct 8B → GRPO − Instruct 8B | short | +0.06 / -0.14 / -0.13 | +0.66 / +0.41 / +0.42 | -0.03 / -0.03 | +0.86 / +0.34 / +0.41 |
| Instruct 8B → GRPO − Instruct 8B | reasoning | +18.22 / +3.12 / +4.21 | +15.82 / +6.55 / +7.12 | +1.23 / +0.87 | +9.28 / +2.70 / +3.79 |
| SFT 8B → GRPO − Instruct 8B | short | +2.06 / +0.80 / +0.88 | +2.24 / +1.40 / +1.43 | -2.67 / -2.67 | +1.19 / +0.59 / +0.62 |
| SFT 8B → GRPO − Instruct 8B | reasoning | +18.36 / +3.04 / +4.14 | +16.37 / +7.12 / +7.69 | +1.37 / +1.00 | +9.83 / +2.57 / +3.77 |
| Instruct 8B → short-prompt GRPO − Instruct 8B | short | +0.71 / +0.27 / +0.30 | +0.76 / +0.41 / +0.43 | -0.87 / -0.87 | +0.81 / +0.17 / +0.25 |
| Instruct 8B → short-prompt GRPO − Instruct 8B | reasoning | +3.30 / +0.66 / +0.86 | +3.20 / +0.61 / +0.75 | -0.17 / +0.07 | +0.29 / -0.03 / +0.05 |
| SFT 8B → short-prompt GRPO − Instruct 8B | short | +2.12 / +0.86 / +0.94 | +2.16 / +1.45 / +1.48 | -17.53 / -17.53 | +1.24 / +0.52 / +0.58 |
| SFT 8B → short-prompt GRPO − Instruct 8B | reasoning | +6.56 / +0.82 / +1.24 | +5.10 / +2.69 / +2.33 | -0.53 / -0.50 | +1.37 / +0.77 / +0.79 |
| SFT 8B → GRPO − SFT 8B | short | -0.21 / -0.17 / -0.17 | +0.04 / -0.19 / -0.18 | +8.17 / +8.17 | -0.06 / +0.02 / +0.00 |
| SFT 8B → GRPO − SFT 8B | reasoning | +13.66 / +2.56 / +3.36 | +14.50 / +5.18 / +6.39 | +1.63 / +1.17 | +8.67 / +1.83 / +3.05 |
| Instruct 8B → short-prompt GRPO − SFT 8B | short | -1.55 / -0.69 / -0.75 | -1.45 / -1.18 / -1.19 | +9.97 / +9.97 | -0.44 / -0.40 / -0.37 |
| Instruct 8B → short-prompt GRPO − SFT 8B | reasoning | -1.39 / +0.19 / +0.09 | +1.34 / -1.33 / -0.55 | +0.10 / +0.23 | -0.87 / -0.77 / -0.68 |
| SFT 8B → short-prompt GRPO − SFT 8B | short | -0.14 / -0.10 / -0.11 | -0.04 / -0.14 / -0.14 | -6.70 / -6.70 | -0.01 / -0.05 / -0.03 |
| SFT 8B → short-prompt GRPO − SFT 8B | reasoning | +1.87 / +0.35 / +0.47 | +3.23 / +0.75 / +1.03 | -0.27 / -0.33 | +0.21 / +0.03 / +0.06 |
| Instruct → short-prompt GRPO − Instruct → GRPO | short | +0.68 / +0.32 / +0.35 | +0.29 / -0.18 / -0.16 | +0.03 / +0.03 | -0.01 / -0.07 / -0.07 |
| Instruct → short-prompt GRPO − Instruct → GRPO | reasoning | -16.42 / -1.54 / -2.64 | -17.34 / -3.69 / -4.37 | -12.77 / -12.67 | -7.64 / -1.66 / -2.76 |
| SFT → short-prompt GRPO − SFT → GRPO | short | +0.07 / -0.07 / -0.06 | -0.32 / -0.21 / -0.22 | -0.20 / -0.20 | +0.70 / +0.54 / +0.56 |
| SFT → short-prompt GRPO − SFT → GRPO | reasoning | -10.71 / -0.71 / -1.44 | -4.64 / -1.46 / -1.79 | +2.07 / +2.30 | -8.30 / -1.28 / -2.46 |
| Instruct 8B → short-prompt GRPO − Instruct 8B → GRPO | short | +0.65 / +0.41 / +0.43 | +0.10 / +0.01 / +0.01 | -0.83 / -0.83 | -0.05 / -0.17 / -0.16 |
| Instruct 8B → short-prompt GRPO − Instruct 8B → GRPO | reasoning | -14.91 / -2.45 / -3.34 | -12.62 / -5.94 / -6.38 | -1.40 / -0.80 | -8.99 / -2.73 / -3.74 |
| SFT 8B → short-prompt GRPO − SFT 8B → GRPO | short | +0.06 / +0.06 / +0.06 | -0.08 / +0.05 / +0.04 | -14.87 / -14.87 | +0.05 / -0.07 / -0.03 |
| SFT 8B → short-prompt GRPO − SFT 8B → GRPO | reasoning | -11.79 / -2.21 / -2.90 | -11.27 / -4.43 / -5.36 | -1.90 / -1.50 | -8.46 / -1.80 / -2.98 |
<!-- /report:judge-deltas -->

The judge accepts 16–33% of short-prompt misses on VQAv2, GQA and
TextVQA, and 31–61% of the Instruct models' reasoning-prompt misses, whose answers are longer
and freely phrased ("Yes" for "yes", "a large kite flying" for "kite"). On exact hits it agrees
in 98–100% of every cell's controls. An image-aware audit then checked a seeded sample of its
verdicts: Opus 5.5 at low effort, shown the image, judged 50 verdicts per stratum
(`python -m scripts.audit_judge`, $1.31). It confirms 92–94% of the judge's acceptances on
VQAv2, 86–96% on GQA and 72–88% on TextVQA, where partial OCR strings ("angeles" for "los
angeles") get through; untagged step-by-step text that mentions the answer is the other common
false acceptance. The audited accuracy discounts each acceptance by its stratum's
confirmation rate. The audit also accepts 12–24% of the misses the judge rejected, often where
GQA's single reference is one of several right answers, so even the audited accuracy
undercounts. The judge cost $4.21 for 51,169 distinct answers through the Batch API.

Three readings change under the audited scorer. Most of what exact match credited to GRPO
under the reasoning prompt is answer form: Instruct → GRPO gains 4.4 points on VQAv2 at 2B
(interval [+3.5, +5.4]) instead of 23.2, 4.8 on GQA instead of 24.1, and 3.1 on TextVQA
instead of 10.2; at 8B, 4.2, 7.1 and 3.8. A real gain remains while the reasoning disappears.
The reasoning prompt's cost for Instruct 2B shrinks from 22.8 to 3.8 points on VQAv2 and from
23.3 to 4.7 on GQA, but stays negative, as the July follow-up found. And short-prompt GRPO's
in-domain gains over Instruct fall inside their intervals at 2B (VQAv2 +0.23 [−0.23, +0.68],
GQA +0.36 [−0.03, +0.74]), while SFT's halve and hold (+1.15 and +1.53). Short-prompt GRPO
keeps its TextVQA lead over SFT (+0.84 [+0.22, +1.46]) and SFT keeps its in-domain lead. The
intervals resample questions with the audit rates held fixed; they leave out the audit's own
sampling error.

## Response behavior under the reasoning prompt

Mean output tokens · share of responses that close an `<answer>` tag.

<!-- report:behavior -->
| Arm | VQAv2 | GQA | CLEVR counting | TextVQA (far-OOD) |
| --- | ---: | ---: | ---: | ---: |
| Instruct 2B | 148 · 95% | 123 · 97% | 186 · 98% | 107 · 95% |
| SFT 2B | 21 · 40% | 8 · 20% | 62 · 98% | 34 · 47% |
| Instruct → GRPO | 8 · 100% | 8 · 100% | 8 · 100% | 10 · 100% |
| SFT → GRPO | 11 · 100% | 11 · 100% | 14 · 100% | 17 · 100% |
| Instruct → short-prompt GRPO | 132 · 98% | 116 · 98% | 169 · 99% | 101 · 96% |
| SFT → short-prompt GRPO | 27 · 50% | 11 · 28% | 146 · 99% | 40 · 52% |
| Instruct → gold-conditioned OPD | 141 · 93% | 117 · 93% | 197 · 99% | 111 · 94% |
| Instruct 8B | 118 · 99% | 104 · 99% | 193 · 99% | 111 · 99% |
| SFT 8B | 69 · 78% | 72 · 82% | 96 · 82% | 46 · 71% |
| Instruct 8B → GRPO | 8 · 100% | 8 · 100% | 9 · 100% | 10 · 100% |
| SFT 8B → GRPO | 8 · 100% | 8 · 100% | 8 · 100% | 10 · 100% |
| Instruct 8B → short-prompt GRPO | 113 · 99% | 100 · 99% | 189 · 99% | 110 · 99% |
| SFT 8B → short-prompt GRPO | 71 · 81% | 68 · 86% | 113 · 90% | 55 · 76% |
<!-- /report:behavior -->

GRPO under the reasoning prompt removes the reasoning at both sizes and from both starts:
8–17 tokens per response, every one closing the answer tag. GRPO under the short prompt
leaves the reasoning-prompt behavior of its starting model almost unchanged (Instruct 2B
148 → 132 tokens on VQAv2), and so does OPD (141 tokens). Short-answer SFT shortens
reasoning-prompt responses (VQAv2 148 → 21 tokens at 2B) and many lose the answer tag
(40% keep it); GRPO after SFT restores the tag in every response without restoring the
length. Visible reasoning length follows what the reward pays for; it is not by itself a
measure of better answers. The per-update view of the same collapse is on the
[monitoring](monitoring.md) page.

Responses cut off at the token budget · responses with an empty answer, same prompt.

<!-- report:failures -->
| Arm | VQAv2 | GQA | CLEVR counting | TextVQA (far-OOD) |
| --- | ---: | ---: | ---: | ---: |
| Instruct 2B | 3.7% · 0.0% | 2.3% · 0.0% | 1.8% · 0.0% | 1.3% · 0.0% |
| SFT 2B | 1.8% · 0.8% | 0.5% · 0.3% | 1.3% · 0.0% | 2.1% · 0.2% |
| Instruct → GRPO | 0.0% · 0.0% | 0.0% · 0.0% | 0.0% · 0.0% | 0.0% · 0.0% |
| SFT → GRPO | 0.0% · 0.0% | 0.1% · 0.0% | 0.0% · 0.0% | 0.4% · 0.0% |
| Instruct → short-prompt GRPO | 1.5% · 0.0% | 0.9% · 0.0% | 0.9% · 0.0% | 0.7% · 0.0% |
| SFT → short-prompt GRPO | 1.0% · 0.1% | 0.3% · 0.1% | 0.7% · 0.0% | 0.7% · 0.1% |
| Instruct → gold-conditioned OPD | 4.5% · 0.0% | 3.0% · 0.0% | 1.4% · 0.0% | 2.2% · 0.0% |
| Instruct 8B | 0.9% · 0.0% | 0.4% · 0.0% | 0.7% · 0.0% | 0.5% · 0.0% |
| SFT 8B | 3.5% · 0.0% | 5.0% · 0.0% | 0.3% · 0.0% | 0.7% · 0.0% |
| Instruct 8B → GRPO | 0.0% · 0.0% | 0.0% · 0.0% | 0.0% · 0.0% | 0.0% · 0.0% |
| SFT 8B → GRPO | 0.0% · 0.0% | 0.0% · 0.0% | 0.0% · 0.0% | 0.0% · 0.0% |
| Instruct 8B → short-prompt GRPO | 0.8% · 0.0% | 0.3% · 0.0% | 0.8% · 0.0% | 0.5% · 0.0% |
| SFT 8B → short-prompt GRPO | 2.3% · 0.0% | 3.0% · 0.0% | 0.5% · 0.0% | 0.5% · 0.0% |
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
| SFT → GRPO | VQAv2 | 78.76 | 78.86 | 78.88 | 6 |
| SFT → GRPO | GQA | 61.52 | 61.53 | 61.54 | 3 |
| SFT → GRPO | CLEVR counting | 86.90 | 86.90 | 86.90 | 0 |
| SFT → GRPO | TextVQA | 79.02 | 79.02 | 79.02 | 3 |
| Instruct → short-prompt GRPO | VQAv2 | 61.85 | 69.71 | 69.81 | 398 |
| Instruct → short-prompt GRPO | GQA | 42.89 | 53.12 | 54.01 | 1,398 |
| Instruct → short-prompt GRPO | CLEVR counting | 86.50 | 86.50 | 86.50 | 0 |
| Instruct → short-prompt GRPO | TextVQA | 72.34 | 72.34 | 72.35 | 2 |
| SFT → short-prompt GRPO | VQAv2 | 68.05 | 68.27 | 68.31 | 13 |
| SFT → short-prompt GRPO | GQA | 56.88 | 56.95 | 57.26 | 48 |
| SFT → short-prompt GRPO | CLEVR counting | 88.97 | 88.97 | 88.97 | 0 |
| SFT → short-prompt GRPO | TextVQA | 70.72 | 70.72 | 70.73 | 2 |
| Instruct → gold-conditioned OPD | VQAv2 | 70.41 | 70.77 | 70.79 | 19 |
| Instruct → gold-conditioned OPD | GQA | 52.53 | 52.60 | 52.73 | 25 |
| Instruct → gold-conditioned OPD | CLEVR counting | 86.50 | 86.50 | 86.50 | 0 |
| Instruct → gold-conditioned OPD | TextVQA | 72.75 | 72.75 | 72.76 | 2 |
| Instruct 8B | VQAv2 | 63.67 | 68.71 | 68.75 | 254 |
| Instruct 8B | GQA | 46.84 | 51.28 | 52.27 | 682 |
| Instruct 8B | CLEVR counting | 96.97 | 96.97 | 96.97 | 0 |
| Instruct 8B | TextVQA | 74.00 | 74.00 | 74.00 | 1 |
| SFT 8B | VQAv2 | 68.36 | 69.22 | 69.26 | 45 |
| SFT 8B | GQA | 48.71 | 49.00 | 49.18 | 59 |
| SFT 8B | CLEVR counting | 96.70 | 96.70 | 96.70 | 0 |
| SFT 8B | TextVQA | 75.16 | 75.16 | 75.18 | 3 |
| Instruct 8B → GRPO | VQAv2 | 81.89 | 82.17 | 82.19 | 15 |
| Instruct 8B → GRPO | GQA | 62.66 | 62.72 | 62.72 | 7 |
| Instruct 8B → GRPO | CLEVR counting | 98.20 | 98.20 | 98.20 | 0 |
| Instruct 8B → GRPO | TextVQA | 83.28 | 83.28 | 83.30 | 3 |
| SFT 8B → GRPO | VQAv2 | 82.03 | 82.27 | 82.29 | 13 |
| SFT 8B → GRPO | GQA | 63.21 | 63.32 | 63.32 | 14 |
| SFT 8B → GRPO | CLEVR counting | 98.33 | 98.33 | 98.33 | 0 |
| SFT 8B → GRPO | TextVQA | 83.83 | 83.83 | 83.85 | 2 |
| Instruct 8B → short-prompt GRPO | VQAv2 | 66.97 | 70.27 | 70.31 | 167 |
| Instruct 8B → short-prompt GRPO | GQA | 50.05 | 52.54 | 53.59 | 445 |
| Instruct 8B → short-prompt GRPO | CLEVR counting | 96.80 | 96.80 | 96.80 | 0 |
| Instruct 8B → short-prompt GRPO | TextVQA | 74.29 | 74.29 | 74.30 | 2 |
| SFT 8B → short-prompt GRPO | VQAv2 | 70.23 | 70.85 | 70.87 | 32 |
| SFT 8B → short-prompt GRPO | GQA | 51.94 | 52.26 | 52.36 | 53 |
| SFT 8B → short-prompt GRPO | CLEVR counting | 96.43 | 96.43 | 96.43 | 0 |
| SFT 8B → short-prompt GRPO | TextVQA | 75.37 | 75.37 | 75.38 | 3 |
<!-- /report:casing -->

## Short-prompt RL on the full pool

The headline short-prompt GRPO run stopped at 1,000 updates, 4,000 of the 55,438 pool
questions. Run for one full epoch of the pool instead (13,859 updates of 4 questions, each
question drawn once, otherwise the same recipe; 43.8 GPU-hours on free A30s in 13 chained
segments), it shows what fourteen times the updates and all of the data buy on VQAv2 and GQA.
The end-of-epoch adapter is the reported model, chosen before training
(`[matrices.fullpool]`; `runs/fullpool-20261007/`):

<!-- report:fullpool -->
| Arm | Prompt | VQAv2 | GQA | CLEVR counting | TextVQA (far-OOD) |
| --- | --- | ---: | ---: | ---: | ---: |
| Instruct 2B | short | 77.87 (77.91) | 59.45 (59.46) | 97.67 (97.67) | 80.53 (80.55) |
| Instruct 2B | reasoning | 55.06 (64.68) | 36.13 (51.95) | 87.13 (87.13) | 69.74 (69.75) |
| SFT 2B | short | 79.87 (79.89) | 62.35 (62.35) | 99.40 (99.40) | 80.08 (80.10) |
| SFT 2B | reasoning | 67.99 (68.17) | 56.61 (57.00) | 92.10 (92.10) | 68.86 (68.87) |
| Instruct → short-prompt GRPO | short | 78.62 (78.66) | 60.34 (60.35) | 99.50 (99.50) | 81.01 (81.02) |
| Instruct → short-prompt GRPO | reasoning | 61.85 (69.81) | 42.89 (54.01) | 86.50 (86.50) | 72.34 (72.35) |
| Instruct → short-prompt GRPO, full pool | short | 79.82 (79.86) | 62.39 (62.39) | 99.60 (99.60) | 80.69 (80.69) |
| Instruct → short-prompt GRPO, full pool | reasoning | 64.92 (70.80) | 45.64 (54.47) | 86.30 (86.30) | 73.43 (73.43) |
<!-- /report:fullpool -->

Candidate minus reference, official scorer, 95% intervals over questions:

<!-- report:fullpool-deltas -->
| Comparison | Prompt | Scorer | VQAv2 | GQA | CLEVR counting | TextVQA (far-OOD) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Instruct → short-prompt GRPO, full pool − Instruct 2B | short | official | +1.95 [+1.33, +2.58] | +2.93 [+2.39, +3.47] | +1.93 [+1.43, +2.47] | +0.16 [-0.45, +0.78] |
| Instruct → short-prompt GRPO, full pool − Instruct 2B | reasoning | official | +9.87 [+8.78, +10.99] | +9.51 [+8.70, +10.29] | -0.83 [-2.27, +0.53] | +3.68 [+2.72, +4.65] |
| Instruct → short-prompt GRPO, full pool − SFT 2B | short | official | -0.06 [-0.66, +0.56] | +0.04 [-0.48, +0.55] | +0.20 [+0.00, +0.43] | +0.60 [-0.03, +1.25] |
| Instruct → short-prompt GRPO, full pool − SFT 2B | reasoning | official | -3.07 [-4.29, -1.82] | -10.96 [-11.81, -10.13] | -5.80 [-7.13, -4.47] | +4.57 [+3.43, +5.68] |
| Instruct → short-prompt GRPO, full pool − Instruct → short-prompt GRPO | short | official | +1.19 [+0.62, +1.76] | +2.04 [+1.55, +2.54] | +0.10 [+0.00, +0.23] | -0.32 [-0.89, +0.26] |
| Instruct → short-prompt GRPO, full pool − Instruct → short-prompt GRPO | reasoning | official | +3.07 [+2.06, +4.05] | +2.75 [+1.99, +3.50] | -0.20 [-1.60, +1.20] | +1.08 [+0.16, +2.01] |
<!-- /report:fullpool-deltas -->

The extra data closes SFT's in-domain lead. Short-prompt accuracy rises 1.2 points on
VQAv2 and 2.0 on GQA over the 1,000-update run, to +2.0 and +2.9 over Instruct, level with
one SFT epoch on both (−0.06 [−0.66, +0.56] and +0.04 [−0.48, +0.55]). This is the
matched-question control's conclusion reached from the other side: given the same questions,
RL and SFT end at the same in-domain ceiling. The far-OOD edge does not grow: on TextVQA the
full-pool run sits 0.3 points below the 1,000-update run and 0.6 above SFT, an interval that
now includes zero ([−0.03, +1.25]). Training shows why so little moves. Every question is
new when it is drawn, so the training reward is accuracy on unseen pool questions: it rises
from 0.70 over the first 1,000 updates to 0.73 over the last, while the share of groups whose
eight answers all score alike, which give no gradient, grows from 79% to 89% and entropy
falls from 0.17 to 0.07. On held-out questions the narrowing continues: against Instruct,
pass@1 gains 4.7, 7.1 and 3.4 points on VQAv2, GQA and TextVQA while pass@32 falls 5.8,
10.8 and 4.6, against 3.2, 7.6 and 1.9 after 1,000 updates (table below). On short-answer VQA
more RL buys a small, SFT-sized in-domain gain and a further loss of coverage; on ChartQA,
with reasoning, the extra data kept paying ([main results](results.md#chartqa-rl-where-reasoning-pays)).

## Held-out pass@k before and after RL

pass@k asks whether training lets a model answer questions it could not answer before, or
concentrates probability on answers it could already sample. For text reasoning,
[Yue et al. (2025)](https://arxiv.org/abs/2504.13837) report the second for RL with verifiable
rewards: higher pass@1, the base model level or ahead at large k. Here each arm answers the
first 1,000 test questions of VQAv2, GQA and TextVQA 32 times at temperature 1 (seed 42, the
template's token budget; `evaluate.py --tier first-1000 --temperature 1 --samples 32`). With c of
n = 32 samples fully correct under the official scorer, pass@k is estimated without bias as
1 − C(n−c, k) / C(n, k) and averaged over questions (`python -m scripts.pass_at_k --declared heldout`).
pass@k is an oracle quantity: it needs a grader to pick the right sample, so it is not a
deployable best-of-k, and pass@1 at temperature 1 sits below greedy accuracy. A VQA answer that
matches only some annotators counts in the mean score but never as a pass. Official scorer, %:

<!-- report:passk -->
| Arm | Prompt | Dataset | Mean score | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Instruct 2B | short | VQAv2 | 71.95 | 62.94 | 68.19 | 72.38 | 75.75 | 78.22 | 79.80 |
| SFT 2B | short | VQAv2 | 71.87 | 62.75 | 70.04 | 75.06 | 78.59 | 80.98 | 82.40 |
| Instruct → short-prompt GRPO | short | VQAv2 | 75.49 | 65.94 | 69.10 | 71.77 | 74.00 | 75.64 | 76.60 |
| SFT → short-prompt GRPO | short | VQAv2 | 76.59 | 66.88 | 70.14 | 72.74 | 74.99 | 76.84 | 78.00 |
| SFT on the 4,000 RL questions | short | VQAv2 | 70.42 | 61.67 | 69.19 | 74.45 | 78.09 | 80.53 | 82.30 |
| Instruct → short-prompt GRPO, full pool | short | VQAv2 | 77.26 | 67.68 | 69.64 | 71.22 | 72.49 | 73.44 | 74.00 |
| Instruct 2B | short | GQA | 55.55 | 55.55 | 63.17 | 69.63 | 75.05 | 79.43 | 82.70 |
| SFT 2B | short | GQA | 56.04 | 56.04 | 66.22 | 74.30 | 80.26 | 84.56 | 87.30 |
| Instruct → short-prompt GRPO | short | GQA | 58.12 | 58.12 | 62.70 | 66.59 | 69.84 | 72.62 | 75.10 |
| SFT → short-prompt GRPO | short | GQA | 60.73 | 60.73 | 65.35 | 69.09 | 72.22 | 74.80 | 77.00 |
| SFT on the 4,000 RL questions | short | GQA | 53.50 | 53.50 | 63.93 | 72.27 | 78.83 | 83.93 | 87.90 |
| Instruct → short-prompt GRPO, full pool | short | GQA | 62.62 | 62.62 | 65.11 | 67.02 | 68.69 | 70.30 | 71.90 |
| Instruct 2B | short | TextVQA | 75.64 | 67.77 | 72.70 | 76.08 | 78.71 | 80.62 | 82.00 |
| SFT 2B | short | TextVQA | 70.70 | 63.48 | 70.92 | 75.78 | 79.19 | 81.73 | 83.70 |
| Instruct → short-prompt GRPO | short | TextVQA | 78.16 | 70.18 | 73.28 | 75.58 | 77.44 | 78.89 | 80.10 |
| SFT → short-prompt GRPO | short | TextVQA | 76.19 | 68.28 | 72.51 | 75.42 | 77.76 | 79.78 | 81.60 |
| SFT on the 4,000 RL questions | short | TextVQA | 72.81 | 65.06 | 71.57 | 75.91 | 79.12 | 81.59 | 83.50 |
| Instruct → short-prompt GRPO, full pool | short | TextVQA | 79.12 | 71.12 | 73.25 | 74.70 | 75.83 | 76.71 | 77.40 |
| Instruct 2B | reasoning | VQAv2 | 52.00 | 45.65 | 55.04 | 62.13 | 67.69 | 72.36 | 76.20 |
| Instruct → GRPO | reasoning | VQAv2 | 75.31 | 65.54 | 68.08 | 70.01 | 71.56 | 72.92 | 74.30 |
| Instruct 2B | reasoning | GQA | 34.02 | 34.02 | 44.91 | 54.87 | 63.85 | 71.67 | 78.40 |
| Instruct → GRPO | reasoning | GQA | 59.04 | 59.04 | 62.82 | 65.84 | 68.31 | 70.47 | 72.60 |
| Instruct 2B | reasoning | TextVQA | 67.07 | 59.67 | 67.43 | 72.34 | 75.86 | 78.61 | 80.80 |
| Instruct → GRPO | reasoning | TextVQA | 77.38 | 69.48 | 72.44 | 74.66 | 76.55 | 78.20 | 79.50 |
<!-- /report:passk -->

Candidate minus reference, points, with 95% intervals that resample images:

<!-- report:passk-deltas -->
| Comparison | Prompt | Dataset | Mean score | pass@1 | pass@4 | pass@32 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Instruct → short-prompt GRPO − Instruct 2B | short | VQAv2 | +3.54 [+2.89, +4.20] | +3.01 [+2.30, +3.74] | -0.61 [-1.41, +0.17] | -3.20 [-4.55, -1.99] |
| Instruct → short-prompt GRPO − Instruct 2B | short | GQA | +2.57 [+1.66, +3.48] | +2.57 [+1.66, +3.48] | -3.04 [-4.10, -2.04] | -7.60 [-9.27, -5.99] |
| Instruct → short-prompt GRPO − Instruct 2B | short | TextVQA | +2.52 [+1.86, +3.21] | +2.41 [+1.62, +3.23] | -0.50 [-1.25, +0.22] | -1.90 [-2.92, -0.90] |
| SFT 2B − Instruct 2B | short | VQAv2 | -0.08 [-0.92, +0.79] | -0.18 [-1.10, +0.73] | +2.68 [+1.80, +3.61] | +2.60 [+1.51, +3.73] |
| SFT 2B − Instruct 2B | short | GQA | +0.49 [-0.64, +1.62] | +0.49 [-0.64, +1.62] | +4.67 [+3.34, +5.97] | +4.60 [+2.78, +6.48] |
| SFT 2B − Instruct 2B | short | TextVQA | -4.94 [-5.91, -3.95] | -4.29 [-5.39, -3.18] | -0.30 [-1.38, +0.80] | +1.70 [+0.50, +2.91] |
| SFT on the 4,000 RL questions − Instruct 2B | short | VQAv2 | -1.53 [-2.17, -0.87] | -1.27 [-1.98, -0.55] | +2.07 [+1.28, +2.87] | +2.50 [+1.49, +3.59] |
| SFT on the 4,000 RL questions − Instruct 2B | short | GQA | -2.05 [-3.04, -1.04] | -2.05 [-3.04, -1.04] | +2.64 [+1.43, +3.82] | +5.20 [+3.47, +6.98] |
| SFT on the 4,000 RL questions − Instruct 2B | short | TextVQA | -2.83 [-3.58, -2.06] | -2.72 [-3.63, -1.82] | -0.17 [-1.05, +0.72] | +1.50 [+0.50, +2.52] |
| SFT on the 4,000 RL questions − Instruct → short-prompt GRPO | short | VQAv2 | -5.07 [-5.92, -4.25] | -4.28 [-5.18, -3.37] | +2.68 [+1.71, +3.68] | +5.70 [+4.23, +7.25] |
| SFT on the 4,000 RL questions − Instruct → short-prompt GRPO | short | GQA | -4.62 [-5.86, -3.35] | -4.62 [-5.86, -3.35] | +5.68 [+4.24, +7.17] | +12.80 [+10.56, +15.11] |
| SFT on the 4,000 RL questions − Instruct → short-prompt GRPO | short | TextVQA | -5.35 [-6.20, -4.54] | -5.12 [-6.01, -4.27] | +0.33 [-0.48, +1.14] | +3.40 [+2.18, +4.70] |
| Instruct → short-prompt GRPO, full pool − Instruct → short-prompt GRPO | short | VQAv2 | +1.77 [+0.83, +2.73] | +1.74 [+0.74, +2.76] | -0.55 [-1.64, +0.52] | -2.60 [-3.81, -1.41] |
| Instruct → short-prompt GRPO, full pool − Instruct → short-prompt GRPO | short | GQA | +4.50 [+2.91, +6.17] | +4.50 [+2.91, +6.17] | +0.43 [-1.27, +2.19] | -3.20 [-5.09, -1.31] |
| Instruct → short-prompt GRPO, full pool − Instruct → short-prompt GRPO | short | TextVQA | +0.96 [+0.14, +1.83] | +0.94 [+0.07, +1.85] | -0.88 [-1.82, +0.08] | -2.70 [-3.97, -1.41] |
| Instruct → short-prompt GRPO, full pool − Instruct 2B | short | VQAv2 | +5.31 [+4.25, +6.37] | +4.74 [+3.61, +5.88] | -1.16 [-2.34, -0.02] | -5.80 [-7.41, -4.30] |
| Instruct → short-prompt GRPO, full pool − Instruct 2B | short | GQA | +7.07 [+5.47, +8.73] | +7.07 [+5.47, +8.73] | -2.61 [-4.24, -0.95] | -10.80 [-13.16, -8.52] |
| Instruct → short-prompt GRPO, full pool − Instruct 2B | short | TextVQA | +3.48 [+2.60, +4.45] | +3.35 [+2.33, +4.43] | -1.38 [-2.50, -0.25] | -4.60 [-6.13, -3.12] |
| SFT → short-prompt GRPO − SFT 2B | short | VQAv2 | +4.72 [+4.04, +5.41] | +4.12 [+3.34, +4.88] | -2.32 [-3.24, -1.51] | -4.40 [-5.71, -3.17] |
| SFT → short-prompt GRPO − SFT 2B | short | GQA | +4.69 [+3.60, +5.80] | +4.69 [+3.60, +5.80] | -5.21 [-6.46, -3.96] | -10.30 [-12.36, -8.30] |
| SFT → short-prompt GRPO − SFT 2B | short | TextVQA | +5.49 [+4.84, +6.16] | +4.79 [+4.13, +5.48] | -0.36 [-0.93, +0.21] | -2.10 [-3.20, -1.10] |
| Instruct → GRPO − Instruct 2B | reasoning | VQAv2 | +23.31 [+21.40, +25.28] | +19.90 [+17.87, +21.97] | +7.88 [+5.95, +9.85] | -1.90 [-3.89, +0.10] |
| Instruct → GRPO − Instruct 2B | reasoning | GQA | +25.03 [+22.44, +27.59] | +25.03 [+22.44, +27.59] | +10.97 [+8.55, +13.35] | -5.80 [-8.47, -3.17] |
| Instruct → GRPO − Instruct 2B | reasoning | TextVQA | +10.31 [+8.85, +11.82] | +9.81 [+8.19, +11.44] | +2.32 [+0.77, +3.82] | -1.30 [-3.11, +0.50] |
<!-- /report:passk-deltas -->

RL sharpens; SFT broadens. Short-prompt GRPO raises pass@1 over Instruct by 2.4–3.0 points on
the three datasets, is level or behind by k = 4, and trails by 1.9–7.6 points at k = 32: sampled
32 times, the base model answers more of these questions than the RL model does. GRPO after SFT
repeats the pattern (pass@1 +4.1 to +4.8, pass@32 −2.1 to −10.3 against SFT). Under the
reasoning prompt, GRPO's large pass@1 gain (+9.8 to +25.0) turns into −1.3 to −5.8 at k = 32,
with the VQAv2 and TextVQA intervals including zero. One epoch of SFT does the opposite: pass@1
stays within 0.5 points of Instruct on VQAv2 and GQA (−4.3 on TextVQA) while pass@32 rises by
1.7–4.6 points. With the training questions matched the contrast is sharpest: SFT on the 4,000
questions short-prompt GRPO drew trails it by 4.3–5.1 points at pass@1 and leads it by 3.4–12.8
at pass@32, although under greedy decoding the two are within a point of each other.
The normalized scorer gives the same crossover
(`python -m scripts.pass_at_k --declared heldout --variant normalized`), so casing does not explain it.
This matches what Yue et al. report for text: RL with a verifiable reward makes the model choose
answers it could already produce more often rather than adding new ones, and here it costs some
coverage. One training seed per arm; the intervals cover which questions and images were asked,
not a fresh draw of samples.

## One model on both families

The transfer matrix shows each single-family model carrying its habit everywhere. Trained on
both families at once, does a model learn when to reason? One 2B model trained with the
reasoning prompt on the union of the two specialists' training questions, each drawn once:
the 4,000 VQAv2 and GQA questions every Part 1 RL arm drew and all 7,378 ChartQA questions
(pool `mix-chartqa-vqa4k`, 2,844 updates, 65% ChartQA; 7.8 GPU-hours on the RTX 4090 with the
dev curve). The comparisons and the end-of-run model were fixed before training
(`[matrices.mixture]`, `runs/mix-20261009/`). It did not learn when to reason. At the end it
answers in 8–11 tokens on every dataset and behaves like the VQA-trained model: within a point
of it on VQAv2, GQA and TextVQA, 3.4 points lower on CLEVR under the reasoning prompt. On
ChartQA it loses the reasoning that paid there: 65.9 under the reasoning prompt, 14.2 points
below the ChartQA-trained model ([−16.9, −11.4]) and 3.8 below Instruct ([−6.8, −0.8]). Against
each specialist, official scorer:

<!-- report:mixture-deltas -->
| Comparison | Prompt | Scorer | VQAv2 | GQA | CLEVR counting | TextVQA (far-OOD) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Instruct → GRPO (VQAv2 + GQA + ChartQA) − Instruct → GRPO | short | official | -0.17 [-0.57, +0.22] | -0.20 [-0.54, +0.14] | +0.10 [-0.03, +0.27] | -0.33 [-0.69, +0.04] |
| Instruct → GRPO (VQAv2 + GQA + ChartQA) − Instruct → GRPO | reasoning | official | +0.46 [-0.06, +0.98] | -0.80 [-1.21, -0.37] | -3.37 [-4.03, -2.73] | -0.26 [-0.70, +0.19] |
| Instruct → GRPO (VQAv2 + GQA + ChartQA) − Instruct → GRPO (ChartQA, 1 epoch) | short | official | +0.15 [-0.26, +0.55] | +0.42 [+0.10, +0.75] | +2.50 [+1.93, +3.10] | +0.04 [-0.35, +0.44] |
| Instruct → GRPO (VQAv2 + GQA + ChartQA) − Instruct → GRPO (ChartQA, 1 epoch) | reasoning | official | +8.47 [+7.51, +9.43] | +7.36 [+6.61, +8.12] | +6.10 [+4.83, +7.37] | +4.14 [+3.25, +5.03] |
<!-- /report:mixture-deltas -->

The dev curve shows how. Reasoning-prompt accuracy (normalized) and mean output tokens on the
dev roles, from the adapters kept every 250 updates:

| Update | VQAv2 | GQA | ChartQA |
| ---: | ---: | ---: | ---: |
| 250 | 60.6 (177 tokens) | 50.8 (166) | 74.0 (229) |
| 1,000 | 65.3 (178) | 61.0 (163) | 75.4 (233) |
| 1,500 | 69.8 (125) | 61.0 (122) | 77.2 (197) |
| 1,750 | 70.0 (88) | 64.4 (93) | 76.6 (206) |
| 2,000 | 74.5 (11) | 66.1 (8) | 65.2 (10) |
| 2,844 | 75.6 (9) | 64.4 (8) | 66.2 (10) |

For 1,750 updates the model was separating the two families: its photo answers shortened to
about 90 tokens while chart answers stayed above 200, and ChartQA dev accuracy rose to 77.2.
Then, within 250 updates, responses collapsed to about 10 tokens on every dataset at once, and
ChartQA lost 11 points. The VQA-trained model made the same collapse within its first 200
updates; mixing in nearly two chart questions per photo question delayed it by about 1,500 updates and
did not prevent it. Untested explanations: under one prompt the policy has no cue but the
image to condition its length on, and the VQA questions reward dropping the reasoning by more
(about 23 points) than ChartQA rewards keeping it (about 10). One seed, one mixing ratio.

## Training knobs (ChartQA)

Each knob changes one training setting against a fixed baseline: the pilot's Instruct 2B →
GRPO run (300 updates, the frozen recipe). Knobs run on ChartQA rather than on VQAv2 and
GQA because there the responses keep their reasoning and vary in length, so loss and length
settings have something to act on; on the general data a reasoning-prompt run collapses to
8-token answers within 200 updates either way. Each knob trains for the same 300 updates and
its final adapter answers ChartQA's full dev role (958 questions) greedily under both
prompts; the test cells are kept for knobs that get promoted to a full run. The expected
direction was written before each run. A knob reads + or − only when its paired 95%
interval excludes zero and the change is larger than the repeat's, a rerun of the baseline
with the same seed whose difference is run-to-run noise. A 300-update budget favors knobs
that speed early learning, so a + is a candidate for a full-length run, not a recipe change.
Official relaxed accuracy, points; `python -m scripts.knobs summarize chartqa`.

<!-- report:knobs-chartqa -->
| Knob | Change | Expected | Reasoning Δ | Short Δ | Reasoning tokens | Train GPU-h | Verdict |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| repeat | the baseline again, same seed | the noise floor: GPU kernels make same-seed runs differ | +0.52 [-1.25, +2.40] | +0.31 [-0.21, +0.84] | 198 → 186 | 1.1 | — |
| loss-grpo | average each response's token losses first (loss_type grpo) | 0 or −, longer responses: a long wrong answer is penalized less per token | +0.21 [-1.57, +1.98] | +0.00 [-0.52, +0.52] | 198 → 211 | 1.1 | 0 |
| length-1024 | 1,024-token response cap (4 completions per micro-batch, same 32 per update) | small +, longer responses: 4.7% of pilot responses hit the 640 cap and score 0 | -0.21 [-2.09, +1.67] | +0.21 [-0.31, +0.73] | 198 → 227 | 1.3 | 0 |
| scale-none | advantages not divided by the group's reward spread (Dr. GRPO) | about 0 | -1.46 [-3.34, +0.42] | +0.42 [+0.00, +0.94] | 198 → 213 | 1.1 | 0 |
| kl-0.04 | KL penalty 0.04 to the starting model (original GRPO) | slight − at 300 updates: the penalty slows learning | -0.94 [-2.92, +1.04] | +0.21 [-0.31, +0.84] | 198 → 210 | 1.3 | 0 |
| loss-dr-grpo | token losses divided by a fixed constant (loss_type dr_grpo) | − at 300 updates: dividing by the 640-token cap instead of ~200 written tokens shrinks each step about 3× | -1.15 [-3.13, +0.84] | +0.21 [-0.31, +0.84] | 198 → 211 | 1.1 | 0 |
| reward-lists | list-aware ChartQA reward: Unicode dashes as hyphens, bracketed lists as unordered sets | small +: about 2% of questions have list or dash answers the current reward marks wrong | -0.52 [-2.51, +1.46] | +0.42 [-0.31, +1.15] | 198 → 197 | 1.1 | 0 |
| canvas-896 | 896-pixel training canvas instead of 608 (784 vision tokens instead of 361; 4 completions per micro-batch) | + if the 608 canvas costs accuracy in the resolution check | +1.04 [-0.94, +3.03] | +0.21 [-0.21, +0.63] | 198 → 197 | 1.6 | 0 |
| image-budget | each image kept in shape within the 0.79 MP evaluation budget, no padding (up to 768 vision tokens; 4 completions per micro-batch) | +, about canvas-896's: same pixels on most charts, no padding tokens | +0.10 [-1.77, +1.98] | +0.73 [+0.21, +1.36] | 198 → 195 | 1.2 | 0 |
<!-- /report:knobs-chartqa -->

No knob moves reasoning-prompt accuracy beyond the repeat at this budget; the largest
changes, −1.5 without reward scaling and +1.0 with the 896-pixel canvas, sit inside
intervals about ±2 points wide. The last three knobs came from the overfitting check
([validation](validation.md#part-2-checks-2026-10-08)). The list-aware reward, which
concerns about 2% of questions, moves dev accuracy by −0.5. Training on the evaluation image
geometry instead of the 608 canvas moves it by +0.1, although that canvas costs Instruct 4.3
points when it is the image evaluated: every trained model is evaluated on the original
images, so at 300 updates the training geometry does not limit what RL learns about reading
charts. The same run's short answers rise 0.7 points [+0.2, +1.4], against the repeat's 0.3,
a small difference outside the reasoning-prompt verdict.

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

Three questions decided the recipe. **Does training on questions with a signal help?**
The filtered pool starts with fewer dead groups (0.33 against 0.58 in the first 100
updates) but ends level on the short prompt and worse on the reasoning prompt: it removes
the reasoning later and less completely (about 20 tokens instead of 8), and the reasoning it
keeps miscounts CLEVR (−18.1). **Do larger batches, refilled groups, or a higher learning
rate help?** Refilling and a 3× learning rate stay within 0.9 points on every dataset
but CLEVR under the reasoning prompt; 16 questions per update at the same number of questions trains 4× fewer
steps and ends worse on the reasoning prompt; a 10× learning rate costs accuracy everywhere
and 11 points of CLEVR counting under the short prompt, the pattern of the 8B SFT at too
high a rate. **Does the sampler matter?** Unmerged-adapter rollouts remove the
sampler–trainer gap ([method](method.md#sampler-and-trained-policy)) without changing any
cell by more than 0.52 points, so the frozen recipe keeps them as the correct on-policy
setup. The rollout prompt is the one change that moves the short prompt: short-prompt GRPO
adds 0.5–0.6 points on VQAv2 and GQA over reasoning-prompt GRPO and keeps the reasoning
behavior of its start; both are part of the final rerun. At 8B, a 0.2 format reward keeps
the reasoning (about 130 tokens) and costs 9.7, 9.6, and 5.6 points under the reasoning
prompt against correctness-only GRPO of the same 300 updates, leaving the short prompt
unchanged.

The July runs, which used a defective VQAv2 reward, are tabulated with the erratum in
[history](history.md).
## Image resolution

The headline cells resize images to at most 0.79 megapixels, which no VQAv2, GQA, or CLEVR
test image exceeds but 826 of the 5,000 TextVQA images do. The same arm evaluated at the
model's own budget (16.8 megapixels) shows what the cap costs, with a paired 95% interval.
For Instruct 2B it costs nothing measurable: the full budget changes short-prompt TextVQA by
−0.1 points and reasoning-prompt TextVQA by −0.7, both intervals including zero. An arm
evaluated on the 608×608 canvas RL trains on appears here too, paired with its
original-image cell. On that canvas the four corrected-campaign 2B arms change VQAv2 and
GQA by at most 1 point, including both GRPO arms that trained on it, while TextVQA loses 5
to 7 points for every arm: text needs pixels, and the evaluation budget does not hide a
gain that RL learned at its training resolution. SLAKE cells evaluated on images upscaled to
the budget belong to the [vision-side options](#vision-side-options-slake).

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
| Instruct 2B | SLAKE (English) | reasoning | official | ≤ 0.79 → 0.79 (upscaled) | +1.23 [-0.85, +3.20] |
| Instruct 2B | SLAKE (English) | reasoning | normalized | ≤ 0.79 → 0.79 (upscaled) | +1.13 [-0.85, +3.11] |
| Instruct 2B | SLAKE (English) | short | official | ≤ 0.79 → 0.79 (upscaled) | +0.47 [-0.75, +1.70] |
| Instruct 2B | SLAKE (English) | short | normalized | ≤ 0.79 → 0.79 (upscaled) | +0.47 [-0.75, +1.70] |
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
| SFT 2B (SLAKE) | SLAKE (English) | reasoning | official | ≤ 0.79 → 0.79 (upscaled) | -0.66 [-1.98, +0.66] |
| SFT 2B (SLAKE) | SLAKE (English) | reasoning | normalized | ≤ 0.79 → 0.79 (upscaled) | -0.75 [-2.07, +0.57] |
| SFT 2B (SLAKE) | SLAKE (English) | short | official | ≤ 0.79 → 0.79 (upscaled) | -0.38 [-1.70, +0.94] |
| SFT 2B (SLAKE) | SLAKE (English) | short | normalized | ≤ 0.79 → 0.79 (upscaled) | -0.38 [-1.70, +0.94] |
| SFT 2B, upscaled images (SLAKE) | SLAKE (English) | reasoning | official | ≤ 0.79 → 0.79 (upscaled) | -0.85 [-2.26, +0.57] |
| SFT 2B, upscaled images (SLAKE) | SLAKE (English) | reasoning | normalized | ≤ 0.79 → 0.79 (upscaled) | -0.85 [-2.26, +0.57] |
| SFT 2B, upscaled images (SLAKE) | SLAKE (English) | short | official | ≤ 0.79 → 0.79 (upscaled) | +1.13 [-0.09, +2.36] |
| SFT 2B, upscaled images (SLAKE) | SLAKE (English) | short | normalized | ≤ 0.79 → 0.79 (upscaled) | +1.13 [-0.09, +2.36] |
| Instruct → short-prompt GRPO (SLAKE, 1 epoch) | SLAKE (English) | reasoning | official | ≤ 0.79 → 0.79 (upscaled) | +0.94 [-1.13, +3.02] |
| Instruct → short-prompt GRPO (SLAKE, 1 epoch) | SLAKE (English) | reasoning | normalized | ≤ 0.79 → 0.79 (upscaled) | +0.85 [-1.23, +2.92] |
| Instruct → short-prompt GRPO (SLAKE, 1 epoch) | SLAKE (English) | short | official | ≤ 0.79 → 0.79 (upscaled) | +1.13 [-0.09, +2.36] |
| Instruct → short-prompt GRPO (SLAKE, 1 epoch) | SLAKE (English) | short | normalized | ≤ 0.79 → 0.79 (upscaled) | +1.13 [-0.09, +2.36] |
| Instruct → short-prompt GRPO, upscaled images (SLAKE, 1 epoch) | SLAKE (English) | reasoning | official | ≤ 0.79 → 0.79 (upscaled) | +2.64 [+0.57, +4.71] |
| Instruct → short-prompt GRPO, upscaled images (SLAKE, 1 epoch) | SLAKE (English) | reasoning | normalized | ≤ 0.79 → 0.79 (upscaled) | +2.54 [+0.47, +4.62] |
| Instruct → short-prompt GRPO, upscaled images (SLAKE, 1 epoch) | SLAKE (English) | short | official | ≤ 0.79 → 0.79 (upscaled) | -0.09 [-1.23, +1.04] |
| Instruct → short-prompt GRPO, upscaled images (SLAKE, 1 epoch) | SLAKE (English) | short | normalized | ≤ 0.79 → 0.79 (upscaled) | +0.00 [-1.13, +1.13] |
<!-- /report:resolution -->

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
| SFT → GRPO − Instruct 2B | short | 231 / 136 | 869 / 447 | 63 / 2 | 170 / 185 |
| SFT → GRPO − Instruct 2B | reasoning | 893 / 158 | 1,889 / 682 | 253 / 260 | 680 / 203 |
| SFT → GRPO − SFT 2B | short | 43 / 52 | 209 / 150 | 11 / 2 | 42 / 36 |
| SFT → GRPO − SFT 2B | reasoning | 665 / 107 | 1,055 / 484 | 113 / 269 | 669 / 148 |
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
| SFT → short-prompt GRPO − SFT → GRPO | short | 84 / 75 | 228 / 268 | 1 / 7 | 83 / 54 |
| SFT → short-prompt GRPO − SFT → GRPO | reasoning | 119 / 667 | 455 / 994 | 282 / 220 | 149 / 577 |
| Instruct → gold-conditioned OPD − Instruct 2B | short | 46 / 40 | 155 / 112 | 8 / 0 | 62 / 62 |
| Instruct → gold-conditioned OPD − Instruct 2B | reasoning | 596 / 274 | 1,059 / 961 | 146 / 165 | 452 / 285 |
| SFT 8B − Instruct 8B | short | 220 / 92 | 669 / 392 | 2 / 327 | 182 / 120 |
| SFT 8B − Instruct 8B | reasoning | 454 / 430 | 1,068 / 1,456 | 68 / 76 | 405 / 351 |
| Instruct 8B → GRPO − Instruct 8B | short | 47 / 46 | 178 / 95 | 1 / 2 | 74 / 36 |
| Instruct 8B → GRPO − Instruct 8B | reasoning | 782 / 91 | 1,889 / 574 | 83 / 46 | 603 / 128 |
| SFT 8B → GRPO − Instruct 8B | short | 222 / 106 | 674 / 392 | 2 / 82 | 185 / 124 |
| SFT 8B → GRPO − Instruct 8B | reasoning | 802 / 103 | 1,934 / 543 | 82 / 41 | 664 / 163 |
| SFT 8B → GRPO − SFT 8B | short | 24 / 36 | 120 / 115 | 245 / 0 | 27 / 28 |
| SFT 8B → GRPO − SFT 8B | reasoning | 780 / 105 | 2,289 / 510 | 82 / 33 | 576 / 129 |
| Instruct 8B → short-prompt GRPO − Instruct 8B | short | 83 / 41 | 241 / 146 | 2 / 28 | 70 / 33 |
| Instruct 8B → short-prompt GRPO − Instruct 8B | reasoning | 191 / 114 | 466 / 300 | 36 / 41 | 134 / 124 |
| Instruct 8B → short-prompt GRPO − SFT 8B | short | 88 / 174 | 398 / 580 | 299 / 0 | 117 / 142 |
| Instruct 8B → short-prompt GRPO − SFT 8B | reasoning | 456 / 403 | 1,523 / 969 | 75 / 72 | 349 / 393 |
| Instruct 8B → short-prompt GRPO − Instruct 8B → GRPO | short | 59 / 18 | 151 / 139 | 1 / 26 | 21 / 22 |
| Instruct 8B → short-prompt GRPO − Instruct 8B → GRPO | reasoning | 90 / 704 | 609 / 1,758 | 45 / 87 | 116 / 581 |
| SFT 8B → short-prompt GRPO − Instruct 8B | short | 223 / 106 | 672 / 400 | 2 / 528 | 186 / 127 |
| SFT 8B → short-prompt GRPO − Instruct 8B | reasoning | 455 / 352 | 1,154 / 1,142 | 65 / 81 | 384 / 321 |
| SFT 8B → short-prompt GRPO − SFT 8B | short | 45 / 56 | 162 / 167 | 0 / 201 | 43 / 46 |
| SFT 8B → short-prompt GRPO − SFT 8B | reasoning | 296 / 217 | 876 / 476 | 33 / 41 | 192 / 183 |
| SFT 8B → short-prompt GRPO − SFT 8B → GRPO | short | 47 / 46 | 135 / 145 | 0 / 446 | 25 / 27 |
| SFT 8B → short-prompt GRPO − SFT 8B → GRPO | reasoning | 96 / 692 | 535 / 1,914 | 34 / 91 | 127 / 565 |
<!-- /report:flips -->

Under the reasoning prompt, Instruct → GRPO fixes 894 VQAv2 questions for every 148 it
breaks at 2B, and 1,791 against 740 on GQA; the 8B run fixes 782 against 91 on VQAv2.
Under the short prompt RL changes few verdicts in either direction (69 / 66 on VQAv2 at 2B,
47 / 46 at 8B). Against SFT under the short prompt, short-prompt GRPO from Instruct gets
fewer questions right than it loses on VQAv2 (142 / 203) and GQA (469 / 720), the
question-level view of SFT's higher in-domain ceiling.

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
| SFT → GRPO | VQAv2 | 79.4 | 0.1 | 0.0 | 0.0 | 10.6 | 9.9 |
| SFT → GRPO | GQA | 61.5 | 0.0 | 0.1 | 0.0 | 3.8 | 34.6 |
| SFT → GRPO | CLEVR counting | 86.9 | 0.0 | 0.0 | 0.0 | 0.0 | 13.1 |
| SFT → GRPO | TextVQA | 79.8 | 0.0 | 0.4 | 0.0 | 10.1 | 9.7 |
| Instruct → short-prompt GRPO | VQAv2 | 62.1 | 8.0 | 1.5 | 0.8 | 15.3 | 12.4 |
| Instruct → short-prompt GRPO | GQA | 42.9 | 11.1 | 0.9 | 0.7 | 6.3 | 38.2 |
| Instruct → short-prompt GRPO | CLEVR counting | 86.5 | 0.0 | 0.9 | 0.0 | 0.0 | 12.6 |
| Instruct → short-prompt GRPO | TextVQA | 73.1 | 0.0 | 0.7 | 2.9 | 11.4 | 11.9 |
| SFT → short-prompt GRPO | VQAv2 | 68.3 | 0.3 | 1.0 | 18.6 | 6.0 | 5.9 |
| SFT → short-prompt GRPO | GQA | 56.9 | 0.4 | 0.3 | 30.5 | 1.6 | 10.3 |
| SFT → short-prompt GRPO | CLEVR counting | 89.0 | 0.0 | 0.7 | 0.0 | 0.0 | 10.4 |
| SFT → short-prompt GRPO | TextVQA | 71.2 | 0.0 | 0.7 | 14.3 | 6.8 | 6.9 |
| Instruct → gold-conditioned OPD | VQAv2 | 70.8 | 0.4 | 4.5 | 2.9 | 10.1 | 11.2 |
| Instruct → gold-conditioned OPD | GQA | 52.5 | 0.2 | 3.0 | 4.1 | 4.7 | 35.5 |
| Instruct → gold-conditioned OPD | CLEVR counting | 86.5 | 0.0 | 1.4 | 0.0 | 0.0 | 12.1 |
| Instruct → gold-conditioned OPD | TextVQA | 73.6 | 0.0 | 2.2 | 3.3 | 9.8 | 11.1 |
| Instruct 8B | VQAv2 | 64.0 | 5.1 | 0.9 | 0.3 | 17.8 | 11.9 |
| Instruct 8B | GQA | 46.8 | 5.4 | 0.4 | 0.7 | 8.2 | 38.5 |
| Instruct 8B | CLEVR counting | 97.0 | 0.0 | 0.7 | 0.0 | 0.0 | 2.4 |
| Instruct 8B | TextVQA | 74.8 | 0.0 | 0.5 | 0.6 | 14.0 | 10.1 |
| SFT 8B | VQAv2 | 68.7 | 0.9 | 3.5 | 8.1 | 10.9 | 7.9 |
| SFT 8B | GQA | 48.7 | 0.5 | 5.0 | 11.5 | 5.0 | 29.3 |
| SFT 8B | CLEVR counting | 96.7 | 0.0 | 0.3 | 1.1 | 0.0 | 1.9 |
| SFT 8B | TextVQA | 75.8 | 0.0 | 0.7 | 8.1 | 8.7 | 6.7 |
| Instruct 8B → GRPO | VQAv2 | 82.6 | 0.3 | 0.0 | 0.0 | 9.1 | 8.0 |
| Instruct 8B → GRPO | GQA | 62.7 | 0.1 | 0.0 | 0.0 | 4.4 | 32.9 |
| Instruct 8B → GRPO | CLEVR counting | 98.2 | 0.0 | 0.0 | 0.0 | 0.0 | 1.8 |
| Instruct 8B → GRPO | TextVQA | 84.3 | 0.0 | 0.0 | 0.0 | 8.6 | 7.1 |
| SFT 8B → GRPO | VQAv2 | 82.8 | 0.3 | 0.0 | 0.0 | 9.0 | 7.9 |
| SFT 8B → GRPO | GQA | 63.2 | 0.1 | 0.0 | 0.0 | 4.1 | 32.5 |
| SFT 8B → GRPO | CLEVR counting | 98.3 | 0.0 | 0.0 | 0.0 | 0.0 | 1.7 |
| SFT 8B → GRPO | TextVQA | 84.8 | 0.0 | 0.0 | 0.0 | 8.0 | 7.2 |
| Instruct 8B → short-prompt GRPO | VQAv2 | 67.3 | 3.3 | 0.8 | 0.5 | 16.4 | 11.6 |
| Instruct 8B → short-prompt GRPO | GQA | 50.0 | 3.5 | 0.3 | 0.5 | 7.6 | 38.1 |
| Instruct 8B → short-prompt GRPO | CLEVR counting | 96.8 | 0.0 | 0.8 | 0.0 | 0.0 | 2.4 |
| Instruct 8B → short-prompt GRPO | TextVQA | 75.0 | 0.0 | 0.5 | 0.5 | 13.7 | 10.3 |
| SFT 8B → short-prompt GRPO | VQAv2 | 70.5 | 0.6 | 2.3 | 6.9 | 10.7 | 8.9 |
| SFT 8B → short-prompt GRPO | GQA | 51.9 | 0.4 | 3.0 | 9.7 | 4.9 | 30.1 |
| SFT 8B → short-prompt GRPO | CLEVR counting | 96.4 | 0.0 | 0.5 | 0.8 | 0.0 | 2.3 |
| SFT 8B → short-prompt GRPO | TextVQA | 76.0 | 0.0 | 0.5 | 7.0 | 9.3 | 7.1 |
<!-- /report:errors -->

SFT's reasoning-prompt errors are mostly missing tags (34.6% of GQA questions). From
Instruct 2B, GRPO removes almost every error of form: casing misses fall from 9.6% to 0.7%
of VQAv2 questions and from 15.8% to 0.1% of GQA, truncation and missing tags go to zero,
and partial matches halve (VQAv2 18.2 → 10.2%). Wrong content barely moves (VQAv2 11.9 →
10.1%, GQA 37.0 → 35.4%), so on GQA most of the official gain is answer form, which is why
its normalized gain (+8.4) is a third of the official one.

Answer forms that could game an answer checker, as a share of reasoning-prompt responses:
hedged answers ("red or blue", "red/blue") · several `<answer>` tags · answers over six
words. The checks read the tagged answer the RL reward reads, so an untagged response is
never flagged. None of the three rises after GRPO from either start at either size; from
Instruct, hedged and long answers nearly disappear (VQAv2 2.1 → 0.1% and 5.7 → 0.2% at 2B),
so the reward was not gained by answering several ways at once.

<!-- report:answer-form -->
| Arm | VQAv2 | GQA | CLEVR counting | TextVQA (far-OOD) |
| --- | ---: | ---: | ---: | ---: |
| Instruct 2B | 2.1 · 0.0 · 5.7 | 0.8 · 0.0 · 3.9 | 0.0 · 0.0 · 0.0 | 0.8 · 0.0 · 2.4 |
| SFT 2B | 0.2 · 0.0 · 0.5 | 0.0 · 0.0 · 0.2 | 0.0 · 0.0 · 0.0 | 0.5 · 0.0 · 1.3 |
| Instruct → GRPO | 0.1 · 0.0 · 0.2 | 0.0 · 0.0 · 0.0 | 0.0 · 0.0 · 0.0 | 0.5 · 0.0 · 0.8 |
| SFT → GRPO | 0.0 · 0.0 · 0.1 | 0.0 · 0.0 · 0.0 | 0.0 · 0.0 · 0.0 | 0.5 · 0.0 · 1.1 |
| Instruct → short-prompt GRPO | 0.9 · 0.0 · 3.5 | 0.3 · 0.0 · 2.1 | 0.0 · 0.0 · 0.0 | 0.8 · 0.0 · 2.0 |
| SFT → short-prompt GRPO | 0.4 · 0.0 · 0.6 | 0.0 · 0.0 · 0.3 | 0.0 · 0.0 · 0.0 | 0.4 · 0.0 · 1.2 |
| Instruct → gold-conditioned OPD | 0.2 · 0.0 · 0.7 | 0.0 · 0.0 · 0.2 | 0.0 · 0.0 · 0.0 | 0.5 · 0.0 · 1.0 |
| Instruct 8B | 2.6 · 0.0 · 5.4 | 1.5 · 0.0 · 4.6 | 0.0 · 0.0 · 0.0 | 0.9 · 0.0 · 2.6 |
| SFT 8B | 0.6 · 0.0 · 1.4 | 0.2 · 0.0 · 0.8 | 0.0 · 0.0 · 0.0 | 0.5 · 0.0 · 1.3 |
| Instruct 8B → GRPO | 0.1 · 0.0 · 0.1 | 0.0 · 0.0 · 0.1 | 0.0 · 0.0 · 0.0 | 0.6 · 0.0 · 1.0 |
| SFT 8B → GRPO | 0.1 · 0.0 · 0.1 | 0.0 · 0.0 · 0.0 | 0.0 · 0.0 · 0.0 | 0.5 · 0.0 · 0.7 |
| Instruct 8B → short-prompt GRPO | 2.1 · 0.0 · 4.3 | 1.1 · 0.0 · 3.7 | 0.0 · 0.0 · 0.0 | 0.9 · 0.0 · 2.5 |
| SFT 8B → short-prompt GRPO | 0.6 · 0.0 · 1.1 | 0.2 · 0.0 · 0.7 | 0.0 · 0.0 · 0.0 | 0.6 · 0.0 · 1.3 |
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
| Instruct 8B | short | black | — | — | — | 83.8 → 6.3 (94%) |
| Instruct 8B | short | swapped | — | — | — | — |
<!-- /report:interventions -->

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
([recipe experiments](#recipe-experiments)).

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

## Medical pilots in detail

Each medical pilot on its own test role, with the other dataset's models as an unseen-data
test (official, normalized in parentheses; tokens under each prompt). SLAKE, 1,061 test
questions:

<!-- report:slake-pilot -->
| Arm | Prompt | Accuracy | closed answers | open answers | Output tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| Instruct 2B | short | 49.76 (49.76) | 75.96 | 32.87 | 3 |
| Instruct 2B | reasoning | 40.34 (40.53) | 57.69 | 29.15 | 138 |
| SFT 2B (SLAKE) | short | 74.84 (74.84) | 82.93 | 69.61 | 3 |
| SFT 2B (SLAKE) | reasoning | 69.18 (69.27) | 81.73 | 61.09 | 25 |
| Instruct → short-prompt GRPO (SLAKE) | short | 60.41 (60.41) | 78.12 | 48.99 | 3 |
| Instruct → short-prompt GRPO (SLAKE) | reasoning | 46.56 (46.65) | 62.50 | 36.28 | 128 |
| Instruct → short-prompt GRPO (SLAKE, 1 epoch) | short | 65.60 (65.60) | 80.05 | 56.28 | 2 |
| Instruct → short-prompt GRPO (SLAKE, 1 epoch) | reasoning | 52.78 (52.87) | 67.31 | 43.41 | 127 |
| SFT → short-prompt GRPO (SLAKE, 1 epoch) | short | 76.15 (76.34) | 85.34 | 70.23 | 3 |
| SFT → short-prompt GRPO (SLAKE, 1 epoch) | reasoning | 70.88 (71.07) | 83.89 | 62.48 | 23 |
| Instruct → GRPO (SLAKE) | short | 52.12 (52.12) | 77.64 | 35.66 | 3 |
| Instruct → GRPO (SLAKE) | reasoning | 55.70 (56.17) | 78.37 | 41.09 | 12 |
| SFT 2B (OmniMedVQA) | short | 42.88 (42.88) | 63.70 | 29.46 | 3 |
| SFT 2B (OmniMedVQA) | reasoning | 38.93 (39.11) | 59.86 | 25.43 | 166 |
| Instruct → short-prompt GRPO (OmniMedVQA) | short | 49.01 (49.01) | 75.24 | 32.09 | 3 |
| Instruct → short-prompt GRPO (OmniMedVQA) | reasoning | 40.72 (40.72) | 57.45 | 29.92 | 139 |
<!-- /report:slake-pilot -->

<!-- report:slake-pilot-deltas -->
| Comparison | Prompt | Official |
| --- | --- | ---: |
| SFT 2B (SLAKE) − Instruct 2B | short | +25.07 [+21.87, +28.18] |
| SFT 2B (SLAKE) − Instruct 2B | reasoning | +28.84 [+25.54, +32.14] |
| Instruct → short-prompt GRPO (SLAKE) − Instruct 2B | short | +10.65 [+8.39, +12.82] |
| Instruct → short-prompt GRPO (SLAKE) − Instruct 2B | reasoning | +6.22 [+4.05, +8.48] |
| Instruct → short-prompt GRPO (SLAKE, 1 epoch) − Instruct 2B | short | +15.83 [+13.10, +18.47] |
| Instruct → short-prompt GRPO (SLAKE, 1 epoch) − Instruct 2B | reasoning | +12.44 [+9.71, +15.27] |
| SFT → short-prompt GRPO (SLAKE, 1 epoch) − Instruct 2B | short | +26.39 [+23.37, +29.50] |
| SFT → short-prompt GRPO (SLAKE, 1 epoch) − Instruct 2B | reasoning | +30.54 [+27.33, +33.65] |
| Instruct → GRPO (SLAKE) − Instruct 2B | short | +2.36 [+1.13, +3.58] |
| Instruct → GRPO (SLAKE) − Instruct 2B | reasoning | +15.36 [+12.63, +18.10] |
| SFT 2B (OmniMedVQA) − Instruct 2B | short | -6.88 [-9.33, -4.43] |
| SFT 2B (OmniMedVQA) − Instruct 2B | reasoning | -1.41 [-4.05, +1.23] |
| Instruct → short-prompt GRPO (OmniMedVQA) − Instruct 2B | short | -0.75 [-1.70, +0.19] |
| Instruct → short-prompt GRPO (OmniMedVQA) − Instruct 2B | reasoning | +0.38 [-1.60, +2.36] |
| Instruct → short-prompt GRPO (SLAKE) − SFT 2B (SLAKE) | short | -14.42 [-17.06, -11.69] |
| Instruct → short-prompt GRPO (SLAKE) − SFT 2B (SLAKE) | reasoning | -22.62 [-25.64, -19.42] |
| Instruct → short-prompt GRPO (SLAKE, 1 epoch) − SFT 2B (SLAKE) | short | -9.24 [-11.59, -6.88] |
| Instruct → short-prompt GRPO (SLAKE, 1 epoch) − SFT 2B (SLAKE) | reasoning | -16.40 [-19.32, -13.38] |
| SFT → short-prompt GRPO (SLAKE, 1 epoch) − SFT 2B (SLAKE) | short | +1.32 [+0.00, +2.73] |
| SFT → short-prompt GRPO (SLAKE, 1 epoch) − SFT 2B (SLAKE) | reasoning | +1.70 [+0.00, +3.39] |
| Instruct → short-prompt GRPO (OmniMedVQA) − SFT 2B (OmniMedVQA) | short | +6.13 [+3.77, +8.48] |
| Instruct → short-prompt GRPO (OmniMedVQA) − SFT 2B (OmniMedVQA) | reasoning | +1.79 [-0.85, +4.43] |
<!-- /report:slake-pilot-deltas -->

OmniMedVQA, 8,914 test questions. SFT's 99.7% shows the image-level re-split leaves test
images too close to training ones; the official scorer accepts only the option letter, which
Instruct often leaves out (47.2 official, 69.5 normalized):

<!-- report:omnimedvqa-pilot -->
| Arm | Prompt | Accuracy | Anatomy Identification answers | Disease Diagnosis answers | Lesion Grading answers | Modality Recognition answers | Other Biological Attributes answers | Output tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Instruct 2B | short | 47.22 (69.45) | 58.24 | 37.15 | 61.16 | 79.10 | 45.16 | 3 |
| Instruct 2B | reasoning | 64.76 (64.83) | 73.89 | 55.76 | 75.45 | 99.01 | 48.97 | 139 |
| SFT 2B (OmniMedVQA) | short | 99.66 (99.66) | 99.76 | 99.55 | 99.55 | 100.00 | 100.00 | 2 |
| SFT 2B (OmniMedVQA) | reasoning | 91.08 (92.60) | 91.46 | 89.98 | 92.86 | 97.48 | 85.34 | 90 |
| Instruct → short-prompt GRPO (OmniMedVQA) | short | 78.27 (79.52) | 79.96 | 73.96 | 83.48 | 99.37 | 68.33 | 2 |
| Instruct → short-prompt GRPO (OmniMedVQA) | reasoning | 73.92 (73.92) | 77.50 | 68.76 | 81.70 | 98.92 | 54.25 | 148 |
| Instruct → GRPO (OmniMedVQA) | short | 44.70 (70.18) | 54.81 | 34.55 | 60.71 | 77.75 | 43.40 | 4 |
| Instruct → GRPO (OmniMedVQA) | reasoning | 73.15 (73.15) | 78.88 | 67.04 | 71.43 | 99.28 | 61.29 | 259 |
| SFT 2B (SLAKE) | short | 69.14 (69.16) | 82.01 | 59.05 | 74.11 | 99.28 | 70.09 | 2 |
| SFT 2B (SLAKE) | reasoning | 53.90 (67.62) | 61.25 | 42.64 | 81.70 | 91.35 | 62.17 | 37 |
| Instruct → short-prompt GRPO (SLAKE) | short | 60.18 (69.59) | 73.83 | 50.67 | 65.18 | 90.27 | 47.80 | 3 |
| Instruct → short-prompt GRPO (SLAKE) | reasoning | 64.77 (64.79) | 73.77 | 55.62 | 76.34 | 98.92 | 51.91 | 88 |
| Instruct → short-prompt GRPO (SLAKE, 1 epoch) | short | 67.99 (69.49) | 81.11 | 58.80 | 68.75 | 98.38 | 55.13 | 2 |
| Instruct → short-prompt GRPO (SLAKE, 1 epoch) | reasoning | 66.92 (66.92) | 76.90 | 57.86 | 79.91 | 98.47 | 55.13 | 156 |
| SFT → short-prompt GRPO (SLAKE, 1 epoch) | short | 69.56 (69.68) | 82.01 | 59.69 | 75.89 | 99.01 | 70.38 | 2 |
| SFT → short-prompt GRPO (SLAKE, 1 epoch) | reasoning | 54.67 (58.83) | 69.31 | 45.74 | 81.25 | 72.70 | 53.08 | 49 |
<!-- /report:omnimedvqa-pilot -->

<!-- report:omnimedvqa-pilot-deltas -->
| Comparison | Prompt | Official |
| --- | --- | ---: |
| SFT 2B (OmniMedVQA) − Instruct 2B | short | +52.45 [+51.40, +53.48] |
| SFT 2B (OmniMedVQA) − Instruct 2B | reasoning | +26.32 [+25.22, +27.42] |
| Instruct → short-prompt GRPO (OmniMedVQA) − Instruct 2B | short | +31.05 [+30.07, +32.04] |
| Instruct → short-prompt GRPO (OmniMedVQA) − Instruct 2B | reasoning | +9.15 [+8.31, +10.02] |
| Instruct → GRPO (OmniMedVQA) − Instruct 2B | short | -2.51 [-2.93, -2.11] |
| Instruct → GRPO (OmniMedVQA) − Instruct 2B | reasoning | +8.39 [+7.42, +9.39] |
| SFT 2B (SLAKE) − Instruct 2B | short | +21.92 [+20.90, +22.95] |
| SFT 2B (SLAKE) − Instruct 2B | reasoning | -10.86 [-11.98, -9.75] |
| Instruct → short-prompt GRPO (SLAKE) − Instruct 2B | short | +12.96 [+12.26, +13.69] |
| Instruct → short-prompt GRPO (SLAKE) − Instruct 2B | reasoning | +0.01 [-0.65, +0.68] |
| Instruct → short-prompt GRPO (SLAKE, 1 epoch) − Instruct 2B | short | +20.78 [+19.90, +21.67] |
| Instruct → short-prompt GRPO (SLAKE, 1 epoch) − Instruct 2B | reasoning | +2.15 [+1.30, +3.03] |
| SFT → short-prompt GRPO (SLAKE, 1 epoch) − Instruct 2B | short | +22.35 [+21.34, +23.36] |
| SFT → short-prompt GRPO (SLAKE, 1 epoch) − Instruct 2B | reasoning | -10.10 [-11.20, -9.00] |
| Instruct → short-prompt GRPO (OmniMedVQA) − SFT 2B (OmniMedVQA) | short | -21.39 [-22.27, -20.54] |
| Instruct → short-prompt GRPO (OmniMedVQA) − SFT 2B (OmniMedVQA) | reasoning | -17.16 [-18.24, -16.11] |
| Instruct → short-prompt GRPO (SLAKE) − SFT 2B (SLAKE) | short | -8.96 [-9.83, -8.09] |
| Instruct → short-prompt GRPO (SLAKE) − SFT 2B (SLAKE) | reasoning | +10.87 [+9.78, +11.97] |
| Instruct → short-prompt GRPO (SLAKE, 1 epoch) − SFT 2B (SLAKE) | short | -1.14 [-1.83, -0.45] |
| Instruct → short-prompt GRPO (SLAKE, 1 epoch) − SFT 2B (SLAKE) | reasoning | +13.01 [+11.86, +14.16] |
| SFT → short-prompt GRPO (SLAKE, 1 epoch) − SFT 2B (SLAKE) | short | +0.43 [+0.03, +0.83] |
| SFT → short-prompt GRPO (SLAKE, 1 epoch) − SFT 2B (SLAKE) | reasoning | +0.76 [-0.18, +1.72] |
<!-- /report:omnimedvqa-pilot-deltas -->

PathVQA, 6,719 test questions, trained on by no model (it failed the screen's image gate):

<!-- report:pathvqa-pilot -->
| Arm | Prompt | Accuracy | open answers | yes/no answers | Output tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| Instruct 2B | short | 33.52 (33.64) | 3.90 | 63.09 | 3 |
| Instruct 2B | reasoning | 30.99 (31.09) | 1.22 | 60.71 | 292 |
| SFT 2B (SLAKE) | short | 31.69 (31.81) | 4.41 | 58.92 | 3 |
| SFT 2B (SLAKE) | reasoning | 25.69 (25.79) | 1.01 | 50.33 | 117 |
| Instruct → short-prompt GRPO (SLAKE) | short | 32.95 (33.09) | 4.11 | 61.75 | 3 |
| Instruct → short-prompt GRPO (SLAKE) | reasoning | 30.64 (30.72) | 1.43 | 59.82 | 301 |
| Instruct → short-prompt GRPO (SLAKE, 1 epoch) | short | 32.52 (32.67) | 4.05 | 60.95 | 3 |
| Instruct → short-prompt GRPO (SLAKE, 1 epoch) | reasoning | 30.15 (30.23) | 1.34 | 58.92 | 299 |
| SFT → short-prompt GRPO (SLAKE, 1 epoch) | short | 32.58 (32.68) | 4.68 | 60.44 | 3 |
| SFT → short-prompt GRPO (SLAKE, 1 epoch) | reasoning | 24.01 (24.07) | 1.16 | 46.82 | 123 |
| SFT 2B (OmniMedVQA) | short | 30.09 (30.24) | 3.13 | 57.02 | 3 |
| SFT 2B (OmniMedVQA) | reasoning | 27.92 (28.00) | 1.13 | 54.67 | 210 |
| Instruct → short-prompt GRPO (OmniMedVQA) | short | 33.68 (33.81) | 3.93 | 63.38 | 3 |
| Instruct → short-prompt GRPO (OmniMedVQA) | reasoning | 31.00 (31.06) | 1.79 | 60.17 | 273 |
<!-- /report:pathvqa-pilot -->

<!-- report:pathvqa-pilot-deltas -->
| Comparison | Prompt | Official |
| --- | --- | ---: |
| SFT 2B (SLAKE) − Instruct 2B | short | -1.83 [-2.83, -0.85] |
| SFT 2B (SLAKE) − Instruct 2B | reasoning | -5.31 [-6.38, -4.23] |
| Instruct → short-prompt GRPO (SLAKE) − Instruct 2B | short | -0.57 [-1.06, -0.09] |
| Instruct → short-prompt GRPO (SLAKE) − Instruct 2B | reasoning | -0.34 [-1.12, +0.45] |
| Instruct → short-prompt GRPO (SLAKE, 1 epoch) − Instruct 2B | short | -1.00 [-1.79, -0.21] |
| Instruct → short-prompt GRPO (SLAKE, 1 epoch) − Instruct 2B | reasoning | -0.83 [-1.68, -0.01] |
| SFT → short-prompt GRPO (SLAKE, 1 epoch) − Instruct 2B | short | -0.94 [-1.85, -0.04] |
| SFT → short-prompt GRPO (SLAKE, 1 epoch) − Instruct 2B | reasoning | -6.99 [-8.02, -5.94] |
| SFT 2B (OmniMedVQA) − Instruct 2B | short | -3.42 [-4.23, -2.65] |
| SFT 2B (OmniMedVQA) − Instruct 2B | reasoning | -3.07 [-3.99, -2.14] |
| Instruct → short-prompt GRPO (OmniMedVQA) − Instruct 2B | short | +0.16 [-0.28, +0.61] |
| Instruct → short-prompt GRPO (OmniMedVQA) − Instruct 2B | reasoning | +0.01 [-0.77, +0.79] |
| Instruct → short-prompt GRPO (SLAKE) − SFT 2B (SLAKE) | short | +1.26 [+0.39, +2.16] |
| Instruct → short-prompt GRPO (SLAKE) − SFT 2B (SLAKE) | reasoning | +4.96 [+3.91, +6.04] |
| Instruct → short-prompt GRPO (SLAKE, 1 epoch) − SFT 2B (SLAKE) | short | +0.83 [+0.18, +1.50] |
| Instruct → short-prompt GRPO (SLAKE, 1 epoch) − SFT 2B (SLAKE) | reasoning | +4.47 [+3.50, +5.48] |
| SFT → short-prompt GRPO (SLAKE, 1 epoch) − SFT 2B (SLAKE) | short | +0.89 [+0.43, +1.34] |
| SFT → short-prompt GRPO (SLAKE, 1 epoch) − SFT 2B (SLAKE) | reasoning | -1.68 [-2.51, -0.88] |
| Instruct → short-prompt GRPO (OmniMedVQA) − SFT 2B (OmniMedVQA) | short | +3.59 [+2.78, +4.40] |
| Instruct → short-prompt GRPO (OmniMedVQA) − SFT 2B (OmniMedVQA) | reasoning | +3.09 [+2.17, +4.03] |
<!-- /report:pathvqa-pilot-deltas -->

Under the reasoning prompt, each model minus Instruct:

<!-- report:medical-transfer-reasoning -->
| Arm | Trained on | SLAKE (English) | OmniMedVQA (open access) | PathVQA |
| --- | --- | ---: | ---: | ---: |
| SFT 2B (SLAKE) | SLAKE | +28.75 [+25.45, +32.05] | +2.79 [+1.82, +3.80] | -5.31 [-6.38, -4.23] |
| Instruct → short-prompt GRPO (SLAKE, 1 epoch) | SLAKE | +12.35 [+9.71, +15.08] | +2.09 [+1.23, +2.97] | -0.86 [-1.71, -0.03] |
| SFT → short-prompt GRPO (SLAKE, 1 epoch) | SLAKE (SFT, then RL) | +30.54 [+27.33, +33.65] | -6.00 [-7.07, -4.91] | -7.03 [-8.07, -5.97] |
| SFT 2B (OmniMedVQA) | OmniMedVQA | -1.41 [-4.05, +1.23] | +27.77 [+26.67, +28.86] | -3.10 [-4.02, -2.17] |
| Instruct → short-prompt GRPO (OmniMedVQA) | OmniMedVQA | +0.19 [-1.70, +2.17] | +9.09 [+8.25, +9.95] | -0.03 [-0.82, +0.76] |
| SFT 2B | VQAv2 + GQA | +1.60 [-0.94, +4.15] | +5.64 [+4.66, +6.63] | — |
| Instruct → GRPO (ChartQA, 1 epoch) | ChartQA | +7.45 [+5.00, +9.90] | +5.80 [+4.79, +6.81] | — |
<!-- /report:medical-transfer-reasoning -->

Reasoning does not pay on these datasets. Reasoning-prompt GRPO on SLAKE deletes the
reasoning as it did on VQAv2 and GQA (138 → 12 tokens) and gains 15.4 points under the
reasoning prompt, yet scores 4.7 below short-prompt GRPO's short answers. On OmniMedVQA it
keeps 259 tokens and gains 8.4 under the reasoning prompt, still 5 points below short-prompt
GRPO. The general models help only under the reasoning prompt: ChartQA-trained GRPO adds 7.2
points on SLAKE and 5.9 on OmniMedVQA there, and nothing to short answers. The medical SFT
models also lose most under the reasoning prompt off their data (SLAKE SFT: −10.9 on
OmniMedVQA, −5.3 on PathVQA).

OmniMedVQA with held-out source collections, 1,156 test questions from 17 collections no model
trained on. Short-prompt GRPO from Instruct learned to answer with the option text instead of
the letter (7 tokens; the normalized reward accepts both), so the official letter scorer gives
it 0.9% against 79.1% normalized; compare it on the normalized scores in parentheses. Against
SFT on the same 4,916 questions it trails by 4.1 points normalized [−6.1, −1.9], and against the
full SFT epoch by 6.0: on collections never trained on, SFT generalizes better than RL in this
task family. SFT followed by RL adds 0.5 points to SFT [+0.1, +1.0]. For multiple-choice data a
reward that requires the letter would keep RL aligned with the official metric:

<!-- report:omnimedx-pilot -->
| Arm | Prompt | Accuracy | Anatomy Identification answers | Disease Diagnosis answers | Lesion Grading answers | Modality Recognition answers | Other Biological Attributes answers | Output tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Instruct 2B | short | 53.81 (74.57) | 63.79 | 43.12 | 14.89 | 93.25 | 26.19 | 3 |
| Instruct 2B | reasoning | 69.81 (69.81) | 81.40 | 60.36 | 40.43 | 100.00 | 38.10 | 99 |
| SFT 2B on the RL questions (OmniMedVQA collections) | short | 83.13 (83.13) | 94.68 | 80.43 | 27.66 | 100.00 | 35.71 | 2 |
| SFT 2B on the RL questions (OmniMedVQA collections) | reasoning | 80.36 (80.36) | 93.02 | 76.12 | 42.55 | 100.00 | 16.67 | 7 |
| Instruct → short-prompt GRPO (OmniMedVQA collections) | short | 0.87 (79.07) | 0.00 | 0.50 | 0.00 | 4.29 | 0.00 | 7 |
| Instruct → short-prompt GRPO (OmniMedVQA collections) | reasoning | 67.21 (70.24) | 76.41 | 58.04 | 42.55 | 100.00 | 33.33 | 136 |
| SFT 2B (OmniMedVQA collections) | short | 85.03 (85.03) | 93.02 | 85.90 | 31.91 | 100.00 | 16.67 | 2 |
| SFT 2B (OmniMedVQA collections) | reasoning | 82.09 (82.09) | 92.69 | 79.44 | 40.43 | 99.39 | 23.81 | 41 |
| SFT → short-prompt GRPO (OmniMedVQA collections) | short | 85.55 (85.55) | 93.02 | 86.57 | 36.17 | 100.00 | 16.67 | 2 |
| SFT → short-prompt GRPO (OmniMedVQA collections) | reasoning | 81.92 (81.92) | 92.36 | 78.77 | 42.55 | 99.39 | 28.57 | 55 |
<!-- /report:omnimedx-pilot -->

<!-- report:omnimedx-pilot-deltas -->
| Comparison | Prompt | Official |
| --- | --- | ---: |
| SFT 2B on the RL questions (OmniMedVQA collections) − Instruct 2B | short | +29.33 [+26.38, +32.35] |
| SFT 2B on the RL questions (OmniMedVQA collections) − Instruct 2B | reasoning | +10.55 [+7.96, +13.06] |
| Instruct → short-prompt GRPO (OmniMedVQA collections) − Instruct 2B | short | -52.94 [-55.80, -50.09] |
| Instruct → short-prompt GRPO (OmniMedVQA collections) − Instruct 2B | reasoning | -2.60 [-4.58, -0.52] |
| SFT 2B (OmniMedVQA collections) − Instruct 2B | short | +31.23 [+28.29, +34.17] |
| SFT 2B (OmniMedVQA collections) − Instruct 2B | reasoning | +12.28 [+9.43, +15.05] |
| SFT → short-prompt GRPO (OmniMedVQA collections) − Instruct 2B | short | +31.75 [+28.81, +34.69] |
| SFT → short-prompt GRPO (OmniMedVQA collections) − Instruct 2B | reasoning | +12.11 [+9.26, +14.88] |
| Instruct → short-prompt GRPO (OmniMedVQA collections) − SFT 2B on the RL questions (OmniMedVQA collections) | short | -82.27 [-84.52, -80.10] |
| Instruct → short-prompt GRPO (OmniMedVQA collections) − SFT 2B on the RL questions (OmniMedVQA collections) | reasoning | -13.15 [-15.74, -10.55] |
| Instruct → short-prompt GRPO (OmniMedVQA collections) − SFT 2B (OmniMedVQA collections) | short | -84.17 [-86.33, -82.01] |
| Instruct → short-prompt GRPO (OmniMedVQA collections) − SFT 2B (OmniMedVQA collections) | reasoning | -14.88 [-17.65, -12.02] |
| SFT → short-prompt GRPO (OmniMedVQA collections) − SFT 2B (OmniMedVQA collections) | short | +0.52 [+0.09, +1.04] |
| SFT → short-prompt GRPO (OmniMedVQA collections) − SFT 2B (OmniMedVQA collections) | reasoning | -0.17 [-1.30, +0.95] |
<!-- /report:omnimedx-pilot-deltas -->

Under the reasoning prompt, the held-out-collection models against Instruct 2B (normalized):

<!-- report:collections-transfer-reasoning -->
| Arm | Trained on | OmniMedVQA (held-out collections) | SLAKE (English) | PathVQA |
| --- | --- | ---: | ---: | ---: |
| SFT 2B on the RL questions (OmniMedVQA collections) | OmniMedVQA collections (4,916 questions) | +10.55 [+7.96, +13.06] | +2.45 [-0.19, +5.09] | -2.75 [-3.74, -1.75] |
| Instruct → short-prompt GRPO (OmniMedVQA collections) | OmniMedVQA collections (same 4,916) | +0.43 [-1.38, +2.25] | -3.11 [-5.28, -0.85] | -0.50 [-1.35, +0.35] |
| SFT 2B (OmniMedVQA collections) | OmniMedVQA collections (62,678) | +12.28 [+9.43, +15.05] | +0.09 [-2.73, +2.92] | -2.36 [-3.30, -1.41] |
| SFT → short-prompt GRPO (OmniMedVQA collections) | OmniMedVQA collections (SFT, then RL) | +12.11 [+9.26, +14.88] | +1.98 [-0.85, +4.81] | -2.28 [-3.24, -1.33] |
<!-- /report:collections-transfer-reasoning -->

The 8B SLAKE models on each medical test set (official, normalized in parentheses), and their
differences from Instruct 8B and from the 8B SFT model:

<!-- report:slake8-pilot -->
| Arm | Prompt | Accuracy | closed answers | open answers | Output tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| Instruct 8B | short | 58.06 (58.06) | 78.85 | 44.65 | 3 |
| Instruct 8B | reasoning | 37.79 (37.79) | 63.94 | 20.93 | 70 |
| SFT 8B (SLAKE) | short | 77.19 (77.19) | 84.38 | 72.56 | 3 |
| SFT 8B (SLAKE) | reasoning | 55.61 (55.61) | 72.12 | 44.96 | 97 |
| Instruct 8B → short-prompt GRPO (SLAKE, 1 epoch) | short | 68.33 (68.33) | 84.13 | 58.14 | 2 |
| Instruct 8B → short-prompt GRPO (SLAKE, 1 epoch) | reasoning | 42.04 (42.04) | 65.87 | 26.67 | 69 |
| SFT 8B → short-prompt GRPO (SLAKE, 1 epoch) | short | 78.23 (78.23) | 83.65 | 74.73 | 3 |
| SFT 8B → short-prompt GRPO (SLAKE, 1 epoch) | reasoning | 61.36 (61.36) | 77.64 | 50.85 | 105 |
<!-- /report:slake8-pilot -->

<!-- report:slake8-pilot-deltas -->
| Comparison | Prompt | Official |
| --- | --- | ---: |
| SFT 8B (SLAKE) − Instruct 8B | short | +19.13 [+16.49, +21.77] |
| SFT 8B (SLAKE) − Instruct 8B | reasoning | +17.81 [+14.89, +20.74] |
| Instruct 8B → short-prompt GRPO (SLAKE, 1 epoch) − Instruct 8B | short | +10.27 [+8.29, +12.25] |
| Instruct 8B → short-prompt GRPO (SLAKE, 1 epoch) − Instruct 8B | reasoning | +4.24 [+2.64, +5.94] |
| SFT 8B → short-prompt GRPO (SLAKE, 1 epoch) − Instruct 8B | short | +20.17 [+17.34, +23.00] |
| SFT 8B → short-prompt GRPO (SLAKE, 1 epoch) − Instruct 8B | reasoning | +23.56 [+20.55, +26.67] |
| Instruct 8B → short-prompt GRPO (SLAKE, 1 epoch) − SFT 8B (SLAKE) | short | -8.86 [-11.03, -6.79] |
| Instruct 8B → short-prompt GRPO (SLAKE, 1 epoch) − SFT 8B (SLAKE) | reasoning | -13.57 [-16.49, -10.74] |
| SFT 8B → short-prompt GRPO (SLAKE, 1 epoch) − SFT 8B (SLAKE) | short | +1.04 [-0.28, +2.45] |
| SFT 8B → short-prompt GRPO (SLAKE, 1 epoch) − SFT 8B (SLAKE) | reasoning | +5.75 [+3.39, +8.11] |
<!-- /report:slake8-pilot-deltas -->

<!-- report:omnimedvqa8-pilot -->
| Arm | Prompt | Accuracy | Anatomy Identification answers | Disease Diagnosis answers | Lesion Grading answers | Modality Recognition answers | Other Biological Attributes answers | Output tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Instruct 8B | short | 73.79 (75.98) | 67.09 | 71.45 | 70.54 | 99.10 | 64.52 | 2 |
| Instruct 8B | reasoning | 68.40 (68.41) | 65.16 | 64.77 | 55.80 | 98.74 | 53.08 | 291 |
| SFT 8B (SLAKE) | short | 67.37 (78.03) | 53.25 | 70.16 | 72.32 | 80.54 | 44.28 | 3 |
| SFT 8B (SLAKE) | reasoning | 66.12 (66.14) | 65.76 | 61.13 | 51.79 | 96.22 | 61.00 | 271 |
| Instruct 8B → short-prompt GRPO (SLAKE, 1 epoch) | short | 72.81 (76.39) | 66.19 | 70.63 | 67.86 | 98.56 | 60.12 | 2 |
| Instruct 8B → short-prompt GRPO (SLAKE, 1 epoch) | reasoning | 67.94 (68.01) | 65.16 | 64.28 | 53.12 | 98.92 | 50.15 | 284 |
| SFT 8B → short-prompt GRPO (SLAKE, 1 epoch) | short | 64.72 (77.61) | 50.42 | 67.20 | 75.00 | 78.47 | 42.23 | 3 |
| SFT 8B → short-prompt GRPO (SLAKE, 1 epoch) | reasoning | 64.66 (64.66) | 64.32 | 59.96 | 40.62 | 96.49 | 55.43 | 306 |
<!-- /report:omnimedvqa8-pilot -->

<!-- report:omnimedvqa8-pilot-deltas -->
| Comparison | Prompt | Official |
| --- | --- | ---: |
| Instruct 8B → short-prompt GRPO (SLAKE, 1 epoch) − SFT 8B (SLAKE) | short | +5.44 [+4.59, +6.29] |
| Instruct 8B → short-prompt GRPO (SLAKE, 1 epoch) − SFT 8B (SLAKE) | reasoning | +1.82 [+0.86, +2.76] |
| SFT 8B → short-prompt GRPO (SLAKE, 1 epoch) − SFT 8B (SLAKE) | short | -2.65 [-3.11, -2.18] |
| SFT 8B → short-prompt GRPO (SLAKE, 1 epoch) − SFT 8B (SLAKE) | reasoning | -1.46 [-2.34, -0.56] |
<!-- /report:omnimedvqa8-pilot-deltas -->

<!-- report:pathvqa8-pilot -->
| Arm | Prompt | Accuracy | open answers | yes/no answers | Output tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| Instruct 8B | short | 36.00 (36.17) | 3.66 | 68.29 | 3 |
| Instruct 8B | reasoning | 33.89 (33.98) | 1.22 | 66.51 | 101 |
| SFT 8B (SLAKE) | short | 35.18 (35.35) | 4.08 | 66.24 | 3 |
| SFT 8B (SLAKE) | reasoning | 28.34 (28.50) | 1.94 | 54.70 | 130 |
| Instruct 8B → short-prompt GRPO (SLAKE, 1 epoch) | short | 36.60 (36.76) | 3.66 | 69.48 | 3 |
| Instruct 8B → short-prompt GRPO (SLAKE, 1 epoch) | reasoning | 34.14 (34.26) | 1.55 | 66.69 | 84 |
| SFT 8B → short-prompt GRPO (SLAKE, 1 epoch) | short | 36.29 (36.46) | 3.87 | 68.65 | 3 |
| SFT 8B → short-prompt GRPO (SLAKE, 1 epoch) | reasoning | 27.89 (28.01) | 1.88 | 53.87 | 133 |
<!-- /report:pathvqa8-pilot -->

<!-- report:pathvqa8-pilot-deltas -->
| Comparison | Prompt | Official |
| --- | --- | ---: |
| Instruct 8B → short-prompt GRPO (SLAKE, 1 epoch) − SFT 8B (SLAKE) | short | +1.41 [+0.79, +2.04] |
| Instruct 8B → short-prompt GRPO (SLAKE, 1 epoch) − SFT 8B (SLAKE) | reasoning | +5.80 [+4.79, +6.83] |
| SFT 8B → short-prompt GRPO (SLAKE, 1 epoch) − SFT 8B (SLAKE) | short | +1.11 [+0.55, +1.68] |
| SFT 8B → short-prompt GRPO (SLAKE, 1 epoch) − SFT 8B (SLAKE) | reasoning | -0.45 [-1.17, +0.29] |
<!-- /report:pathvqa8-pilot-deltas -->

<!-- report:medical8-transfer-reasoning -->
| Arm | Trained on | SLAKE (English) | OmniMedVQA (open access) | PathVQA |
| --- | --- | ---: | ---: | ---: |
| SFT 8B (SLAKE) | SLAKE | +17.81 [+14.89, +20.74] | -2.27 [-3.22, -1.32] | -5.48 [-6.53, -4.42] |
| Instruct 8B → short-prompt GRPO (SLAKE, 1 epoch) | SLAKE | +4.24 [+2.64, +5.94] | -0.40 [-1.17, +0.34] | +0.28 [-0.24, +0.79] |
| SFT 8B → short-prompt GRPO (SLAKE, 1 epoch) | SLAKE (SFT, then RL) | +23.56 [+20.55, +26.67] | -3.75 [-4.72, -2.77] | -5.98 [-7.05, -4.94] |
<!-- /report:medical8-transfer-reasoning -->

At 8B the SLAKE results repeat the 2B ones: SFT +19.1, RL +10.3 and SFT followed by RL +20.2
points over Instruct 8B on SLAKE, and on PathVQA RL and SFT followed by RL lead SFT by 1.4 and 1.1
points. On OmniMedVQA, which the 8B SFT model was not trained on either, SFT gains 2.1 points
normalized and RL 0.4: there SFT's domain knowledge transfers and RL's does not. On OmniMedVQA's
official letter scorer the order flips (SFT −6.4, RL −1.0), because SFT on SLAKE's short labels
makes the model answer multiple-choice questions with text instead of letters.

## Beyond accuracy

Three measurements, declared in `experiments.toml` (`[[beyond_accuracy]]`) with their expected
outcomes before they ran, each on the task the models were trained on (test role; VQAv2's first
1,000 questions) under the prompt they were trained with:

- **How far training moved the model:** the forward KL from the starting model,
  E[log π₀(y|x) − log π(y|x)] over 8 responses π₀ samples at T = 1 for each of the first 500
  questions, summed over a response's tokens (`scripts/kl_to_base.py`; vLLM scores each trained
  model on exactly the sampled tokens). The starting model scored against its own samples is the
  check: 0.00–0.01 nats under the short prompt and 0.13 under the reasoning prompt, where 200-token
  responses accumulate the small numerical differences between vLLM's sampling and scoring
  passes. Responses whose text does not re-tokenize to the sampled tokens are left out: 0–5 of
  4,000 under the short prompt, 66 under the reasoning prompt. Because the KL is summed over
  tokens, it compares models within a setting, not across prompts.
- **How much the answers depend on the image:** accuracy with a black image (questions the model
  answers without seeing anything), its change from the starting model's, and the accuracy lost
  with another test question's image.
- **Whether confidence tracks accuracy:** the majority answer of the first 8 samples at T = 1,
  with the share of samples giving it as the confidence; the expected calibration error over 10
  equal-width bins, and the accuracy on the half and on the 80% of questions the model is most
  sure of.

`scripts/beyond_accuracy.py` writes [`results/beyond-accuracy.json`](../../results/beyond-accuracy.json);
all accuracies use the normalized scorer, the training reward's.

<!-- report:beyond-accuracy -->
| Setting | Model | Accuracy | KL from start (nats) | Black-image accuracy | Black image, change from start | Lost, swapped image |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SLAKE (English), short | Instruct 2B | 49.76 | 0.00 [-0.00, 0.00] | 25.07 | +0.00 [+0.00, +0.00] | +7.45 [+5.18, +9.80] |
| SLAKE (English), short | SFT 2B (SLAKE) | 74.84 | 10.85 [10.43, 11.26] | 28.75 | +3.68 [+1.70, +5.75] | +12.91 [+10.27, +15.55] |
| SLAKE (English), short | Instruct → short-prompt GRPO (SLAKE, 1 epoch) | 65.60 | 2.14 [1.90, 2.38] | 24.79 | -0.28 [-1.32, +0.75] | +8.48 [+6.13, +10.84] |
| SLAKE (English), short | SFT → short-prompt GRPO (SLAKE, 1 epoch) | 76.34 | 16.60 [15.98, 17.22] | 31.57 | +6.50 [+4.52, +8.58] | +13.38 [+10.84, +16.02] |
| OmniMedVQA (held-out collections), short | Instruct 2B | 74.57 | 0.01 [0.00, 0.01] | 43.34 | +0.00 [+0.00, +0.00] | +29.15 [+25.95, +32.36] |
| OmniMedVQA (held-out collections), short | SFT 2B on the RL questions (OmniMedVQA collections) | 83.13 | 14.11 [13.23, 15.01] | 58.74 | +15.40 [+12.20, +18.77] | +23.88 [+21.19, +26.73] |
| OmniMedVQA (held-out collections), short | Instruct → short-prompt GRPO (OmniMedVQA collections) | 79.07 | 0.90 [0.69, 1.12] | 46.54 | +3.20 [+0.95, +5.45] | +28.98 [+25.78, +32.27] |
| OmniMedVQA (held-out collections), short | SFT 2B (OmniMedVQA collections) | 85.03 | 8.82 [8.24, 9.42] | 67.30 | +23.96 [+20.67, +27.34] | +19.46 [+16.87, +22.06] |
| OmniMedVQA (held-out collections), short | SFT → short-prompt GRPO (OmniMedVQA collections) | 85.55 | 10.10 [9.42, 10.81] | 67.30 | +23.96 [+20.67, +27.34] | +20.50 [+17.91, +23.10] |
| VQAv2, short | Instruct 2B | 76.19 | 0.00 [0.00, 0.01] | 32.75 | +0.00 [+0.00, +0.00] | +44.26 [+41.07, +47.43] |
| VQAv2, short | SFT on the 4,000 RL questions | 77.75 | 0.14 [0.12, 0.16] | 34.64 | +1.89 [+0.58, +3.22] | +46.94 [+43.86, +49.97] |
| VQAv2, short | Instruct → short-prompt GRPO | 76.99 | 0.23 [0.19, 0.28] | 32.78 | +0.03 [-1.05, +1.12] | +45.32 [+42.15, +48.48] |
| VQAv2, short | SFT 2B | 78.75 | 0.20 [0.17, 0.24] | 34.53 | +1.78 [+0.37, +3.25] | +47.37 [+44.30, +50.45] |
| VQAv2, short | SFT → short-prompt GRPO | 78.46 | 0.49 [0.40, 0.59] | 34.34 | +1.59 [+0.11, +3.08] | +47.25 [+44.17, +50.40] |
| ChartQA (human), short | Instruct 2B | 66.24 | 0.01 [-0.00, 0.02] | 13.44 | +0.00 [+0.00, +0.00] | +53.20 [+50.24, +56.16] |
| ChartQA (human), short | SFT 2B (ChartQA) | 68.96 | 0.39 [0.33, 0.45] | 14.40 | +0.96 [-0.08, +2.00] | +56.40 [+53.52, +59.36] |
| ChartQA (human), short | Instruct → short-prompt GRPO (ChartQA, 1 epoch) | 67.76 | 1.02 [0.87, 1.19] | 13.92 | +0.48 [-0.40, +1.36] | +54.96 [+51.92, +57.92] |
| ChartQA (human), reasoning | Instruct 2B | 70.40 | 0.13 [0.11, 0.16] | 5.52 | +0.00 [+0.00, +0.00] | +62.88 [+60.08, +65.76] |
| ChartQA (human), reasoning | Instruct → GRPO (ChartQA, 1 epoch) | 80.96 | 25.35 [24.66, 26.06] | 5.52 | +0.00 [-1.28, +1.28] | +72.16 [+69.52, +74.80] |
| ChartQA (human), reasoning | SFT → GRPO (ChartQA, 1 epoch) | 80.56 | 18.78 [18.36, 19.19] | 5.28 | -0.24 [-1.52, +1.04] | +72.40 [+69.76, +75.04] |
| SLAKE (English), short | Instruct 8B | 58.06 | 0.00 [-0.00, 0.00] | 25.16 | +0.00 [+0.00, +0.00] | +9.71 [+7.16, +12.35] |
| SLAKE (English), short | SFT 8B (SLAKE) | 77.19 | 8.68 [8.27, 9.09] | 35.53 | +10.37 [+8.29, +12.54] | +13.95 [+11.50, +16.40] |
| SLAKE (English), short | Instruct 8B → short-prompt GRPO (SLAKE, 1 epoch) | 68.33 | 1.40 [1.18, 1.65] | 25.54 | +0.38 [-0.85, +1.60] | +11.03 [+8.58, +13.48] |
| SLAKE (English), short | SFT 8B → short-prompt GRPO (SLAKE, 1 epoch) | 78.23 | 12.25 [11.69, 12.78] | 36.00 | +10.84 [+8.39, +13.29] | +12.91 [+10.46, +15.46] |
| SLAKE (English), short | SFT 2B + vision LoRA (SLAKE) | 75.12 | 11.18 [10.76, 11.60] | 28.28 | +3.20 [+1.13, +5.28] | +13.20 [+10.56, +15.83] |
| SLAKE (English), short | SFT 2B, upscaled images (SLAKE) | 74.08 | 10.77 [10.36, 11.18] | 30.35 | +5.28 [+3.30, +7.35] | +11.50 [+8.95, +14.14] |
| SLAKE (English), short | Instruct → short-prompt GRPO, upscaled images (SLAKE, 1 epoch) | 66.07 | 2.68 [2.42, 2.96] | 26.20 | +1.13 [-0.09, +2.36] | +9.33 [+6.97, +11.69] |
| SLAKE (English), short | Instruct → short-prompt GRPO + perception loss (SLAKE, 1 epoch) | 56.27 | 8.32 [7.69, 9.00] | 3.68 | -21.39 [-23.94, -18.94] | +8.58 [+6.22, +10.93] |
<!-- /report:beyond-accuracy -->

**RL's gains come from the image.** With a black image, RL from Instruct answers as many
questions as Instruct in five of the six settings (changes of −0.3 to +0.5 points, every
interval including zero) and 3.2 more on the held-out OmniMedVQA collections. SFT answers more
without the image in every setting it was trained in: +3.7 on SLAKE and +10.4 at 8B, +15.4 on
the held-out collections trained on the same 4,916 questions as RL (+24.0 on the full training
set), +1.8 to +1.9 on VQAv2, and +1.0 on ChartQA, significant in all but ChartQA. At 8B, 10.4 of
SFT's 19.1-point gain on SLAKE is still there with a black image, against 0.4 of RL's 10.3. On the
held-out collections, which are multiple choice, SFT's black-image accuracy rises more than its
accuracy does (+15.4 against +8.6), and the gap between the real image and a black one shrinks
from 31.2 points to 24.4 (17.7 after SFT on the full set): much of what SFT learned there is which
options are likely. RL after SFT keeps what SFT learned, and on SLAKE at 2B adds to it (+6.5).
The expectation declared for this measurement named no direction.

**On medical data, RL moves the model five to sixteen times less than SFT.** The forward KL from
Instruct is 2.1 nats for RL against 10.8 for SFT on SLAKE at 2B, 1.4 against 8.7 at 8B, and 0.9
against 14.1 for SFT on the same held-out-collection questions (8.8 for SFT on the full set),
as expected from RL's Razor (Shenfeld et al., 2025), which finds forgetting tracks this KL, and
consistent with RL's smaller losses on unseen medical sets. The expectation held only where SFT
moves the model far. On VQAv2 and on short-answer ChartQA, where SFT's labels are close to what
Instruct already says, SFT moves the model less than RL does (0.14 and 0.20 against 0.23 nats on
VQAv2, 0.39 against 1.02 on ChartQA); the declared expectation (RL below SFT) fails there.
Reasoning-prompt RL on ChartQA moves the model 25.4 nats; the reasoning itself changes (162-token
responses against Instruct's 205). RL after SFT moves the model further than SFT on every set
where both were measured, since it starts from SFT.

<!-- report:calibration -->
| Setting | Model | Majority accuracy | Mean confidence | ECE | Accuracy, top 50% | Accuracy, top 80% |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SLAKE (English), short | Instruct 2B | 49.5 | 74.2 | 24.7 | 74.9 | 59.6 |
| SLAKE (English), short | SFT 2B (SLAKE) | 74.3 | 76.1 | 2.2 | 97.9 | 86.6 |
| SLAKE (English), short | Instruct → short-prompt GRPO (SLAKE, 1 epoch) | 65.7 | 89.4 | 23.7 | 85.3 | 78.3 |
| SLAKE (English), short | SFT → short-prompt GRPO (SLAKE, 1 epoch) | 76.1 | 89.1 | 13.1 | 92.3 | 87.3 |
| OmniMedVQA (held-out collections), short | Instruct 2B | 74.0 | 69.8 | 7.5 | 86.9 | 81.2 |
| OmniMedVQA (held-out collections), short | SFT 2B on the RL questions (OmniMedVQA collections) | 82.3 | 88.9 | 6.7 | 95.2 | 91.4 |
| OmniMedVQA (held-out collections), short | Instruct → short-prompt GRPO (OmniMedVQA collections) | 78.6 | 94.3 | 16.6 | 81.3 | 82.4 |
| OmniMedVQA (held-out collections), short | SFT 2B (OmniMedVQA collections) | 84.9 | 93.4 | 8.6 | 95.5 | 95.4 |
| OmniMedVQA (held-out collections), short | SFT → short-prompt GRPO (OmniMedVQA collections) | 84.6 | 95.6 | 11.0 | 92.4 | 93.1 |
| VQAv2, short | Instruct 2B | 74.7 | 80.9 | 6.4 | 94.6 | 85.8 |
| VQAv2, short | SFT on the 4,000 RL questions | 76.7 | 76.8 | 4.9 | 95.7 | 87.4 |
| VQAv2, short | Instruct → short-prompt GRPO | 76.5 | 88.6 | 12.1 | 90.8 | 86.4 |
| VQAv2, short | SFT 2B | 77.4 | 77.7 | 3.7 | 95.7 | 88.6 |
| VQAv2, short | SFT → short-prompt GRPO | 78.4 | 88.0 | 10.3 | 92.3 | 86.8 |
| ChartQA (human), short | Instruct 2B | 65.2 | 73.3 | 12.0 | 87.4 | 73.9 |
| ChartQA (human), short | SFT 2B (ChartQA) | 68.1 | 65.3 | 6.0 | 90.1 | 76.7 |
| ChartQA (human), short | Instruct → short-prompt GRPO (ChartQA, 1 epoch) | 68.5 | 86.5 | 18.5 | 82.1 | 76.7 |
| ChartQA (human), reasoning | Instruct 2B | 72.1 | 76.4 | 6.2 | 89.1 | 81.1 |
| ChartQA (human), reasoning | Instruct → GRPO (ChartQA, 1 epoch) | 82.7 | 88.4 | 8.1 | 93.1 | 89.8 |
| ChartQA (human), reasoning | SFT → GRPO (ChartQA, 1 epoch) | 84.6 | 84.9 | 6.7 | 95.0 | 91.3 |
| SLAKE (English), short | Instruct 8B | 58.3 | 85.5 | 27.2 | 79.2 | 70.0 |
| SLAKE (English), short | SFT 8B (SLAKE) | 74.6 | 78.2 | 4.3 | 97.5 | 88.5 |
| SLAKE (English), short | Instruct 8B → short-prompt GRPO (SLAKE, 1 epoch) | 68.0 | 90.2 | 22.3 | 87.9 | 80.4 |
| SLAKE (English), short | SFT 8B → short-prompt GRPO (SLAKE, 1 epoch) | 77.7 | 88.2 | 10.5 | 95.3 | 89.3 |
| SLAKE (English), short | SFT 2B + vision LoRA (SLAKE) | 74.6 | 77.0 | 3.4 | 97.9 | 86.7 |
| SLAKE (English), short | SFT 2B, upscaled images (SLAKE) | 73.6 | 76.2 | 2.6 | 97.2 | 86.0 |
| SLAKE (English), short | Instruct → short-prompt GRPO, upscaled images (SLAKE, 1 epoch) | 66.0 | 89.6 | 23.6 | 84.3 | 77.9 |
| SLAKE (English), short | Instruct → short-prompt GRPO + perception loss (SLAKE, 1 epoch) | 56.5 | 82.5 | 26.0 | 81.3 | 68.8 |
<!-- /report:calibration -->

**RL is overconfident.** Short-prompt RL agrees with itself more than Instruct does in every
setting (86.5–94.3% of samples give the majority answer, against 69.8–85.5% for Instruct), as
expected, but its accuracy does not rise as much: its expected calibration error is 12.1 to
23.7 points, against 2.2 to 8.6 for SFT, and on the half of the questions each model is most sure
of, SFT is right 90.1–97.9% of the time and RL 81.3–90.8%. On SLAKE, Instruct is already
overconfident (calibration error 24.7); SFT corrects that (2.2) and RL keeps it (23.7). RL after
SFT sits between the two in every setting (10.3–13.1). Under the reasoning prompt on ChartQA, RL raises confidence and
accuracy together (calibration error 8.1 against Instruct's 6.2). Agreement among samples is one
measure of confidence; RL makes the samples agree by construction, so a confidence read from the
model's token probabilities could rank models differently.

## Vision-side options (SLAKE)

Every model above trains only the language model; the vision encoder and its merger stay
frozen, and SLAKE's images (240 to 1024 pixels square) reach the encoder at 56 to 768 vision
tokens. Three options act on the vision side, each one change against a recorded 2B recipe,
declared in `experiments.toml` before training (`[pilots.slake-vision]`): SFT with LoRA also
on the vision blocks and merger (`--vision-lora`), SFT and short-prompt GRPO on images
upscaled to the pixel budget (`--image-upscale`, `--image-fit upscale`), and short-prompt GRPO
with a perception loss after PAPO (`--perception-coef 0.02`, `vqar/perception_trainer.py`),
which rewards answers that change when 60% of the image's patches are masked.

Expected, as declared: upscaling and vision LoRA help SLAKE (small images, a domain the
encoder saw little), and the perception loss makes answers depend more on the image. None of the
three improves SLAKE. The arms on its test set, and their differences from the recipe each one
changes (official scorer, 95% intervals over questions):

<!-- report:slake-vision-pilot -->
| Arm | Prompt | Accuracy | closed answers | open answers | Output tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| Instruct 2B | short | 49.76 (49.76) | 75.96 | 32.87 | 3 |
| Instruct 2B | reasoning | 40.34 (40.53) | 57.69 | 29.15 | 138 |
| SFT 2B (SLAKE) | short | 74.84 (74.84) | 82.93 | 69.61 | 3 |
| SFT 2B (SLAKE) | reasoning | 69.18 (69.27) | 81.73 | 61.09 | 25 |
| SFT 2B + vision LoRA (SLAKE) | short | 75.12 (75.12) | 83.17 | 69.92 | 3 |
| SFT 2B + vision LoRA (SLAKE) | reasoning | 66.07 (66.07) | 80.53 | 56.74 | 46 |
| SFT 2B, upscaled images (SLAKE) | short | 74.08 (74.08) | 81.97 | 68.99 | 3 |
| SFT 2B, upscaled images (SLAKE) | reasoning | 69.65 (69.75) | 82.21 | 61.55 | 21 |
| Instruct → short-prompt GRPO (SLAKE, 1 epoch) | short | 65.60 (65.60) | 80.05 | 56.28 | 2 |
| Instruct → short-prompt GRPO (SLAKE, 1 epoch) | reasoning | 52.78 (52.87) | 67.31 | 43.41 | 127 |
| Instruct → short-prompt GRPO, upscaled images (SLAKE, 1 epoch) | short | 65.98 (66.07) | 81.25 | 56.12 | 2 |
| Instruct → short-prompt GRPO, upscaled images (SLAKE, 1 epoch) | reasoning | 49.58 (49.67) | 62.50 | 41.24 | 153 |
| Instruct → short-prompt GRPO + perception loss (SLAKE, 1 epoch) | short | 55.98 (56.27) | 73.80 | 44.50 | 5 |
| Instruct → short-prompt GRPO + perception loss (SLAKE, 1 epoch) | reasoning | 35.16 (35.34) | 43.99 | 29.46 | 121 |
<!-- /report:slake-vision-pilot -->

<!-- report:slake-vision-pilot-deltas -->
| Comparison | Prompt | Official |
| --- | --- | ---: |
| SFT 2B + vision LoRA (SLAKE) − SFT 2B (SLAKE) | short | +0.28 [-0.85, +1.32] |
| SFT 2B + vision LoRA (SLAKE) − SFT 2B (SLAKE) | reasoning | -3.11 [-5.00, -1.23] |
| SFT 2B, upscaled images (SLAKE) − SFT 2B (SLAKE) | short | -0.75 [-1.89, +0.28] |
| SFT 2B, upscaled images (SLAKE) − SFT 2B (SLAKE) | reasoning | +0.47 [-0.75, +1.79] |
| Instruct → short-prompt GRPO, upscaled images (SLAKE, 1 epoch) − Instruct → short-prompt GRPO (SLAKE, 1 epoch) | short | +0.38 [-1.32, +2.07] |
| Instruct → short-prompt GRPO, upscaled images (SLAKE, 1 epoch) − Instruct → short-prompt GRPO (SLAKE, 1 epoch) | reasoning | -3.20 [-5.56, -0.85] |
| Instruct → short-prompt GRPO + perception loss (SLAKE, 1 epoch) − Instruct → short-prompt GRPO (SLAKE, 1 epoch) | short | -9.61 [-11.97, -7.26] |
| Instruct → short-prompt GRPO + perception loss (SLAKE, 1 epoch) − Instruct → short-prompt GRPO (SLAKE, 1 epoch) | reasoning | -17.62 [-20.64, -14.61] |
<!-- /report:slake-vision-pilot-deltas -->

Each model against Instruct 2B on SLAKE and on the two unseen medical sets (normalized):

<!-- report:vision-transfer-short -->
| Arm | Trained on | SLAKE (English) | OmniMedVQA (open access) | PathVQA |
| --- | --- | ---: | ---: | ---: |
| SFT 2B (SLAKE) | SLAKE | +25.07 [+21.87, +28.18] | -0.29 [-0.99, +0.38] | -1.83 [-2.83, -0.85] |
| SFT 2B + vision LoRA (SLAKE) | SLAKE (+ vision LoRA) | +25.35 [+22.15, +28.56] | +0.79 [+0.10, +1.48] | -2.34 [-3.33, -1.32] |
| SFT 2B, upscaled images (SLAKE) | SLAKE (upscaled) | +24.32 [+21.11, +27.43] | -0.08 [-0.79, +0.61] | -2.51 [-3.56, -1.46] |
| Instruct → short-prompt GRPO (SLAKE, 1 epoch) | SLAKE | +15.83 [+13.10, +18.47] | +0.03 [-0.34, +0.40] | -0.97 [-1.76, -0.18] |
| Instruct → short-prompt GRPO, upscaled images (SLAKE, 1 epoch) | SLAKE (upscaled) | +16.31 [+13.67, +18.94] | +0.13 [-0.21, +0.47] | -0.57 [-1.15, -0.00] |
| Instruct → short-prompt GRPO + perception loss (SLAKE, 1 epoch) | SLAKE (+ perception loss) | +6.50 [+3.49, +9.52] | -2.21 [-2.96, -1.47] | -9.29 [-10.37, -8.17] |
<!-- /report:vision-transfer-short -->

<!-- report:vision-transfer-reasoning -->
| Arm | Trained on | SLAKE (English) | OmniMedVQA (open access) | PathVQA |
| --- | --- | ---: | ---: | ---: |
| SFT 2B (SLAKE) | SLAKE | +28.75 [+25.45, +32.05] | +2.79 [+1.82, +3.80] | -5.31 [-6.38, -4.23] |
| SFT 2B + vision LoRA (SLAKE) | SLAKE (+ vision LoRA) | +25.54 [+22.15, +28.84] | -2.89 [-3.86, -1.87] | -9.19 [-10.26, -8.12] |
| SFT 2B, upscaled images (SLAKE) | SLAKE (upscaled) | +29.22 [+25.92, +32.42] | +1.82 [+0.88, +2.78] | -6.69 [-7.81, -5.60] |
| Instruct → short-prompt GRPO (SLAKE, 1 epoch) | SLAKE | +12.35 [+9.71, +15.08] | +2.09 [+1.23, +2.97] | -0.86 [-1.71, -0.03] |
| Instruct → short-prompt GRPO, upscaled images (SLAKE, 1 epoch) | SLAKE (upscaled) | +9.14 [+6.41, +11.88] | +3.16 [+2.24, +4.11] | +0.54 [-0.27, +1.34] |
| Instruct → short-prompt GRPO + perception loss (SLAKE, 1 epoch) | SLAKE (+ perception loss) | -5.18 [-8.39, -2.07] | -19.62 [-20.75, -18.47] | -22.52 [-23.62, -21.42] |
<!-- /report:vision-transfer-reasoning -->

**Upscaling.** Training on images upscaled to the pixel budget changes short-prompt SLAKE by
−0.8 points for SFT and +0.4 for GRPO, both intervals including zero, and costs GRPO 3.2 points
under the reasoning prompt. Upscaling only at test time does not help either: Instruct, SFT and
GRPO move by −0.4 to +1.1 points (the resolution table), and the upscaled GRPO model gains 2.6
under the reasoning prompt when tested on the upscaled images it trained on, which recovers its
own loss and no more. 79% of SLAKE's test questions come with images 512 pixels square or
larger (256 to 768 vision tokens); the answers do not seem limited by resolution. Upscaled
GRPO took 8.3 A30-hours for its epoch; the recipe takes 2.0 hours on the RTX 4090.

**Vision LoRA.** LoRA on the vision blocks and mergers leaves short-prompt SLAKE where SFT has
it (+0.3 [−0.9, +1.3]) and costs 3.1 points under the reasoning prompt. Off SLAKE it moves
OmniMedVQA up by 1.1 points over SFT under the short prompt and down by 5.7 under the
reasoning prompt, and costs PathVQA 0.5 and 3.9 more than SFT: adapting the encoder to SLAKE
narrows the model rather than sharpening its view.

**Perception loss.** At the weight PAPO uses (0.02), GRPO with the perception loss ends 9.6
points below plain GRPO on SLAKE under the short prompt and 17.6 under the reasoning prompt,
and falls 9.3 points below Instruct on PathVQA (normalized). Its training log shows how: the masked-image KL it maximizes
rose from 0.15 to 0.84 nats per token, and from about update 700 the responses grew from 2.6 to
13 tokens while the training reward fell from 0.6–0.7 to 0.45–0.6. The model found the
cheapest way to make its answer depend on the image: say the image is missing. With a black
image it now answers 3.7% of SLAKE correctly (Instruct 25.1%), replying "I can't see the image,
so I can't answer this question. Please upload a clear image again." or "Cannot be determined,
the image is missing." to most questions; with another question's image it answers as before
(8.6 points lost, against 8.5 for plain GRPO), and on real images "cannot be identified"
became one of its five most common answers (40 of 1,061). It learned to detect a missing image,
not to use the image more, and it moved four times as far from Instruct as plain GRPO (KL 8.3
against 2.1 nats). Our KL is estimated on the sampled tokens of 2-token answers, where PAPO
trained on long reasoning; a smaller weight, PAPO's entropy regularization, or a cap tied to
the answer length are the untested remedies.

One seed per arm; the reliance, distance and calibration of each model are in the
[beyond-accuracy](#beyond-accuracy) tables. Compute: 16.9 cluster GPU-hours (one free RTX 6000
for the vision-LoRA SFT, free A30s for the rest) and 1.9 on the local RTX 4090 for the smoke
test and every evaluation.

## Shortcut audit of 3D CT benchmarks

Two 3D CT benchmarks, DeepTumorVQA v2 (abdominal CT, 2D views of each scan) and M3D-VQA (CT and
MRI from Radiopaedia, questions only here), measured for how much of their published accuracy is
reachable without the scan, then DeepTumorVQA rebuilt without its option shortcuts to ask what a
2B model and RL learn from the scan itself. Every run was declared in `experiments.toml` before it
trained; text-only rules come from `python -m scripts.option_priors --dataset <name>` and write
`results/screening/<name>-option-prior.json`. Accuracy (%), one training seed, 2B models trained
on 40,000 questions for one epoch unless noted.

| Benchmark, test set | Text-only rule | SFT, without the image | SFT, with the image | Published fine-tuned |
| --- | ---: | ---: | ---: | ---: |
| DeepTumorVQA, multiple choice (9,996) | 60.7 | 59.0 (the sighted model given black images) | 61.4 (8,000 questions) | 64.6 (Qwen3-VL-4B), 66.3 |
| DeepTumorVQA, free form (10,000) | 57.4 | 57.6 (trained and tested on black images) | 60.3 | 63.4 (Qwen3-VL-4B) |
| M3D-VQA, multiple choice (13,790) | 50.6 | 68.7, mean of types 70.2 (trained and tested on black images) | — | 75.8 (M3D-LaMed, mean of types) |

The DeepTumorVQA rule reads the options: its distractors for a continuous value are drawn around
the true value (0.7 to 1.3 times it), so the true value is the second or third smallest of four
options 88% of the time, and binary and categorical answers follow the patients' prevalence. The
free-form rule answers each subtype and organ with the median (numbers) or most common (text)
training answer, scored with the authors' free-form rules (mean relative accuracy for numbers,
exact match for text). In free form the image adds 2.7 points [+2.1, +3.3] for SFT (sighted
model with real images against the blind model with black ones), almost all of it on lesion
existence (78.9% against 69.8%). The M3D-VQA rule is the most common answer letter per question
type: A is correct in 47.5% of its test questions. By type, the blind M3D model scores 86.4
(plane), 65.0 (phase), 72.4 (organ), 66.0 (abnormality) and 61.0 (location) against M3D-LaMed's
98.80, 79.75, 74.75, 66.65 and 58.94 with the volume: abnormality and location questions are
answered as well without the image.

With the shortcuts removed (`deeptumorx`: options regenerated so the true value sits at a uniformly
random rank, every option list shuffled, discrete answers balanced within each subtype; 6,607 test
questions, text-only rule 35.2% against 33.7% chance), little is left for a 2B model to learn from
2D views of the scan (strict letter scorer; Instruct 34.6% under the lenient scorer, which also
accepts option text):

| Model on debiased DeepTumorVQA | Montage (9 slices) | Black image | 32-slice grid | Black grid |
| --- | ---: | ---: | ---: | ---: |
| SFT trained on the released options | 36.0 | — | — | — |
| SFT, 8,000 debiased questions | 36.7 | 34.4 | 36.6 | 34.3 |
| Instruct → GRPO, 2,000 updates | 36.4 | 33.7 | — | — |
| SFT → GRPO, 2,000 updates | 37.5 | 33.9 | — | — |

The 32-slice grid (evenly spaced axial slices of the released CT video, 2,048 vision tokens)
does no better than the nine-slice montage (36.6% against 36.7%), short of the declared 3-point
margin, so RL trained on the montage. GRPO with a reward for the option letter alone (on the same
8,000-question exposure) matches SFT: −0.3 [−1.6, +1.0] from Instruct and +0.7 [−0.4, +1.8] after
SFT, both inside the noise, so RL does not beat SFT here. Every model's one visual skill is lesion
existence: SFT 60.7%, GRPO 57.1%, SFT → GRPO 60.5%, each about 51% with a black image (chance
50%); measurement stays at chance (24.7–26.2% against 25%). Training reward for both GRPO runs
hovered around 0.3–0.5 against 0.34 for guessing.
