# Full tables

The paired differences behind every comparison and the error analysis, generated from the saved
records. The full test matrix under both prompts is in
[Prompt template × scorer variant](findings/form-not-ability.md#prompt-template--scorer-variant).

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

## Error analysis

`python -m scripts.error_analysis` reads every declared arm's saved predictions and writes
[`results/error-analysis.json`](../results/error-analysis.json); the rules are defined
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
