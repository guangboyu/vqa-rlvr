# History: earlier campaigns, the reward erratum, and the claims ledger

The project's first campaign ran in July 2026. Two defects in it were found later: an
inert VQAv2 training reward and training pools that shared images with the evaluation
sets. The corrected campaign of September 13 fixed both, with 300-update RL runs sampled
from adapter-merged weights. The final rerun of late September froze a recipe from
one-change experiments (1,000 updates, unmerged-adapter rollouts) and retrained every arm
under it; it supplies every headline number. This page keeps the corrected campaign and
the July results as labeled history, documents the reward defect in full because the July
adapters on the Hub were trained with it, and lists every published claim with its
current status.

## Erratum: VQAv2 training-reward defect (found 2026-07-16)

**What happened.** The GRPO correctness reward scored VQAv2 examples with the official
ten-annotator, leave-one-out VQA accuracy. The VQAv2 RL training subset, built from
the_cauldron, stores exactly one reference answer per question. Leave-one-out over a
single annotator compares each prediction with an empty set and returns 0.0 for every
prediction, so in every July GRPO run whose mix included VQAv2, the VQAv2 examples carried
no correctness signal. The correctness gradient came from GQA (and CLEVR where used);
VQAv2 examples contributed only the format reward, or nothing in the correctness-only arm.

**What was not affected.** Evaluation. Evaluation uses the ten-annotator VQAv2 validation
data, where the official metric behaves as intended.

**What was affected.** The interpretation of July training arms that included VQAv2:

| July run | Training mix | Correctness signal actually received |
|---|---|---|
| `grpo_2b_main_base` | VQAv2 + GQA | GQA only |
| `grpo_2b_main_sft` | VQAv2 + GQA | GQA only |
| `grpo_2b_fmt0` (format weight 0) | VQAv2 + GQA | GQA only; VQAv2 groups had zero advantage |
| `grpo_2b_fmt05` (format weight 0.5) | VQAv2 + GQA | GQA only |
| `grpo_2b_vqav2only` | VQAv2 | none: format-only training in effect |
| `grpo_2b_gqaonly` | GQA | GQA (unaffected) |
| `grpo_2b_clevr*` | CLEVR | CLEVR (unaffected) |

**The fix.** `vqar.metrics.vqa_accuracy`, which the normalized scorer and the reward use,
falls back to normalized exact match when fewer than two references exist, where
leave-one-out is undefined. Before the fix, `score("vqav2", "yes", ["yes"])` returned 0.0;
it now returns 1.0. Regression tests cover the one- and ten-reference cases and the reward
itself (`pytest tests/test_metrics.py -k SingleReference`). The corrected campaign trained
with the fixed reward. Of the four adapters published on the Hub in July, the two GRPO
adapters carry this defect; all four were trained on the July pools.

**The July re-validation.** A July re-run of the correctness-only arm with the fixed reward
(`grpo_2b_fmt0_fixedreward`) still removed most visible reasoning: 12.7% of VQAv2
completions stayed longer than 100 characters, against 23.5% with the defective reward, at
slightly higher accuracy (74.7 against 72.9, normalized scorer, reasoning prompt). The corrected campaign,
with held-out image pools and the multimodal padding fix, did not reproduce the removal at
2B; the ledger below records this as open.

## The corrected campaign (September 13 recipe)

The corrected campaign supplied the headline numbers until the final rerun under the frozen
recipe. Its RL arms ran 300 updates and sampled rollouts from adapter-merged bf16 weights;
the Instruct and SFT 2B arms are shared with the final rerun. Official accuracy (%), with
the normalized scorer in parentheses, on the full test subsets.

<!-- report:corrected -->
| Arm | Prompt | VQAv2 | GQA | CLEVR counting | TextVQA (far-OOD) |
| --- | --- | ---: | ---: | ---: | ---: |
| Instruct 2B | short | 77.87 (77.91) | 59.45 (59.46) | 97.67 (97.67) | 80.53 (80.55) |
| Instruct 2B | reasoning | 55.06 (64.68) | 36.13 (51.95) | 87.13 (87.13) | 69.74 (69.75) |
| SFT 2B | short | 79.87 (79.89) | 62.35 (62.35) | 99.40 (99.40) | 80.08 (80.10) |
| SFT 2B | reasoning | 67.99 (68.17) | 56.61 (57.00) | 92.10 (92.10) | 68.86 (68.87) |
| Instruct → GRPO (Sep 13) | short | 78.02 (78.06) | 59.55 (59.55) | 98.27 (98.27) | 80.56 (80.57) |
| Instruct → GRPO (Sep 13) | reasoning | 61.79 (69.01) | 44.90 (54.09) | 86.10 (86.10) | 74.06 (74.06) |
| SFT → GRPO (Sep 13) | short | 79.85 (79.87) | 62.44 (62.44) | 99.40 (99.40) | 80.10 (80.12) |
| SFT → GRPO (Sep 13) | reasoning | 69.18 (70.80) | 53.70 (54.86) | 97.80 (97.80) | 74.58 (74.59) |
| Instruct → gold-conditioned OPD (Sep 13) | short | 77.79 (77.83) | 59.72 (59.72) | 97.73 (97.73) | 80.59 (80.60) |
| Instruct → gold-conditioned OPD (Sep 13) | reasoning | 68.61 (69.23) | 51.35 (51.58) | 87.10 (87.10) | 72.87 (72.87) |
| Instruct 8B | short | 81.18 (81.22) | 61.66 (61.66) | 99.67 (99.67) | 83.49 (83.51) |
| Instruct 8B | reasoning | 63.67 (68.75) | 46.84 (52.27) | 96.97 (96.97) | 74.00 (74.00) |
| Instruct 8B → GRPO (Sep 13) | short | 81.12 (81.16) | 61.78 (61.78) | 99.57 (99.57) | 83.67 (83.69) |
| Instruct 8B → GRPO (Sep 13) | reasoning | 77.37 (78.63) | 59.43 (59.82) | 97.30 (97.30) | 80.46 (80.47) |
<!-- /report:corrected -->

Paired differences, candidate − reference, with 95% bootstrap intervals over questions.

<!-- report:corrected-deltas -->
| Comparison | Prompt | Scorer | VQAv2 | GQA | CLEVR counting | TextVQA (far-OOD) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| SFT 2B − Instruct 2B | short | normalized | +1.98 [+1.33, +2.65] | +2.89 [+2.33, +3.46] | +1.73 [+1.27, +2.20] | -0.45 [-1.11, +0.21] |
| SFT 2B − Instruct 2B | short | official | +2.00 [+1.36, +2.67] | +2.89 [+2.34, +3.47] | +1.73 [+1.27, +2.20] | -0.45 [-1.11, +0.22] |
| SFT 2B − Instruct 2B | reasoning | normalized | +3.49 [+2.32, +4.65] | +5.06 [+4.26, +5.84] | +4.97 [+3.57, +6.30] | -0.89 [-2.05, +0.29] |
| SFT 2B − Instruct 2B | reasoning | official | +12.93 [+11.59, +14.28] | +20.47 [+19.56, +21.36] | +4.97 [+3.57, +6.30] | -0.89 [-2.05, +0.29] |
| Instruct → GRPO (Sep 13) − Instruct 2B | short | normalized | +0.15 [-0.04, +0.35] | +0.09 [-0.06, +0.23] | +0.60 [+0.33, +0.90] | +0.02 [-0.20, +0.26] |
| Instruct → GRPO (Sep 13) − Instruct 2B | short | official | +0.15 [-0.04, +0.35] | +0.10 [-0.05, +0.24] | +0.60 [+0.33, +0.90] | +0.03 [-0.20, +0.26] |
| Instruct → GRPO (Sep 13) − Instruct 2B | reasoning | normalized | +4.33 [+3.43, +5.22] | +2.15 [+1.55, +2.74] | -1.03 [-2.30, +0.20] | +4.31 [+3.47, +5.17] |
| Instruct → GRPO (Sep 13) − Instruct 2B | reasoning | official | +6.73 [+5.77, +7.72] | +8.76 [+8.06, +9.44] | -1.03 [-2.30, +0.20] | +4.31 [+3.48, +5.18] |
| SFT → GRPO (Sep 13) − Instruct 2B | short | normalized | +1.96 [+1.32, +2.61] | +2.98 [+2.42, +3.55] | +1.73 [+1.27, +2.20] | -0.43 [-1.09, +0.24] |
| SFT → GRPO (Sep 13) − Instruct 2B | short | official | +1.98 [+1.34, +2.63] | +2.99 [+2.43, +3.55] | +1.73 [+1.27, +2.20] | -0.42 [-1.08, +0.24] |
| SFT → GRPO (Sep 13) − Instruct 2B | reasoning | normalized | +6.13 [+5.15, +7.13] | +2.91 [+2.23, +3.59] | +10.67 [+9.40, +11.93] | +4.84 [+3.85, +5.82] |
| SFT → GRPO (Sep 13) − Instruct 2B | reasoning | official | +14.13 [+12.95, +15.33] | +17.57 [+16.73, +18.37] | +10.67 [+9.40, +11.93] | +4.84 [+3.85, +5.82] |
| SFT → GRPO (Sep 13) − SFT 2B | short | normalized | -0.02 [-0.21, +0.16] | +0.10 [-0.07, +0.25] | +0.00 [+0.00, +0.00] | +0.02 [-0.21, +0.26] |
| SFT → GRPO (Sep 13) − SFT 2B | short | official | -0.02 [-0.21, +0.16] | +0.10 [-0.07, +0.25] | +0.00 [+0.00, +0.00] | +0.02 [-0.21, +0.26] |
| SFT → GRPO (Sep 13) − SFT 2B | reasoning | normalized | +2.63 [+1.57, +3.72] | -2.15 [-2.88, -1.42] | +5.70 [+4.73, +6.67] | +5.73 [+4.68, +6.76] |
| SFT → GRPO (Sep 13) − SFT 2B | reasoning | official | +1.19 [+0.10, +2.30] | -2.90 [-3.63, -2.17] | +5.70 [+4.73, +6.67] | +5.73 [+4.69, +6.76] |
| Instruct → gold-conditioned OPD (Sep 13) − Instruct 2B | short | normalized | -0.08 [-0.37, +0.21] | +0.25 [+0.06, +0.46] | +0.07 [-0.07, +0.20] | +0.05 [-0.29, +0.39] |
| Instruct → gold-conditioned OPD (Sep 13) − Instruct 2B | short | official | -0.08 [-0.37, +0.21] | +0.26 [+0.06, +0.46] | +0.07 [-0.07, +0.20] | +0.06 [-0.28, +0.40] |
| Instruct → gold-conditioned OPD (Sep 13) − Instruct 2B | reasoning | normalized | +4.55 [+3.53, +5.59] | -0.37 [-1.03, +0.30] | -0.03 [-1.20, +1.13] | +3.12 [+2.18, +4.06] |
| Instruct → gold-conditioned OPD (Sep 13) − Instruct 2B | reasoning | official | +13.55 [+12.31, +14.78] | +15.22 [+14.41, +16.03] | -0.03 [-1.20, +1.13] | +3.13 [+2.19, +4.06] |
| Instruct 8B → GRPO (Sep 13) − Instruct 8B | short | normalized | -0.06 [-0.22, +0.10] | +0.13 [-0.02, +0.28] | -0.10 [-0.23, +0.00] | +0.18 [-0.01, +0.38] |
| Instruct 8B → GRPO (Sep 13) − Instruct 8B | short | official | -0.06 [-0.22, +0.10] | +0.13 [-0.02, +0.28] | -0.10 [-0.23, +0.00] | +0.18 [-0.01, +0.38] |
| Instruct 8B → GRPO (Sep 13) − Instruct 8B | reasoning | normalized | +9.88 [+8.96, +10.79] | +7.55 [+6.84, +8.28] | +0.33 [-0.27, +0.93] | +6.46 [+5.60, +7.36] |
| Instruct 8B → GRPO (Sep 13) − Instruct 8B | reasoning | official | +13.70 [+12.67, +14.73] | +12.59 [+11.82, +13.36] | +0.33 [-0.27, +0.93] | +6.45 [+5.59, +7.35] |
<!-- /report:corrected-deltas -->

Mean output tokens · share of responses that close an `<answer>` tag, reasoning prompt.

<!-- report:corrected-behavior -->
| Arm | VQAv2 | GQA | CLEVR counting | TextVQA (far-OOD) |
| --- | ---: | ---: | ---: | ---: |
| Instruct 2B | 148 · 95% | 123 · 97% | 186 · 98% | 107 · 95% |
| SFT 2B | 21 · 40% | 8 · 20% | 62 · 98% | 34 · 47% |
| Instruct → GRPO (Sep 13) | 141 · 96% | 119 · 98% | 197 · 97% | 100 · 99% |
| SFT → GRPO (Sep 13) | 102 · 97% | 79 · 98% | 24 · 100% | 61 · 99% |
| Instruct → gold-conditioned OPD (Sep 13) | 141 · 93% | 119 · 93% | 206 · 98% | 106 · 96% |
| Instruct 8B | 118 · 99% | 104 · 99% | 193 · 99% | 111 · 99% |
| Instruct 8B → GRPO (Sep 13) | 9 · 100% | 9 · 100% | 83 · 99% | 11 · 100% |
<!-- /report:corrected-behavior -->

What each RL run trained on, and its training signals (defined on the
[training evidence](harness/training-evidence.md#what-each-rl-run-trained-on) page).

<!-- report:corrected-exposure -->
| Run | Updates | Questions | Responses | Groups with varied rewards (%) | Updates without reward variation | Mean reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `grpo2-base` | 300 | 1,200 | 9,600 | 38.67 | 43 | 0.599 |
| `grpo2-sft` | 300 | 1,200 | 9,600 | 49.00 | 16 | 0.581 |
| `grpo8-base` | 300 | 1,200 | 9,600 | 23.33 | 110 | 0.646 |
| `opd2-self` | 300 | 1,200 | 9,600 | 49.25 | 21 | 0.588 |
<!-- /report:corrected-exposure -->

<!-- report:corrected-diagnostics -->
| Run | Reward (first → last 10 updates) | Dead groups (mean) | All right (first → last 10) | All wrong (first → last 10) | Entropy | Clip fraction (mean) | Completion tokens | Distinct prompts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `grpo2-base` | 0.62 → 0.61 | 0.613 | 0.33 → 0.40 | 0.15 → 0.20 | 0.73 → 0.63 | 0.000 | 154 → 139 | 1,200 |
| `grpo2-sft` | 0.27 → 0.63 | 0.510 | 0.03 → 0.35 | 0.30 → 0.15 | 1.08 → 0.90 | 0.000 | 30 → 110 | 1,200 |
| `grpo8-base` | 0.62 → 0.67 | 0.767 | 0.42 → 0.57 | 0.23 → 0.28 | 0.40 → 0.12 | 0.000 | 133 → 13 | 1,200 |
| `opd2-self` | 0.64 → 0.61 | 0.507 | 0.40 → 0.20 | 0.17 → 0.15 | 0.75 → 0.68 | 0.000 | 150 → 132 | 1,200 |
<!-- /report:corrected-diagnostics -->

Under this recipe GRPO kept the reasoning at 2B (148 → 141 tokens on VQAv2) and gained
6.7 points on VQAv2 and 8.8 on GQA under the reasoning prompt, while the 8B run already
removed it (118 → 9 tokens). With 1,000 updates the final rerun removes it at 2B too, between
updates 100 and 150 (training responses 129 → 8 tokens); the recipe experiments' baseline,
sampled from merged weights like the corrected run, tracked that run's reward on the same
question order for its first updates and removed the reasoning between updates 110 and
200. Only 23–49% of the
corrected runs' groups had varied rewards; the final runs' shares are on the
[training evidence](harness/training-evidence.md#what-each-rl-run-trained-on) page.

## The July runs

These runs used the defective reward above and training pools that overlapped the VQAv2
evaluation set on 47 images (53 questions). They are not comparable with the corrected
campaign. Official score (normalized score), %; the official column comes from rescoring the
saved predictions.

<!-- report:history -->
| July run | Prompt | VQAv2 | GQA | CLEVR counting | TextVQA (far-OOD) |
| --- | --- | ---: | ---: | ---: | ---: |
| `grpo_2b_clevr_base_reasoning` | reasoning | 60.03 (68.05) | 43.58 (53.20) | 91.40 (91.40) | 75.50 (75.50) |
| `grpo_2b_clevr_reasoning` | reasoning | — | — | 99.83 (99.83) | — |
| `grpo_2b_fmt0` | short | 77.90 (77.94) | 59.56 (59.57) | — | — |
| `grpo_2b_fmt05` | short | 78.12 (78.16) | 59.47 (59.48) | — | — |
| `grpo_2b_fmt05_reasoning` | reasoning | 58.40 (67.76) | 39.47 (53.68) | 86.77 (86.77) | — |
| `grpo_2b_fmt0_fixedreward` | short | 77.94 (77.98) | 59.46 (59.46) | — | — |
| `grpo_2b_fmt0_fixedreward_reasoning` | reasoning | 72.77 (74.69) | 55.16 (56.51) | — | — |
| `grpo_2b_fmt0_reasoning` | reasoning | 70.28 (72.86) | 54.14 (56.17) | 99.57 (99.57) | — |
| `grpo_2b_gqaonly` | short | 78.05 (78.09) | 59.62 (59.63) | — | — |
| `grpo_2b_gqaonly_reasoning` | reasoning | 60.29 (69.01) | 41.80 (53.76) | 86.60 (86.60) | 72.74 (72.74) |
| `grpo_2b_main_base` | short | 77.75 (77.79) | 59.66 (59.66) | 98.47 (98.47) | 80.73 (80.73) |
| `grpo_2b_main_base_reasoning` | reasoning | 62.84 (69.54) | 46.06 (53.86) | 83.90 (83.90) | 72.69 (72.70) |
| `grpo_2b_main_sft` | short | 79.77 (79.79) | 62.29 (62.29) | 99.77 (99.77) | 79.55 (79.57) |
| `grpo_2b_main_sft_reasoning` | reasoning | 64.03 (69.09) | 48.02 (54.31) | 87.13 (87.13) | 72.30 (72.30) |
| `grpo_2b_vqav2only` | short | 77.93 (77.97) | 59.48 (59.48) | — | — |
| `grpo_2b_vqav2only_reasoning` | reasoning | 59.95 (68.87) | 40.41 (53.08) | 87.23 (87.23) | 72.64 (72.64) |
| `grpo_8b_fmt0` | short | 81.20 (81.24) | 61.73 (61.73) | — | — |
| `grpo_8b_fmt0_reasoning` | reasoning | 77.22 (78.34) | 59.31 (59.75) | 98.27 (98.27) | 80.44 (80.47) |
| `opd_2b_main` | short | 78.03 (78.07) | 59.41 (59.42) | 97.67 (97.67) | 80.46 (80.48) |
| `opd_2b_main_reasoning` | reasoning | 55.86 (65.70) | 34.97 (51.03) | 88.13 (88.13) | 69.67 (69.69) |
| `opd_self_2b_main` | short | 77.95 (77.99) | 59.58 (59.58) | 97.90 (97.90) | 80.42 (80.43) |
| `opd_self_2b_main_reasoning` | reasoning | 67.81 (68.47) | 51.22 (51.46) | 86.27 (86.27) | 72.06 (72.08) |
| `opd_self_2b_main_s43` | short | 77.86 (77.90) | 59.71 (59.71) | — | — |
| `opd_self_2b_main_s43_reasoning` | reasoning | 67.85 (68.47) | 51.02 (51.26) | 86.83 (86.83) | 71.69 (71.70) |
| `opd_self_8b_main` | short | 81.09 (81.11) | 61.73 (61.73) | — | — |
| `opd_self_8b_main_reasoning` | reasoning | 73.06 (73.76) | 54.85 (55.37) | 97.10 (97.10) | 77.95 (77.95) |
| `sft_2b` | short | 79.87 (79.89) | 62.26 (62.26) | 99.77 (99.77) | 79.60 (79.62) |
| `sft_2b_reasoning` | reasoning | 71.64 (71.82) | 57.52 (57.88) | 99.80 (99.80) | 71.28 (71.30) |
| `sft_8b` | short | 82.94 (82.96) | 63.63 (63.63) | 92.83 (92.83) | 85.00 (85.02) |
| `sft_8b_reasoning` | reasoning | 69.70 (70.24) | 50.45 (50.79) | 96.17 (96.17) | 75.40 (75.42) |
| `zero_shot_2b` | short | 77.98 (78.02) | 59.48 (59.49) | 97.70 (97.70) | 80.41 (80.43) |
| `zero_shot_2b_reasoning` | reasoning | 55.02 (64.92) | 35.90 (51.90) | 88.23 (88.23) | 70.20 (70.20) |
| `zero_shot_8b` | short | 81.19 (81.23) | 61.69 (61.69) | 99.63 (99.63) | 83.95 (83.97) |
| `zero_shot_8b_reasoning` | reasoning | 62.95 (68.17) | 46.79 (52.35) | 96.80 (96.80) | 75.65 (75.65) |
<!-- /report:history -->

The adapters published on the Hugging Face Hub in July come from these runs. All four were
trained on the July pools, and the two GRPO adapters also carry the reward defect:

| Adapter | Download |
|---|---|
| 2B SFT | [vqa-rlvr-sft-2b](https://huggingface.co/omnifish123/vqa-rlvr-sft-2b) |
| 8B SFT | [vqa-rlvr-sft-8b](https://huggingface.co/omnifish123/vqa-rlvr-sft-8b) |
| 2B GRPO from Instruct | [vqa-rlvr-grpo-2b-main-base](https://huggingface.co/omnifish123/vqa-rlvr-grpo-2b-main-base) |
| 2B SFT then GRPO | [vqa-rlvr-grpo-2b-main-sft](https://huggingface.co/omnifish123/vqa-rlvr-grpo-2b-main-sft) |

## Claims ledger

Every claim the project published, with its status after the final rerun. "July README" is
the public README of July 2026; "research branch" claims were written on an unpublished
branch in July–August; "corrected README" is the README of the corrected campaign
(September 13). Replicated: holds in the final rerun. Qualified: holds only under a stated
condition. Not replicated: the final rerun did not reproduce it. Retracted: contradicted by later evidence. Open: not re-tested, or the cause is
not yet known. Confirmed: a reported defect that the tests reproduce.

| Claim | Where | Status | Evidence |
|---|---|---|---|
| GRPO raises 2B reasoning-prompt accuracy (VQAv2 +4.6, GQA +2.0, TextVQA +2.5, normalized) | July README | Replicated | Final Instruct → GRPO, reasoning prompt: +23.2, +24.1, +10.2 official. The gain comes from deleting the reasoning (148 → 8 tokens), which brings the reasoning prompt level with the short prompt in-domain. |
| Correctness-only GRPO deletes the chain of thought at 2B | July README; research branch | Replicated (1,000 updates) | Final 2B run: VQAv2 reasoning-prompt output 148 → 8 tokens, tags 100%; removed between updates 100 and 150. The corrected 300-update run (148 → 141) stopped before this point; that it did so because of its shorter learning-rate schedule is a proposed explanation, not a tested one. |
| The reasoning "cost" is a metric artifact: an LLM judge reverses the ranking (81.8 vs 81.1) | July README | Retracted | A pre-registered follow-up on an independent judged sample found no reversal: VQAv2 gap −0.23 points, 95% CI [−1.84, +1.34]; pooled −0.12, CI [−1.27, +0.92]. Exact match undercounts, but equally for both arms. |
| Training data are contamination-free | July README | Retracted | July pools shared 47 images (53 questions) with VQAv2 evaluation, and the SFT dev slice shared 317 of its 487 images with SFT training. The corrected split files exclude every dev and test image from every training pool by decoded-pixel hash. |
| CLEVR-only RL gives the largest out-of-domain gain (TextVQA +5.3) | July README | Open | A single July run; not re-run in the corrected pipeline. |
| Short-answer SFT removes visible reasoning, and GRPO revives it | July README | Qualified | The July emission count (0/3000) looked for literal `<think>` tags, which the Instruct model never emits. GRPO after SFT restores the answer tag (40% → 100% on VQAv2) but, at 1,000 updates, not the reasoning: 21 → 11 tokens. |
| The harness matches lmms-eval (GQA 59.5 vs 58.7 ± 1.6) | July README | Qualified | A sanity check on 1,000 GQA questions with the historical normalized scorer. The official scorers now match the dataset authors' code on 626,074 saved predictions. |
| VQAv2 training reward was inert (erratum above) | July README | Confirmed, fixed | Regression tests; the corrected campaign trained with the fixed reward. |
| The learning signal type decides whether reasoning survives: exact-match GRPO deletes, dense signals keep ~99% | Research branch | Replicated, qualified | After 1,000 updates, correctness-only GRPO leaves 8 tokens at both sizes, while gold-conditioned OPD keeps 141 (93% tags). A 0.2 format reward also keeps 8B reasoning (about 130 tokens), so the reward, not only its density, decides. |
| Signal grounding beats teacher capacity; the gold-conditioned self-teacher has no capability ceiling | Research branch | Retracted | The teacher comparison confounds capacity, grounding, and signal density. |
| Short-prompt accuracy is flat across signal types and scales | Research branch | Qualified | GRPO trained on the reasoning prompt moves short-prompt accuracy by at most 0.7 points on VQAv2 and GQA; GRPO trained on the short prompt adds 0.7–0.9; SFT adds 2.0–2.9. |
| Empty reasoning is equivalent to full reasoning (pilot P2) | Research branch | Retracted | The equivalence test fails: −2.32 points, 90% CI [−5.48, +0.84], n = 487. Swapped reasoning changes answers in 63.6% of the 143 no-leak pairs, so the answers depend on the reasoning text. |
| A static predictor of RL gains fails (pilot P3) | Research branch | Open | The reanalysis is inconclusive under the corrected retention proxy. |
| Public RLVR checkpoints show no grounding drift (ten models, 2,300 probes) | Research branch | Qualified | No degradation was detected under that short-prompt protocol; image sensitivity alone does not establish correct grounding. |
| No arm passes its own short-answer ceiling (28 of 28 arm and dataset pairs) | Corrected README | Qualified | After 1,000 correctness-only updates from Instruct, the reasoning prompt reaches the short prompt in-domain (2B VQAv2 +0.3, 95% CI [−0.3, +1.0]; 8B +0.7, [+0.1, +1.2], within the 0.5-point decoding noise); TextVQA stays about 1 point below. |
| SFT, not RL, moves the short-answer ceiling | Corrected README | Replicated, qualified | Short-prompt GRPO adds 0.7–0.9 points in-domain, SFT 2.0–2.9 at both sizes; on far-OOD TextVQA short-prompt GRPO beats SFT by 0.9 at 2B. |

The July judge study behind the retracted reversal claim is `scripts/judge_study.py`; its
output, `results/judge_study.json`, appears in the dashboard's history view.
