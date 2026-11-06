# Knowledge missing: SFT first, RL polishes

*One branch of the [report](https://guangboyu.github.io/vqa-rlvr/report/): whether RL beats SFT depends on what the task pays for.*

On SLAKE at 2B, SFT reaches <!--n:med.slk2.sft-->74.8<!--/n-->%, GRPO from Instruct <!--n:med.slk2.grpo-->65.6<!--/n-->%, and GRPO after SFT
<!--n:med.slk2.sftgrpo-->76.2<!--/n-->%. The order holds at 8B (<!--n:med.slk8.sft-->77.2<!--/n-->, <!--n:med.slk8.grpo-->68.3<!--/n-->, <!--n:med.slk8.sftgrpo-->78.2<!--/n-->) and on
OmniMedVQA collections held out of training (<!--n:med.omx.sft-->85.0<!--/n-->, <!--n:med.omx.grpo-->79.1<!--/n-->, <!--n:med.omx.sftgrpo-->85.6<!--/n-->). What RL adds
after SFT is polish: <!--n:med.slk2.sftgrpo_sft-->+1.3 [+0.0, +2.7]<!--/n-->, <!--n:med.slk8.sftgrpo_sft-->+1.0 [−0.3, +2.5]<!--/n--> and <!--n:med.omx.sftgrpo_sft-->+0.5 [+0.1, +1.0]<!--/n--> points.

On SLAKE's open-answer questions Instruct's pass@8 is <!--n:reach.slk2.reach-->53.7<!--/n-->%. GRPO ends close to it at
<!--n:reach.slk2.rl-->56.1<!--/n-->%, and SFT passes it at <!--n:reach.slk2.sft-->68.6<!--/n-->%; 8B repeats this (<!--n:reach.slk8.reach-->56.7<!--/n-->, <!--n:reach.slk8.rl-->59.5<!--/n-->,
<!--n:reach.slk8.sft-->72.1<!--/n-->). The proposed mechanism is that RL can only reward answers the model samples while
SFT can teach the others; the pattern is measured on every set in the [reach table](what-the-gain-is.md#reach),
and the mechanism itself is untested.

## Medical images: SFT then RL is the best model, and RL forgets less

Two medical datasets passed the screen ([evaluation](../harness/evaluation.md#dataset-screening)): SLAKE
(radiology questions, English, 4,919 training questions) and OmniMedVQA (multiple choice across
imaging types). OmniMedVQA is also split a second way, with whole source collections held out
for testing (1,156 test questions from 17 collections across X-ray, microscopy, dermoscopy, CT
and fundus images), because its image-level split lets SFT reach 99.7%. Each dataset got the
same three treatments: one SFT epoch, short-prompt GRPO from Instruct, and SFT followed by the
same GRPO; SLAKE at 2B and 8B (one epoch each, 1,229 updates of GRPO, the 8B runs on the
cluster's RTX 6000s), the held-out collections at 2B with GRPO on the SLAKE budget and SFT both
on exactly the questions GRPO drew and for a full epoch. The correctness reward is each
dataset's normalized scorer. Every model was also scored on the other medical test sets and on
PathVQA, which none trained on. Short prompt, each model minus its Instruct start, normalized
scorer (OmniMedVQA's official scorer counts only option letters), 95% intervals over
questions. SLAKE and the image-level OmniMedVQA split at 2B:

<!-- report:medical-transfer-short -->
| Arm | Trained on | SLAKE (English) | OmniMedVQA (open access) | PathVQA |
| --- | --- | ---: | ---: | ---: |
| SFT 2B (SLAKE) | SLAKE | +25.07 [+21.87, +28.18] | -0.29 [-0.99, +0.38] | -1.83 [-2.83, -0.85] |
| Instruct → short-prompt GRPO (SLAKE, 1 epoch) | SLAKE | +15.83 [+13.10, +18.47] | +0.03 [-0.34, +0.40] | -0.97 [-1.76, -0.18] |
| SFT → short-prompt GRPO (SLAKE, 1 epoch) | SLAKE (SFT, then RL) | +26.58 [+23.56, +29.69] | +0.22 [-0.45, +0.89] | -0.95 [-1.86, -0.04] |
| SFT 2B (OmniMedVQA) | OmniMedVQA | -6.88 [-9.33, -4.43] | +30.21 [+29.23, +31.18] | -3.39 [-4.20, -2.62] |
| Instruct → short-prompt GRPO (OmniMedVQA) | OmniMedVQA | -0.75 [-1.70, +0.19] | +10.06 [+9.38, +10.76] | +0.18 [-0.27, +0.62] |
| SFT 2B | VQAv2 + GQA | +1.13 [-0.66, +2.92] | +1.82 [+1.31, +2.31] | — |
| Instruct → GRPO (ChartQA, 1 epoch) | ChartQA | +0.09 [-0.75, +1.04] | +0.30 [+0.07, +0.55] | — |
<!-- /report:medical-transfer-short -->

SLAKE at 8B:

<!-- report:medical8-transfer-short -->
| Arm | Trained on | SLAKE (English) | OmniMedVQA (open access) | PathVQA |
| --- | --- | ---: | ---: | ---: |
| SFT 8B (SLAKE) | SLAKE | +19.13 [+16.49, +21.77] | +2.05 [+1.45, +2.68] | -0.82 [-1.47, -0.19] |
| Instruct 8B → short-prompt GRPO (SLAKE, 1 epoch) | SLAKE | +10.27 [+8.29, +12.25] | +0.40 [+0.06, +0.74] | +0.60 [+0.25, +0.94] |
| SFT 8B → short-prompt GRPO (SLAKE, 1 epoch) | SLAKE (SFT, then RL) | +20.17 [+17.34, +23.00] | +1.63 [+0.99, +2.27] | +0.31 [-0.40, +1.01] |
<!-- /report:medical8-transfer-short -->

OmniMedVQA with held-out collections at 2B:

<!-- report:collections-transfer-short -->
| Arm | Trained on | OmniMedVQA (held-out collections) | SLAKE (English) | PathVQA |
| --- | --- | ---: | ---: | ---: |
| SFT 2B on the RL questions (OmniMedVQA collections) | OmniMedVQA collections (4,916 questions) | +8.56 [+6.14, +10.99] | -3.02 [-5.47, -0.66] | -3.01 [-3.79, -2.22] |
| Instruct → short-prompt GRPO (OmniMedVQA collections) | OmniMedVQA collections (same 4,916) | +4.50 [+2.68, +6.40] | -2.45 [-4.05, -0.85] | -1.79 [-2.62, -0.95] |
| SFT 2B (OmniMedVQA collections) | OmniMedVQA collections (62,678) | +10.47 [+7.96, +12.98] | -6.22 [-8.67, -3.77] | -4.08 [-4.90, -3.26] |
| SFT → short-prompt GRPO (OmniMedVQA collections) | OmniMedVQA collections (SFT, then RL) | +10.99 [+8.48, +13.41] | -6.79 [-9.24, -4.34] | -4.03 [-4.84, -3.23] |
<!-- /report:collections-transfer-short -->

SFT followed by RL is the best medical model in every setting. On SLAKE it scores 76.2 at 2B
and 78.2 at 8B, 1.3 [+0.0, +2.7] and 1.0 [−0.3, +2.5] points above SFT; on the held-out
collections it scores 85.6, 0.5 above SFT [+0.1, +1.0] and 11.0 above Instruct. Each margin
over SFT is small and the 8B interval includes zero, but all three point the same way, and on
ChartQA the same order added 13.5 points.

RL alone gains less in domain, 15.8 points over Instruct on SLAKE at 2B and 10.3 at 8B, nine
points below SFT at both sizes, but it loses less on medical data it did not train on. On
PathVQA every SFT model scores below its Instruct start (−0.8 to −4.1 points) while RL from
Instruct stays within 1.8 of it: RL leads SFT there by 0.8 [+0.2, +1.5] trained on SLAKE at 2B,
1.4 [+0.8, +2.0] at 8B, and 2.3 [+1.2, +3.4] trained on the held-out-collection split, 1.2
[+0.5, +2.0] against SFT on the same questions. SFT on OmniMedVQA costs 6.9 points on SLAKE
where RL trained on it costs 0.8. Part of SFT's loss comes from seeing more data:
matched to RL's questions, SFT costs 3.0 points on SLAKE and RL 2.5. How much of SFT's off-domain loss
RL after SFT wins back varies: half on PathVQA from SLAKE at 2B, all of it at 8B, none from the
held-out collections. The edge is about forgetting, not about learning the task better: on
OmniMedVQA collections never trained on, SFT beats RL even with the same questions, and at 8B,
SFT on SLAKE transfers to OmniMedVQA better than RL does
([below](#medical-pilots-in-detail)). Neither dataset releases patient IDs;
one training seed.

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
