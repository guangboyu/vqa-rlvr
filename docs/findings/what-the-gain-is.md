# What RL's gain is made of

*One branch of the [report](https://guangboyu.github.io/vqa-rlvr/report/): whether RL beats SFT depends on what the task pays for.*

**RL mostly sharpens what the model can already sample.** Across every set below, RL from Instruct
raises pass@1 and ends level or behind at pass@32, while SFT raises pass@32
([pass@k](#held-out-passk-before-and-after-rl)), as Yue et al. (2025) report for text. Where the answer
is out of reach, SFT has to go first ([reach](#reach)).

**RL's gains come from the image.** At 8B on SLAKE, with the same training questions,
<!--n:ba.slk8.sft.blackgain-->10.4<!--/n--> of SFT's <!--n:ba.slk8.sft.gain-->19.1<!--/n-->-point gain survives a black image, against
<!--n:ba.slk8.grpo.blackgain-->0.4<!--/n--> of RL's <!--n:ba.slk8.grpo.gain-->10.3<!--/n-->. RL from Instruct answers no more blind than Instruct
in <!--n:ba.rl.level-->5<!--/n--> of <!--n:ba.rl.settings-->6<!--/n--> settings, and <!--n:ba.rl.most-->+3.2<!--/n--> points in the other.

**RL moves the model less.** On medical data its forward KL from Instruct is <!--n:kl.med.lo-->5<!--/n--> to <!--n:kl.med.hi-->16<!--/n-->
times smaller than SFT's, and on sets no model trained on it keeps a small lead over SFT
(<!--n:tv.rl_sft.v-->+0.9<!--/n--> on TextVQA, <!--n:vw.rl_sft.v-->+1.0<!--/n--> on VizWiz).

Two limits go with these. In short-answer settings RL is overconfident, with a calibration error
of <!--n:cal.grpo.lo-->12.1<!--/n-->–<!--n:cal.grpo.hi-->23.7<!--/n--> points against SFT's <!--n:cal.sft.lo-->2.2<!--/n-->–<!--n:cal.sft.hi-->8.6<!--/n-->. And all of this describes RL
from Instruct: GRPO after SFT keeps SFT's priors, answering <!--n:ba.omx.sftgrpo.blackgain-->+24.0<!--/n--> points more blind on
OmniMedVQA's held-out collections.

## Reach

How far does each method get past what Instruct can already sample? Reach is Instruct's pass@8 (8 is
GRPO's group size) on open-answer questions, full credit under the normalized scorer, against the
greedy accuracy of each trained model on the same questions and the same rule. Yes/no and
multiple-choice questions are left out, since random picks answer them by chance; OmniMedVQA is all
multiple choice, so it has no row. `python -m scripts.reach`, no GPU.

<!-- report:reach -->
| Test set | Prompt | Open questions | Instruct, greedy | Instruct, pass@8 | RL from Instruct | SFT | SFT → RL |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SLAKE, 2B | short | 706 | 34.3 | **53.7** | 56.1 | 68.6 | 71.0 |
| SLAKE, 8B | short | 706 | 46.6 | **56.7** | 59.5 | 72.1 | 74.1 |
| ChartQA | reasoning | 1,146 | 70.2 | **87.1** | 80.9 | 66.5 | 80.6 |
| ChartQA | short | 1,146 | 66.6 | **78.2** | 68.5 | 69.6 | — |
| VQAv2 | reasoning | 637 | 43.0 | **61.1** | 58.1 | 48.2 | 56.5 |
| GQA | reasoning | 644 | 39.8 | **61.3** | 51.6 | 49.5 | 54.3 |
| TextVQA | reasoning | 948 | 60.5 | **75.1** | 69.4 | 59.3 | 68.1 |
| VQAv2 | short | 637 | 55.4 | **66.9** | 56.8 | 57.5 | 56.0 |
| GQA | short | 644 | 50.0 | **67.8** | 50.3 | 55.1 | 54.0 |
| TextVQA | short | 948 | 70.1 | **78.1** | 70.4 | 67.9 | 69.5 |
<!-- /report:reach -->

On these <!--n:reach.sets-->10<!--/n--> rows RL from Instruct ends at most <!--n:reach.rl.most-->2.8<!--/n--> points above reach. SFT passes it only
on SLAKE, by <!--n:reach.slk2.sft.over-->14.9<!--/n--> points at 2B and <!--n:reach.slk8.sft.over-->15.4<!--/n--> at 8B. Low reach points to SFT first; high
reach only leaves RL room, since on GQA under the short prompt reach is <!--n:reach.gqa.reach-->67.8<!--/n-->% and RL ends at
<!--n:reach.gqa.rl-->50.3<!--/n-->%. One domain supports the rule so far, and new training sources will test it.

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

## RL's medical gains come from the image

Two measurements beyond accuracy, declared before they ran, show what the medical models
learned: how many questions each model still answers with a black image in place of the real
one, and how far training moved it from Instruct (forward KL on Instruct's own samples for the
task's questions, summed over a response). Short prompt, normalized scorer, 95% intervals:

<!-- report:beyond-accuracy-medical -->
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
| SLAKE (English), short | Instruct 8B | 58.06 | 0.00 [-0.00, 0.00] | 25.16 | +0.00 [+0.00, +0.00] | +9.71 [+7.16, +12.35] |
| SLAKE (English), short | SFT 8B (SLAKE) | 77.19 | 8.68 [8.27, 9.09] | 35.53 | +10.37 [+8.29, +12.54] | +13.95 [+11.50, +16.40] |
| SLAKE (English), short | Instruct 8B → short-prompt GRPO (SLAKE, 1 epoch) | 68.33 | 1.40 [1.18, 1.65] | 25.54 | +0.38 [-0.85, +1.60] | +11.03 [+8.58, +13.48] |
| SLAKE (English), short | SFT 8B → short-prompt GRPO (SLAKE, 1 epoch) | 78.23 | 12.25 [11.69, 12.78] | 36.00 | +10.84 [+8.39, +13.29] | +12.91 [+10.46, +15.46] |
<!-- /report:beyond-accuracy-medical -->

RL's gains need the image. With a black image, RL from Instruct answers as many questions as
Instruct on SLAKE at both sizes (−0.3 and +0.4 points, intervals including zero) and 3.2 more
on the held-out collections. SFT answers 3.7 more on SLAKE at 2B, 10.4 more at 8B, and 15.4
more on the held-out collections trained on the same questions as RL: at 8B, 10.4 of SFT's
19.1-point gain on SLAKE is still there without the image, against 0.4 of RL's 10.3. On the
same data, RL also moves the model five to sixteen times less than SFT: 2.1 against 10.8 nats on
SLAKE at 2B, 1.4 against 8.7 at 8B, 0.9 against 14.1 on the held-out collections. RL's Razor
(Shenfeld et al., 2025) finds that forgetting tracks this distance, and it matches what the
tables above show: RL loses less than SFT on the medical sets it did not train on. The same
black-image pattern holds on VQAv2 and ChartQA; the full measurements, including where RL
does not move the model less and where it is less well calibrated than SFT, are in the
[below](#beyond-accuracy).

## Image interventions

Official score with the original image → with a black or swapped image, on the first 128
test questions, and the share of normalized answers that changed. The references stay those
of the original image, so the changed-image score measures agreement with them, not
accuracy on the new image. Every arm depends on the image: a black image changes 60–100% of
answers and lowers scores by 18 to 98 points.
These records exist only for the corrected campaign's arms (September 13 recipe; see
[history](../history.md#the-corrected-campaign-september-13-recipe)); the final-rerun arms
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

## Out of domain, RL keeps an edge over SFT

On the training distribution, short-answer RL and SFT end at the same ceiling ([full
pool](form-not-ability.md#short-prompt-rl-on-the-full-pool)). Away from it, RL is ahead. On
far-OOD TextVQA, short-prompt GRPO beats one SFT epoch by 0.9 points [+0.2, +1.6] at 2B, and on
a second far-OOD set, VizWiz, by 1.0 [+0.5, +1.6]. The two controls below rule out the obvious
explanations: SFT trained on exactly the questions RL drew does not close the gap, and the
audited judge keeps it (+0.8 [+0.2, +1.5] on TextVQA;
[judge](form-not-ability.md#exact-match-misses-rescored-by-an-llm-judge)). The edge is about one point
at 2B with one training seed; at 8B it is not significant on TextVQA.

### SFT on exactly the RL questions

SFT in the headline trains on 39,458 questions; short-prompt GRPO from Instruct drew 4,000. To
separate the amount of supervised data from the training method, one more 2B SFT run trains
one epoch on exactly the 4,000 questions that GRPO run drew (2,022 VQAv2 and 1,978 GQA, 3,535
images, none in a dev or test role), with the unchanged SFT recipe: 125 updates. The pool is
`data/pools/part1-short2-actual4000.json`, built from the run's prompt-draw log by
`scripts/build_draw_pool.py`; the final step was chosen in advance, not by dev score.

<!-- report:matched -->
| Arm | Prompt | VQAv2 | GQA | CLEVR counting | TextVQA (far-OOD) |
| --- | --- | ---: | ---: | ---: | ---: |
| Instruct 2B | short | 77.87 (77.91) | 59.45 (59.46) | 97.67 (97.67) | 80.53 (80.55) |
| Instruct 2B | reasoning | 55.06 (64.68) | 36.13 (51.95) | 87.13 (87.13) | 69.74 (69.75) |
| SFT on the 4,000 RL questions | short | 78.83 (78.87) | 60.84 (60.84) | 99.73 (99.73) | 80.12 (80.13) |
| SFT on the 4,000 RL questions | reasoning | 56.26 (59.84) | 41.44 (45.36) | 88.07 (88.07) | 73.24 (73.25) |
| Instruct → short-prompt GRPO | short | 78.62 (78.66) | 60.34 (60.35) | 99.50 (99.50) | 81.01 (81.02) |
| Instruct → short-prompt GRPO | reasoning | 61.85 (69.81) | 42.89 (54.01) | 86.50 (86.50) | 72.34 (72.35) |
| SFT 2B | short | 79.87 (79.89) | 62.35 (62.35) | 99.40 (99.40) | 80.08 (80.10) |
| SFT 2B | reasoning | 67.99 (68.17) | 56.61 (57.00) | 92.10 (92.10) | 68.86 (68.87) |
<!-- /report:matched -->

Official scorer, candidate minus reference, with 95% intervals over questions:

<!-- report:matched-deltas -->
| Comparison | Prompt | Scorer | VQAv2 | GQA | CLEVR counting | TextVQA (far-OOD) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| SFT on the 4,000 RL questions − Instruct 2B | short | official | +0.96 [+0.42, +1.51] | +1.38 [+0.91, +1.87] | +2.07 [+1.53, +2.63] | -0.41 [-0.99, +0.17] |
| SFT on the 4,000 RL questions − Instruct 2B | reasoning | official | +1.20 [-0.13, +2.52] | +5.30 [+4.41, +6.17] | +0.93 [-0.43, +2.33] | +3.49 [+2.55, +4.43] |
| SFT on the 4,000 RL questions − Instruct → short-prompt GRPO | short | official | +0.21 [-0.32, +0.73] | +0.49 [+0.07, +0.92] | +0.23 [+0.07, +0.43] | -0.89 [-1.42, -0.35] |
| SFT on the 4,000 RL questions − Instruct → short-prompt GRPO | reasoning | official | -5.60 [-6.84, -4.34] | -1.45 [-2.33, -0.60] | +1.57 [+0.13, +3.00] | +0.89 [-0.02, +1.82] |
| SFT on the 4,000 RL questions − SFT 2B | short | official | -1.04 [-1.59, -0.51] | -1.51 [-2.01, -1.03] | +0.33 [+0.07, +0.63] | +0.04 [-0.56, +0.64] |
| SFT on the 4,000 RL questions − SFT 2B | reasoning | official | -11.73 [-13.04, -10.42] | -15.17 [-16.04, -14.34] | -4.03 [-5.33, -2.73] | +4.38 [+3.30, +5.47] |
<!-- /report:matched-deltas -->

Matching the questions closes most of the in-domain gap between SFT and RL. Full SFT led
short-prompt GRPO by 1.25 points on VQAv2 and 2.00 on GQA; trained on the same 4,000 questions,
SFT leads by 0.21 (an interval that includes zero) and 0.49. Far-OOD TextVQA keeps the order of
the headline: short-prompt GRPO stays 0.89 points ahead. Only the question identities are
matched. SFT fits one supplied answer per question in 125 updates on images resized to the
pixel budget; GRPO samples eight answers per question over 1,000 updates on the 608×608 RL
canvas. So this is not a comparison at equal compute, and the full-SFT model predates the fix
that seeds the adapter initialization, so the two SFT runs do not share an initialization.

### Extra OOD sets

Two more far-OOD test sets were frozen on 2026-10-05, before any prediction, and evaluated with
the short prompt and greedy decoding. VizWiz (8,000 questions on 7,928 photos taken by blind
people, unanswerable questions included) is scored with the authors' evaluator, reproduced
on 24,006 pre-declared cases. RealWorldQA (765 questions on 762 images, many multiple choice)
has no verified official evaluator, so it is scored by project-defined exact match that ignores
case and whitespace. No image of either set appears in a training pool (decoded-pixel hash).
Primary score, normalized companion in parentheses (`python -m scripts.compare_ood`):

<!-- report:ood -->
| Arm | VizWiz (authors' evaluator) | RealWorldQA (project exact match) |
| --- | ---: | ---: |
| Instruct 2B | 42.50 (42.53) | 60.13 (60.13) |
| SFT 2B | 42.53 (42.55) | 62.35 (62.35) |
| Instruct → short-prompt GRPO | 43.58 (43.60) | 62.88 (62.88) |
| SFT → short-prompt GRPO | 43.01 (43.04) | 63.40 (63.40) |
| SFT on the 4,000 RL questions | 42.39 (42.42) | 63.79 (63.79) |
<!-- /report:ood -->

Candidate minus reference, points, with 95% intervals that resample images:

<!-- report:ood-deltas -->
| Comparison | VizWiz | RealWorldQA |
| --- | ---: | ---: |
| SFT 2B − Instruct 2B | +0.03 [-0.56, +0.62] | +2.22 [+0.13, +4.31] |
| Instruct → short-prompt GRPO − Instruct 2B | +1.07 [+0.65, +1.51] | +2.75 [+1.17, +4.45] |
| SFT on the 4,000 RL questions − Instruct 2B | -0.11 [-0.67, +0.45] | +3.66 [+1.70, +5.64] |
| Instruct → short-prompt GRPO − SFT 2B | +1.04 [+0.51, +1.58] | +0.52 [-1.57, +2.74] |
| SFT → short-prompt GRPO − SFT 2B | +0.48 [+0.10, +0.86] | +1.05 [-0.39, +2.49] |
| SFT on the 4,000 RL questions − Instruct → short-prompt GRPO | -1.18 [-1.68, -0.70] | +0.92 [-0.91, +2.74] |
<!-- /report:ood-deltas -->

Short-prompt GRPO's far-OOD edge over SFT, seen on TextVQA, replicates on VizWiz: +1.04 points
over SFT and +1.07 over Instruct, while SFT leaves VizWiz unchanged (+0.03). On RealWorldQA both
methods help by 2–3 points over Instruct and do not differ from each other (+0.52, interval
[−1.57, +2.74]); with 765 questions it resolves only differences of about 2 points. VizWiz scores
are low for every arm because 37% of its questions have "unanswerable" as the majority
reference and the models almost never say so: Instruct 2B does in 69 of 8,000 answers, scoring
15.8 on those questions against 58.3 on the rest. SFT on the same 4,000 questions short-prompt
GRPO drew is level with Instruct on VizWiz and 1.18 points behind GRPO (interval [−1.68,
−0.70]), so GRPO's VizWiz lead is not a matter of data volume.
