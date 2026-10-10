# Main results

Where RL with a verifiable reward beats the starting model, and where it beats SFT on the
same data. That is the bar for every result on this page: an RL arm counts as a win when it
improves on Instruct, and ideally on SFT, on the dataset's test questions. Reasoning is one
way to get there, not the goal. The negative controls, null results and diagnostics that
explain the rest (reasoning deleted on VQAv2 and GQA, RL level with SFT in domain, training
knobs that did not matter, a mixture that did not learn when to reason) are in the
[supplementary results](supplement.md), with the full test matrix.

Every table here is generated from the stored evaluation records by
`python -m scripts.report`; a test fails if a committed copy drifts from the records.
Scores are percentages on the materialized test subsets with greedy decoding and one
training seed per arm. The official scorer is the dataset authors' evaluator and is the
primary number; the normalized scorer, which also defines the training reward, appears
beside it. At milestones an LLM judge, audited by an image-aware model, rescores the
exact-match misses. The [method](method.md) describes the arms and how to rerun them.

## ChartQA: RL where reasoning pays

On VQAv2, GQA, CLEVR and TextVQA, reasoning lowered accuracy and correctness-only GRPO deleted
it ([supplement](supplement.md#prompt-template--scorer-variant)). ChartQA's
human-written questions are the first candidate where it does not: in screening on training
questions, reasoning helped numeric answers by 5.6 points and cost text answers 5.5
([screening](evaluation.md#dataset-screening)). The 2B arms train on its 7,378-question pool
under the frozen recipe, with ChartQA's relaxed accuracy (5% tolerance on numbers) as the
correctness reward, on one RTX 4090: one SFT epoch, then GRPO from Instruct with the reasoning
prompt, from Instruct with the short prompt, and from SFT with the reasoning prompt. Each GRPO
arm ran twice: a 300-update pilot (3.3 GPU-hours for every arm, with merges and evaluation)
and one full epoch, 1,844 updates in which each pool question is drawn once, the exposure SFT
had (15.7 GPU-hours of training for the three arms). The epoch length was fixed before
training, and its final checkpoint is the reported model. Official (normalized) accuracy on
all 1,250 human test questions, the official score by answer type, and mean output tokens:

<!-- report:chartqa-pilot -->
| Arm | Prompt | Accuracy | number answers | text answers | Output tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| Instruct 2B | short | 65.84 (66.24) | 62.87 | 71.81 | 4 |
| Instruct 2B | reasoning | 69.68 (70.40) | 68.02 | 73.01 | 205 |
| SFT 2B (ChartQA) | short | 68.80 (68.96) | 66.71 | 73.01 | 4 |
| SFT 2B (ChartQA) | reasoning | 66.24 (66.56) | 63.11 | 72.53 | 76 |
| Instruct → GRPO (ChartQA) | short | 66.16 (66.56) | 63.35 | 71.81 | 4 |
| Instruct → GRPO (ChartQA) | reasoning | 76.40 (77.52) | 76.41 | 76.39 | 201 |
| Instruct → short-prompt GRPO (ChartQA) | short | 66.96 (67.20) | 63.95 | 73.01 | 4 |
| Instruct → short-prompt GRPO (ChartQA) | reasoning | 72.08 (72.96) | 71.86 | 72.53 | 198 |
| SFT → GRPO (ChartQA) | short | 69.44 (69.60) | 66.95 | 74.46 | 4 |
| SFT → GRPO (ChartQA) | reasoning | 76.48 (77.28) | 76.89 | 75.66 | 168 |
| Instruct → GRPO (ChartQA, 1 epoch) | short | 66.72 (66.96) | 63.95 | 72.29 | 4 |
| Instruct → GRPO (ChartQA, 1 epoch) | reasoning | 80.08 (80.96) | 80.60 | 79.04 | 162 |
| Instruct → short-prompt GRPO (ChartQA, 1 epoch) | short | 67.52 (67.76) | 65.03 | 72.53 | 4 |
| Instruct → short-prompt GRPO (ChartQA, 1 epoch) | reasoning | 75.60 (76.48) | 76.29 | 74.22 | 177 |
| SFT → GRPO (ChartQA, 1 epoch) | short | 69.68 (69.84) | 66.47 | 76.14 | 4 |
| SFT → GRPO (ChartQA, 1 epoch) | reasoning | 79.76 (80.56) | 80.48 | 78.31 | 197 |
| Instruct → GRPO (VQAv2 + GQA + ChartQA) | short | 67.04 (67.28) | 64.19 | 72.77 | 4 |
| Instruct → GRPO (VQAv2 + GQA + ChartQA) | reasoning | 65.92 (66.48) | 63.35 | 71.08 | 11 |
<!-- /report:chartqa-pilot -->

Candidate minus reference, official scorer, with 95% intervals over questions:

<!-- report:chartqa-pilot-deltas -->
| Comparison | Prompt | Official |
| --- | --- | ---: |
| SFT 2B (ChartQA) − Instruct 2B | short | +2.96 [+1.12, +4.80] |
| SFT 2B (ChartQA) − Instruct 2B | reasoning | -3.44 [-6.24, -0.64] |
| Instruct → GRPO (ChartQA) − Instruct 2B | short | +0.32 [-0.32, +0.96] |
| Instruct → GRPO (ChartQA) − Instruct 2B | reasoning | +6.72 [+4.56, +8.80] |
| Instruct → short-prompt GRPO (ChartQA) − Instruct 2B | short | +1.12 [+0.32, +1.92] |
| Instruct → short-prompt GRPO (ChartQA) − Instruct 2B | reasoning | +2.40 [+0.56, +4.16] |
| Instruct → GRPO (ChartQA, 1 epoch) − Instruct 2B | short | +0.88 [-0.16, +2.00] |
| Instruct → GRPO (ChartQA, 1 epoch) − Instruct 2B | reasoning | +10.40 [+8.16, +12.64] |
| Instruct → short-prompt GRPO (ChartQA, 1 epoch) − Instruct 2B | short | +1.68 [+0.32, +3.04] |
| Instruct → short-prompt GRPO (ChartQA, 1 epoch) − Instruct 2B | reasoning | +5.92 [+3.68, +8.08] |
| SFT → GRPO (ChartQA) − SFT 2B (ChartQA) | short | +0.64 [-0.16, +1.44] |
| SFT → GRPO (ChartQA) − SFT 2B (ChartQA) | reasoning | +10.24 [+7.68, +12.72] |
| Instruct → short-prompt GRPO (ChartQA) − SFT 2B (ChartQA) | short | -1.84 [-3.60, -0.08] |
| Instruct → short-prompt GRPO (ChartQA) − SFT 2B (ChartQA) | reasoning | +5.84 [+3.12, +8.56] |
| SFT → GRPO (ChartQA, 1 epoch) − SFT 2B (ChartQA) | short | +0.88 [-0.24, +2.00] |
| SFT → GRPO (ChartQA, 1 epoch) − SFT 2B (ChartQA) | reasoning | +13.52 [+10.88, +16.08] |
| Instruct → GRPO (ChartQA, 1 epoch) − Instruct → GRPO (ChartQA) | short | +0.56 [-0.48, +1.60] |
| Instruct → GRPO (ChartQA, 1 epoch) − Instruct → GRPO (ChartQA) | reasoning | +3.68 [+1.76, +5.60] |
| Instruct → short-prompt GRPO (ChartQA, 1 epoch) − Instruct → short-prompt GRPO (ChartQA) | short | +0.56 [-0.72, +1.84] |
| Instruct → short-prompt GRPO (ChartQA, 1 epoch) − Instruct → short-prompt GRPO (ChartQA) | reasoning | +3.52 [+1.44, +5.60] |
| SFT → GRPO (ChartQA, 1 epoch) − SFT → GRPO (ChartQA) | short | +0.24 [-0.80, +1.28] |
| SFT → GRPO (ChartQA, 1 epoch) − SFT → GRPO (ChartQA) | reasoning | +3.28 [+1.44, +5.20] |
| Instruct → GRPO (VQAv2 + GQA + ChartQA) − Instruct → GRPO (ChartQA, 1 epoch) | short | +0.32 [-0.88, +1.52] |
| Instruct → GRPO (VQAv2 + GQA + ChartQA) − Instruct → GRPO (ChartQA, 1 epoch) | reasoning | -14.16 [-16.88, -11.44] |
<!-- /report:chartqa-pilot-deltas -->

Here GRPO keeps the reasoning and improves it, and more data keeps paying. Under the reasoning
prompt, GRPO from Instruct gains 6.7 points in 300 updates and 10.4 [+8.2, +12.6] after the
epoch (69.7 → 80.1, and +3.7 [+1.8, +5.6] over the pilot), on numeric answers (68.0 → 80.6)
and text answers (73.0 → 79.0) alike. It still reasons, at 162 tokens per response against
Instruct's 205, where the same recipe cut VQAv2 and GQA responses to 8 tokens; in the pilot's
training its completions went from 207 to 202 tokens and its entropy from 0.24 to 0.23,
against 152 → 8 tokens and 0.73 → 0.06 on VQAv2 and GQA. GRPO after SFT, whose start had
shortened its reasoning to 76 tokens, grows it back to 197 and ends 13.5 [+10.9, +16.1]
points above SFT, level with GRPO from Instruct (79.8 against 80.1). Short answers gain far
less from the extra data: short-prompt GRPO adds 1.1 points in 300 updates and 1.7 [+0.3,
+3.0] after the epoch, +0.6 [−0.7, +1.8] over the pilot. The best short answer (SFT → GRPO
after the epoch, 69.7) stays 10 points below both reasoning-trained arms, so on this dataset RL
with reasoning lifts accuracy above the short-answer ceiling. Short-prompt training also raises
the reasoning prompt's score, by 5.9 [+3.7, +8.1] points after the epoch, at 177 tokens. One
training seed and one dataset throughout.

pass@k on all 1,250 test questions, 32 samples at temperature 1 (official scorer, %;
`python -m scripts.pass_at_k --declared chartqa`):

<!-- report:chartqa-passk -->
| Arm | Prompt | Dataset | Mean score | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Instruct 2B | short | ChartQA (human) | 62.68 | 62.68 | 68.79 | 73.83 | 78.30 | 82.38 | 86.08 |
| SFT 2B (ChartQA) | short | ChartQA (human) | 60.97 | 60.97 | 71.05 | 78.09 | 83.22 | 87.01 | 89.76 |
| Instruct → short-prompt GRPO (ChartQA) | short | ChartQA (human) | 64.93 | 64.93 | 69.61 | 73.46 | 76.86 | 79.91 | 82.56 |
| Instruct → short-prompt GRPO (ChartQA, 1 epoch) | short | ChartQA (human) | 67.75 | 67.75 | 70.04 | 71.94 | 73.71 | 75.39 | 77.12 |
| Instruct 2B | reasoning | ChartQA (human) | 66.44 | 66.44 | 75.68 | 82.10 | 86.46 | 89.24 | 91.04 |
| SFT 2B (ChartQA) | reasoning | ChartQA (human) | 58.00 | 58.00 | 72.32 | 81.99 | 88.15 | 92.07 | 94.56 |
| Instruct → GRPO (ChartQA) | reasoning | ChartQA (human) | 74.86 | 74.86 | 81.66 | 86.08 | 89.08 | 91.14 | 92.56 |
| SFT → GRPO (ChartQA) | reasoning | ChartQA (human) | 73.14 | 73.14 | 82.41 | 87.89 | 91.32 | 93.50 | 95.12 |
| Instruct → GRPO (ChartQA, 1 epoch) | reasoning | ChartQA (human) | 79.78 | 79.78 | 83.77 | 86.58 | 88.69 | 90.24 | 91.52 |
| SFT → GRPO (ChartQA, 1 epoch) | reasoning | ChartQA (human) | 79.41 | 79.41 | 84.81 | 88.29 | 90.61 | 92.19 | 93.28 |
<!-- /report:chartqa-passk -->

<!-- report:chartqa-passk-deltas -->
| Comparison | Prompt | Dataset | Mean score | pass@1 | pass@4 | pass@32 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Instruct → GRPO (ChartQA) − Instruct 2B | reasoning | ChartQA (human) | +8.43 [+7.38, +9.49] | +8.43 [+7.38, +9.49] | +3.98 [+3.16, +4.83] | +1.52 [+0.56, +2.56] |
| SFT → GRPO (ChartQA) − SFT 2B (ChartQA) | reasoning | ChartQA (human) | +15.14 [+13.96, +16.33] | +15.14 [+13.96, +16.33] | +5.90 [+4.85, +6.98] | +0.56 [-0.56, +1.68] |
| Instruct → short-prompt GRPO (ChartQA) − Instruct 2B | short | ChartQA (human) | +2.25 [+1.79, +2.72] | +2.25 [+1.79, +2.72] | -0.37 [-0.90, +0.15] | -3.52 [-4.72, -2.32] |
| SFT 2B (ChartQA) − Instruct 2B | short | ChartQA (human) | -1.70 [-2.77, -0.65] | -1.70 [-2.77, -0.65] | +4.26 [+3.21, +5.36] | +3.68 [+2.32, +5.04] |
| Instruct → GRPO (ChartQA, 1 epoch) − Instruct 2B | reasoning | ChartQA (human) | +13.35 [+12.00, +14.69] | +13.35 [+12.00, +14.69] | +4.48 [+3.40, +5.59] | +0.48 [-0.72, +1.68] |
| SFT → GRPO (ChartQA, 1 epoch) − SFT 2B (ChartQA) | reasoning | ChartQA (human) | +21.41 [+19.96, +22.85] | +21.41 [+19.96, +22.85] | +6.30 [+5.08, +7.58] | -1.28 [-2.48, -0.08] |
| Instruct → short-prompt GRPO (ChartQA, 1 epoch) − Instruct 2B | short | ChartQA (human) | +5.08 [+4.02, +6.16] | +5.08 [+4.02, +6.16] | -1.89 [-2.97, -0.84] | -8.96 [-10.72, -7.28] |
<!-- /report:chartqa-passk-deltas -->

The full epoch separates the two kinds of RL further. Trained with reasoning, GRPO from
Instruct gains 13.3 [+12.0, +14.7] points at pass@1 and ends level with Instruct at pass@32
(+0.5 [−0.7, +1.7]; the pilot was +1.5 [+0.6, +2.6]): the extra updates concentrate
probability on answers Instruct could already sample, without losing them. GRPO after SFT
gains 21.4 at pass@1 and ends 1.3 points below SFT at pass@32 ([−2.5, −0.1]). Short-prompt
GRPO narrows with every update: its pass@1 gain grows from 2.2 to 5.1 while its pass@32 gap
widens from −3.5 to −9.0 [−10.7, −7.3], below Instruct from k = 4 on. SFT does the reverse,
−1.7 at pass@1 and +3.7 at pass@32. At the same data and update budget, then, reasoning RL
keeps Instruct's coverage at k = 32 and short-answer RL gives up 9 points of it.

The exact-match misses of these 16 cells were rescored by the judge that rescored the headline
cells, and an image-aware audit checked 50 of its verdicts per stratum on ChartQA ($0.24 for
the judge, $0.45 for the audit; `python -m scripts.judge_records --pilot chartqa`, then
`python -m scripts.audit_judge --pilot chartqa`). The auditor confirms 80% of the judge's
one-word acceptances, 96% of its two-to-five-word ones and 18% of longer ones, mostly
unfinished reasoning the judge took for an answer, and finds 10% of its rejections correct.
Official / judge-adjusted / audited accuracy, with the share of misses the judge accepts:

<!-- report:chartqa-judge -->
| Arm | Prompt | ChartQA (human) |
| --- | --- | ---: |
| Instruct 2B | short | 65.84 / 68.16 / 67.89 (7%) |
| Instruct 2B | reasoning | 69.68 / 80.00 / 77.17 (34%) |
| SFT 2B (ChartQA) | short | 68.80 / 70.88 / 70.58 (7%) |
| SFT 2B (ChartQA) | reasoning | 66.24 / 72.00 / 69.83 (17%) |
| Instruct → GRPO (ChartQA) | short | 66.16 / 68.16 / 67.94 (6%) |
| Instruct → GRPO (ChartQA) | reasoning | 76.40 / 81.84 / 80.11 (23%) |
| Instruct → short-prompt GRPO (ChartQA) | short | 66.96 / 68.88 / 68.60 (6%) |
| Instruct → short-prompt GRPO (ChartQA) | reasoning | 72.08 / 80.56 / 78.20 (31%) |
| SFT → GRPO (ChartQA) | short | 69.44 / 71.20 / 71.00 (6%) |
| SFT → GRPO (ChartQA) | reasoning | 76.48 / 80.96 / 79.66 (19%) |
| Instruct → GRPO (ChartQA, 1 epoch) | short | 66.72 / 68.96 / 68.65 (7%) |
| Instruct → GRPO (ChartQA, 1 epoch) | reasoning | 80.08 / 83.52 / 82.62 (17%) |
| Instruct → short-prompt GRPO (ChartQA, 1 epoch) | short | 67.52 / 69.44 / 69.17 (6%) |
| Instruct → short-prompt GRPO (ChartQA, 1 epoch) | reasoning | 75.60 / 81.28 / 79.57 (24%) |
| SFT → GRPO (ChartQA, 1 epoch) | short | 69.68 / 71.20 / 70.96 (5%) |
| SFT → GRPO (ChartQA, 1 epoch) | reasoning | 79.76 / 83.44 / 81.98 (18%) |
| Instruct → GRPO (VQAv2 + GQA + ChartQA) | short | — |
| Instruct → GRPO (VQAv2 + GQA + ChartQA) | reasoning | — |
<!-- /report:chartqa-judge -->

Paired differences in points, official / adjusted / audited:

<!-- report:chartqa-judge-deltas -->
| Comparison | Prompt | ChartQA (human) |
| --- | --- | ---: |
| SFT 2B (ChartQA) − Instruct 2B | short | +2.96 / +2.72 / +2.69 |
| SFT 2B (ChartQA) − Instruct 2B | reasoning | -3.44 / -8.00 / -7.34 |
| Instruct → GRPO (ChartQA) − Instruct 2B | short | +0.32 / +0.00 / +0.05 |
| Instruct → GRPO (ChartQA) − Instruct 2B | reasoning | +6.72 / +1.84 / +2.94 |
| Instruct → short-prompt GRPO (ChartQA) − Instruct 2B | short | +1.12 / +0.72 / +0.71 |
| Instruct → short-prompt GRPO (ChartQA) − Instruct 2B | reasoning | +2.40 / +0.56 / +1.03 |
| Instruct → GRPO (ChartQA, 1 epoch) − Instruct 2B | short | +0.88 / +0.80 / +0.77 |
| Instruct → GRPO (ChartQA, 1 epoch) − Instruct 2B | reasoning | +10.40 / +3.52 / +5.45 |
| Instruct → short-prompt GRPO (ChartQA, 1 epoch) − Instruct 2B | short | +1.68 / +1.28 / +1.28 |
| Instruct → short-prompt GRPO (ChartQA, 1 epoch) − Instruct 2B | reasoning | +5.92 / +1.28 / +2.40 |
| SFT → GRPO (ChartQA) − SFT 2B (ChartQA) | short | +0.64 / +0.32 / +0.42 |
| SFT → GRPO (ChartQA) − SFT 2B (ChartQA) | reasoning | +10.24 / +8.96 / +9.84 |
| Instruct → short-prompt GRPO (ChartQA) − SFT 2B (ChartQA) | short | -1.84 / -2.00 / -1.98 |
| Instruct → short-prompt GRPO (ChartQA) − SFT 2B (ChartQA) | reasoning | +5.84 / +8.56 / +8.37 |
| SFT → GRPO (ChartQA, 1 epoch) − SFT 2B (ChartQA) | short | +0.88 / +0.32 / +0.38 |
| SFT → GRPO (ChartQA, 1 epoch) − SFT 2B (ChartQA) | reasoning | +13.52 / +11.44 / +12.15 |
| Instruct → GRPO (ChartQA, 1 epoch) − Instruct → GRPO (ChartQA) | short | +0.56 / +0.80 / +0.72 |
| Instruct → GRPO (ChartQA, 1 epoch) − Instruct → GRPO (ChartQA) | reasoning | +3.68 / +1.68 / +2.51 |
| Instruct → short-prompt GRPO (ChartQA, 1 epoch) − Instruct → short-prompt GRPO (ChartQA) | short | +0.56 / +0.56 / +0.57 |
| Instruct → short-prompt GRPO (ChartQA, 1 epoch) − Instruct → short-prompt GRPO (ChartQA) | reasoning | +3.52 / +0.72 / +1.38 |
| SFT → GRPO (ChartQA, 1 epoch) − SFT → GRPO (ChartQA) | short | +0.24 / +0.00 / -0.04 |
| SFT → GRPO (ChartQA, 1 epoch) − SFT → GRPO (ChartQA) | reasoning | +3.28 / +2.48 / +2.32 |
<!-- /report:chartqa-judge-deltas -->

On ChartQA, as on VQAv2, part of what exact match credits to RL is answer form, but here half
of the reasoning gain survives. Instruct's reasoning-prompt misses are often right answers in
the wrong form, such as a unit added ("20%" for 20) or a list without brackets, and the judge
accepts 34% of them; GRPO learns the reward's form, and the accepted share falls to 17%. Under
the audited scorer, one epoch of GRPO from Instruct gains 5.5 points [+3.5, +7.4] under the
reasoning prompt instead of 10.4, still 2.5 [+0.7, +4.3] over the pilot; GRPO after SFT gains
12.2 [+9.6, +14.7] over SFT, whose short reasoning had few form errors to correct. On VQAv2 the
same scorer kept about a fifth of the official gain (4.4 of 23.2). Reasoning itself pays more
under the audited scorer: Instruct scores 77.2 with reasoning against 67.9 with a short
answer, and one epoch of reasoning-trained GRPO reaches 82.6, 11.7 points above the best
short-answer arm (SFT → GRPO, 71.0). Short-prompt GRPO's +1.7 becomes +1.3 [−0.1, +2.6].

## Out of domain, RL keeps an edge over SFT

On the training distribution, short-answer RL and SFT end at the same ceiling (supplement:
[full pool](supplement.md#short-prompt-rl-on-the-full-pool)). Away from it, RL is ahead. On
far-OOD TextVQA, short-prompt GRPO beats one SFT epoch by 0.9 points [+0.2, +1.6] at 2B, and on
a second far-OOD set, VizWiz, by 1.0 [+0.5, +1.6]. The two controls below rule out the obvious
explanations: SFT trained on exactly the questions RL drew does not close the gap, and the
audited judge keeps it (+0.8 [+0.2, +1.5] on TextVQA;
[judge](supplement.md#exact-match-misses-rescored-by-an-llm-judge)). The edge is about one point
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

## Transfer between training data

Each model trained on one dataset family was also scored on the other family's test sets,
against Instruct 2B: full test roles, official scorer, 95% intervals over questions
(`[transfer]` in experiments.toml; `python -m scripts.paired_deltas`). Under the reasoning
prompt:

<!-- report:transfer-reasoning -->
| Arm | Trained on | VQAv2 | GQA | TextVQA (far-OOD) | CLEVR counting | ChartQA (human) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Instruct → GRPO (ChartQA, 1 epoch) | ChartQA | +15.21 [+13.99, +16.40] | +15.94 [+15.11, +16.76] | +5.85 [+4.90, +6.80] | +2.67 [+1.27, +4.07] | +10.40 [+8.16, +12.64] |
| Instruct → short-prompt GRPO (ChartQA, 1 epoch) | ChartQA | +4.83 [+3.79, +5.88] | +6.58 [+5.87, +7.31] | +0.86 [-0.03, +1.80] | +0.53 [-0.67, +1.77] | +5.92 [+3.68, +8.08] |
| Instruct → GRPO | VQAv2 + GQA | +23.22 [+21.95, +24.47] | +24.10 [+23.17, +24.98] | +10.24 [+9.17, +11.29] | +12.13 [+10.90, +13.37] | -6.72 [-9.68, -3.84] |
| Instruct → short-prompt GRPO | VQAv2 + GQA | +6.80 [+5.85, +7.78] | +6.76 [+6.07, +7.43] | +2.60 [+1.75, +3.45] | -0.63 [-1.90, +0.60] | +0.48 [-1.20, +2.16] |
| SFT 2B | VQAv2 + GQA | +12.93 [+11.59, +14.28] | +20.47 [+19.56, +21.36] | -0.89 [-2.05, +0.29] | +4.97 [+3.57, +6.30] | -7.92 [-10.80, -5.12] |
| Instruct → GRPO (VQAv2 + GQA + ChartQA) | VQAv2 + GQA + ChartQA | +23.68 [+22.40, +24.95] | +23.30 [+22.40, +24.19] | +9.99 [+8.93, +11.02] | +8.77 [+7.43, +10.10] | -3.76 [-6.80, -0.80] |
<!-- /report:transfer-reasoning -->

Under the short prompt:

<!-- report:transfer-short -->
| Arm | Trained on | VQAv2 | GQA | TextVQA (far-OOD) | CLEVR counting | ChartQA (human) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Instruct → GRPO (ChartQA, 1 epoch) | ChartQA | -0.25 [-0.57, +0.07] | -0.02 [-0.25, +0.22] | +0.12 [-0.25, +0.49] | -0.60 [-0.90, -0.33] | +0.88 [-0.16, +2.00] |
| Instruct → short-prompt GRPO (ChartQA, 1 epoch) | ChartQA | -0.35 [-0.74, +0.02] | +0.05 [-0.21, +0.31] | +0.16 [-0.31, +0.63] | +1.77 [+1.30, +2.27] | +1.68 [+0.32, +3.04] |
| Instruct → GRPO | VQAv2 + GQA | +0.07 [-0.32, +0.47] | +0.60 [+0.29, +0.91] | +0.49 [+0.05, +0.94] | +1.80 [+1.30, +2.33] | +0.00 [-0.88, +0.88] |
| Instruct → short-prompt GRPO | VQAv2 + GQA | +0.75 [+0.27, +1.23] | +0.89 [+0.48, +1.30] | +0.48 [-0.00, +0.97] | +1.83 [+1.33, +2.37] | +0.56 [-0.40, +1.52] |
| SFT 2B | VQAv2 + GQA | +2.00 [+1.36, +2.67] | +2.89 [+2.34, +3.47] | -0.45 [-1.11, +0.22] | +1.73 [+1.27, +2.20] | -0.16 [-1.60, +1.28] |
| Instruct → GRPO (VQAv2 + GQA + ChartQA) | VQAv2 + GQA + ChartQA | -0.10 [-0.52, +0.30] | +0.41 [+0.08, +0.72] | +0.16 [-0.30, +0.62] | +1.90 [+1.40, +2.43] | +1.20 [+0.00, +2.40] |
<!-- /report:transfer-short -->

What GRPO teaches here is a response habit, and the habit travels. Trained on ChartQA with
reasoning, the model keeps reasoning on photos, shorter and more accurate: 66–83 tokens on
VQAv2, GQA and TextVQA against Instruct's 107–148, and +15.2 points on VQAv2, +15.9 on GQA
and +5.9 on TextVQA under the reasoning prompt. Trained on VQAv2 and GQA, GRPO answers in
8–10 tokens under the reasoning prompt on every test set; the habit pays on photos (+23 to
+24 points on its training datasets) and costs 6.7 points on ChartQA ([−9.7, −3.8]), where
reasoning pays. SFT on the same VQA questions, at 36 tokens on ChartQA, costs 7.9 there
([−10.8, −5.1]). Under the short prompt no model loses more than 0.6 points on the other
family's test sets, so training on one family left the other's short answers intact. Each
model still leads on its own data: under the reasoning prompt the VQA-trained model scores
78.3 on VQAv2 against 70.3 for the ChartQA-trained one, and 63.0 on ChartQA against 80.1.
One model trained on both families at once did not learn when to reason: it kept chart
reasoning for 1,750 updates, then dropped reasoning everywhere
([supplement](supplement.md#one-model-on-both-families)).
