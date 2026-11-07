# Reasoning pays: RL wins and keeps it

*One branch of the [report](https://guangboyu.github.io/vqa-rlvr/report/): whether RL beats SFT depends on what the task pays for.*

On ChartQA's human-written questions, GRPO from Instruct raises accuracy from <!--n:cq.base.r-->69.7<!--/n-->% to
<!--n:cq.grpo.r-->80.1<!--/n-->% under the reasoning prompt, a gain of <!--n:cq.gain-->10.4<!--/n--> points (<!--n:cq.audgain-->5.4<!--/n--> under the
audited judge). Its answers stay long (<!--n:cq.base.tok-->205<!--/n--> → <!--n:cq.grpo.tok-->162<!--/n--> tokens), and it ends
<!--n:cq.oversft-->11.3<!--/n--> points above SFT's best score on the same questions.

The answers were already within reach. On open-answer questions Instruct's greedy accuracy is
<!--n:reach.cq.instruct-->70.2<!--/n-->% while its pass@8 is <!--n:reach.cq.reach-->87.1<!--/n-->%, and GRPO stays level with Instruct at
pass@32 (<!--n:pk.base.r.32-->91.0<!--/n-->% → <!--n:pk.grpo.32-->91.5<!--/n-->%): it learns to pick what Instruct could already sample.

GRPO also beats SFT on reasoning traces. SFT on Instruct 2B's own correct reasoning reaches <!--n:cq.traces.self-->74.5<!--/n-->%,
and on Instruct 8B's <!--n:cq.traces.8b-->73.4<!--/n-->%, so RL leads by <!--n:cq.traces.self.lead.ci-->+5.6 [+3.5, +7.6]<!--/n--> and <!--n:cq.traces.8b.lead.ci-->+6.6 [+4.6, +8.7]<!--/n--> points.

ChartQA is the one chart source trained on in this study; a second document source is the next test.

## ChartQA: RL where reasoning pays

On VQAv2, GQA, CLEVR and TextVQA, reasoning lowered accuracy and correctness-only GRPO deleted
it ([form, not ability](form-not-ability.md#prompt-template--scorer-variant)). ChartQA's
human-written questions are the first candidate where it does not: in screening on training
questions, reasoning helped numeric answers by 5.6 points and cost text answers 5.5
([screening](../harness/evaluation.md#dataset-screening)). The 2B arms train on its 7,378-question pool
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
| Instruct → GRPO (ChartQA) − Instruct 2B | reasoning | ChartQA (human) | +8.43 [+7.28, +9.59] | +8.43 [+7.28, +9.59] | +3.98 [+3.09, +4.91] | +1.52 [+0.48, +2.64] |
| SFT → GRPO (ChartQA) − SFT 2B (ChartQA) | reasoning | ChartQA (human) | +15.14 [+13.96, +16.35] | +15.14 [+13.96, +16.35] | +5.90 [+4.88, +6.92] | +0.56 [-0.48, +1.60] |
| Instruct → short-prompt GRPO (ChartQA) − Instruct 2B | short | ChartQA (human) | +2.25 [+1.78, +2.76] | +2.25 [+1.78, +2.76] | -0.37 [-0.92, +0.17] | -3.52 [-4.80, -2.32] |
| SFT 2B (ChartQA) − Instruct 2B | short | ChartQA (human) | -1.70 [-2.79, -0.61] | -1.70 [-2.79, -0.61] | +4.26 [+3.16, +5.36] | +3.68 [+2.40, +5.04] |
| Instruct → GRPO (ChartQA, 1 epoch) − Instruct 2B | reasoning | ChartQA (human) | +13.35 [+11.94, +14.82] | +13.35 [+11.94, +14.82] | +4.48 [+3.33, +5.68] | +0.48 [-0.72, +1.76] |
| SFT → GRPO (ChartQA, 1 epoch) − SFT 2B (ChartQA) | reasoning | ChartQA (human) | +21.41 [+19.95, +22.82] | +21.41 [+19.95, +22.82] | +6.30 [+5.07, +7.50] | -1.28 [-2.48, -0.08] |
| Instruct → short-prompt GRPO (ChartQA, 1 epoch) − Instruct 2B | short | ChartQA (human) | +5.08 [+4.00, +6.20] | +5.08 [+4.00, +6.20] | -1.89 [-2.97, -0.82] | -8.96 [-10.72, -7.20] |
<!-- /report:chartqa-passk-deltas -->

The full epoch separates the two kinds of RL further. Trained with reasoning, GRPO from
Instruct gains 13.3 [+11.9, +14.8] points at pass@1 and ends level with Instruct at pass@32
(+0.5 [−0.7, +1.8]; the pilot was +1.5 [+0.5, +2.6]): the extra updates concentrate
probability on answers Instruct could already sample, without losing them. GRPO after SFT
gains 21.4 at pass@1 and ends 1.3 points below SFT at pass@32 ([−2.5, −0.1]). Short-prompt
GRPO narrows with every update: its pass@1 gain grows from 2.2 to 5.1 while its pass@32 gap
widens from −3.5 to −9.0 [−10.7, −7.2], below Instruct from k = 4 on. SFT does the reverse,
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
