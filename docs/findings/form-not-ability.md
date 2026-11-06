# Reasoning doesn't pay: RL changes form, not ability

*One branch of the [report](https://guangboyu.github.io/vqa-rlvr/report/): whether RL beats SFT depends on what the task pays for.*

On VQAv2, GRPO under the reasoning prompt goes from <!--n:ph.vqav2.base.r-->55.1<!--/n-->% at <!--n:ph.vqav2.base.tok-->148<!--/n--> tokens
to <!--n:ph.vqav2.grpor.r-->78.3<!--/n-->% at <!--n:ph.vqav2.grpor.tok-->8<!--/n--> tokens, the level of Instruct's short answers
(<!--n:ph.vqav2.base.s-->77.9<!--/n-->%). The exact-match gain of <!--n:ph.vqav2.gain-->23.2<!--/n--> points becomes <!--n:ph.vqav2.audgain-->4.4<!--/n--> under the
audited judge: a correctness reward that does not pay for reasoning removes it, and most of the
gain is the answer getting shorter.

Short-prompt RL meets SFT on the same data rather than passing it: <!--n:ph.vqav2.grpo.s-->78.6<!--/n-->% against
<!--n:ph.vqav2.sft.s-->79.9<!--/n-->% on VQAv2 and <!--n:ph.gqa.grpo.s-->60.3<!--/n-->% against <!--n:ph.gqa.sft.s-->62.3<!--/n-->% on GQA. SFT's lead comes mostly
from seeing more questions; see [SFT on exactly the RL questions](what-the-gain-is.md#sft-on-exactly-the-rl-questions).

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
[monitoring](../harness/monitoring.md) page.

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
with reasoning, the extra data kept paying ([ChartQA](reasoning-pays.md#chartqa-rl-where-reasoning-pays)).
