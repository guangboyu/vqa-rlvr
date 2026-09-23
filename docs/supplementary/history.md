# History: the July runs, the reward erratum, and the claims ledger

The project's first campaign ran in July 2026. Two defects in it were found later: an
inert VQAv2 training reward and training pools that shared images with the evaluation
sets. The corrected campaign of September 2026 fixed both and supplies every headline
number. This page keeps the July results as labeled history, documents the reward defect
in full because the July adapters on the Hub were trained with it, and lists every
published claim with its current status.

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

## Claims ledger

Every claim the project published, with its status after the corrected campaign. "July
README" is the public README of July 2026; "research branch" claims were written on an
unpublished branch in July–August. Replicated: holds in the corrected campaign. Qualified:
holds only under a stated condition. Not replicated: the corrected campaign did not
reproduce it. Retracted: contradicted by later evidence. Open: not re-tested, or the cause is
not yet known. Confirmed: a reported defect that the tests reproduce.

| Claim | Where | Status | Evidence |
|---|---|---|---|
| GRPO raises 2B reasoning-prompt accuracy (VQAv2 +4.6, GQA +2.0, TextVQA +2.5, normalized) | July README | Replicated, qualified | Corrected Instruct → GRPO, reasoning prompt: +4.3, +2.1, +4.6 normalized; +6.7, +8.8, +4.6 official. Short-prompt accuracy moves by at most 0.6 points, and every arm scores higher with the short prompt. |
| Correctness-only GRPO deletes the chain of thought at 2B | July README; research branch | Not replicated at 2B; replicated at 8B; cause open | Corrected 2B run: VQAv2 reasoning-prompt output 148 → 141 tokens, answer tags 95% → 96%. Corrected 8B run: 118 → 9 tokens. Candidate causes of the 2B difference (the multimodal padding fix, the held-out image pool, the vLLM memory fraction, the pinned revision) are untested. |
| The reasoning "cost" is a metric artifact: an LLM judge reverses the ranking (81.8 vs 81.1) | July README | Retracted | A pre-registered follow-up on an independent judged sample found no reversal: VQAv2 gap −0.23 points, 95% CI [−1.84, +1.34]; pooled −0.12, CI [−1.27, +0.92]. Exact match undercounts, but equally for both arms. |
| Training data are contamination-free | July README | Retracted | July pools shared 47 images (53 questions) with VQAv2 evaluation, and the SFT dev slice shared 317 of its 487 images with SFT training. The corrected split files exclude every dev and test image from every training pool by decoded-pixel hash. |
| CLEVR-only RL gives the largest out-of-domain gain (TextVQA +5.3) | July README | Open | A single July run; not re-run in the corrected pipeline. |
| Short-answer SFT removes visible reasoning, and GRPO revives it | July README | Qualified | The July emission count (0/3000) looked for literal `<think>` tags, which the Instruct model never emits. In the corrected campaign SFT → GRPO restores length and tags under the reasoning prompt (VQAv2 21 → 102 tokens, tags 40% → 97%). |
| The harness matches lmms-eval (GQA 59.5 vs 58.7 ± 1.6) | July README | Qualified | A sanity check on 1,000 GQA questions with the historical normalized scorer. The official scorers now match the dataset authors' code on 626,074 saved predictions. |
| VQAv2 training reward was inert (erratum above) | July README | Confirmed, fixed | Regression tests; the corrected campaign trained with the fixed reward. |
| The learning signal type decides whether reasoning survives: exact-match GRPO deletes, dense signals keep ~99% | Research branch | Not replicated at 2B; replicated at 8B | See the deletion row. |
| Signal grounding beats teacher capacity; the gold-conditioned self-teacher has no capability ceiling | Research branch | Retracted | The teacher comparison confounds capacity, grounding, and signal density. |
| Short-prompt accuracy is flat across signal types and scales | Research branch | Replicated, qualified | After GRPO or OPD, short-prompt official accuracy moves by at most 0.6 points at 2B and 0.13 at 8B. SFT does move it: +2.0 VQAv2, +2.9 GQA. |
| Empty reasoning is equivalent to full reasoning (pilot P2) | Research branch | Retracted | The equivalence test fails: −2.32 points, 90% CI [−5.48, +0.84], n = 487. Swapped reasoning changes answers in 63.6% of the 143 no-leak pairs, so the answers depend on the reasoning text. |
| A static predictor of RL gains fails (pilot P3) | Research branch | Open | The reanalysis is inconclusive under the corrected retention proxy. |
| Public RLVR checkpoints show no grounding drift (ten models, 2,300 probes) | Research branch | Qualified | No degradation was detected under that short-prompt protocol; image sensitivity alone does not establish correct grounding. |

The July judge study behind the retracted reversal claim is `scripts/judge_study.py`; its
output, `results/judge_study.json`, appears in the dashboard's history view.
