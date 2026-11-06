# Open: one model for photos and charts

*The [report](https://guangboyu.github.io/vqa-rlvr/report/)'s open question.*

Does one model learn when to reason? Trained on photos and charts together, a 2B model kept its chart
reasoning until update 1,750, lost it by update 2,000, and ended at <!--n:ph.cq.mix.r-->65.9<!--/n-->% on ChartQA against
<!--n:cq.grpo.r-->80.1<!--/n-->% for the chart-only model, at <!--n:ph.cq.mix.tok-->11<!--/n--> tokens.

The cause is mixing, not training length. A chart-only model trained for the same <!--n:mix.updates-->2,844<!--/n--> updates keeps
<!--n:ctrl.tok-->179<!--/n-->-token answers and its dev accuracy (<!--n:ctrl.base.acc-->75.8<!--/n-->% → <!--n:ctrl.acc-->75.8<!--/n-->%). Routing prompts by domain is the next
test.

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
