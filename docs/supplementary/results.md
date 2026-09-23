# Results

Every table on this page is generated from the stored evaluation records by
`python -m scripts.report`; a test fails if a committed copy drifts from the records.
Scores are percentages on the materialized test subsets with greedy decoding and one
training seed. The official column uses the dataset authors' evaluators; the normalized
scorer, which also defines the training reward, appears beside it.

## Prompt template × scorer variant

Official accuracy, with the normalized scorer in parentheses.

<!-- report:matrix -->
| Arm | Prompt | VQAv2 | GQA | CLEVR counting | TextVQA (far-OOD) |
| --- | --- | ---: | ---: | ---: | ---: |
| Instruct 2B | short | 77.87 (77.91) | 59.45 (59.46) | 97.67 (97.67) | 80.55 (80.56) |
| Instruct 2B | reasoning | 55.06 (64.68) | 36.13 (51.95) | 87.13 (87.13) | 69.60 (69.61) |
| SFT | short | 79.87 (79.89) | 62.35 (62.35) | 99.40 (99.40) | 79.83 (79.86) |
| SFT | reasoning | 67.99 (68.17) | 56.61 (57.00) | 92.10 (92.10) | 68.78 (68.79) |
| Instruct → GRPO | short | 78.02 (78.06) | 59.55 (59.55) | 98.27 (98.27) | 80.38 (80.39) |
| Instruct → GRPO | reasoning | 61.79 (69.01) | 44.90 (54.09) | 86.10 (86.10) | 74.23 (74.22) |
| SFT → GRPO | short | 79.85 (79.87) | 62.44 (62.44) | 99.40 (99.40) | 79.82 (79.84) |
| SFT → GRPO | reasoning | 69.18 (70.80) | 53.70 (54.86) | 97.80 (97.80) | 74.28 (74.28) |
| Instruct → gold-conditioned OPD | short | 77.79 (77.83) | 59.72 (59.72) | 97.73 (97.73) | 80.36 (80.36) |
| Instruct → gold-conditioned OPD | reasoning | 68.61 (69.23) | 51.35 (51.58) | 87.10 (87.10) | 72.65 (72.65) |
| Instruct 8B | short | 81.18 (81.22) | 61.66 (61.66) | 99.67 (99.67) | 83.98 (84.00) |
| Instruct 8B | reasoning | 63.67 (68.75) | 46.84 (52.27) | 96.97 (96.97) | 75.31 (75.31) |
| Instruct 8B → GRPO | short | 81.12 (81.16) | 61.78 (61.78) | 99.57 (99.57) | 84.07 (84.09) |
| Instruct 8B → GRPO | reasoning | 77.37 (78.63) | 59.43 (59.82) | 97.30 (97.30) | 80.78 (80.80) |
<!-- /report:matrix -->

## Retention under the reasoning prompt

Mean output tokens · share of responses that close an `<answer>` tag.

<!-- report:behavior -->
| Arm | VQAv2 | GQA | CLEVR counting | TextVQA (far-OOD) |
| --- | ---: | ---: | ---: | ---: |
| Instruct 2B | 148 · 95% | 123 · 97% | 186 · 98% | 106 · 95% |
| SFT | 21 · 40% | 8 · 20% | 62 · 98% | 33 · 47% |
| Instruct → GRPO | 141 · 96% | 119 · 98% | 197 · 97% | 101 · 98% |
| SFT → GRPO | 102 · 97% | 79 · 98% | 24 · 100% | 59 · 99% |
| Instruct → gold-conditioned OPD | 141 · 93% | 119 · 93% | 206 · 98% | 107 · 96% |
| Instruct 8B | 118 · 99% | 104 · 99% | 193 · 99% | 109 · 99% |
| Instruct 8B → GRPO | 9 · 100% | 9 · 100% | 83 · 99% | 12 · 100% |
<!-- /report:behavior -->

Responses cut off at the token budget · responses with an empty answer, same prompt.

<!-- report:failures -->
| Arm | VQAv2 | GQA | CLEVR counting | TextVQA (far-OOD) |
| --- | ---: | ---: | ---: | ---: |
| Instruct 2B | 3.7% · 0.0% | 2.3% · 0.0% | 1.8% · 0.0% | 1.5% · 0.0% |
| SFT | 1.8% · 0.8% | 0.5% · 0.3% | 1.3% · 0.0% | 2.0% · 0.1% |
| Instruct → GRPO | 3.7% · 0.0% | 1.8% · 0.0% | 2.7% · 0.0% | 1.4% · 0.0% |
| SFT → GRPO | 2.8% · 0.0% | 1.9% · 0.0% | 0.2% · 0.0% | 1.1% · 0.0% |
| Instruct → gold-conditioned OPD | 4.3% · 0.0% | 2.8% · 0.0% | 1.5% · 0.0% | 1.9% · 0.0% |
| Instruct 8B | 0.9% · 0.0% | 0.4% · 0.0% | 0.7% · 0.0% | 0.4% · 0.0% |
| Instruct 8B → GRPO | 0.0% · 0.0% | 0.0% · 0.0% | 0.9% · 0.0% | 0.1% · 0.0% |
<!-- /report:failures -->

## Capitalization under the reasoning prompt

The official evaluators keep case when annotators agree, so a capitalized answer can miss.
Lowercasing the answers before official scoring shows how much of the gap to the normalized
scorer is casing.

<!-- report:casing -->
| Arm | Dataset | Official | Official, lowercased answers | Normalized | Answers scored differently |
| --- | --- | ---: | ---: | ---: | ---: |
| Instruct 2B | VQAv2 | 55.06 | 64.64 | 64.68 | 481 |
| Instruct 2B | GQA | 36.13 | 50.80 | 51.95 | 1,989 |
| Instruct 2B | CLEVR counting | 87.13 | 87.13 | 87.13 | 0 |
| Instruct 2B | TextVQA | 69.60 | 69.60 | 69.61 | 1 |
| SFT | VQAv2 | 67.99 | 68.15 | 68.17 | 9 |
| SFT | GQA | 56.61 | 56.68 | 57.00 | 50 |
| SFT | CLEVR counting | 92.10 | 92.10 | 92.10 | 0 |
| SFT | TextVQA | 68.78 | 68.78 | 68.79 | 2 |
| Instruct → GRPO | VQAv2 | 61.79 | 68.99 | 69.01 | 361 |
| Instruct → GRPO | GQA | 44.90 | 53.43 | 54.09 | 1,157 |
| Instruct → GRPO | CLEVR counting | 86.10 | 86.10 | 86.10 | 0 |
| Instruct → GRPO | TextVQA | 74.23 | 74.23 | 74.22 | 1 |
| SFT → GRPO | VQAv2 | 69.18 | 70.76 | 70.80 | 81 |
| SFT → GRPO | GQA | 53.70 | 54.58 | 54.86 | 145 |
| SFT → GRPO | CLEVR counting | 97.80 | 97.80 | 97.80 | 0 |
| SFT → GRPO | TextVQA | 74.28 | 74.28 | 74.28 | 1 |
| Instruct → gold-conditioned OPD | VQAv2 | 68.61 | 69.21 | 69.23 | 31 |
| Instruct → gold-conditioned OPD | GQA | 51.35 | 51.43 | 51.58 | 29 |
| Instruct → gold-conditioned OPD | CLEVR counting | 87.10 | 87.10 | 87.10 | 0 |
| Instruct → gold-conditioned OPD | TextVQA | 72.65 | 72.65 | 72.65 | 2 |
| Instruct 8B | VQAv2 | 63.67 | 68.71 | 68.75 | 254 |
| Instruct 8B | GQA | 46.84 | 51.28 | 52.27 | 682 |
| Instruct 8B | CLEVR counting | 96.97 | 96.97 | 96.97 | 0 |
| Instruct 8B | TextVQA | 75.31 | 75.31 | 75.31 | 1 |
| Instruct 8B → GRPO | VQAv2 | 77.37 | 78.61 | 78.63 | 63 |
| Instruct 8B → GRPO | GQA | 59.43 | 59.64 | 59.82 | 49 |
| Instruct 8B → GRPO | CLEVR counting | 97.30 | 97.30 | 97.30 | 0 |
| Instruct 8B → GRPO | TextVQA | 80.78 | 80.78 | 80.80 | 2 |
<!-- /report:casing -->

## Image interventions

Official score with the original image → with a black or swapped image, on the first 128
test questions, and the share of normalized answers that changed. The references stay those
of the original image, so the changed-image score measures agreement with them, not
accuracy on the new image.

<!-- report:interventions -->
| Arm | Prompt | Image | VQAv2 | GQA | CLEVR counting | TextVQA (far-OOD) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Instruct 2B | short | black | 68.8 → 31.2 (74%) | 57.8 → 32.8 (61%) | 97.7 → 0.0 (100%) | 84.5 → 4.8 (97%) |
| Instruct 2B | short | swapped | 68.8 → 32.1 (75%) | 57.8 → 32.8 (63%) | 97.7 → 17.2 (84%) | 84.5 → 2.0 (99%) |
| Instruct 2B | reasoning | black | 47.3 → 10.5 (92%) | 40.6 → 8.6 (89%) | 83.6 → 0.0 (100%) | 75.0 → 1.2 (99%) |
| Instruct 2B | reasoning | swapped | 47.3 → 23.1 (80%) | 40.6 → 18.8 (77%) | 83.6 → 17.2 (84%) | 75.0 → 1.8 (99%) |
| SFT | short | black | 73.4 → 35.5 (74%) | 57.0 → 39.1 (64%) | 98.4 → 0.0 (100%) | 83.5 → 4.8 (97%) |
| SFT | short | swapped | 73.4 → 33.0 (74%) | 57.0 → 29.7 (71%) | 98.4 → 17.2 (83%) | 83.5 → 2.7 (98%) |
| SFT | reasoning | black | 58.7 → 26.0 (78%) | 57.0 → 24.2 (69%) | 89.8 → 0.0 (100%) | 74.4 → 2.7 (98%) |
| SFT | reasoning | swapped | 58.7 → 26.3 (77%) | 57.0 → 25.8 (67%) | 89.8 → 15.6 (85%) | 74.4 → 0.0 (99%) |
| Instruct → GRPO | short | black | 69.0 → 31.1 (74%) | 57.8 → 34.4 (60%) | 98.4 → 0.0 (100%) | 83.7 → 4.8 (97%) |
| Instruct → GRPO | short | swapped | 69.0 → 31.6 (74%) | 57.8 → 32.0 (64%) | 98.4 → 17.2 (83%) | 83.7 → 2.0 (99%) |
| Instruct → GRPO | reasoning | black | 57.6 → 17.7 (84%) | 47.7 → 13.3 (83%) | 85.9 → 0.0 (100%) | 80.0 → 2.0 (98%) |
| Instruct → GRPO | reasoning | swapped | 57.6 → 26.5 (77%) | 47.7 → 22.7 (73%) | 85.9 → 14.1 (87%) | 80.0 → 1.8 (98%) |
| SFT → GRPO | short | black | 74.1 → 35.1 (75%) | 57.0 → 37.5 (62%) | 98.4 → 0.0 (100%) | 83.0 → 4.8 (97%) |
| SFT → GRPO | short | swapped | 74.1 → 32.3 (75%) | 57.0 → 30.5 (72%) | 98.4 → 17.2 (83%) | 83.0 → 2.5 (98%) |
| SFT → GRPO | reasoning | black | 57.5 → 26.1 (80%) | 53.9 → 28.1 (65%) | 97.7 → 0.0 (100%) | 78.2 → 2.0 (98%) |
| SFT → GRPO | reasoning | swapped | 57.5 → 27.4 (80%) | 53.9 → 21.1 (76%) | 97.7 → 16.4 (84%) | 78.2 → 1.0 (99%) |
<!-- /report:interventions -->

## What each RL run trained on

The OPD run logs the same correctness reward for diagnosis; its learning signal is the
teacher's token scores, so its reward columns do not drive its updates.

<!-- report:exposure -->
| Run | Updates | Questions | Responses | Groups with varied rewards (%) | Updates without reward variation | Mean reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `grpo2-base` | 300 | 1,200 | 9,600 | 38.67 | 43 | 0.599 |
| `grpo2-sft` | 300 | 1,200 | 9,600 | 49.00 | 16 | 0.581 |
| `grpo8-base` | 300 | 1,200 | 9,600 | 23.33 | 110 | 0.646 |
| `opd2-self` | 300 | 1,200 | 9,600 | 49.25 | 21 | 0.588 |
<!-- /report:exposure -->

Training signals. The reward (mean of the first and of the last ten updates) and the share
of prompt groups whose samples all scored the same (no gradient) come from every update's
saved draws; entropy and completion length are the trainer's logs (every ten updates),
first → last. KL is absent because these runs used no KL penalty.

<!-- report:diagnostics -->
| Run | Reward (first → last 10 updates) | Dead groups (mean) | Entropy | Clip fraction (mean) | Completion tokens | Distinct prompts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `grpo2-base` | 0.62 → 0.61 | 0.613 | 0.73 → 0.63 | 0.000 | 154 → 139 | 1,200 |
| `grpo2-sft` | 0.27 → 0.63 | 0.510 | 1.08 → 0.90 | 0.000 | 30 → 110 | 1,200 |
| `grpo8-base` | 0.62 → 0.67 | 0.767 | 0.40 → 0.12 | 0.000 | 133 → 13 | 1,200 |
| `opd2-self` | 0.64 → 0.61 | 0.507 | 0.75 → 0.68 | 0.000 | 150 → 132 | 1,200 |
<!-- /report:diagnostics -->

## Learning curves

Each point evaluates a saved adapter on the dev tier through vLLM's LoRA serving rather
than a merged copy; on 500 VQAv2 dev questions the two agreed on 496 answers.

<!-- report:curves -->
No checkpoint dev evaluations have been recorded yet.
<!-- /report:curves -->

## History: July runs

These runs used an inert VQAv2 reward and overlapping image pools; they are kept as
labeled history and are not comparable with the tables above. Official (normalized),
where the official rescoring exists.

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
