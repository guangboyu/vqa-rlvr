# What RL adds to a vision-language model's answers

A hands-on study of short-answer SFT, GRPO with a verifiable reward, and on-policy
distillation for Qwen3-VL at 2B and 8B, on VQAv2, GQA, CLEVR counting, and far-OOD
TextVQA. Every model is evaluated on the same fixed test questions under two prompt
templates, a short answer and step-by-step reasoning, and scored twice, with the dataset
authors' evaluator and with a normalized matcher. Reading the two prompts side by side
separates what each training method changes: **with a correctness reward, GRPO turns the
reasoning prompt into a direct answer and brings it up to the short-answer ceiling, and
short-answer SFT is what raises that ceiling on the training distribution.** On ChartQA,
where reasoning pays, the same recipe keeps the reasoning and improves it.

The 2B pipeline runs on one 24 GB GPU; separate 80/96 GB profiles run the 8B recipes. The
recorded demo runs without a GPU.

## Results

Official accuracy (%), with the normalized scorer in parentheses, on the full test subsets:
5,000 VQAv2, 12,578 GQA, 3,000 CLEVR counting, and 5,000 TextVQA questions (its whole
validation split). Greedy decoding, one training seed per arm. The short prompt asks for a
single word or phrase; the reasoning prompt asks for step-by-step thinking and a tagged final
answer.

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

**1. RL removes reasoning that does not pay, and the reasoning prompt reaches the
short-answer ceiling.** Under the reasoning prompt, 1,000 GRPO updates from Instruct raise
official accuracy by 23.2 points on VQAv2, 24.1 on GQA, and 10.2 on far-OOD TextVQA at 2B,
and by 18.2, 15.8, and 9.3 at 8B. The policy gets there by deleting the reasoning: mean
responses fall from 148 tokens to 8 at 2B (118 to 8 at 8B) while the answer tag stays in
100% of them. In-domain, the reasoning prompt now matches the short prompt of the same model
(2B VQAv2 +0.3 points, 95% interval [−0.3, +1.0]; GQA +0.2). The corrected campaign's
300-update runs stopped before this point (2B +6.7 on VQAv2). Paying for reasoning keeps it:
a 0.2 format reward holds 8B responses near 130 tokens, at 9.7 points below
correctness-only GRPO of the same length on VQAv2 and 9.6 on GQA ([results](docs/supplementary/results.md#recipe-experiments)).

**2. RL lifts the short-answer ceiling a little; SFT lifts it more in-domain.** GRPO trained on the short prompt from Instruct adds 0.8 points
on VQAv2 and 0.9 on GQA at 2B (0.7 and 0.8 at 8B); one epoch of short-answer SFT adds 2.0
and 2.9 at 2B (2.3 and 2.2 at 8B). On far-OOD TextVQA at 2B the order reverses: short-prompt GRPO beats
SFT by 0.9 points (interval [+0.2, +1.6]), while SFT itself moves TextVQA by −0.5; at 8B
both add about 1 point there. Most of SFT's in-domain lead is the amount of data: trained on
exactly the 4,000 questions short-prompt GRPO drew, SFT leads it by 0.2 points on VQAv2
(interval [−0.3, +0.7]) and 0.5 on GQA, and GRPO stays 0.9 ahead on TextVQA
([results](docs/supplementary/results.md#sft-on-exactly-the-rl-questions)). Run on the whole
55,438-question pool (13,859 updates), short-prompt GRPO reaches SFT's level, +2.0 and +2.9
over Instruct (within 0.1 points of SFT), with no further TextVQA gain and a wider pass@32
deficit ([results](docs/supplementary/results.md#short-prompt-rl-on-the-full-pool)).

**3. RL after SFT keeps the SFT ceiling.** GRPO from the SFT model keeps short-prompt
accuracy within 0.5 points of SFT on VQAv2 and GQA at both sizes, and adds up to 0.8 on
TextVQA. Where SFT hurt, the rollout prompt decides: at 8B, SFT costs 10.8 points of CLEVR
counting under the short prompt (the casual counts in VQAv2 and GQA answers); GRPO with the
reasoning prompt wins back 8.2 of them, while GRPO with the short prompt loses 6.7 more.

**4. The standard RL recipe knobs do not move these results.** Against a fixed 2B baseline,
a pool filtered to questions with a learning signal, refilling groups whose answers all
scored the same, 16 instead of 4 questions per update at the same number of questions, and
a 3× learning rate leave short-prompt accuracy on VQAv2, GQA, and TextVQA within 0.35
points; a 10× learning rate costs 0.7–2.2 points there and 11 on CLEVR counting. One engineering change matters for correctness
rather than accuracy: rollouts sampled from LoRA-merged bf16 weights come from a lagging copy
of the trained policy (a mean log-probability gap of up to 0.2 nats per token, 1.4 on the
near-tied token before `</answer>`); serving the adapter unmerged brings the gap under 0.01
and is part of the final recipe ([method](docs/supplementary/method.md#sampler-and-trained-policy)).

**5. Much of what exact match credits to training is answer form.** An LLM judge (Anthropic's
Haiku 4.5) rescored every exact-match miss of the headline cells, and an image-aware audit by a
stronger model measured how often its acceptances hold. Under that audited scorer, GRPO from
Instruct gains 4.4 points on VQAv2 under the reasoning prompt at 2B instead of 23.2 (4.8
instead of 24.1 on GQA), and the reasoning prompt still costs Instruct 3.8 points instead of
22.8. Short-prompt GRPO's in-domain gains over Instruct (+0.2 and +0.4 points) fall inside
their intervals, SFT's halve to +1.2 and +1.5, and short-prompt GRPO's TextVQA lead over SFT
holds (+0.8, interval [+0.2, +1.5]) ([results](docs/supplementary/results.md#exact-match-misses-rescored-by-an-llm-judge)).
The same lead appears on a second far-OOD set, VizWiz: short-prompt GRPO beats SFT by 1.0 point
(interval [+0.5, +1.6]) and SFT on its own 4,000 questions by 1.2
([results](docs/supplementary/results.md#extra-ood-sets)).

**6. RL sharpens what the model already samples; it does not add answers.** On the first 1,000
test questions of each dataset, sampled 32 times at temperature 1, short-prompt GRPO raises
pass@1 over Instruct by 2.4–3.0 points but trails it by 1.9–7.6 at pass@32, and reasoning-prompt
GRPO turns a pass@1 gain of 10 to 25 points into −1 to −6 at k = 32. One epoch of SFT does the
reverse: pass@1 about level, pass@32 up 1.7–4.6 points
([results](docs/supplementary/results.md#held-out-passk-before-and-after-rl)).

**7. Where reasoning pays, RL keeps it, and more data keeps paying.** ChartQA's human-written
questions are the first dataset where reasoning helped in screening (numeric answers, +5.6
points). There, GRPO at 2B keeps the reasoning (160–200 tokens) and raises reasoning-prompt
accuracy from 69.7 to 76.4 in 300 updates and to 80.1 after one epoch of the pool (+10.4,
interval [+8.2, +12.6]), ten points above the best short-answer arm, while short-prompt GRPO
gains 1.7. Under the audited judge half of the reasoning gain remains (+5.5, interval [+3.5,
+7.4]), against a fifth on VQAv2. Unlike finding 6, reasoning-trained GRPO keeps Instruct's
coverage at k = 32 (+0.5, interval [−0.7, +1.7]); short-prompt GRPO on the same data falls
9.0 points behind. One seed, one dataset
([results](docs/supplementary/results.md#chartqa-rl-where-reasoning-pays)).

**8. The habit RL teaches carries across datasets.** GRPO trained on ChartQA keeps a shorter
chain of reasoning on photos and gains 15–16 points on VQAv2 and GQA under the reasoning
prompt; GRPO trained on VQAv2 and GQA answers in under ten tokens everywhere and loses 6.7
points on ChartQA. Short answers move by at most 0.6 points across the two families
([results](docs/supplementary/results.md#transfer-between-training-data)).

Differences below about 0.5 points under the reasoning prompt are within the run-to-run
noise of greedy vLLM decoding (three repeats of Instruct 2B span 0.4 points) and are not
claimed.

The scorer matters under the reasoning prompt. The authors' evaluators compare some
answers case-sensitively (VQAv2 when all annotators agree, GQA always), and for Instruct
2B lowercasing the answers closes nearly all of the official-to-normalized gap: 9.6 of 9.6
points on VQAv2 and 14.7 of 15.8 on GQA. Both numbers are always reported side by side.

The [results](docs/supplementary/results.md) page has the full tables: response length and
answer tags, truncation, casing, image interventions, what each RL run trained on, training
diagnostics, and a learnability pre-check of the RL pool.

## What was trained

| Arm | Supervision |
|---|---|
| SFT | one epoch on 39,458 VQAv2 and GQA short answers |
| Instruct → GRPO, SFT → GRPO | 1,000 GRPO updates under the reasoning prompt, correctness reward: the tagged answer, scored by the normalized matcher |
| Instruct → short-prompt GRPO, SFT → short-prompt GRPO | the same 1,000 updates under the short prompt, rewarding the whole answer (64-token budget) |
| Instruct → gold-conditioned OPD (2B) | 1,000 updates of token-level feedback from the frozen Instruct model, shown the reference answer |

Every RL arm draws its questions in the same seeded order from a 55,438-question pool,
four questions and eight sampled responses per update, learning rate 1e-5, no KL penalty.
Rollouts come from vLLM serving the current LoRA adapter unmerged, so the sampled policy is
the trained one. SFT → GRPO adds the SFT epoch on top, so it measures the value of SFT
before the same RL. The recipe was frozen after one-change experiments against a fixed
baseline ([results](docs/supplementary/results.md#recipe-experiments)). No dev or test image
appears in any training pool; the split files in `data/splits/` list every row and
`check_data` verifies it. The [method](docs/supplementary/method.md) has the full recipe and
the commands that reproduce each arm.

## Evaluation

| Dataset | Test questions | Official scorer |
|---|---:|---|
| VQAv2 | 5,000 | VQA ten-annotator accuracy with its answer processing |
| GQA | 12,578 | exact answer match |
| CLEVR counting | 3,000 | exact count |
| TextVQA (far-OOD) | 5,000 (all of validation) | MMF/EvalAI answer processing, ten-annotator accuracy |

These are fixed subsets materialized by this repository, not benchmark-server
submissions; GQA's consistency and grounding metrics are outside an accuracy score. The
official scorers reproduce the authors' evaluator code on 626,074 saved predictions from
89 runs with no per-question mismatch ([validation](docs/supplementary/validation.md)).
The [evaluation](docs/supplementary/evaluation.md) page describes the harness in the terms
public harnesses use and lists where it departs from their practice. Run side by side with
lmms-eval on the same model, the scorers agree exactly and GQA matches within 0.09 points;
the 1.15-point TextVQA gap comes from lmms-eval capitalizing each question.

`evaluate.py` runs one model under a declared protocol (dataset, role, tier, prompt,
decoding, image mode) and writes a record with both scorer variants and every prediction.
A cell that already has a record is skipped, and an interrupted one resumes. Every table
in this repository, and the dashboard, is generated from those records:

```bash
uv run --frozen python -m scripts.report
```

## Try it

Browse recorded examples and the results dashboard, no GPU needed:

```bash
uv sync --frozen --group demo
cd demo/frontend && npm ci && npm run build && cd ../..
uv run --frozen uvicorn demo.backend.app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. The [demo notes](demo/README.md) cover live inference with
the three 2B models on a 24 GB GPU.

![Recorded VQA examples](demo/assets/gallery.png)

The [tutorial](docs/tutorial.md) runs the 2B pipeline on one 24 GB GPU: data, split check,
evaluation, the learnability pre-check, two-step SFT and GRPO runs with their dev-tier
learning curves, and the tables. The [hardware notes](docs/supplementary/hardware.md)
cover the 80/96 GB profiles, resumption, and measured memory; [monitoring](docs/supplementary/monitoring.md)
covers what to watch while an RL run trains and how to plot it.

## History and corrections

The first campaign, in July 2026, trained GRPO with a VQAv2 reward that paid nothing on
single-reference training questions, and its training pools shared 47 images with the
VQAv2 test subset. The corrected campaign of September 13 fixed both, with 300-update RL
runs sampled from merged weights; the final rerun above replaced it under the frozen recipe.
The [history](docs/supplementary/history.md) keeps both earlier campaigns as labeled
history, the full erratum for the reward defect, and a ledger of every published claim with
its status: replicated, qualified, not replicated, retracted, or open.

The adapters published in July come from those runs. All four were trained on the July
pools, and the two GRPO adapters also carry the reward defect:

| Adapter | Download |
|---|---|
| 2B SFT | [vqa-rlvr-sft-2b](https://huggingface.co/omnifish123/vqa-rlvr-sft-2b) |
| 8B SFT | [vqa-rlvr-sft-8b](https://huggingface.co/omnifish123/vqa-rlvr-sft-8b) |
| 2B GRPO from Instruct | [vqa-rlvr-grpo-2b-main-base](https://huggingface.co/omnifish123/vqa-rlvr-grpo-2b-main-base) |
| 2B SFT then GRPO | [vqa-rlvr-grpo-2b-main-sft](https://huggingface.co/omnifish123/vqa-rlvr-grpo-2b-main-sft) |

## Repository map

- `train_sft.py`, `train_grpo.py`, `train_opd.py`: training entry points on shared
  plumbing in `vqar/training.py` and `vqar/rl.py`; presets and hardware profiles in
  `vqar/config.py`.
- `vqar/registry.py`: datasets, their roles and scorers, and prompt templates, registered
  once.
- `vqar/eval/`: the evaluation harness, records, the experiment matrix, and the report
  builder.
- `evaluate.py`, `scripts/`: evaluation, data checks, the learnability pre-check,
  reports, and the gallery.
- `data/splits/`, `experiments.toml`, `results/records/`: the split files, the declared
  arms and protocols, and the evaluation records behind every table.
- `demo/`: recorded examples, the results dashboard, and optional live comparison.
- `tests/`: CPU tests; GitHub Actions runs them without PyTorch.

## Acknowledgments

Built with Qwen, Hugging Face Transformers, TRL, PEFT, and vLLM. Dataset and model
licenses apply to their respective assets. See [LICENSE](LICENSE) for this code.
