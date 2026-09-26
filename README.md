# What RL adds to a vision-language model's answers

A hands-on study of short-answer SFT, GRPO with a verifiable reward, and on-policy
distillation for Qwen3-VL at 2B and 8B, on VQAv2, GQA, CLEVR counting, and far-OOD
TextVQA. Every model is evaluated on the same fixed test questions under two prompt
templates, a short answer and step-by-step reasoning, and scored twice, with the dataset
authors' evaluator and with a normalized matcher. Reading the two prompts side by side
separates what each training method changes: **RL wins back accuracy that the reasoning
prompt costs, and only SFT raises the short-answer ceiling.**

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
| SFT → GRPO | short | — | — | — | — |
| SFT → GRPO | reasoning | — | — | — | — |
| Instruct → short-prompt GRPO | short | 78.62 (78.66) | 60.34 (60.35) | 99.50 (99.50) | 81.01 (81.02) |
| Instruct → short-prompt GRPO | reasoning | 61.85 (69.81) | 42.89 (54.01) | 86.50 (86.50) | 72.34 (72.35) |
| SFT → short-prompt GRPO | short | 79.83 (79.85) | 62.50 (62.50) | 99.50 (99.50) | 80.89 (80.90) |
| SFT → short-prompt GRPO | reasoning | 68.05 (68.31) | 56.88 (57.26) | 88.97 (88.97) | 70.72 (70.73) |
| Instruct → gold-conditioned OPD | short | — | — | — | — |
| Instruct → gold-conditioned OPD | reasoning | — | — | — | — |
| Instruct 8B | short | 81.18 (81.22) | 61.66 (61.66) | 99.67 (99.67) | 83.49 (83.51) |
| Instruct 8B | reasoning | 63.67 (68.75) | 46.84 (52.27) | 96.97 (96.97) | 74.00 (74.00) |
| SFT 8B | short | 83.44 (83.46) | 63.86 (63.86) | 88.83 (88.83) | 84.73 (84.76) |
| SFT 8B | reasoning | 68.36 (69.26) | 48.71 (49.18) | 96.70 (96.70) | 75.16 (75.18) |
| Instruct 8B → GRPO | short | — | — | — | — |
| Instruct 8B → GRPO | reasoning | — | — | — | — |
| SFT 8B → GRPO | short | — | — | — | — |
| SFT 8B → GRPO | reasoning | — | — | — | — |
| Instruct 8B → short-prompt GRPO | short | — | — | — | — |
| Instruct 8B → short-prompt GRPO | reasoning | — | — | — | — |
| SFT 8B → short-prompt GRPO | short | — | — | — | — |
| SFT 8B → short-prompt GRPO | reasoning | — | — | — | — |
<!-- /report:matrix -->

<!-- TODO(final rerun): findings 1-3 and the scorer paragraph quote the corrected campaign (now in docs/supplementary/history.md); rewrite them for the final-rerun arms in the table above. The intro's bold summary may need the same. -->
**1. RL recovers reasoning-prompt accuracy.** Under the reasoning prompt, 300 GRPO updates
from Instruct raise official accuracy by 6.7 points on VQAv2, 8.8 on GQA, and 4.3 on
TextVQA at 2B, and by 13.7, 12.6, and 6.5 at 8B. The normalized scorer confirms gains
beyond answer formatting (2B: +4.3, +2.1, +4.3; 8B: +9.9, +7.6, +6.5). At 8B the
reasoning prompt ends within 3.8 points of the short prompt on VQAv2, down from 17.5.

**2. No arm passes its own short-answer ceiling.** In all 28 arm and dataset pairs, the
short prompt scores higher than the reasoning prompt, under both scorers. GRPO and OPD
leave short-prompt accuracy where it was: within 0.26 points on VQAv2 and GQA, and 0.6 on
any dataset (official).

**3. SFT is what moves the ceiling.** One epoch of short-answer SFT raises short-prompt
official accuracy by 2.0 points on VQAv2 and 2.9 on GQA, and GRPO after SFT keeps that
gain (within 0.1 points) while adding reasoning-prompt accuracy on CLEVR counting (+5.7)
and TextVQA (+5.7), under both scorers.

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
| Instruct → GRPO, SFT → GRPO | 300 GRPO updates with the correctness reward: the tagged answer, scored by the normalized matcher |
| Instruct → gold-conditioned OPD | 300 updates of token-level feedback from the frozen Instruct model, shown the reference answer |

All RL arms drew exactly the same 1,200 questions in the same order from a 55,438-question
pool, eight sampled responses per question. SFT → GRPO adds the SFT epoch on top, so it
measures the value of SFT before the same RL. No dev or test image appears in any training
pool; the split files in `data/splits/` list every row and `check_data` verifies it. The
[method](docs/supplementary/method.md) has the full recipe and the commands that
reproduce each arm.

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

<!-- TODO(final rerun): this section says the corrected campaign supplies every number above; the final rerun now does. -->
The first campaign, in July 2026, trained GRPO with a VQAv2 reward that paid nothing on
single-reference training questions, and its training pools shared 47 images with the
VQAv2 test subset. The corrected campaign fixes both and supplies every number above. The
[history](docs/supplementary/history.md) keeps the July results as labeled history, the
full erratum for the reward defect, and a ledger of every published claim with its status
after the corrected campaign: replicated, qualified, not replicated, retracted, or open.

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
