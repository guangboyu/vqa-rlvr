# What RL adds to a vision-language model

When does reinforcement learning with a verifiable reward improve a vision-language model
beyond supervised fine-tuning on the same data? This repository trains Qwen3-VL at 2B and 8B
with SFT, with GRPO and a 0/1 correctness reward, and with SFT followed by GRPO, on photo,
chart and medical questions, and scores every model through one evaluation harness.

**Report:** [guangboyu.github.io/vqa-rlvr/report](https://guangboyu.github.io/vqa-rlvr/report/)
has the findings, figures and evaluation harness on one page.

## Results at a glance

Accuracy (%) on fixed test sets, best in each row in bold. Instruct is the released Qwen3-VL
Instruct model; the other columns were trained on each row's own training data, except the
unseen rows, which test models trained elsewhere.

<!-- report:overview -->
| Test set | Prompt | Instruct | SFT | GRPO | SFT → GRPO |
| --- | --- | ---: | ---: | ---: | ---: |
| ChartQA | reasoning | 69.7 | 66.2 | **80.1** | 79.8 |
| ChartQA | reasoning, audited judge | 77.2 | 69.8 | **82.6** | 82.0 |
| SLAKE, 2B | short | 49.8 | 74.8 | 65.6 | **76.2** |
| SLAKE, 8B | short | 58.1 | 77.2 | 68.3 | **78.2** |
| OmniMedVQA, held-out collections | short, normalized scorer | 74.6 | 85.0 | 79.1 | **85.6** |
| PathVQA, unseen (SLAKE 2B models) | short | **33.5** | 31.7 | 32.5 | 32.6 |
| TextVQA, unseen (VQAv2 + GQA models) | short | 80.5 | 80.1 | **81.0** | 80.9 |
| VizWiz, unseen (VQAv2 + GQA models) | short | 42.5 | 42.5 | **43.6** | 43.0 |
<!-- /report:overview -->

RL helps most where reasoning helps. On ChartQA, GRPO keeps its step-by-step reasoning and
gains 10.4 points, about ten points above every short-answer model, while its pass@32 stays
level with Instruct's. On medical images SFT teaches the domain's answers, and GRPO started
from SFT is the best model in every setting. On data that no model trained on, GRPO keeps a
lead of about one to two points over SFT.

RL's gains depend on the image. With a black image in place of the real one, GRPO answers no
more questions than Instruct in five of six settings, while SFT answers up to 24 points more.
On medical data GRPO also moves the model five to sixteen times less than SFT, measured by
forward KL from Instruct.

RL does not help everywhere. On short-answer photo questions (VQAv2, GQA) it drops the
reasoning and only matches SFT; on medical data, SFT beats GRPO started from Instruct; and RL
is overconfident, with a calibration error of 12 to 24 points against SFT's 2 to 9. These
results and the full test matrix are in the [supplement](docs/supplementary/supplement.md).

## Do 3D CT benchmarks need the scan?

The same harness measures how much of two 3D CT benchmarks can be answered without looking.
Accuracy (%) of 2B models trained with SFT; "without the image" means black images at test
time (and in training, except for DeepTumorVQA multiple choice):

| Benchmark | Text-only rule | Without the image | With the image | Published fine-tuned model |
| --- | ---: | ---: | ---: | ---: |
| DeepTumorVQA v2, multiple choice | 60.7 | 59.0 | 61.4 | 64.6 |
| DeepTumorVQA v2, free form | 57.4 | 57.6 | 60.3 | 63.4 |
| M3D-VQA, mean over question types | — | 70.2 | — | 75.8 |

DeepTumorVQA's answer options and answer frequencies give most answers away, and M3D-VQA's
correct option is A in nearly half of its questions. A debiased DeepTumorVQA, built here,
brings the text-only rule down to chance (35.2% against 33.7%). On it, GRPO and SFT reach
the same accuracy, and the one thing a 2B model learns from 2D views of the scan is whether
a lesion is present ([details](docs/supplementary/supplement.md#shortcut-audit-of-3d-ct-benchmarks)).

## Quick start

Python 3.12 and [uv](https://docs.astral.sh/uv/); the lockfile pins CUDA 12.9 builds of
PyTorch and vLLM. The 2B pipeline runs on one 24 GB GPU.

```bash
uv sync --frozen
uv run --frozen python prepare_data.py           # pinned sources, fixed subsets (about 13 GB)
uv run --frozen python -m scripts.check_data     # fails if a test image is in a training pool
uv run --frozen python evaluate.py --model Qwen/Qwen3-VL-2B-Instruct --label base2 \
  --tier smoke --prompts short reasoning
uv run --frozen python train_grpo.py --preset grpo_2b_main --hardware 24gb \
  --max-steps 2 --save-steps 1 --run-dir runs/grpo-smoke --no-wandb
```

The [tutorial](docs/tutorial.md) goes through data, evaluation, SFT and GRPO with the
expected output of each step, and [method](docs/supplementary/method.md) has the command
behind every model in the tables. All tables are generated from the saved evaluation records
by `python -m scripts.report`, and a test fails if a committed table drifts from them.

## Demo

A local app shows recorded answers and the results without a GPU, and compares the 2B
models side by side on a 24 GB GPU ([demo notes](demo/README.md)):

```bash
uv sync --frozen --group demo
(cd demo/frontend && npm ci && npm run build)
uv run --frozen uvicorn demo.backend.app:app --port 8000
```

![Recorded examples in the demo](demo/assets/gallery.png)

## Documentation

- [Results](docs/supplementary/results.md) and [supplement](docs/supplementary/supplement.md): every table, with paired 95% intervals, including the negative results.
- [Evaluation](docs/supplementary/evaluation.md) and [validation](docs/supplementary/validation.md): the harness, the dataset screen, and how the scorers were checked against the authors' code.
- [Hardware](docs/supplementary/hardware.md) and [monitoring](docs/supplementary/monitoring.md): GPU profiles, memory, resuming, and what to watch while RL trains.
- [History](docs/supplementary/history.md): the July 2026 version of this repository had a defective VQAv2 reward and 47 test images in its training pools; this page records both, the corrections, and the status of every earlier claim.

## Next steps

RL restricted to the CT questions that need the image (lesion presence); 3D perception tools
that return measurements to the language model; a second domain where reasoning pays; and
calibrated RL that is rewarded for abstaining when unsure.

## Acknowledgments

Built with Qwen, Hugging Face Transformers, TRL, PEFT and vLLM. Dataset and model licenses
apply to their assets; the code is under the [Apache 2.0 license](LICENSE).
