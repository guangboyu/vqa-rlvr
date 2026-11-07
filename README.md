# What RL adds to a vision-language model

When does reinforcement learning with a verifiable reward improve a vision-language model beyond
supervised fine-tuning on the same data? This repository trains Qwen3-VL at 2B and 8B with SFT, with
GRPO and a 0/1 correctness reward, and with SFT followed by GRPO, and scores every model through one
evaluation harness.

**Whether RL beats SFT depends on what the task pays for.** RL mostly sharpens what the model can
already sample; where the answer is out of reach, SFT has to go first.

Across <!--n:cat.benchmarks-->18<!--/n--> public benchmarks, reasoning beats short answers on only <!--n:cat.reasoning.pays-->3<!--/n-->.

**Report:** [guangboyu.github.io/vqa-rlvr/report](https://guangboyu.github.io/vqa-rlvr/report/)

## Results

- **Where reasoning pays, RL wins and keeps it.** On ChartQA, GRPO raises accuracy from <!--n:cq.base.r-->69.7<!--/n-->% to
  <!--n:cq.grpo.r-->80.1<!--/n-->%, <!--n:cq.oversft-->11.3<!--/n--> points above SFT on short answers and <!--n:cq.traces.self.lead-->5.6<!--/n--> above SFT on reasoning traces,
  with answers still <!--n:cq.grpo.tok-->162<!--/n--> tokens long.
  ([evidence](docs/findings/reasoning-pays.md))
- **Where it doesn't, RL changes form, not ability.** On VQAv2 under the reasoning prompt, answers go
  from <!--n:ph.vqav2.base.tok-->148<!--/n--> tokens to <!--n:ph.vqav2.grpor.tok-->8<!--/n-->, and accuracy ends at <!--n:ph.vqav2.grpor.r-->78.3<!--/n-->%, the level of
  short answers (<!--n:ph.vqav2.base.s-->77.9<!--/n-->%). ([evidence](docs/findings/form-not-ability.md))
- **RL's gains come from the image.** At 8B on SLAKE, <!--n:ba.slk8.sft.blackgain-->10.4<!--/n--> of SFT's <!--n:ba.slk8.sft.gain-->19.1<!--/n-->-point gain
  survives a black image, against <!--n:ba.slk8.grpo.blackgain-->0.4<!--/n--> of RL's <!--n:ba.slk8.grpo.gain-->10.3<!--/n-->.
  ([evidence](docs/findings/what-the-gain-is.md))

On medical images, GRPO after SFT is the best model in every setting (<!--n:med.slk2.sftgrpo-->76.2<!--/n-->% on SLAKE at 2B).
The answers are out of reach at first, so SFT goes first: <!--n:med.slk2.sft-->74.8<!--/n-->% for SFT against
<!--n:med.slk2.grpo-->65.6<!--/n-->% for GRPO from Instruct. ([evidence](docs/findings/sft-then-rl.md))

Accuracy (%) on fixed test sets, best in each row in bold. Instruct is the released Qwen3-VL Instruct
model; the other columns were trained on each row's own training data, except the unseen rows, which test
models trained elsewhere. SFT trains on short reference answers, and there is one training seed per model.

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
expected output of each step, and [method](docs/harness/method.md) has the command
behind every model in the tables. All tables are generated from the saved evaluation records
by `python -m scripts.report`, quoted numbers come from `results/numbers.json`, and a test fails if
either drifts from the records.

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

- **Findings:** [reasoning pays](docs/findings/reasoning-pays.md), [form, not ability](docs/findings/form-not-ability.md), [SFT first](docs/findings/sft-then-rl.md), [what the gain is made of](docs/findings/what-the-gain-is.md), and the open question of [one model for photos and charts](docs/findings/one-model.md).
- **[What did not work](docs/negatives.md):** each thing tried, what was expected, what happened and what it implies.
- **[Do 3D CT benchmarks need the scan?](docs/ct-audit.md)** A separate note: on DeepTumorVQA a text-only rule scores <!--n:au.mc.rule-->60.7<!--/n-->%, and on a debiased version built here GRPO and SFT reach the same accuracy.
- **Harness:** [method](docs/harness/method.md), [evaluation](docs/harness/evaluation.md), [validation](docs/harness/validation.md), [training evidence](docs/harness/training-evidence.md), [hardware](docs/harness/hardware.md), [monitoring](docs/harness/monitoring.md).
- **[Category table](docs/matrix.md#category-table)** across 18 benchmarks, and [full tables](docs/matrix.md) with paired 95% intervals, and [history](docs/history.md): earlier versions of this repository, the corrections made since, and the status of every earlier claim.

## Acknowledgments

Built with Qwen, Hugging Face Transformers, TRL, PEFT and vLLM. Dataset and model licenses
apply to their assets; the code is under the [Apache 2.0 license](LICENSE).
