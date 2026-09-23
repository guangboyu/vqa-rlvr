# Tutorial: the 2B pipeline on one 24 GB GPU

This walks through every stage with Qwen3-VL-2B-Instruct on a single 24 GB card: materialize
the data, check the splits, evaluate the starting model, measure how much of the RL pool can
teach it, train with SFT and GRPO, evaluate the result, and regenerate the tables. Every
command below was run on an RTX 4090; the timings and outputs are from that card, after the
model download. Larger cards use the `80gb` or `96gb` hardware profiles described in the
[hardware notes](supplementary/hardware.md).

## 1. Environment

Python 3.12 and the committed lockfile, which pins CUDA 12.9 builds of PyTorch and vLLM:

```bash
uv sync --frozen
uv run --frozen pytest -q            # CPU tests, a minute or two
```

## 2. Data

`prepare_data.py` downloads the pinned source revisions and materializes fixed, seeded
subsets under `data/` (about 13 GB). `check_data` then validates each dataset's split file
in `data/splits/` against the materialized rows and fails if any dev or test image appears
in any training pool.

```bash
uv run --frozen python prepare_data.py
uv run --frozen python -m scripts.check_data
```

Expected:

```text
vqav2 (in-domain): {'sft': 19553, 'rl': 27553, 'dev': 447, 'test': 5000}
gqa (in-domain): {'sft': 19905, 'rl': 27885, 'dev': 59, 'test': 12578}
clevr (in-domain): {'rl': 8000, 'dev': 3000, 'test': 3000}
textvqa (far-ood): {'test': 2000}
no held-out image in any training pool; report: data/splits/overlap-report.json
```

The committed split files describe the recorded materialization. If `check_data` reports
that a subset changed since its split file was written, your download differs from it:
move `data/splits/*.json` aside and run `check_data` again to build split files for your
data with the same policy and seed. Numbers from your data are then not row-for-row
comparable with the recorded tables.

## 3. Evaluate the starting model

Every evaluation is one cell: a model under a protocol (dataset, role, tier, prompt
template, decoding, image mode). The `smoke` tier takes 32 questions per dataset:

```bash
uv run --frozen python evaluate.py --model Qwen/Qwen3-VL-2B-Instruct --label base2 \
  --tier smoke --prompts short reasoning
```

Each cell prints both scorer variants and writes a record under `results/records/`, with
the per-question predictions beside it. About a minute for all eight cells:

```text
vqav2-test-smoke-short: official 0.7031, normalized 0.7031, n=32 -> results/records/base2-dec3ad6e3a/vqav2-test-smoke-short-dbf5c8ce22.json
vqav2-test-smoke-reasoning: official 0.5437, normalized 0.6375, n=32 -> ...
...
```

Short-prompt outputs repeat exactly from run to run; reasoning-prompt outputs can differ
slightly, because long batched generations in vLLM are not bitwise deterministic. Running
the command again finds the records, prints `(cached)`, and skips the work.

## 4. How much of the RL pool can teach this model?

GRPO learns only from prompts whose sampled responses disagree: if all eight responses are
right, or all wrong, the group has no advantage. The pre-check takes a seeded random sample
of each dataset's RL pool and scores a group of responses per prompt with the training
reward, using the rollout settings of the run's preset:

```bash
uv run --frozen python -m scripts.learnability --preset grpo_2b_main --label base2 --tier dev
```

About four minutes for 500 prompts on each of VQAv2, GQA, and CLEVR:

```text
vqav2: 47.8% of 500 prompts learnable; correct samples per prompt {'0/8': 99, ..., '8/8': 162}
gqa: 42.0% of 500 prompts learnable; correct samples per prompt {'0/8': 133, ..., '8/8': 157}
clevr: 48.4% of 500 prompts learnable; correct samples per prompt {'0/8': 0, ..., '8/8': 258}
```

The rest of the prompts are always or never answered correctly. These records are already
in the repository, so on the recorded data this command returns them from the cache.

## 5. Train

SFT on the short-answer pool, then GRPO from either starting point. A two-step run checks
the whole path in a few minutes; the full recipe is one SFT epoch (no `--max-steps`) and
300 GRPO updates (`--max-steps 300`), as in the [method](supplementary/method.md).

```bash
uv run --frozen python train_sft.py --preset sft_2b --hardware 24gb \
  --max-steps 2 --save-steps 1 --run-dir runs/sft-smoke --no-wandb
uv run --frozen python train_grpo.py --preset grpo_2b_main --hardware 24gb \
  --max-steps 2 --save-steps 1 --run-dir runs/grpo-smoke --no-wandb
```

Each run keeps the adapter of every scheduled save and writes `diagnostics.json` (loss,
reward, reward spread, groups without reward variation, entropy, completion length). Once
training completes, it evaluates every kept adapter on the dev tier of its datasets with
both prompts; the two-step GRPO run above takes about three minutes including that step
and prints lines such as `step 2 vqav2-dev-dev-short: normalized 0.7181`. Those records
form the run's learning curve and stay in `runs/<run>/checkpoints/dev-records/`;
`--no-dev-eval` skips the step. A run interrupted at any point resumes from its last
complete checkpoint when started again with the same `--run-dir`.

To start GRPO from the SFT model instead, merge the adapter and pass the merge:

```bash
uv run --frozen python scripts/merge_adapter.py runs/sft-smoke/checkpoints \
  --out checkpoints/merged/sft-smoke
uv run --frozen python train_grpo.py --preset grpo_2b_main --hardware 24gb \
  --model checkpoints/merged/sft-smoke --max-steps 2 --save-steps 1 \
  --run-dir runs/grpo-sft-smoke --no-wandb
```

Add `--dry-run` to a training command to check its data and reward without a GPU.

## 6. Evaluate a trained model

```bash
uv run --frozen python scripts/merge_adapter.py runs/grpo-smoke/checkpoints \
  --out checkpoints/merged/grpo-smoke
uv run --frozen python evaluate.py --model checkpoints/merged/grpo-smoke \
  --label grpo-smoke --prompts short reasoning --tier smoke
uv run --frozen python evaluate.py --model checkpoints/merged/grpo-smoke \
  --label grpo-smoke --prompts short --tier smoke --image-mode black
```

The last command repeats the short-prompt cells with every image replaced by a black image
of the same size. Its record pairs each answer with the original-image answer, so run it
after the original cells. On the smoke tier, CLEVR counting drops from 0.97 to 0.

## 7. Tables and the dashboard

The tables in the README and the supplementary pages are generated from the records of the
arms declared in `experiments.toml`:

```bash
uv run --frozen python -m scripts.report            # regenerate tables and the demo export
uv run --frozen python -m scripts.report --check    # list declared cells without a record
```

Add an arm to `experiments.toml` to give it a row; `--run` evaluates its missing cells.
The dashboard reads the same export; see the [demo](../demo/README.md) to run it without a
GPU.
