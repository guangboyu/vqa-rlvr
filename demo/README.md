# Demo: recorded examples, results and a live comparison

The app has three views. Examples shows recorded answers to the same test questions,
including scorer disagreements and counterexamples; Results shows the evaluation tables; Live
comparison asks the 2B models a question about your own image. The first two need no GPU or
downloaded model. The examples are illustrative and do not estimate failure rates.

## Start

```bash
uv sync --frozen --group demo
cd demo/frontend
npm ci
npm run build
cd ../..
uv run --frozen uvicorn demo.backend.app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. For frontend development, `npm run dev` serves Vite
on port 5173 and proxies `/api` and `/examples` to port 8000.

The Results view reads `results/report.json`, which `python -m scripts.report` writes from
the same evaluation records as every table in the README and `docs/`, so the
two cannot disagree. Switch between short and reasoning prompts and between scorers, and
inspect response behavior and image interventions. The July 2026 experiments remain
reachable from the bottom of the view, with their limitations stated.
`node screenshot.mjs` (in `demo/frontend`, with the server on port 8000) refreshes the
screenshots in `demo/assets/`.

## Optional live inference

Prepare the three 2B model paths listed in `backend/model_manager.py`. The default
lineup uses the pinned Instruct revision and the corrected SFT/base-to-GRPO models
from the [corrected campaign](../docs/history.md#the-corrected-campaign-september-13-recipe). Fresh adapters are prepared
locally and have not been published. Historical Hub adapters correspond to older runs.
Then enable model loading:

```bash
VQA_LIVE_MODELS=1 uv run --frozen uvicorn demo.backend.app:app \
  --host 127.0.0.1 --port 8000
```

This loads the base, SFT, and GRPO models in bf16. The corrected three-model lineup
passed a small concurrent streaming check on an RTX 4090: 11.89 GiB resident
PyTorch allocation and 12.10 GiB peak. The check also exercised black-image and
reasoning-prompt responses. [Recorded outputs and scope](../results/baselines/live-demo-24gb.json)
are included; larger images and longer responses need additional memory.
Models and short/reasoning prompts are selectable. An optional black-image input
keeps the image dimensions fixed for a simple sensitivity check. The app reports unavailable
models and generation failures instead of leaving the stream waiting forever.
The current live comparison is the 2B path; 8B training profiles do not enable
three simultaneous 8B models in this app.
Live inference uses Transformers and bounds image size for responsiveness, so its
responses can differ from the recorded vLLM evaluations.

Recorded examples retain their prediction run IDs and questions. Refresh them with
`uv run --frozen python -m scripts.make_gallery` after restoring
the predictions and VQAv2 evaluation subset. The generator prefers the completed
four-arm comparison; historical examples remain available when those files are absent.
