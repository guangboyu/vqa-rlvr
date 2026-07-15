"""VQA Arena backend: side-by-side model comparison + results dashboard API.

Run:
    uv run --group demo uvicorn demo.backend.app:app --host 0.0.0.0 --port 8000
"""

import asyncio
import io
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from demo.backend.model_manager import LINEUP, ModelManager

ROOT = Path(__file__).parent.parent.parent
DIST = Path(__file__).parent.parent / "frontend" / "dist"
manager = ModelManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(manager.load_all)
    yield


app = FastAPI(title="VQA Arena", lifespan=lifespan)


@app.get("/api/models")
def models() -> list[dict]:
    return manager.specs()


@app.post("/api/ask")
async def ask(
    image: UploadFile = File(...),
    question: str = Form(...),
    template: str = Form("reasoning"),
    models: str = Form(""),  # comma-separated keys; empty = all
):
    pil = Image.open(io.BytesIO(await image.read())).convert("RGB")
    pil.thumbnail((1024, 1024))  # bound vision tokens for interactive latency
    keys = [k for k in models.split(",") if k] or [s.key for s in LINEUP]

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    for key in keys:
        asyncio.create_task(
            asyncio.to_thread(manager.stream_sync, key, pil, question, template, queue, loop)
        )

    async def events():
        done = 0
        while done < len(keys):
            key, token = await queue.get()
            if token is None:
                done += 1
                payload = {"model": key, "done": True}
            else:
                payload = {"model": key, "token": token}
            yield f"data: {json.dumps(payload)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/api/results")
def results() -> dict:
    runs = {}
    for path in sorted((ROOT / "results" / "runs").glob("*.json")):
        run = json.loads(path.read_text())
        if "eval" not in run:
            continue
        dataset = run["eval"]["dataset"]
        run_id = run["run_id"].removesuffix(f"-{dataset}")
        runs.setdefault(run_id, {})[dataset] = round(run["metrics"]["overall"] * 100, 1)
    study_path = ROOT / "results" / "judge_study.json"
    judge = json.loads(study_path.read_text()) if study_path.exists() else None
    # CoT retention measured from results/preds (fraction of GQA completions >100 chars)
    ablation = [
        {"weight": 0.0, "steps": 300, "retention": 17, "run_id": "grpo_2b_fmt0_reasoning"},
        {"weight": 0.2, "steps": 500, "retention": 100, "run_id": "grpo_2b_main_base_reasoning"},
        {"weight": 0.5, "steps": 300, "retention": 100, "run_id": "grpo_2b_fmt05_reasoning"},
    ]
    return {"runs": runs, "judge_study": judge, "ablation": ablation}


if DIST.exists():  # serve the built frontend
    app.mount("/", StaticFiles(directory=str(DIST), html=True), name="frontend")
