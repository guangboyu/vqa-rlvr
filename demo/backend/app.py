"""VQA Arena backend: side-by-side model comparison + results dashboard API.

Run:
    uv run --group demo uvicorn demo.backend.app:app --host 0.0.0.0 --port 8000
"""

import asyncio
import io
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from demo.backend.model_manager import LINEUP, PROMPTS, ModelManager

ROOT = Path(__file__).parent.parent.parent
DIST = Path(__file__).parent.parent / "frontend" / "dist"
manager = ModelManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.environ.get("VQA_LIVE_MODELS") == "1":
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
    image_condition: str = Form("original"),
    models: str = Form(""),  # comma-separated keys; empty = all
):
    if template not in PROMPTS:
        raise HTTPException(400, "Unknown prompt template")
    if image_condition not in ("original", "black"):
        raise HTTPException(400, "Unknown image condition")
    keys = list(dict.fromkeys(k for k in models.split(",") if k)) or [s.key for s in LINEUP]
    available = {s["key"]: s["loaded"] for s in manager.specs()}
    if any(key not in available for key in keys):
        raise HTTPException(400, "Unknown model")
    if not all(available[key] for key in keys):
        raise HTTPException(503, "Live models are unavailable. The recorded gallery needs no GPU.")
    pil = Image.open(io.BytesIO(await image.read())).convert("RGB")
    pil.thumbnail((1024, 1024))  # bound vision tokens for interactive latency
    if image_condition == "black":
        pil = Image.new("RGB", pil.size)

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    async def generate(key):
        try:
            await asyncio.to_thread(manager.stream_sync, key, pil, question, template, queue, loop)
        except Exception:
            logging.exception("Generation failed for %s", key)
            await queue.put((key, {"error": "Generation failed. Check the server log."}))
        finally:
            await queue.put((key, None))

    tasks = [asyncio.create_task(generate(key)) for key in keys]

    async def events():
        done = set()
        try:
            while len(done) < len(keys):
                key, token = await queue.get()
                if token is None:
                    done.add(key)
                    payload = {"model": key, "done": True}
                elif isinstance(token, dict):
                    payload = {"model": key, **token}
                else:
                    payload = {"model": key, "token": token}
                yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            await asyncio.gather(*tasks, return_exceptions=True)

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
    diagnostic_path = ROOT / "results" / "diagnostics.json"
    diagnostics = json.loads(diagnostic_path.read_text()) if diagnostic_path.exists() else {}
    comparison_path = ROOT / "results" / "baselines" / "comparison.json"
    comparison = json.loads(comparison_path.read_text()) if comparison_path.exists() else None
    return {"runs": runs, "judge_study": judge, "ablation": diagnostics.get("ablation", []),
            "official_runs": diagnostics.get("official_runs", {}), "comparison": comparison}


@app.get("/api/gallery")
def gallery():
    path = ROOT / "demo" / "examples" / "gallery.json"
    return json.loads(path.read_text()) if path.exists() else []


examples = ROOT / "demo" / "examples"
if examples.exists():
    app.mount("/examples", StaticFiles(directory=str(examples)), name="examples")


if DIST.exists():  # serve the built frontend
    app.mount("/", StaticFiles(directory=str(DIST), html=True), name="frontend")
