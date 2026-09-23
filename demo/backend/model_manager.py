"""Keeps the demo's model lineup resident on the GPU and streams generations.

Three 2B checkpoints (~4.5GB bf16 each) fit a 24GB card together, so every arena
request fans out to all requested models concurrently — each generation runs in a
worker thread with a TextIteratorStreamer feeding an asyncio queue.
"""

import asyncio
import threading
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from vqar.registry import PROMPTS as TEMPLATES

ROOT = Path(__file__).parent.parent.parent

# The demo offers the two headline templates, not the diagnostic ones.
PROMPTS = {name: p.text for name, p in TEMPLATES.items() if not p.diagnostic}


@dataclass
class ModelSpec:
    key: str
    label: str
    path: str
    description: str
    revision: str | None = None


LINEUP = [
    ModelSpec(
        "zero_shot",
        "Qwen3-VL-2B (zero-shot)",
        "Qwen/Qwen3-VL-2B-Instruct",
        "The released Instruct checkpoint, before our VQA fine-tuning.",
        "89644892e4d85e24eaac8bacfd4f463576704203",
    ),
    ModelSpec(
        "sft",
        "+ SFT (LoRA)",
        str(ROOT / "checkpoints/merged/sharedpool_20260913_sft_2b"),
        "One epoch on 39,458 VQAv2/GQA short answers, with held-out images excluded.",
    ),
    ModelSpec(
        "grpo",
        "+ GRPO (RLVR)",
        str(ROOT / "checkpoints/merged/sharedpool_20260913_grpo_2b_base"),
        "300 correctness-only GRPO updates from Instruct on the shared VQAv2/GQA pool.",
    ),
]


class ModelManager:
    def __init__(self):
        self._models: dict[str, tuple] = {}
        self._lock = threading.Lock()

    def load_all(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        for spec in LINEUP:
            if spec.key in self._models:
                continue
            processor = AutoProcessor.from_pretrained(spec.path, revision=spec.revision)
            model = AutoModelForImageTextToText.from_pretrained(
                spec.path, revision=spec.revision, dtype=torch.bfloat16, device_map="cuda"
            ).eval()
            self._models[spec.key] = (model, processor)

    def specs(self) -> list[dict]:
        return [
            {
                "key": s.key,
                "label": s.label,
                "description": s.description,
                "loaded": s.key in self._models,
            }
            for s in LINEUP
        ]

    def stream_sync(
        self,
        key: str,
        image: Image.Image,
        question: str,
        template: str,
        queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
        max_new_tokens: int = 512,
    ) -> None:
        """Blocking generation (run via asyncio.to_thread); pushes (key, token|None).

        All requested models generate concurrently — CUDA interleaves their kernels,
        so the arena cards stream side by side.
        """
        import torch
        from transformers import TextIteratorStreamer

        model, processor = self._models[key]
        text = PROMPTS[template].format(question=question)
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": text}],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda")
        streamer = TextIteratorStreamer(
            processor.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        failure = []

        def generate():
            try:
                with torch.inference_mode():
                    model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        streamer=streamer,
                    )
            except Exception as error:
                failure.append(error)
                streamer.end()

        thread = threading.Thread(target=generate, daemon=True)
        thread.start()
        try:
            for token in streamer:
                asyncio.run_coroutine_threadsafe(queue.put((key, token)), loop).result()
            if failure:
                raise failure[0]
        finally:
            thread.join()
