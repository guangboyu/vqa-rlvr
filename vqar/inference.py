"""Generation helpers and answer extraction.

Answer extraction is shared by eval and RL rewards: models trained with the
`reasoning` template answer inside <answer></answer> tags; models using the
`short` template answer directly. Both are scored identically after extraction.

vLLM notes (single RTX 4090, WSL2):
- FlashInfer (vLLM 0.23's default attention backend) JIT-compiles with nvcc, which
  needs a full CUDA toolkit this box doesn't have. FLASH_ATTN ships precompiled,
  so we force it before vLLM is imported.
- Images are sent as base64 data URIs built from the stored bytes — the original
  JPEG/PNG payloads pass through without a decode/re-encode round trip.
"""

import base64
import os
import re

os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def extract_answer(text: str) -> str:
    """Return the content of the last <answer> tag, or the stripped full text."""
    matches = _ANSWER_TAG.findall(text)
    if matches:
        return matches[-1].strip()
    return text.strip()


def image_to_data_uri(image_bytes: bytes) -> str:
    mime = "image/png" if image_bytes[:4] == b"\x89PNG" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(image_bytes).decode()


def build_messages(question: str, image_bytes: bytes, template: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_to_data_uri(image_bytes)}},
                {"type": "text", "text": template.format(question=question)},
            ],
        }
    ]


def generate_vllm(llm, conversations: list[list[dict]], max_tokens: int) -> list[str]:
    """Greedy batch generation; chunked so data-URI payloads never all sit in RAM."""
    from vllm import SamplingParams

    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    outputs = []
    chunk = 256
    for i in range(0, len(conversations), chunk):
        results = llm.chat(conversations[i : i + chunk], params, use_tqdm=False)
        outputs.extend(r.outputs[0].text for r in results)
    return outputs


def load_vllm(model: str, max_model_len: int, gpu_memory_utilization: float):
    from vllm import LLM

    return LLM(
        model=model,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1},
    )
