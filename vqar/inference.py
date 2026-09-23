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


def build_messages(question: str, images: list[bytes], template: str) -> list[dict]:
    content = [{"type": "image_url", "image_url": {"url": image_to_data_uri(image)}}
               for image in images]
    content.append({"type": "text", "text": template.format(question=question)})
    return [{"role": "user", "content": content}]


class VLLMGenerator:
    """The production generator for `vqar.eval.run_eval`: one vLLM engine, loaded on first
    use so that cached cells never start it."""

    def __init__(self, model: str, revision: str | None = None, max_model_len: int = 4096,
                 gpu_memory_utilization: float = 0.8):
        self.settings = dict(model=model, revision=revision, max_model_len=max_model_len,
                             gpu_memory_utilization=gpu_memory_utilization)
        self._llm = None

    def generate(self, conversations, *, max_tokens, temperature, samples, seed):
        """Per conversation, `samples` completions with token counts and stop reasons."""
        from vllm import SamplingParams

        if self._llm is None:
            self._llm = load_vllm(**self.settings)
        params = SamplingParams(temperature=temperature, max_tokens=max_tokens, n=samples,
                                seed=seed)
        outputs = []
        chunk = 256
        for i in range(0, len(conversations), chunk):
            for result in self._llm.chat(conversations[i : i + chunk], params, use_tqdm=False):
                outputs.append([
                    {"text": response.text, "output_tokens": len(response.token_ids),
                     "finish_reason": response.finish_reason}
                    for response in result.outputs
                ])
        return outputs


def load_vllm(model: str, max_model_len: int, gpu_memory_utilization: float,
              revision: str | None = None):
    from vllm import LLM

    from vqar.config import MAX_PIXELS

    return LLM(
        model=model,
        revision=revision,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1},
        # Same pixel budget as training; also keeps large images (TextVQA is >1MP)
        # from exceeding max_model_len in vision tokens.
        mm_processor_kwargs={"max_pixels": MAX_PIXELS},
    )
