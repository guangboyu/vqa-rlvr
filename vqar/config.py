"""Run presets: named, frozen dataclass configs selected via --preset <name>.

Every entrypoint takes a preset name plus a handful of override flags; the resolved
config is serialized into the run's results JSON and W&B config for reproducibility.
Presets are added milestone by milestone (eval → sft → grpo).
"""

from dataclasses import asdict, dataclass

MODEL_2B = "Qwen/Qwen3-VL-2B-Instruct"
MODEL_8B = "Qwen/Qwen3-VL-8B-Instruct"

# Pixel budget per image: 768 tokens x (32x32 px per merged vision token) ≈ 0.79 MP.
# Caps VRAM for training and eval; recorded in every results JSON.
MAX_IMAGE_TOKENS = 768
PIXELS_PER_TOKEN = 32 * 32  # patch_size 16, merge_size 2 (verified in M0 spike)
MAX_PIXELS = MAX_IMAGE_TOKENS * PIXELS_PER_TOKEN


@dataclass(frozen=True)
class EvalPreset:
    name: str
    model: str  # HF model id or local path to a merged checkpoint
    backend: str = "vllm"  # "vllm" | "hf" (hf covers models vLLM can't serve)
    template: str = "short"  # "short" | "reasoning" (see vqar.data)
    datasets: tuple[str, ...] = ("vqav2", "gqa", "clevr", "textvqa")
    max_tokens: int = 64  # 256+ needed for the reasoning template
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.8


EVAL_PRESETS: dict[str, EvalPreset] = {
    p.name: p
    for p in [
        EvalPreset(name="zero_shot_2b", model=MODEL_2B),
        EvalPreset(name="zero_shot_8b", model=MODEL_8B),
        EvalPreset(
            name="zero_shot_2b_reasoning", model=MODEL_2B, template="reasoning", max_tokens=512
        ),
        EvalPreset(
            name="zero_shot_8b_reasoning", model=MODEL_8B, template="reasoning", max_tokens=512
        ),
    ]
}


@dataclass(frozen=True)
class SFTPreset:
    name: str
    model: str
    quantize_4bit: bool  # QLoRA (NF4 + paged 8-bit AdamW) for the 8B; plain bf16 LoRA for 2B
    learning_rate: float
    per_device_batch_size: int
    gradient_accumulation: int
    subsets: tuple[str, ...] = ("vqav2_sft", "gqa_sft")
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # LM attention + MLP only; the vision tower stays frozen.
    lora_targets: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    )
    epochs: float = 1.0
    max_length: int = 1024
    warmup_ratio: float = 0.03
    seed: int = 42


SFT_PRESETS: dict[str, SFTPreset] = {
    p.name: p
    for p in [
        SFTPreset(
            name="sft_2b",
            model=MODEL_2B,
            quantize_4bit=False,
            learning_rate=1e-4,
            per_device_batch_size=4,
            gradient_accumulation=8,
        ),
        SFTPreset(
            name="sft_8b",
            model=MODEL_8B,
            quantize_4bit=True,
            learning_rate=1e-4,
            per_device_batch_size=2,
            gradient_accumulation=16,
        ),
    ]
}


@dataclass(frozen=True)
class GRPOPreset:
    name: str
    model: str  # base HF id or an SFT adapter dir (checkpoints/sft_2b) to continue from
    subsets: tuple[str, ...]  # RL prompt pools, e.g. ("clevr_rl",)
    quantize_4bit: bool
    learning_rate: float
    num_generations: int  # group size G
    per_device_batch_size: int
    gradient_accumulation: int
    max_steps: int
    max_completion_length: int = 640  # >512: kill the truncation tail seen in M1
    rewards: tuple[str, ...] = ("correctness", "format")
    reward_weights: tuple[float, ...] = (1.0, 0.2)
    kl_beta: float = 0.0  # modern GRPO practice; one 0.04 sanity run in the sandbox
    vllm_gpu_memory_utilization: float = 0.3
    lora_r: int = 16
    lora_alpha: int = 32
    lora_targets: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    )
    seed: int = 42


GRPO_PRESETS: dict[str, GRPOPreset] = {
    p.name: p
    for p in [
        GRPOPreset(
            name="grpo_2b_clevr",  # the verifier sandbox: pipeline must show rising reward
            model="checkpoints/merged/sft_2b",
            subsets=("clevr_rl",),
            quantize_4bit=False,
            learning_rate=1e-5,
            num_generations=8,
            per_device_batch_size=8,
            gradient_accumulation=4,
            max_steps=400,
        ),
        GRPOPreset(
            name="grpo_2b_main",
            model="checkpoints/merged/sft_2b",
            subsets=("vqav2_rl", "gqa_rl"),
            quantize_4bit=False,
            learning_rate=1e-5,
            num_generations=8,
            per_device_batch_size=8,
            gradient_accumulation=4,
            max_steps=500,
        ),
        GRPOPreset(
            name="grpo_8b_clevr",  # OOM stress test before the 8B main run
            model="checkpoints/merged/sft_8b",
            subsets=("clevr_rl",),
            quantize_4bit=True,
            learning_rate=5e-6,
            num_generations=6,
            per_device_batch_size=6,
            gradient_accumulation=8,
            max_steps=300,
            max_completion_length=512,
            vllm_gpu_memory_utilization=0.25,
        ),
        GRPOPreset(
            name="grpo_8b_main",
            model="checkpoints/merged/sft_8b",
            subsets=("vqav2_rl", "gqa_rl"),
            quantize_4bit=True,
            learning_rate=5e-6,
            num_generations=6,
            per_device_batch_size=6,
            gradient_accumulation=8,
            max_steps=300,
            max_completion_length=512,
            vllm_gpu_memory_utilization=0.25,
        ),
    ]
}


def dump(preset) -> dict:
    """Serialize any preset dataclass to a plain dict for results JSONs and W&B."""
    return asdict(preset)
