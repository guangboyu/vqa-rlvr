"""Run presets: named, frozen dataclass configs selected via --preset <name>.

Every entrypoint takes a preset name plus a handful of override flags; the resolved
config is serialized into the run's results JSON and W&B config for reproducibility.
Presets are added milestone by milestone (eval → sft → grpo).
"""

from dataclasses import asdict, dataclass, replace

from vqar.registry import pools

MODEL_2B = "Qwen/Qwen3-VL-2B-Instruct"
MODEL_8B = "Qwen/Qwen3-VL-8B-Instruct"

# Pixel budget per image: 768 tokens x (32x32 px per merged vision token) ≈ 0.79 MP.
# Caps VRAM for training and eval; recorded in every results JSON.
MAX_IMAGE_TOKENS = 768
PIXELS_PER_TOKEN = 32 * 32  # patch_size 16, merge_size 2 (verified in M0 spike)
MAX_PIXELS = MAX_IMAGE_TOKENS * PIXELS_PER_TOKEN

# Explicit TRL 1.8 defaults. This uses DAPO's loss reduction, not its full recipe.
RL_RECIPE = dict(
    loss_type="dapo",
    scale_rewards="group",
    multi_objective_aggregation="sum_then_normalize",
    importance_sampling_level="token",
    epsilon=0.2,
    epsilon_high=0.2,
    num_iterations=1,
    mask_truncated_completions=False,
    vllm_importance_sampling_correction=True,
    vllm_importance_sampling_mode="sequence_mask",
    vllm_importance_sampling_clip_max=3.0,
    vllm_importance_sampling_clip_min=None,
    top_entropy_quantile=1.0,
    entropy_coef=0.0,
    use_adaptive_entropy=False,
)

HARDWARE_PROFILES = ("legacy", "24gb", "80gb", "96gb")


@dataclass(frozen=True)
class EvalPreset:
    name: str
    model: str  # HF model id or local path to a merged checkpoint
    template: str = "short"  # a vqar.registry.PROMPTS name
    datasets: tuple[str, ...] = ()  # empty: every registered dataset with a test role
    max_tokens: int = 64  # 256+ needed for the reasoning template
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.8


EVAL_PRESETS: dict[str, EvalPreset] = {
    p.name: p
    for p in [
        EvalPreset(name="zero_shot_2b", model=MODEL_2B),
        # 17GB of bf16 weights: give vLLM most of the card or KV-cache allocation fails
        EvalPreset(name="zero_shot_8b", model=MODEL_8B, gpu_memory_utilization=0.88),
        EvalPreset(
            name="zero_shot_2b_reasoning", model=MODEL_2B, template="reasoning", max_tokens=512
        ),
        EvalPreset(
            name="zero_shot_8b_reasoning",
            model=MODEL_8B,
            template="reasoning",
            max_tokens=512,
            gpu_memory_utilization=0.88,
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
    subsets: tuple[str, ...] = pools("sft", "vqav2", "gqa")
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # LM attention + MLP only; the vision tower stays frozen.
    lora_targets: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
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
            # 1e-4 (the 2B recipe) visibly damaged the 8B: counting regressed and
            # instruction-following eroded.
            learning_rate=5e-5,
            per_device_batch_size=2,
            gradient_accumulation=16,
        ),
    ]
}


@dataclass(frozen=True)
class GRPOPreset:
    name: str
    model: str  # base HF id or a merged SFT checkpoint dir to continue from
    subsets: tuple[str, ...]  # RL prompt pools, e.g. pools("rl", "clevr")
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
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    seed: int = 42


GRPO_PRESETS: dict[str, GRPOPreset] = {
    p.name: p
    for p in [
        GRPOPreset(
            name="grpo_2b_clevr_base",  # R1-Zero-style sandbox: base model emits <think>
            model=MODEL_2B,  # zero-shot, so format reward is alive from step 0
            subsets=pools("rl", "clevr"),
            quantize_4bit=False,
            learning_rate=1e-5,
            num_generations=8,
            per_device_batch_size=8,
            gradient_accumulation=4,
            max_steps=400,
        ),
        # The corrected 2B recipe: Instruct start, correctness-only reward, 300 updates.
        # The SFT-start arm passes --model with the merged SFT checkpoint.
        GRPOPreset(
            name="grpo_2b_main",
            model=MODEL_2B,
            subsets=pools("rl", "vqav2", "gqa"),
            quantize_4bit=False,
            learning_rate=1e-5,
            num_generations=8,
            per_device_batch_size=8,
            gradient_accumulation=4,
            max_steps=300,
            rewards=("correctness",),
            reward_weights=(1.0,),
        ),
        # Same-scale EM contrast for the E7 self-distill 8B arm: bf16 student
        # (NF4 corrupts TRL colocate sync — see opd_self_8b_main), matched
        # geometry and lr, correctness-only reward, sized for 48GB+ cards.
        GRPOPreset(
            name="grpo_8b_fmt0",
            model=MODEL_8B,
            subsets=pools("rl", "vqav2", "gqa"),
            quantize_4bit=False,
            learning_rate=1e-5,
            num_generations=8,
            per_device_batch_size=8,
            gradient_accumulation=4,
            max_steps=300,
            rewards=("correctness",),
            reward_weights=(1.0,),
            vllm_gpu_memory_utilization=0.55,
        ),
    ]
}


@dataclass(frozen=True)
class OPDPreset:
    """On-policy distillation reference arm: a frozen teacher scores the student's own
    rollouts per token (reverse KL), replacing the scalar EM advantage. Geometry is
    matched 1:1 to the GRPO arm it's compared against — same student init, data,
    seed, LoRA, lr, and rollouts/step — so speed and quality compare directly."""

    name: str
    model: str  # student init (base HF id or merged checkpoint dir)
    teacher: str  # scored frozen; NF4-quantized to fit next to the student on 24GB
    subsets: tuple[str, ...]
    learning_rate: float
    num_generations: int  # rollouts per prompt; kept only for geometry parity
    per_device_batch_size: int
    gradient_accumulation: int
    max_steps: int
    max_completion_length: int = 640
    quantize_4bit: bool = False  # QLoRA student (NF4 + paged 8-bit AdamW), 8B only
    teacher_quantize_4bit: bool = True
    teacher_chunk_size: int = 2  # sequences per teacher forward (bounds logit memory)
    vllm_gpu_memory_utilization: float = 0.25  # 0.30 minus room for the teacher
    lora_r: int = 16
    lora_alpha: int = 32
    lora_targets: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    seed: int = 42


OPD_PRESETS: dict[str, OPDPreset] = {
    p.name: p
    for p in [
        # Matched to the July correctness-only GRPO run (historical results name
        # grpo_2b_fmt0_fixedreward): Instruct-2B start, VQAv2+GQA RL pools,
        # 300 steps x 32 rollouts. Only the learning signal differs.
        OPDPreset(
            name="opd_2b_main",
            model=MODEL_2B,
            teacher=MODEL_8B,
            subsets=pools("rl", "vqav2", "gqa"),
            learning_rate=1e-5,
            num_generations=8,
            per_device_batch_size=8,
            gradient_accumulation=4,
            max_steps=300,
        ),
        # Size-to-size arm: teacher = the BASE 2B itself (adapters disabled),
        # conditioned on the gold answer. Sharing base weights avoids a resident
        # 8B teacher; the extra gold context changes the information available.
        OPDPreset(
            name="opd_self_2b_main",
            model=MODEL_2B,
            teacher="self",
            subsets=pools("rl", "vqav2", "gqa"),
            learning_rate=1e-5,
            num_generations=8,
            per_device_batch_size=8,
            gradient_accumulation=4,
            max_steps=300,
            teacher_chunk_size=8,
            vllm_gpu_memory_utilization=0.3,
        ),
        # Keep the student bf16: the historical NF4/vLLM weight sync was corrupt.
        # The 80/96 GB profiles adjust engine memory; LR matches grpo_8b_fmt0.
        OPDPreset(
            name="opd_self_8b_main",
            model=MODEL_8B,
            teacher="self",
            subsets=pools("rl", "vqav2", "gqa"),
            learning_rate=1e-5,
            num_generations=8,
            per_device_batch_size=8,
            gradient_accumulation=4,
            max_steps=300,
            teacher_chunk_size=4,
            # 0.40 starved the engine: 16G of weights + the profiling pass left no
            # room for cache blocks (init crash-loop, jobs 55070364..918). 0.55
            # gives KV ~7G; wake peak ≈ 45G, train peak ≈ 24G (engine asleep).
            vllm_gpu_memory_utilization=0.55,
        ),
    ]
}


def dump(preset) -> dict:
    """Serialize any preset dataclass to a plain dict for results JSONs and W&B."""
    return asdict(preset)


def for_hardware(preset, hardware: str):
    """Adjust precision and memory placement, keeping learning rates and batch geometry."""
    if hardware not in HARDWARE_PROFILES:
        raise ValueError(f"unknown hardware profile: {hardware}")
    if hardware in ("legacy", "24gb"):
        return preset
    # Snapshot hashes and merged checkpoint paths do not reliably identify model size.
    large_model = "8b" in preset.name.split("_")
    if isinstance(preset, SFTPreset):
        return replace(preset, quantize_4bit=False)
    # Reserve about 29 GB for an 8B engine or 12 GB for a 2B engine.
    utilization = {"80gb": (0.36, 0.15), "96gb": (0.30, 0.125)}[hardware]
    return replace(
        preset, quantize_4bit=False,
        vllm_gpu_memory_utilization=utilization[0 if large_model else 1],
    )
