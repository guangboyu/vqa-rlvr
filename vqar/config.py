"""Run presets: named, frozen dataclass configs selected via --preset <name>.

Every entrypoint takes a preset name plus a handful of override flags; the resolved
config is serialized into the run's results JSON and W&B config for reproducibility.
Presets are added milestone by milestone (eval → sft → grpo).
"""

MODEL_2B = "Qwen/Qwen3-VL-2B-Instruct"
MODEL_8B = "Qwen/Qwen3-VL-8B-Instruct"

PRESETS: dict[str, object] = {}
