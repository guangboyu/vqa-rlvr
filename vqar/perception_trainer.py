"""GRPO with a perception loss: reward answers that depend on the image (after PAPO).

PAPO (Wang et al., 2025, "Perception-Aware Policy Optimization for Multimodal Reasoning") adds
to the GRPO objective the KL divergence between the policy given the real image and given a
copy with most of its patches masked, so the policy is pushed to answer differently when the
image is gone: an answer the model would give without looking earns nothing extra.

Here the KL is estimated on the sampled completion tokens as log pi(o_t | image) -
log pi(o_t | masked image). The first term is the policy's own no-gradient log-probability on
the real image, which the trainer already has (`old_per_token_logps`, computed for every batch
because vLLM's importance-sampling correction needs it); only the masked forward pass carries
gradient, so maximizing the KL lowers the probability of the same answer without the image.
Each token's term is capped, so the policy cannot raise the loss without bound by producing
nonsense on masked images. Masking zeroes whole 2x2 merge groups of vision patches (the unit
Qwen3-VL's merger reads), a fixed share of them, with a seeded generator per update.
"""

import torch
from trl import GRPOTrainer

MASK_RATIO = 0.6  # PAPO's patch masking ratio
MERGE_GROUP = 4  # Qwen-VL patches arrive in consecutive groups of spatial_merge_size ** 2
KL_CAP = 2.0  # nats per token


def mask_patches(pixel_values: torch.Tensor, ratio: float, generator: torch.Generator,
                 group: int = MERGE_GROUP) -> torch.Tensor:
    """A copy of the flattened vision patches with a `ratio` share of whole merge groups set
    to zero (the normalized mean intensity)."""
    groups = pixel_values.shape[0] // group
    masked = pixel_values.clone()
    chosen = torch.randperm(groups, generator=generator)[: round(ratio * groups)]
    rows = (chosen.unsqueeze(1) * group + torch.arange(group)).flatten()
    masked[rows.to(pixel_values.device)] = 0
    return masked


class PerceptionGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, perception_coef: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.perception_coef = perception_coef
        self._mask_generator = torch.Generator().manual_seed(self.args.seed)

    def _compute_loss(self, model, inputs):
        loss = super()._compute_loss(model, inputs)
        pixel_values = inputs.get("pixel_values")
        reference = inputs.get("old_per_token_logps")
        if self.perception_coef == 0 or pixel_values is None or reference is None:
            return loss
        prompt_ids, completion_ids = inputs["prompt_ids"], inputs["completion_ids"]
        mask = inputs["completion_mask"]
        masked_logps, _, _ = self._get_per_token_logps_and_entropies(
            model,
            torch.cat([prompt_ids, completion_ids], dim=1),
            torch.cat([inputs["prompt_mask"], mask], dim=1),
            completion_ids.size(1),
            pixel_values=mask_patches(pixel_values, MASK_RATIO, self._mask_generator),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
        )
        per_token = (reference - masked_logps).clamp(max=KL_CAP)
        kl = (per_token * mask).sum() / mask.sum().clamp(min=1.0)
        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["perception/kl"].append(self.accelerator.gather(kl.detach()).mean().item())
        return loss - self.perception_coef * kl
