"""Rollouts from the unmerged LoRA adapter (`train_grpo.py --rollout-lora`) on TRL 1.8.

TRL's colocate weight sync merges the adapter into the bf16 base weights and loads them into
vLLM. A learning-rate-1e-5 update is far below one bf16 step of most weights, so the merged
sampler rounds away much of what the trainer learned (docs/harness/method.md). Here
vLLM keeps the base weights it loaded and serves the trainer's adapter unmerged, as the dev
evaluation does: the adapter is written to a local directory once per update and passed to
`llm.chat` as a LoRARequest (`vqar.rl.make_chat_rollout(serve_adapter=True)`).

A separate module because it imports TRL at load time: train_grpo imports it only when the
option is on.
"""

import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from accelerate.utils import is_peft_model
from trl.generation import vllm_generation
from trl.trainer import grpo_trainer
from trl.trainer.utils import ensure_master_addr_port


class AdapterExport:
    """The policy's LoRA adapter as files vLLM can load: written once per update, each time
    to a new directory under a new adapter id, so vLLM never serves a cached older adapter.

    The directory lives under the process's temporary directory (node-local on the cluster,
    where Slurm points TMPDIR). Only the latest adapter is kept: vLLM has read the previous
    one into memory by the time the next update writes its own.
    """

    def __init__(self, parent: str | None = None):
        self._root = tempfile.TemporaryDirectory(prefix="vqar-rollout-adapter-", dir=parent)
        self.root = Path(self._root.name)
        self.adapter_id = 0  # vLLM ids start at 1
        self.path: Path | None = None
        self._step: int | None = None

    def for_step(self, model, step: int) -> tuple[int, Path]:
        """The adapter id and directory for this update, written on its first request."""
        if step != self._step:
            self._write(model, step)
            self._step = step
        return self.adapter_id, self.path

    def _write(self, model, step: int) -> None:
        adapter_id = self.adapter_id + 1
        final = self.root / f"adapter-{adapter_id}-step-{step}"
        partial = final.with_name(final.name + ".tmp")
        # PEFT's own format, as saved in checkpoints (fp32 LoRA weights, alpha, rank,
        # targets); vLLM casts the weights to its LoRA dtype, the model's bf16. No embedding
        # check: it would look up the base model's config on the Hub at every update.
        model.save_pretrained(partial, save_embedding_layers=False)
        partial.rename(final)
        if self.path is not None:
            shutil.rmtree(self.path)
        self.adapter_id, self.path = adapter_id, final

    def close(self) -> None:
        self._root.cleanup()


class AdapterVLLMGeneration(vllm_generation.VLLMGeneration):
    """TRL's colocated vLLM backend with LoRA serving instead of merged-weight syncs.

    Three changes, all in colocate mode on one process:
    - the engine is built with TRL's settings plus LoRA (rank from the policy's adapter);
    - it sleeps at level 1, which offloads the weights to CPU memory and discards the KV
      cache, instead of level 2, which discards the weights TRL would sync again;
    - `sync_weights` loads nothing: it wakes the weights the sleep offloaded, and the
      rollout asks `adapter_request` for the adapter of the current update.
    """

    def _init_vllm(self):
        if self.mode != "colocate" or self.accelerator.num_processes != 1:
            raise ValueError("Unmerged LoRA rollouts support one colocated process.")
        if not is_peft_model(self.model):
            raise ValueError("Unmerged LoRA rollouts need a LoRA policy (a peft_config).")
        # What TRL 1.8's _init_vllm sets for one process (tensor parallel 1, no quantization:
        # train_grpo rejects NF4 students).
        os.environ["RANK"] = os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        ensure_master_addr_port()
        self.llm = vllm_generation.LLM(
            model=self.model.name_or_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_length,
            max_num_seqs=self.max_num_seqs,
            enable_sleep_mode=self.enable_sleep_mode,
            model_impl=self.model_impl,
            distributed_executor_backend="external_launcher",
            seed=0,
            max_num_batched_tokens=4096,
            logprobs_mode="processed_logprobs",
            quantization=None,
            trust_remote_code=self.trust_remote_code,
            enable_lora=True,
            max_lora_rank=self.model.active_peft_config.r,
            max_loras=1,
        )
        if self.enable_sleep_mode:
            self.llm.sleep(level=1)
        self.adapters = AdapterExport()
        self.accelerator.wait_for_everyone()

    def sync_weights(self):
        """TRL calls this before a rollout when the policy may have changed. The base
        weights never change, so only wake them, as TRL's sync does before loading."""
        if self.enable_sleep_mode:
            vllm_generation.empty_cache()
            self.llm.wake_up(tags=["weights"])

    def adapter_request(self, step: int):
        """The trainer's current adapter as a vLLM LoRARequest. Every generation round of
        an update (refill rounds included) shares one export."""
        from vllm.lora.request import LoRARequest

        adapter_id, path = self.adapters.for_step(self.model, step)
        return LoRARequest(f"policy-{adapter_id}", adapter_id, str(path))


@contextmanager
def unmerged_adapter_engine():
    """Build GRPOTrainer (or a subclass) inside this block to give it AdapterVLLMGeneration
    in place of TRL's VLLMGeneration."""
    with patch.object(grpo_trainer, "VLLMGeneration", AdapterVLLMGeneration):
        yield
