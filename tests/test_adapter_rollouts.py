"""Unmerged LoRA rollouts (--rollout-lora): plumbing, adapter export, the rollout's LoRA
request, and the engine settings against TRL's, with fake engines and a tiny PEFT model."""

import json
from argparse import Namespace
from dataclasses import replace
from types import SimpleNamespace

import pytest
from fixtures import every_dataset

from vqar import config
from vqar.training import resolve_preset


def test_off_by_default_and_absent_from_the_recorded_preset():
    preset = resolve_preset(config.GRPO_PRESETS, Namespace(preset="grpo_2b_main",
                                                           hardware="24gb", rollout_lora=None))
    assert preset.rollout_lora is False
    assert "rollout_lora" not in config.dump(preset)
    assert config.dump(replace(preset, rollout_lora=True))["rollout_lora"] is True


def test_the_flag_turns_it_on(tmp_path, monkeypatch, capsys):
    every_dataset(tmp_path / "data")
    import train_grpo
    import vqar.rl

    monkeypatch.setattr(vqar.rl, "ROOT", tmp_path)
    argv = ["train_grpo.py", "--preset", "grpo_2b_main", "--hardware", "24gb", "--dry-run"]
    monkeypatch.setattr("sys.argv", argv + ["--rollout-lora"])
    train_grpo.main()
    assert "vLLM serves the rank-16 adapter unmerged" in capsys.readouterr().out
    monkeypatch.setattr("sys.argv", argv)
    train_grpo.main()
    assert "unmerged" not in capsys.readouterr().out


class FakeLLM:
    """Records what the rollout and the engine setup ask of vLLM."""

    def __init__(self, **settings):
        self.settings, self.calls = settings, []

    def chat(self, conversations, params, use_tqdm, lora_request=None):
        self.calls.append(("chat", lora_request))
        sample = SimpleNamespace(token_ids=[5, 6], logprobs=[{5: SimpleNamespace(logprob=-0.1)},
                                                             {6: SimpleNamespace(logprob=-0.2)}])
        return [SimpleNamespace(prompt_token_ids=[1, 2], outputs=[sample])
                for _ in conversations]

    def wake_up(self, tags=None):
        self.calls.append(("wake_up", tuple(tags)))

    def sleep(self, level):
        self.calls.append(("sleep", level))


def fake_trainer(llm, step, adapter_request=None):
    generation = SimpleNamespace(llm=llm, enable_sleep_mode=True)
    if adapter_request:
        generation.adapter_request = adapter_request
    return SimpleNamespace(vllm_generation=generation, state=SimpleNamespace(global_step=step),
                           args=SimpleNamespace(temperature=1.0, top_p=1.0,
                                                max_completion_length=8))


PROMPTS = [[{"role": "user", "content": [{"type": "text", "text": "q"}]}]] * 2


@pytest.mark.parametrize("serve_adapter", [False, True])
def test_the_rollout_passes_the_updates_adapter_only_when_serving_it(serve_adapter):
    pytest.importorskip("vllm")
    from vqar.rl import make_chat_rollout

    llm, asked = FakeLLM(), []
    trainer = fake_trainer(llm, step=7, adapter_request=lambda step: asked.append(step) or "req")
    output = make_chat_rollout(serve_adapter=serve_adapter)(PROMPTS, trainer)
    assert output["completion_ids"] == [[5, 6], [5, 6]]
    assert output["logprobs"] == [[-0.1, -0.2], [-0.1, -0.2]]
    if serve_adapter:  # level 1 keeps the base weights; nothing re-syncs them
        assert asked == [7]
        assert llm.calls == [("wake_up", ("kv_cache",)), ("chat", "req"), ("sleep", 1)]
    else:
        assert asked == []
        assert llm.calls == [("wake_up", ("kv_cache",)), ("chat", None), ("sleep", 2)]


@pytest.fixture
def tiny_policy():
    """A two-layer PEFT LoRA model with nonzero adapter weights."""
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")

    class Tiny(torch.nn.Module):
        name_or_path = "tiny-policy"  # what the engine loads

        def __init__(self):
            super().__init__()
            self.q_proj, self.v_proj = torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)

        def forward(self, x):
            return self.v_proj(self.q_proj(x))

    torch.manual_seed(0)
    lora = peft.LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"])
    model = peft.get_peft_model(Tiny(), lora)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_()
    return model


def adapter_generation(model, tmp_path):
    pytest.importorskip("trl")
    from vqar.adapter_rollouts import AdapterExport, AdapterVLLMGeneration

    generation = AdapterVLLMGeneration.__new__(AdapterVLLMGeneration)  # no engine
    generation.model, generation.adapters = model, AdapterExport(parent=str(tmp_path))
    return generation


def test_each_update_exports_its_adapter_once_under_a_new_id(tiny_policy, tmp_path):
    pytest.importorskip("vllm")
    generation = adapter_generation(tiny_policy, tmp_path)
    first = generation.adapter_request(step=3)
    again = generation.adapter_request(step=3)  # a refill round of the same update
    assert again == first and again.lora_int_id == first.lora_int_id == 1
    written = generation.adapters.path
    later = generation.adapter_request(step=4)
    assert (later.lora_int_id, later.lora_name) == (2, "policy-2")
    assert later.lora_path != first.lora_path and not written.exists()  # only the latest
    assert sorted(p.name for p in generation.adapters.root.iterdir()) == ["adapter-2-step-4"]
    generation.adapters.close()
    assert not generation.adapters.root.exists()


def test_the_exported_adapter_is_the_trainers_and_vllm_accepts_it(tiny_policy, tmp_path):
    import torch
    from peft import PeftModel

    pytest.importorskip("vllm")
    from vllm.config.lora import LoRAConfig
    from vllm.lora.peft_helper import PEFTHelper

    generation = adapter_generation(tiny_policy, tmp_path)
    path = generation.adapter_request(step=0).lora_path
    saved = json.loads((generation.adapters.path / "adapter_config.json").read_text())
    assert (saved["r"], saved["lora_alpha"]) == (4, 8)
    assert sorted(saved["target_modules"]) == ["q_proj", "v_proj"]

    helper = PEFTHelper.from_local_dir(path, max_position_embeddings=None)
    helper.validate_legal(LoRAConfig(max_lora_rank=8))
    assert helper.vllm_lora_scaling_factor == 8 / 4  # PEFT's alpha / r

    reloaded = PeftModel.from_pretrained(type(tiny_policy.get_base_model())(), path)
    trained = dict(tiny_policy.named_parameters())
    for name, param in reloaded.named_parameters():
        if "lora_" in name:
            assert param.dtype == torch.float32  # saved as trained, not rounded to bf16
            assert torch.equal(param, trained[name])


class FakeAccelerator:
    num_processes, process_index, local_process_index = 1, 0, 0

    def wait_for_everyone(self):
        pass


def engine_settings(generation_class, monkeypatch, model):
    """The LLM settings and sleep calls of a backend's engine setup, with a fake LLM."""
    from trl.generation import vllm_generation

    built = []

    def fake_llm(**settings):
        built.append(FakeLLM(**settings))
        return built[-1]

    monkeypatch.setattr(vllm_generation, "LLM", fake_llm)
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.setenv(name, "")  # restored after the test
        monkeypatch.delenv(name)
    generation = generation_class.__new__(generation_class)
    generation.__dict__.update(
        model=model, accelerator=FakeAccelerator(), mode="colocate", tensor_parallel_size=1,
        gpu_memory_utilization=0.3, max_model_length=2048, max_num_seqs=32,
        enable_sleep_mode=True, model_impl="auto", trust_remote_code=False)
    generation._init_vllm()
    (engine,) = built
    return generation, engine.settings, engine.calls


def test_the_engine_is_trls_plus_lora_and_sleeps_at_level_one(tiny_policy, monkeypatch):
    pytest.importorskip("vllm")
    from trl.generation.vllm_generation import VLLMGeneration

    from vqar.adapter_rollouts import AdapterVLLMGeneration

    _, trl_settings, trl_calls = engine_settings(VLLMGeneration, monkeypatch, tiny_policy)
    generation, settings, calls = engine_settings(AdapterVLLMGeneration, monkeypatch,
                                                  tiny_policy)
    assert settings == {**trl_settings, "enable_lora": True, "max_lora_rank": 4,
                        "max_loras": 1}
    assert (trl_calls, calls) == ([("sleep", 2)], [("sleep", 1)])
    generation.adapters.close()


def test_the_weight_sync_only_wakes_the_offloaded_weights(tiny_policy, monkeypatch):
    pytest.importorskip("trl")
    from vqar.adapter_rollouts import AdapterVLLMGeneration

    generation = AdapterVLLMGeneration.__new__(AdapterVLLMGeneration)
    generation.__dict__.update(model=tiny_policy, llm=FakeLLM(), enable_sleep_mode=True)
    before = {n: p.clone() for n, p in tiny_policy.named_parameters()}
    generation.sync_weights()  # a real sync would merge the adapter and load_weights
    assert generation.llm.calls == [("wake_up", ("weights",))]
    assert all((p == before[n]).all() for n, p in tiny_policy.named_parameters())


def test_trainers_built_in_the_block_get_the_adapter_backend():
    pytest.importorskip("trl")
    from trl.generation.vllm_generation import VLLMGeneration
    from trl.trainer import grpo_trainer

    from vqar.adapter_rollouts import AdapterVLLMGeneration, unmerged_adapter_engine

    with unmerged_adapter_engine():
        assert grpo_trainer.VLLMGeneration is AdapterVLLMGeneration
    assert grpo_trainer.VLLMGeneration is VLLMGeneration
