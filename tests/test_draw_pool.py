"""Actual RL exposure can supervise SFT without admitting held-out images or changing labels."""

import hashlib
import json
from copy import deepcopy

import pytest
from fixtures import every_dataset, png, save_subset, write_splits

from scripts.build_draw_pool import build_draw_pool, draw_keys, ordered_sha256
from vqar import config
from vqar.pools import load_pool, pool_path
from vqar.pools import provenance as pool_provenance
from vqar.splits import load_role
from vqar.splits import provenance as split_provenance


@pytest.fixture
def campaign(tmp_path):
    data_dir = tmp_path / "data"
    every_dataset(data_dir)
    # Distinct raw targets expose any normalization or RL-reward substitution in SFT.
    save_subset(data_dir, "gqa_rl", "gqa", [
        (f"gqa_rl-{i}", png(80 + i), "What is there?", ["A red car.", "car"], "")
        for i in range(3)
    ])
    write_splits(data_dir, ["vqav2", "gqa", "clevr", "textvqa"])
    sft = load_role(data_dir, ("vqav2", "gqa"), "sft", seed=None)
    keys = [(sft[0]["dataset"], sft[0]["qid"]), ("gqa", "gqa_rl-0")]
    draws = tmp_path / "draws.jsonl"
    records = [{"step_before_update": i, "dataset": [d] * 2, "qid": [q] * 2,
                "scores": [0.0, 1.0]} for i, (d, q) in enumerate(keys)]
    draws.write_text("".join(json.dumps(row) + "\n" for row in records))
    source = tmp_path / "run_config.json"
    source.write_text(json.dumps({
        "preset": {"datasets": ["vqav2", "gqa"], "seed": 42, "max_steps": 2,
                   "num_generations": 2, "model": "instruct"},
        "policy_model": "instruct", "data": split_provenance(data_dir, ("vqav2", "gqa")),
        "pool": None,
    }))
    return data_dir, draws, source, keys


def declare(campaign):
    data_dir, draws, source, _ = campaign
    pool = build_draw_pool(data_dir, "actual", draws, source, 2, 2, 1, 2)
    path = pool_path(data_dir, "actual")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(pool))
    return pool


def test_draw_pool_selects_sft_and_rl_only_rows_with_original_targets(campaign, monkeypatch):
    import train_sft

    data_dir, draws, _, keys = campaign
    pool = declare(campaign)
    assert [(m["dataset"], m["qid"]) for m in pool["members"]] == keys
    assert pool["exposure"]["selected_in_full_sft"] == 1
    assert pool["exposure"]["selected_rl_only"] == 1
    assert pool["selection"]["draws"]["sha256"] == hashlib.sha256(draws.read_bytes()).hexdigest()
    rows = load_pool(data_dir, ("vqav2", "gqa"), 42, "actual")
    order = [{"dataset": d, "qid": q} for d, q in zip(rows["dataset"], rows["qid"], strict=True)]
    assert (ordered_sha256(order)
            == pool["selection"]["loaded_dataset_order"]["ordered_questions_sha256"])
    monkeypatch.setattr(train_sft, "ROOT", data_dir.parent)
    train, dev = train_sft.load_datasets(config.SFT_PRESETS["sft_2b"], "actual")
    assert len(train) == len(dev) == 2
    assert sorted(row["completion"][0]["content"] for row in train) == ["2", "A red car."]
    monkeypatch.setattr("sys.argv", ["train_sft.py", "--preset", "sft_2b", "--pool", "actual",
                                     "--dry-run", "--no-wandb"])
    train_sft.main()  # Exercises the public CLI and image/answer transform without a GPU.


@pytest.mark.parametrize("damage, message", [
    ("missing", "consecutive updates"),
    ("repeat", "distinct questions"),
    ("short-group", "distinct groups"),
    ("malformed", "differ in length"),
    ("refill", "Refill draws"),
])
def test_invalid_exposure_is_rejected(campaign, damage, message):
    _, draws, _, _ = campaign
    records = [json.loads(line) for line in draws.read_text().splitlines()]
    if damage == "missing":
        records.pop()
    elif damage == "repeat":
        records[1].update(dataset=records[0]["dataset"], qid=records[0]["qid"])
    elif damage == "short-group":
        for column in ("dataset", "qid", "scores"):
            records[1][column].pop()
    elif damage == "malformed":
        records[1]["qid"].pop()
    else:
        records[1]["round"] = 0
    draws.write_text("".join(json.dumps(row) + "\n" for row in records))
    with pytest.raises(ValueError, match=message):
        draw_keys(draws, 2, 2, 1, 2)


def test_resumed_draws_use_the_last_attempt_of_each_update(campaign):
    _, draws, _, keys = campaign
    abandoned = {"step_before_update": 0, "dataset": ["missing"] * 2,
                 "qid": ["abandoned"] * 2, "scores": [0, 0]}
    draws.write_text(json.dumps(abandoned) + "\n" + draws.read_text())
    assert draw_keys(draws, 2, 2, 1, 2) == keys


def test_held_out_draw_identity_is_rejected(campaign):
    data_dir, draws, source, _ = campaign
    records = [json.loads(line) for line in draws.read_text().splitlines()]
    records[1].update(dataset=["vqav2"] * 2, qid=["vqav2_eval-0"] * 2)
    draws.write_text("".join(json.dumps(row) + "\n" for row in records))
    with pytest.raises(ValueError, match="not in the source RL roles"):
        build_draw_pool(data_dir, "heldout", draws, source, 2, 2, 1, 2)


def test_changed_split_file_or_duplicate_member_invalidates_pool(campaign):
    data_dir, _, _, _ = campaign
    pool = declare(campaign)
    path = pool_path(data_dir, "actual")
    pool["members"].append(pool["members"][0])
    path.write_text(json.dumps(pool))
    with pytest.raises(ValueError, match="nonempty and distinct"):
        load_pool(data_dir, ("vqav2", "gqa"), 42, "actual")
    pool["members"].pop()
    pool["source_data"]["split_sha256"]["gqa"] = "changed"
    path.write_text(json.dumps(pool))
    with pytest.raises(ValueError, match="split files differ"):
        load_pool(data_dir, ("vqav2", "gqa"), 42, "actual")


def test_a_resume_cannot_change_the_selected_questions(campaign, tmp_path):
    from vqar.training import write_run_config

    data_dir, _, _, _ = campaign
    declare(campaign)
    output = tmp_path / "checkpoints"
    record = {"preset": config.dump(config.SFT_PRESETS["sft_2b"]),
              "pool": pool_provenance(data_dir, "actual")}

    class Arguments:
        def to_dict(self):
            return {"max_steps": -1}

    write_run_config(output, record, Arguments())
    write_run_config(output, record, Arguments())
    path = pool_path(data_dir, "actual")
    path.write_text(path.read_text() + "\n")
    record["pool"] = pool_provenance(data_dir, "actual")
    with pytest.raises(ValueError, match="Existing run differs in.*pool"):
        write_run_config(output, record, Arguments())


def test_sft_adapter_initialization_uses_the_declared_seed(tmp_path, monkeypatch):
    """Exercise the actual TRL/PEFT constructor before Trainer seeds its own RNG."""
    torch = pytest.importorskip("torch")  # the GPU stack is absent from the CPU checks
    trl = pytest.importorskip("trl")
    from datasets import Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    import train_sft

    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=3, hidden_size=16,
                                       intermediate_size=32, num_hidden_layers=1,
                                       num_attention_heads=2, num_key_value_heads=2, head_dim=8))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"<pad>": 0, "<eos>": 1, "a": 2}, unk_token="a")),
        pad_token="<pad>", eos_token="<eos>")
    rows = Dataset.from_dict({"input_ids": [[2, 2, 1]]})
    sft_config = trl.SFTConfig

    def cpu_config(**kwargs):
        return sft_config(**dict(kwargs, use_cpu=True, bf16=False))

    initialized = []

    def capture(trainer, *args):
        initialized.append({name: value.detach().clone() for name, value in
                            trainer.model.named_parameters() if "lora_A" in name})
        return False

    monkeypatch.setattr(trl, "SFTConfig", cpu_config)
    monkeypatch.setattr(train_sft, "load_datasets", lambda *args: (rows, rows))
    monkeypatch.setattr(train_sft, "build_model", lambda preset: (deepcopy(model), tokenizer))
    monkeypatch.setattr(train_sft, "write_run_config", lambda *args: None)
    monkeypatch.setattr(train_sft, "run_training", capture)
    monkeypatch.setattr("sys.argv", ["train_sft.py", "--preset", "sft_2b", "--no-wandb",
                                     "--run-dir", str(tmp_path / "seed-check")])
    for inherited_seed in (111, 999):
        torch.manual_seed(inherited_seed)
        train_sft.main()
    assert initialized[0]
    assert all(torch.equal(initialized[0][name], initialized[1][name])
               for name in initialized[0])
