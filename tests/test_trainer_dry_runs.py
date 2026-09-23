"""Each trainer's dry run resolves its preset and loads only its own role (tiny data)."""

import importlib

import pytest
from fixtures import png, save_subset, write_splits

LAYOUT = {  # subset -> (dataset, references per question)
    "vqav2_sft": ("vqav2", 1), "vqav2_rl": ("vqav2", 1), "vqav2_eval": ("vqav2", 10),
    "gqa_sft": ("gqa", 1), "gqa_rl": ("gqa", 1), "gqa_eval": ("gqa", 1),
    "clevr_rl": ("clevr", 1), "clevr_val": ("clevr", 1), "clevr_test": ("clevr", 1),
    "textvqa_eval": ("textvqa", 10),
}


@pytest.fixture
def repo(tmp_path, monkeypatch):
    data = tmp_path / "data"
    for number, (subset, (dataset, references)) in enumerate(LAYOUT.items()):
        save_subset(data, subset, dataset, [
            (f"{subset}-{i}", png(20 * number + i), "How many?", ["2"] * references, "number")
            for i in range(3)
        ])
    write_splits(data, ["vqav2", "gqa", "clevr", "textvqa"])
    import train_sft
    import vqar.rl

    monkeypatch.setattr(train_sft, "ROOT", tmp_path)
    monkeypatch.setattr(vqar.rl, "ROOT", tmp_path)
    return tmp_path


# SFT trains on 2 of each SFT pool's 3 rows (one image is dev); RL adds each pool's 3 RL
# rows to those 2 SFT training rows, so 5 per dataset.
@pytest.mark.parametrize("module, preset, expected", [
    ("train_sft", "sft_2b", "train=4, dev=2"),
    ("train_grpo", "grpo_2b_main", "dataset ok: 10 prompts"),
    ("train_opd", "opd_self_2b_main", "dataset ok: 10 prompts"),
])
def test_dry_run_loads_the_trainers_role(repo, module, preset, expected, monkeypatch, capsys):
    trainer = importlib.import_module(module)
    monkeypatch.setattr("sys.argv", [f"{module}.py", "--preset", preset, "--hardware", "24gb",
                                     "--dry-run", "--no-wandb"])
    trainer.main()
    assert expected in capsys.readouterr().out
