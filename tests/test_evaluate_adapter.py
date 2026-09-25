"""evaluate.py serves a LoRA adapter unmerged at the rank its configuration declares."""

import json

from evaluate import adapter_rank


def test_adapter_rank_reads_the_adapter_configuration(tmp_path):
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 16}))
    assert adapter_rank(str(tmp_path)) == 16
    assert adapter_rank(None) is None
