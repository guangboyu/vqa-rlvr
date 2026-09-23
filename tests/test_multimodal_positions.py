"""Optional integration check: cached processor/config only, no model weights or GPU."""

import types

import pytest
from PIL import Image

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("trl")


def test_left_padding_preserves_native_image_positions():
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
    from trl.data_utils import prepare_multimodal_messages

    model_id = "Qwen/Qwen3-VL-2B-Instruct"
    try:
        processor = transformers.AutoProcessor.from_pretrained(
            model_id, padding_side="left", local_files_only=True
        )
        config = transformers.AutoConfig.from_pretrained(model_id, local_files_only=True)
    except OSError:
        pytest.skip("Qwen processor/config are not cached")
    image = Image.new("RGB", (608, 608))
    messages = [
        prepare_multimodal_messages([{"role": "user", "content": question}], images=[image])
        for question in ("What is this?", "What is the color of the object on the left side?")
    ]
    batch = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        padding=True,
        return_dict=True,
        return_tensors="pt",
    )
    model = types.SimpleNamespace(config=config)
    model.get_vision_position_ids = types.MethodType(Qwen3VLModel.get_vision_position_ids, model)
    position, _ = Qwen3VLModel.get_rope_index(
        model,
        batch["input_ids"],
        batch["mm_token_type_ids"],
        image_grid_thw=batch["image_grid_thw"],
        attention_mask=batch["attention_mask"],
    )
    assert batch["attention_mask"][0, 0] == 0
    for i in range(2):
        mask = batch["attention_mask"][i].bool()
        ids = batch["input_ids"][i, mask].unsqueeze(0)
        expected, _ = Qwen3VLModel.get_rope_index(
            model,
            ids,
            (ids == config.image_token_id).long(),
            image_grid_thw=batch["image_grid_thw"][i : i + 1],
        )
        torch.testing.assert_close(position[:, i, mask], expected[:, 0])
