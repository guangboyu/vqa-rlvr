"""The perception loss's patch masking: whole merge groups, a fixed share, seeded."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")

from vqar.perception_trainer import mask_patches  # noqa: E402


def test_masking_zeroes_a_share_of_whole_merge_groups_and_leaves_the_input_alone():
    pixels = torch.ones(40, 3)  # ten merge groups of four patches
    masked = mask_patches(pixels, 0.6, torch.Generator().manual_seed(0))
    zero_rows = (masked == 0).all(dim=1)
    assert zero_rows.sum().item() == 24  # six groups
    assert all(zero_rows[i:i + 4].all() or not zero_rows[i:i + 4].any() for i in range(0, 40, 4))
    assert pixels.eq(1).all()  # a copy, not in place
    again = mask_patches(pixels, 0.6, torch.Generator().manual_seed(0))
    assert torch.equal(masked, again)
