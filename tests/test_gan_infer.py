import numpy as np
import pytest
import torch

from src.gan.infer import MemoryGAN, slerp
from src.gan.models import Generator, ModelConfig


def model() -> MemoryGAN:
    config = ModelConfig(
        resolution=8, z_dim=4, context_dim=4, relationship_dim=8, emotion_dim=6,
        occasion_dim=8, condition_dim=4, w_dim=4, mapping_layers=2,
        channel_base=64, channel_max=16,
    )
    torch.manual_seed(3)
    return MemoryGAN(Generator(config))


def test_slerp_preserves_endpoints_and_follows_sphere() -> None:
    start = torch.tensor([[1.0, 0.0]])
    end = torch.tensor([[0.0, 1.0]])
    assert torch.equal(slerp(start, end, 0.0), start)
    assert torch.equal(slerp(start, end, 1.0), end)
    assert torch.allclose(slerp(start, end, 0.5), torch.full((1, 2), 2**-0.5))


def test_generate_is_deterministic_and_returns_rgb_image() -> None:
    gan = model()
    context = np.arange(4, dtype=np.float32)
    style = context[::-1].copy()
    kwargs = dict(
        relationship_type="colleague", emotion_tag="gratitude", occasion="promotion",
        agency_slider=0.5, human_style_ref=style, seed=11,
    )
    first = gan.generate(context, **kwargs)
    second = gan.generate(context, **kwargs)
    assert first.mode == "RGB" and first.size == (8, 8)
    assert np.array_equal(np.asarray(first), np.asarray(second))


def test_generate_validates_agency_and_style_reference() -> None:
    gan = model()
    context = np.zeros(4, dtype=np.float32)
    with pytest.raises(ValueError, match="human_style_ref"):
        gan.generate(context, "colleague", "joy", "promotion", 0.5)
    with pytest.raises(ValueError, match="agency_slider"):
        gan.generate(context, "colleague", "joy", "promotion", 1.1)

