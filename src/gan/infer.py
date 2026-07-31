from __future__ import annotations

from dataclasses import fields
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F

from .models import Generator, ModelConfig

RELATIONSHIP_TYPES = ("partner", "parent-child", "sibling", "friend", "colleague", "extended-family", "mentor", "other")
EMOTION_TAGS = ("joy", "nostalgia", "gratitude", "humor", "comfort", "other")
OCCASIONS = ("birthday", "anniversary", "graduation", "housewarming", "promotion", "holiday", "thank-you", "other")


def slerp(start: Tensor, end: Tensor, amount: float, *, eps: float = 1e-7) -> Tensor:
    """Spherically interpolate vectors along their final dimension."""
    if not 0.0 <= amount <= 1.0:
        raise ValueError("amount must be between 0 and 1")
    if start.shape != end.shape:
        raise ValueError("slerp endpoints must have the same shape")
    if amount == 0.0:
        return start
    if amount == 1.0:
        return end
    start_norm = torch.linalg.vector_norm(start, dim=-1, keepdim=True)
    end_norm = torch.linalg.vector_norm(end, dim=-1, keepdim=True)
    safe_start = start / start_norm.clamp_min(eps)
    safe_end = end / end_norm.clamp_min(eps)
    dot = (safe_start * safe_end).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    spherical = (
        torch.sin((1.0 - amount) * theta) / sin_theta.clamp_min(eps) * safe_start
        + torch.sin(amount * theta) / sin_theta.clamp_min(eps) * safe_end
    )
    linear_direction = F.normalize(
        (1.0 - amount) * safe_start + amount * safe_end, dim=-1, eps=eps
    )
    direction = torch.where(sin_theta.abs() > eps, spherical, linear_direction)
    result = direction * torch.lerp(start_norm, end_norm, amount)
    return torch.where(
        (start_norm < eps) | (end_norm < eps), torch.lerp(start, end, amount), result
    )


class MemoryGAN:
    """Inference wrapper for the conditional MemoryGAN generator."""

    def __init__(self, generator: Generator, *, device: torch.device | str = "cpu") -> None:
        self.device = torch.device(device)
        self.generator = generator.to(self.device).eval().requires_grad_(False)
        self.config = generator.config

    @classmethod
    def load(cls, checkpoint_path: str) -> "MemoryGAN":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint must contain a mapping")
        raw_config = checkpoint.get("model_config")
        if not isinstance(raw_config, dict):
            raise ValueError("checkpoint is missing model_config")
        allowed = {field.name for field in fields(ModelConfig)}
        config = ModelConfig(**{key: value for key, value in raw_config.items() if key in allowed})
        state = checkpoint.get("generator_ema", checkpoint.get("generator"))
        if not isinstance(state, dict):
            raise ValueError("checkpoint is missing generator weights")
        generator = Generator(config)
        generator.load_state_dict(state)
        return cls(generator, device=device)

    def generate(
        self,
        context_embedding: np.ndarray,
        relationship_type: str,
        emotion_tag: str,
        occasion: str,
        agency_slider: float,
        human_style_ref: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> Image.Image:
        if not np.isfinite(agency_slider) or not 0.0 <= agency_slider <= 1.0:
            raise ValueError("agency_slider must be a finite value between 0 and 1")
        if agency_slider < 1.0 and human_style_ref is None:
            raise ValueError("human_style_ref is required when agency_slider < 1")
        context = self._embedding(context_embedding, "context_embedding")
        human = self._embedding(human_style_ref, "human_style_ref") if human_style_ref is not None else None
        relationship = self._one_hot(relationship_type, RELATIONSHIP_TYPES, self.config.relationship_dim)
        emotion = self._one_hot(emotion_tag, EMOTION_TAGS, self.config.emotion_dim)
        occasion_vector = self._one_hot(occasion, OCCASIONS, self.config.occasion_dim)

        local_seed = int(seed) if seed is not None else int.from_bytes(np.random.bytes(8), "little")
        random = torch.Generator(device=self.device).manual_seed(local_seed)
        z = torch.randn(1, self.config.z_dim, generator=random, device=self.device)
        noises = self._noises(random)
        with torch.inference_mode():
            w_ai = self.generator.map(z, context, relationship, emotion, occasion_vector)
            if agency_slider == 1.0:
                w_final = w_ai
            else:
                assert human is not None
                w_human = self.generator.map(z, human, relationship, emotion, occasion_vector)
                w_final = slerp(w_human, w_ai, float(agency_slider))
            image = self.generator.synthesis(w_final, noises=noises)[0]
        pixels = image.add(1).mul(127.5).clamp(0, 255).permute(1, 2, 0).byte().cpu().numpy()
        return Image.fromarray(pixels, mode="RGB")

    def _embedding(self, value: np.ndarray, name: str) -> Tensor:
        array = np.asarray(value, dtype=np.float32)
        if array.shape != (self.config.context_dim,):
            raise ValueError(f"{name} must have shape ({self.config.context_dim},)")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
        return torch.from_numpy(array.copy()).unsqueeze(0).to(self.device)

    def _one_hot(self, value: str, vocabulary: tuple[str, ...], width: int) -> Tensor:
        active_vocabulary = vocabulary[:width]
        try:
            index = active_vocabulary.index(value)
        except ValueError as error:
            raise ValueError(f"unknown category {value!r}; expected one of {active_vocabulary}") from error
        output = torch.zeros(1, width, device=self.device)
        output[0, index] = 1.0
        return output

    def _noises(self, random: torch.Generator) -> list[Tensor]:
        resolutions = [4]
        for resolution in self.generator.resolutions[1:]:
            resolutions.extend((resolution, resolution))
        return [
            torch.randn(1, 1, resolution, resolution, generator=random, device=self.device)
            for resolution in resolutions
        ]

