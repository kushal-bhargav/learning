from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, autograd, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    resolution: int = 256
    image_channels: int = 3
    z_dim: int = 512
    context_dim: int = 512
    relationship_dim: int = 8
    emotion_dim: int = 6
    occasion_dim: int = 8
    condition_dim: int = 512
    w_dim: int = 512
    mapping_layers: int = 8
    channel_base: int = 16384
    channel_max: int = 512
    r1_gamma: float = 0.8
    path_length_weight: float = 2.0
    path_length_interval: int = 4
    path_length_decay: float = 0.01
    clip_consistency_weight: float = 0.1
    ada_target: float = 0.6
    ada_interval: int = 4
    ada_kimg: int = 500
    augmentation: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelConfig":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def __post_init__(self) -> None:
        if self.resolution < 8 or self.resolution & (self.resolution - 1):
            raise ValueError("resolution must be a power of two and at least 8")

    def channels(self, resolution: int) -> int:
        return min(self.channel_base // resolution, self.channel_max)


class PixelNorm(nn.Module):
    def forward(self, value: Tensor) -> Tensor:
        return value * torch.rsqrt(value.square().mean(dim=1, keepdim=True) + 1e-8)


class EqualLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.scale = 1 / math.sqrt(in_features)
        self.activation = activation

    def forward(self, value: Tensor) -> Tensor:
        output = F.linear(value, self.weight * self.scale, self.bias)
        return F.leaky_relu(output, 0.2) * math.sqrt(2) if self.activation else output


class ConditioningEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        raw_dim = (
            config.context_dim
            + config.relationship_dim
            + config.emotion_dim
            + config.occasion_dim
        )
        self.network = nn.Sequential(
            EqualLinear(raw_dim, config.condition_dim, activation=True),
            EqualLinear(config.condition_dim, config.condition_dim, activation=True),
        )

    def forward(
        self,
        context_embedding: Tensor,
        relationship_onehot: Tensor,
        emotion_onehot: Tensor,
        occasion_onehot: Tensor,
    ) -> Tensor:
        return self.network(
            torch.cat(
                [context_embedding, relationship_onehot, emotion_onehot, occasion_onehot],
                dim=1,
            )
        )


class MappingNetwork(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.conditioning = ConditioningEncoder(config)
        layers: list[nn.Module] = [PixelNorm()]
        input_dim = config.z_dim + config.condition_dim
        for index in range(config.mapping_layers):
            layers.append(
                EqualLinear(input_dim if index == 0 else config.w_dim, config.w_dim, activation=True)
            )
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        z: Tensor,
        context_embedding: Tensor,
        relationship_onehot: Tensor,
        emotion_onehot: Tensor,
        occasion_onehot: Tensor,
    ) -> Tensor:
        condition = self.conditioning(
            context_embedding, relationship_onehot, emotion_onehot, occasion_onehot
        )
        return self.network(torch.cat([z, condition], dim=1))


class ModulatedConv2d(nn.Module):
    """StyleGAN2 modulated convolution with optional weight demodulation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        w_dim: int,
        *,
        demodulate: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.demodulate = demodulate
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channels * kernel_size * kernel_size)
        self.affine = EqualLinear(w_dim, in_channels)
        nn.init.ones_(self.affine.bias)

    def forward(self, value: Tensor, style: Tensor) -> Tensor:
        batch, _, height, width = value.shape
        modulation = self.affine(style).view(batch, 1, self.in_channels, 1, 1)
        weight = self.weight * self.scale * modulation
        if self.demodulate:
            demodulation = torch.rsqrt(weight.square().sum((2, 3, 4)) + 1e-8)
            weight = weight * demodulation.view(batch, self.out_channels, 1, 1, 1)
        weight = weight.view(
            batch * self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        )
        value = value.view(1, batch * self.in_channels, height, width)
        output = F.conv2d(value, weight, padding=self.padding, groups=batch)
        return output.view(batch, self.out_channels, height, width)


class StyledConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, w_dim: int) -> None:
        super().__init__()
        self.conv = ModulatedConv2d(in_channels, out_channels, 3, w_dim)
        self.noise_strength = nn.Parameter(torch.zeros(()))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, value: Tensor, style: Tensor, noise: Tensor | None = None) -> Tensor:
        output = self.conv(value, style)
        if noise is None:
            noise = torch.randn(
                output.shape[0], 1, output.shape[2], output.shape[3],
                device=output.device, dtype=output.dtype,
            )
        output = output + self.noise_strength * noise
        return F.leaky_relu(output + self.bias.view(1, -1, 1, 1), 0.2) * math.sqrt(2)


class ToRGB(nn.Module):
    def __init__(self, in_channels: int, image_channels: int, w_dim: int) -> None:
        super().__init__()
        self.conv = ModulatedConv2d(
            in_channels, image_channels, 1, w_dim, demodulate=False
        )
        self.bias = nn.Parameter(torch.zeros(image_channels))

    def forward(self, value: Tensor, style: Tensor) -> Tensor:
        return self.conv(value, style) + self.bias.view(1, -1, 1, 1)


class Generator(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.mapping = MappingNetwork(config)
        resolutions = [2**power for power in range(2, int(math.log2(config.resolution)) + 1)]
        self.resolutions = resolutions
        first_channels = config.channels(4)
        self.constant = nn.Parameter(torch.randn(1, first_channels, 4, 4))
        convs: list[nn.Module] = [StyledConv(first_channels, first_channels, config.w_dim)]
        to_rgbs: list[nn.Module] = [ToRGB(first_channels, config.image_channels, config.w_dim)]
        previous = first_channels
        for resolution in resolutions[1:]:
            current = config.channels(resolution)
            convs.extend(
                [StyledConv(previous, current, config.w_dim), StyledConv(current, current, config.w_dim)]
            )
            to_rgbs.append(ToRGB(current, config.image_channels, config.w_dim))
            previous = current
        self.convs = nn.ModuleList(convs)
        self.to_rgbs = nn.ModuleList(to_rgbs)
        self.num_ws = len(convs)

    def map(
        self,
        z: Tensor,
        context_embedding: Tensor,
        relationship_onehot: Tensor,
        emotion_onehot: Tensor,
        occasion_onehot: Tensor,
    ) -> Tensor:
        w = self.mapping(
            z, context_embedding, relationship_onehot, emotion_onehot, occasion_onehot
        )
        return w.unsqueeze(1).repeat(1, self.num_ws, 1)

    def synthesis(self, ws: Tensor, noises: Sequence[Tensor | None] | None = None) -> Tensor:
        if ws.ndim != 3 or ws.shape[1] != self.num_ws:
            raise ValueError(f"ws must have shape [batch, {self.num_ws}, w_dim]")
        batch = ws.shape[0]
        value = self.constant.repeat(batch, 1, 1, 1)
        noises = list(noises) if noises is not None else [None] * self.num_ws
        conv_index = 0
        value = self.convs[conv_index](value, ws[:, conv_index], noises[conv_index])
        image = self.to_rgbs[0](value, ws[:, conv_index])
        conv_index += 1
        for block_index in range(1, len(self.resolutions)):
            value = F.interpolate(value, scale_factor=2, mode="bilinear", align_corners=False)
            image = F.interpolate(image, scale_factor=2, mode="bilinear", align_corners=False)
            value = self.convs[conv_index](value, ws[:, conv_index], noises[conv_index])
            conv_index += 1
            value = self.convs[conv_index](value, ws[:, conv_index], noises[conv_index])
            image = image + self.to_rgbs[block_index](value, ws[:, conv_index])
            conv_index += 1
        return torch.tanh(image)

    def forward(
        self,
        z: Tensor,
        context_embedding: Tensor,
        relationship_onehot: Tensor,
        emotion_onehot: Tensor,
        occasion_onehot: Tensor,
        *,
        return_ws: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        ws = self.map(
            z, context_embedding, relationship_onehot, emotion_onehot, occasion_onehot
        )
        image = self.synthesis(ws)
        return (image, ws) if return_ws else image


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, value: Tensor) -> Tensor:
        residual = F.avg_pool2d(self.skip(value), 2)
        value = F.leaky_relu(self.conv1(value), 0.2) * math.sqrt(2)
        value = F.leaky_relu(self.conv2(value), 0.2) * math.sqrt(2)
        value = F.avg_pool2d(value, 2)
        return (value + residual) / math.sqrt(2)


class Discriminator(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        current = config.channels(config.resolution)
        self.from_rgb = nn.Conv2d(config.image_channels, current, 1)
        blocks: list[nn.Module] = []
        resolution = config.resolution
        while resolution > 4:
            next_channels = config.channels(resolution // 2)
            blocks.append(ConvBlock(current, next_channels))
            current = next_channels
            resolution //= 2
        self.blocks = nn.Sequential(*blocks)
        self.final_conv = nn.Conv2d(current + 1, current, 3, padding=1)
        self.final_linear = EqualLinear(current * 4 * 4, config.w_dim, activation=True)
        self.unconditional = EqualLinear(config.w_dim, 1)
        self.conditioning = ConditioningEncoder(config)
        self.condition_projection = EqualLinear(config.condition_dim, config.w_dim)

    def forward(
        self,
        image: Tensor,
        context_embedding: Tensor,
        relationship_onehot: Tensor,
        emotion_onehot: Tensor,
        occasion_onehot: Tensor,
    ) -> Tensor:
        value = F.leaky_relu(self.from_rgb(image), 0.2) * math.sqrt(2)
        value = self.blocks(value)
        std = value.float().std(dim=0, unbiased=False).mean().to(value.dtype)
        std_feature = std.expand(value.shape[0], 1, value.shape[2], value.shape[3])
        value = F.leaky_relu(self.final_conv(torch.cat([value, std_feature], dim=1)), 0.2)
        features = self.final_linear(value.flatten(1))
        condition = self.condition_projection(
            self.conditioning(
                context_embedding, relationship_onehot, emotion_onehot, occasion_onehot
            )
        )
        projection = (features * condition).sum(dim=1, keepdim=True) / math.sqrt(features.shape[1])
        return self.unconditional(features) + projection


class ADAAugment(nn.Module):
    """Differentiable ADA subset: pixel blitting, geometry, and color transforms."""

    def __init__(self, config: ModelConfig, probability: float = 0.0) -> None:
        super().__init__()
        self.settings = config.augmentation
        self.register_buffer("probability", torch.tensor(float(probability)))

    def forward(self, images: Tensor) -> Tensor:
        if not self.training or float(self.probability) <= 0:
            return images
        batch = images.shape[0]
        p = self.probability.to(images.device)
        output = images
        if self.settings.get("xflip", True):
            mask = (torch.rand(batch, 1, 1, 1, device=images.device) < p)
            output = torch.where(mask, torch.flip(output, dims=(3,)), output)
        if self.settings.get("integer_translation", True):
            limit = max(1, round(images.shape[-1] * self.settings.get("max_translation_fraction", 0.125)))
            transformed = []
            for sample in output:
                if torch.rand((), device=images.device) < p:
                    shifts = torch.randint(-limit, limit + 1, (2,), device=images.device)
                    sample = torch.roll(sample, (int(shifts[0]), int(shifts[1])), (1, 2))
                transformed.append(sample)
            output = torch.stack(transformed)
        if self.settings.get("brightness", True):
            delta = torch.randn(batch, 1, 1, 1, device=images.device) * self.settings.get("brightness_std", 0.2)
            mask = (torch.rand(batch, 1, 1, 1, device=images.device) < p)
            output = output + delta * mask
        if self.settings.get("contrast", True):
            factor = torch.exp(torch.randn(batch, 1, 1, 1, device=images.device) * self.settings.get("contrast_std", 0.5))
            mask = (torch.rand(batch, 1, 1, 1, device=images.device) < p)
            output = torch.where(mask, output * factor, output)
        return output.clamp(-1, 1)


@dataclass
class ADAController:
    target: float = 0.6
    interval: int = 4
    ada_kimg: int = 500

    def update(self, augmenter: ADAAugment, real_logits: Tensor, batch_size: int) -> float:
        statistic = real_logits.detach().sign().mean().item()
        direction = 1 if statistic > self.target else -1
        adjustment = direction * batch_size * self.interval / (self.ada_kimg * 1000)
        augmenter.probability.copy_((augmenter.probability + adjustment).clamp(0, 1))
        return float(augmenter.probability)


def discriminator_logistic_loss(real_logits: Tensor, fake_logits: Tensor) -> Tensor:
    return F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()


def generator_nonsaturating_loss(fake_logits: Tensor) -> Tensor:
    return F.softplus(-fake_logits).mean()


def r1_penalty(real_logits: Tensor, real_images: Tensor) -> Tensor:
    gradients = autograd.grad(
        outputs=real_logits.sum(), inputs=real_images, create_graph=True, only_inputs=True
    )[0]
    return gradients.square().flatten(1).sum(1).mean()


def path_length_regularization(
    images: Tensor,
    ws: Tensor,
    mean_path_length: Tensor,
    *,
    decay: float = 0.01,
) -> tuple[Tensor, Tensor, Tensor]:
    noise = torch.randn_like(images) / math.sqrt(images.shape[2] * images.shape[3])
    gradients = autograd.grad(
        outputs=(images * noise).sum(), inputs=ws, create_graph=True, only_inputs=True
    )[0]
    path_lengths = torch.sqrt(gradients.square().sum(2).mean(1) + 1e-8)
    updated_mean = mean_path_length + decay * (path_lengths.mean().detach() - mean_path_length)
    penalty = (path_lengths - updated_mean).square().mean()
    return penalty, updated_mean.detach(), path_lengths


def clip_consistency_loss(
    generated_images: Tensor,
    descriptions: Sequence[str],
    clip_model: nn.Module,
    tokenizer: Any,
) -> Tensor:
    if len(descriptions) != generated_images.shape[0]:
        raise ValueError("one conditioning description is required per image")
    device = generated_images.device
    resized = F.interpolate(generated_images, size=(224, 224), mode="bicubic", align_corners=False)
    resized = (resized + 1) / 2
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
    normalized = (resized - mean) / std
    image_features = clip_model.encode_image(normalized)
    with torch.no_grad():
        text_features = clip_model.encode_text(tokenizer(list(descriptions)).to(device))
    image_features = F.normalize(image_features, dim=1)
    text_features = F.normalize(text_features, dim=1)
    return (1 - (image_features * text_features).sum(dim=1)).mean()
