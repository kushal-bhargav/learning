import torch

from src.gan.models import (
    ADAAugment,
    ADAController,
    Discriminator,
    Generator,
    ModelConfig,
    discriminator_logistic_loss,
    generator_nonsaturating_loss,
    path_length_regularization,
    r1_penalty,
)


def small_config() -> ModelConfig:
    return ModelConfig(
        resolution=16,
        image_channels=3,
        z_dim=8,
        context_dim=8,
        relationship_dim=3,
        emotion_dim=4,
        occasion_dim=5,
        condition_dim=8,
        w_dim=8,
        mapping_layers=2,
        channel_base=128,
        channel_max=32,
        augmentation={
            "xflip": True,
            "integer_translation": True,
            "brightness": True,
            "contrast": True,
            "max_translation_fraction": 0.125,
            "brightness_std": 0.2,
            "contrast_std": 0.5,
        },
    )


def conditions(config: ModelConfig, batch: int = 2) -> tuple[torch.Tensor, ...]:
    context = torch.randn(batch, config.context_dim)
    relationship = torch.nn.functional.one_hot(
        torch.arange(batch) % config.relationship_dim, config.relationship_dim
    ).float()
    emotion = torch.nn.functional.one_hot(
        torch.arange(batch) % config.emotion_dim, config.emotion_dim
    ).float()
    occasion = torch.nn.functional.one_hot(
        torch.arange(batch) % config.occasion_dim, config.occasion_dim
    ).float()
    return context, relationship, emotion, occasion


def test_conditioned_generator_and_discriminator_shapes() -> None:
    torch.manual_seed(7)
    config = small_config()
    generator = Generator(config)
    discriminator = Discriminator(config)
    context, relationship, emotion, occasion = conditions(config)
    images, ws = generator(
        torch.randn(2, config.z_dim),
        context,
        relationship,
        emotion,
        occasion,
        return_ws=True,
    )
    logits = discriminator(images, context, relationship, emotion, occasion)
    assert images.shape == (2, 3, 16, 16)
    assert ws.shape == (2, generator.num_ws, config.w_dim)
    assert logits.shape == (2, 1)
    assert images.min() >= -1 and images.max() <= 1


def test_adversarial_r1_and_path_length_losses_have_gradients() -> None:
    torch.manual_seed(8)
    config = small_config()
    generator = Generator(config)
    discriminator = Discriminator(config)
    context, relationship, emotion, occasion = conditions(config)
    fake, ws = generator(
        torch.randn(2, config.z_dim),
        context,
        relationship,
        emotion,
        occasion,
        return_ws=True,
    )
    real = torch.randn_like(fake, requires_grad=True)
    real_logits = discriminator(real, context, relationship, emotion, occasion)
    fake_logits = discriminator(fake.detach(), context, relationship, emotion, occasion)
    d_loss = discriminator_logistic_loss(real_logits, fake_logits)
    g_loss = generator_nonsaturating_loss(
        discriminator(fake, context, relationship, emotion, occasion)
    )
    r1 = r1_penalty(real_logits, real)
    path, updated_mean, lengths = path_length_regularization(
        fake, ws, torch.zeros((), dtype=fake.dtype)
    )
    assert all(torch.isfinite(value) for value in (d_loss, g_loss, r1, path))
    assert updated_mean.ndim == 0
    assert lengths.shape == (2,)
    (g_loss + path).backward()
    assert any(parameter.grad is not None for parameter in generator.parameters())


def test_ada_augmentation_and_probability_controller() -> None:
    config = small_config()
    augmenter = ADAAugment(config, probability=1.0).train()
    images = torch.linspace(-1, 1, 2 * 3 * 16 * 16).reshape(2, 3, 16, 16)
    augmented = augmenter(images)
    assert augmented.shape == images.shape
    assert not torch.equal(augmented, images)
    controller = ADAController(target=0.6, interval=4, ada_kimg=1)
    previous = float(augmenter.probability)
    updated = controller.update(augmenter, torch.ones(2, 1), batch_size=2)
    assert updated >= previous
