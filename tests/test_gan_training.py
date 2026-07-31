from pathlib import Path

import torch

from src.gan.models import ModelConfig
from src.gan.train import (
    GANDataset,
    frechet_distance,
    load_training_config,
    polynomial_kid,
)


def test_smoke_config_resolves_recipe_and_overrides() -> None:
    config = load_training_config("src/gan/configs/train_smoke.json")
    assert config["learning_rate_g"] == 0.0025
    assert config["adam_betas"] == [0.0, 0.99]
    assert config["max_steps"] == 200
    assert config["model_overrides"]["resolution"] == 16
    assert config["metric_backend"] == "clip"


def test_real_manifest_dataset_builds_conditioning_tensors() -> None:
    training = load_training_config("src/gan/configs/train_smoke.json")
    base = ModelConfig.from_json(training["model_config"])
    model = ModelConfig(**{**base.__dict__, **training["model_overrides"]})
    dataset = GANDataset(
        training["metadata_path"],
        training["embeddings_path"],
        "train",
        model,
        training["emotion_vocabulary"],
        max_images=2,
        seed=training["seed"],
    )
    item = dataset[0]
    assert item["image"].shape == (3, 16, 16)
    assert item["context"].shape == (512,)
    assert item["emotion"].sum() == 1
    assert item["relationship"].sum() == 0
    assert item["occasion"].sum() == 0


def test_feature_space_fid_and_kid_are_finite() -> None:
    real = torch.tensor(
        [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.8, 0.2, 0.0]]
    )
    fake = torch.tensor(
        [[0.0, 1.0, 0.0], [0.1, 0.9, 0.0], [0.2, 0.8, 0.0]]
    )
    fid = frechet_distance(real, fake)
    kid = polynomial_kid(real, fake)
    assert torch.isfinite(fid) and fid >= 0
    assert torch.isfinite(kid)


def test_completed_smoke_run_contains_required_artifacts() -> None:
    run_dir = Path("experiments/run-002")
    required = {
        "config.json",
        "train_log.jsonl",
        "metrics.jsonl",
        "summary.json",
        "checkpoint-000200.pt",
        "samples-000200.png",
    }
    assert required <= {path.name for path in run_dir.iterdir()}
