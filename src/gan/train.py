from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional at runtime.
    tqdm = None

from .models import (
    ADAAugment,
    ADAController,
    Discriminator,
    Generator,
    ModelConfig,
    clip_consistency_loss,
    discriminator_logistic_loss,
    generator_nonsaturating_loss,
    path_length_regularization,
    r1_penalty,
)


def load_training_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    base_path = config.pop("base_config", None)
    if base_path:
        base = load_training_config(base_path)
        base.update(config)
        config = base
    return config


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def device_report(device: torch.device, amp_enabled: bool) -> dict[str, Any]:
    report: dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "resolved_device": str(device),
        "mixed_precision_active": amp_enabled,
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        report.update({
            "cuda_device_index": index,
            "cuda_device_name": torch.cuda.get_device_name(index),
            "cuda_capability": f"{props.major}.{props.minor}",
            "cuda_total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
        })
    return report


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GANDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        metadata_path: str | Path,
        embeddings_path: str | Path,
        split: str,
        model_config: ModelConfig,
        emotion_vocabulary: list[str],
        *,
        max_images: int | None = None,
        seed: int = 2026,
    ) -> None:
        records = [
            json.loads(line)
            for line in Path(metadata_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        records = [record for record in records if record["split"] == split]
        random.Random(seed).shuffle(records)
        self.records = records[:max_images] if max_images else records
        if not self.records:
            raise ValueError(f"No {split} records available")
        self.embeddings = np.load(embeddings_path, mmap_mode="r")
        self.model_config = model_config
        self.emotion_to_index = {name: index for index, name in enumerate(emotion_vocabulary)}
        if len(self.emotion_to_index) != model_config.emotion_dim:
            raise ValueError("emotion vocabulary size must match model emotion_dim")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        resolution = self.model_config.resolution
        with Image.open(record["processed_path"]) as source:
            image = ImageOps.fit(
                source.convert("RGB"),
                (resolution, resolution),
                method=Image.Resampling.LANCZOS,
            )
            array = np.asarray(image, dtype=np.float32).copy() / 127.5 - 1.0
        emotion = torch.zeros(self.model_config.emotion_dim, dtype=torch.float32)
        emotion_index = self.emotion_to_index.get(
            record["pseudo_emotion_tag"], self.emotion_to_index["other"]
        )
        emotion[emotion_index] = 1.0
        artifact_type = str(record.get("artifact_type") or record.get("visual_gift_type") or "other")
        description = (
            record.get("description")
            or record.get("caption")
            or record.get("title")
            or f"a {record['pseudo_emotion_tag']} {artifact_type} visual gift artifact"
        )
        return {
            "image": torch.from_numpy(array).permute(2, 0, 1),
            "context": torch.from_numpy(
                np.asarray(self.embeddings[record["embedding_index"]], dtype=np.float32).copy()
            ),
            "relationship": torch.zeros(self.model_config.relationship_dim),
            "emotion": emotion,
            "occasion": torch.zeros(self.model_config.occasion_dim),
            "description": str(description),
            "artifact_type": artifact_type,
            "id": record["id"],
        }


def infinite_batches(loader: DataLoader) -> Iterator[dict[str, Any]]:
    while True:
        yield from loader


def allocate_run_dir(experiments_dir: str | Path) -> Path:
    root = Path(experiments_dir)
    root.mkdir(parents=True, exist_ok=True)
    numbers = []
    for child in root.iterdir():
        match = re.fullmatch(r"run-(\d+)", child.name)
        if child.is_dir() and match:
            numbers.append(int(match.group(1)))
    run_dir = root / f"run-{max(numbers, default=0) + 1:03d}"
    run_dir.mkdir()
    return run_dir


def move_batch(batch: dict[str, Any], device: torch.device) -> tuple[Tensor, ...]:
    return tuple(
        batch[name].to(device, non_blocking=True)
        for name in ("image", "context", "relationship", "emotion", "occasion")
    )


def update_ema(ema: Generator, source: Generator, decay: float) -> None:
    with torch.no_grad():
        for ema_parameter, parameter in zip(ema.parameters(), source.parameters()):
            ema_parameter.lerp_(parameter, 1 - decay)
        for ema_buffer, buffer in zip(ema.buffers(), source.buffers()):
            ema_buffer.copy_(buffer)


def to_metric_images(images: Tensor) -> Tensor:
    return ((images.detach().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


def clip_image_features(images: Tensor, clip_model: torch.nn.Module) -> Tensor:
    images = torch.nn.functional.interpolate(
        images, size=(224, 224), mode="bicubic", align_corners=False
    )
    images = (images + 1) / 2
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=images.device).view(1, 3, 1, 1)
    return torch.nn.functional.normalize(clip_model.encode_image((images - mean) / std), dim=1)


def cached_clip_consistency_loss(
    generated_images: Tensor,
    descriptions: Sequence[str],
    clip_model: torch.nn.Module,
    tokenizer: Any,
    text_feature_cache: dict[str, Tensor],
) -> Tensor:
    if len(descriptions) != generated_images.shape[0]:
        raise ValueError("one conditioning description is required per image")
    device = generated_images.device
    missing = [description for description in dict.fromkeys(descriptions) if description not in text_feature_cache]
    if missing:
        with torch.no_grad():
            tokens = tokenizer(missing).to(device)
            features = torch.nn.functional.normalize(clip_model.encode_text(tokens), dim=1).detach()
        for description, feature in zip(missing, features):
            text_feature_cache[description] = feature
    image_features = clip_image_features(generated_images, clip_model)
    text_features = torch.stack([text_feature_cache[description].to(device) for description in descriptions])
    return (1 - (image_features * text_features).sum(dim=1)).mean()


def frechet_distance(features_real: Tensor, features_fake: Tensor) -> Tensor:
    real = features_real.double()
    fake = features_fake.double()
    mean_real, mean_fake = real.mean(0), fake.mean(0)
    centered_real = real - mean_real
    centered_fake = fake - mean_fake
    covariance_real = centered_real.T @ centered_real / max(real.shape[0] - 1, 1)
    covariance_fake = centered_fake.T @ centered_fake / max(fake.shape[0] - 1, 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_real)
    root_real = (eigenvectors * eigenvalues.clamp_min(0).sqrt()) @ eigenvectors.T
    middle = root_real @ covariance_fake @ root_real
    middle_eigenvalues = torch.linalg.eigvalsh(middle).clamp_min(0)
    return (
        (mean_real - mean_fake).square().sum()
        + torch.trace(covariance_real)
        + torch.trace(covariance_fake)
        - 2 * middle_eigenvalues.sqrt().sum()
    ).float().clamp_min(0)


def polynomial_kid(features_real: Tensor, features_fake: Tensor) -> Tensor:
    dimension = features_real.shape[1]
    kernel_real = (features_real @ features_real.T / dimension + 1).pow(3)
    kernel_fake = (features_fake @ features_fake.T / dimension + 1).pow(3)
    kernel_cross = (features_real @ features_fake.T / dimension + 1).pow(3)
    count_real, count_fake = features_real.shape[0], features_fake.shape[0]
    real_term = (kernel_real.sum() - kernel_real.diagonal().sum()) / (count_real * (count_real - 1))
    fake_term = (kernel_fake.sum() - kernel_fake.diagonal().sum()) / (count_fake * (count_fake - 1))
    return real_term + fake_term - 2 * kernel_cross.mean()

@torch.no_grad()
def evaluate_metrics(
    generator: Generator,
    loader: DataLoader,
    config: dict[str, Any],
    model_config: ModelConfig,
    device: torch.device,
    clip_model: torch.nn.Module | None = None,
) -> dict[str, Any]:
    target = min(int(config["metric_num_samples"]), len(loader.dataset))
    if target < 2:
        raise ValueError("FID/KID evaluation requires at least two validation images")
    if config["metric_backend"] == "clip":
        if clip_model is None:
            raise ValueError("CLIP metric backend requires a loaded CLIP model")
        real_features: list[Tensor] = []
        fake_features: list[Tensor] = []
        seen = 0
        generator.eval()
        for batch in loader:
            real, context, relationship, emotion, occasion = move_batch(batch, device)
            count = min(real.shape[0], target - seen)
            if count <= 0:
                break
            real = real[:count]
            fake = generator(
                torch.randn(count, model_config.z_dim, device=device), context[:count],
                relationship[:count], emotion[:count], occasion[:count],
            )
            real_features.append(clip_image_features(real, clip_model).cpu())
            fake_features.append(clip_image_features(fake, clip_model).cpu())
            seen += count
        real_tensor = torch.cat(real_features)
        fake_tensor = torch.cat(fake_features)
        generator.train()
        return {
            "fid": float(frechet_distance(real_tensor, fake_tensor)),
            "kid_mean": float(polynomial_kid(real_tensor, fake_tensor)),
            "kid_std": 0.0,
            "metric_samples": seen,
            "feature_backend": "openclip_vit_b32",
        }

    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance

    target = min(int(config["metric_num_samples"]), len(loader.dataset))
    subset_size = min(int(config["kid_subset_size"]), target)
    if target < 2 or subset_size < 2:
        raise ValueError("FID/KID evaluation requires at least two validation images")
    fid = FrechetInceptionDistance(feature=int(config["fid_feature"]), normalize=False).to(device)
    kid = KernelInceptionDistance(subset_size=subset_size, normalize=False).to(device)
    seen = 0
    generator.eval()
    for batch in loader:
        real, context, relationship, emotion, occasion = move_batch(batch, device)
        count = min(real.shape[0], target - seen)
        if count <= 0:
            break
        real = real[:count]
        context = context[:count]
        relationship = relationship[:count]
        emotion = emotion[:count]
        occasion = occasion[:count]
        fake = generator(
            torch.randn(count, model_config.z_dim, device=device),
            context,
            relationship,
            emotion,
            occasion,
        )
        real_uint8 = to_metric_images(real)
        fake_uint8 = to_metric_images(fake)
        fid.update(real_uint8, real=True)
        fid.update(fake_uint8, real=False)
        kid.update(real_uint8, real=True)
        kid.update(fake_uint8, real=False)
        seen += count
    fid_value = float(fid.compute().cpu())
    kid_mean, kid_std = kid.compute()
    generator.train()
    return {
        "fid": fid_value,
        "kid_mean": float(kid_mean.cpu()),
        "kid_std": float(kid_std.cpu()),
        "metric_samples": seen,
        "feature_backend": "inception_v3",
    }


def save_checkpoint(
    path: Path,
    step: int,
    generator: Generator,
    generator_ema: Generator,
    discriminator: Discriminator,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    augmenter: ADAAugment,
    mean_path_length: Tensor,
    model_config: ModelConfig,
    training_config: dict[str, Any],
) -> None:
    torch.save(
        {
            "step": step,
            "generator": generator.state_dict(),
            "generator_ema": generator_ema.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "ada_probability": float(augmenter.probability),
            "mean_path_length": float(mean_path_length),
            "model_config": asdict(model_config),
            "training_config": training_config,
        },
        path,
    )


def train(config_path: str | Path) -> Path:
    config = load_training_config(config_path)
    base_model_config = ModelConfig.from_json(config["model_config"])
    model_config = replace(base_model_config, **config.get("model_overrides", {}))
    seed_everything(int(config["seed"]))
    device = resolve_device(config["device"])
    amp_enabled = bool(config["mixed_precision"] and device.type == "cuda")
    run_dir = allocate_run_dir(config["experiments_dir"])
    torch_home = Path(config["torch_home"])
    torch_home.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(torch_home.resolve())
    clip_interval = int(config.get("clip_interval", 16))
    if clip_interval < 1:
        raise ValueError("clip_interval must be >= 1")
    metric_interval = int(config.get("metric_interval", 0))
    metrics_enabled = bool(config.get("metrics_enabled", True)) and metric_interval > 0
    resolved_config = {
        **config,
        "clip_interval": clip_interval,
        "metrics_enabled": metrics_enabled,
        "model": asdict(model_config),
        "resolved_device": str(device),
        "mixed_precision_active": amp_enabled,
        "source_config": str(config_path),
    }
    (run_dir / "config.json").write_text(
        json.dumps(resolved_config, indent=2), encoding="utf-8"
    )
    print(json.dumps({"event": "training_start", **device_report(device, amp_enabled)}, indent=2), flush=True)

    train_dataset = GANDataset(
        config["metadata_path"], config["embeddings_path"], "train", model_config,
        config["emotion_vocabulary"], max_images=config["max_train_images"], seed=config["seed"],
    )
    validation_dataset = GANDataset(
        config["metadata_path"], config["embeddings_path"], "validation", model_config,
        config["emotion_vocabulary"], max_images=config["max_validation_images"], seed=config["seed"],
    )
    loader_kwargs = {
        "batch_size": int(config["batch_size"]),
        "num_workers": int(config["num_workers"]),
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **loader_kwargs)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)
    batches = infinite_batches(train_loader)
    print(json.dumps({
        "event": "dataset_ready",
        "train_images": len(train_dataset),
        "validation_images": len(validation_dataset),
        "batch_size": int(config["batch_size"]),
        "max_steps": int(config["max_steps"]),
        "clip_interval": clip_interval,
        "metrics_enabled": metrics_enabled,
        "metric_interval": metric_interval,
        "num_workers": int(config["num_workers"]),
        "pin_memory": device.type == "cuda",
        "run_dir": run_dir.as_posix(),
    }, indent=2), flush=True)

    generator = Generator(model_config).to(device)
    generator_ema = copy.deepcopy(generator).eval().requires_grad_(False)
    discriminator = Discriminator(model_config).to(device)
    augmenter = ADAAugment(model_config).to(device).train()
    ada_controller = ADAController(
        model_config.ada_target, model_config.ada_interval, model_config.ada_kimg
    )
    optimizer_g = torch.optim.Adam(
        generator.parameters(), lr=config["learning_rate_g"], betas=tuple(config["adam_betas"])
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(), lr=config["learning_rate_d"], betas=tuple(config["adam_betas"])
    )
    scaler_g = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    scaler_d = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    clip_model = None
    clip_tokenizer = None
    clip_text_feature_cache: dict[str, Tensor] = {}
    if model_config.clip_consistency_weight > 0:
        import open_clip

        print(json.dumps({
            "event": "loading_clip_model",
            "model_name": config["clip"]["model_name"],
            "pretrained": config["clip"]["pretrained"],
            "clip_interval": clip_interval,
        }), flush=True)
        clip_model, _, _ = open_clip.create_model_and_transforms(
            config["clip"]["model_name"], pretrained=config["clip"]["pretrained"], device=device
        )
        clip_model.eval().requires_grad_(False)
        clip_tokenizer = open_clip.get_tokenizer(config["clip"]["model_name"])

    mean_path_length = torch.zeros((), device=device)
    log_path = run_dir / "train_log.jsonl"
    metrics_path = run_dir / "metrics.jsonl"
    started = time.perf_counter()
    best_fid = float("inf")
    stale_evaluations = 0
    final_step = 0

    max_steps = int(config["max_steps"])
    progress_iterable = range(1, max_steps + 1)
    progress = None
    if tqdm is not None and os.getenv("GMGI_DISABLE_TQDM") != "1":
        progress = tqdm(progress_iterable, total=max_steps, desc="MemoryGAN training", unit="step", dynamic_ncols=True)
        progress_iterable = progress

    for step in progress_iterable:
        final_step = step
        batch = next(batches)
        real, context, relationship, emotion, occasion = move_batch(batch, device)
        batch_size = real.shape[0]

        regularize_d = step % int(config["r1_interval"]) == 0
        real.requires_grad_(regularize_d)
        optimizer_d.zero_grad(set_to_none=True)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            fake = generator(
                torch.randn(batch_size, model_config.z_dim, device=device),
                context, relationship, emotion, occasion,
            )
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            real_logits = discriminator(
                augmenter(real), context, relationship, emotion, occasion
            )
            fake_logits = discriminator(
                augmenter(fake), context, relationship, emotion, occasion
            )
            d_loss = discriminator_logistic_loss(real_logits, fake_logits)
            r1 = torch.zeros((), device=device)
            if regularize_d:
                r1 = r1_penalty(real_logits, real)
                d_loss = d_loss + (model_config.r1_gamma / 2) * int(config["r1_interval"]) * r1
        scaler_d.scale(d_loss).backward()
        scaler_d.step(optimizer_d)
        scaler_d.update()
        if step % model_config.ada_interval == 0:
            ada_controller.update(augmenter, real_logits, batch_size)

        optimizer_g.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            fake, ws = generator(
                torch.randn(batch_size, model_config.z_dim, device=device),
                context, relationship, emotion, occasion, return_ws=True,
            )
            fake_logits_g = discriminator(
                augmenter(fake), context, relationship, emotion, occasion
            )
            g_loss = generator_nonsaturating_loss(fake_logits_g)
            path_loss = torch.zeros((), device=device)
            if step % model_config.path_length_interval == 0:
                path_loss, mean_path_length, _ = path_length_regularization(
                    fake, ws, mean_path_length, decay=model_config.path_length_decay
                )
                g_loss = g_loss + model_config.path_length_weight * model_config.path_length_interval * path_loss
            clip_loss = torch.zeros((), device=device)
            if model_config.clip_consistency_weight > 0 and step % clip_interval == 0:
                assert clip_model is not None and clip_tokenizer is not None
                clip_loss = cached_clip_consistency_loss(
                    fake, list(batch["description"]), clip_model, clip_tokenizer, clip_text_feature_cache
                )
                g_loss = g_loss + model_config.clip_consistency_weight * clip_interval * clip_loss
        scaler_g.scale(g_loss).backward()
        scaler_g.step(optimizer_g)
        scaler_g.update()
        update_ema(generator_ema, generator, float(config["ema_decay"]))

        if step == 1 or step % int(config["log_interval"]) == 0:
            record = {
                "step": step,
                "elapsed_seconds": time.perf_counter() - started,
                "d_loss": float(d_loss.detach()),
                "g_loss": float(g_loss.detach()),
                "r1_penalty": float(r1.detach()),
                "path_length_penalty": float(path_loss.detach()),
                "clip_consistency_loss": float(clip_loss.detach()),
                "ada_probability": float(augmenter.probability),
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            if progress is not None:
                progress.set_postfix({
                    "g": f"{record['g_loss']:.3f}",
                    "d": f"{record['d_loss']:.3f}",
                    "ada": f"{record['ada_probability']:.3f}",
                })

        should_measure = metrics_enabled and (step % metric_interval == 0 or step == int(config["max_steps"]))
        if should_measure:
            save_checkpoint(
                run_dir / f"checkpoint-{step:06d}.pt", step, generator, generator_ema,
                discriminator, optimizer_g, optimizer_d, augmenter, mean_path_length,
                model_config, resolved_config,
            )
            if progress is not None:
                progress.write(json.dumps({"event": "metric_evaluation_start", "step": step, "backend": config["metric_backend"]}))
            else:
                print(json.dumps({"event": "metric_evaluation_start", "step": step, "backend": config["metric_backend"]}), flush=True)
            metrics = {"step": step, **evaluate_metrics(generator_ema, validation_loader, config, model_config, device, clip_model)}
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics) + "\n")
            print(json.dumps(metrics), flush=True)
            if progress is not None:
                progress.write(json.dumps({"event": "metrics", **metrics}))
            early = config["early_stopping"]
            if metrics["fid"] < best_fid - float(early["fid_min_delta"]):
                best_fid = metrics["fid"]
                stale_evaluations = 0
            else:
                stale_evaluations += 1
            if early["enabled"] and stale_evaluations >= int(early["fid_patience_evaluations"]):
                if progress is not None:
                    progress.write("Early stopping: FID did not improve enough.")
                break

        if step % int(config["sample_interval"]) == 0 or step == int(config["max_steps"]):
            save_image((fake[: min(16, batch_size)] + 1) / 2, run_dir / f"samples-{step:06d}.png", nrow=4)
        if step % int(config["checkpoint_interval"]) == 0 or step == int(config["max_steps"]):
            save_checkpoint(
                run_dir / f"checkpoint-{step:06d}.pt", step, generator, generator_ema,
                discriminator, optimizer_g, optimizer_d, augmenter, mean_path_length,
                model_config, resolved_config,
            )

    if progress is not None:
        progress.close()

    summary = {
        "final_step": final_step,
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device),
        "mixed_precision_active": amp_enabled,
        "train_images": len(train_dataset),
        "validation_images": len(validation_dataset),
        "best_fid": None if best_fid == float("inf") else best_fid,
        "training_complete": final_step >= int(config["max_steps"]),
        "smoke_test": "smoke" in Path(config_path).stem,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the conditional MemoryGAN")
    parser.add_argument("--config", type=Path, default=Path("src/gan/configs/train.json"))
    args = parser.parse_args()
    print(train(args.config))


if __name__ == "__main__":
    main()


