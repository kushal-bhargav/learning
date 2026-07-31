from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gan.infer import EMOTION_TAGS, MemoryGAN
from src.gan.models import Generator, ModelConfig
from src.gan.train import GANDataset, evaluate_metrics, move_batch

STEPS = (0.0, 0.25, 0.5, 0.75, 1.0)


def latest_checkpoint(experiments_dir: str | Path = "experiments") -> Path:
    candidates: list[tuple[int, int, Path]] = []
    for checkpoint in Path(experiments_dir).glob("run-*/checkpoint-*.pt"):
        run_match = re.fullmatch(r"run-(\d+)", checkpoint.parent.name)
        step_match = re.fullmatch(r"checkpoint-(\d+)\.pt", checkpoint.name)
        if run_match and step_match:
            candidates.append((int(run_match.group(1)), int(step_match.group(1)), checkpoint))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found under {experiments_dir}")
    return max(candidates)[2]


def load_checkpoint_generator(checkpoint_path: str | Path, device: torch.device) -> tuple[Generator, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a mapping")
    raw_config = checkpoint.get("model_config")
    if not isinstance(raw_config, dict):
        raise ValueError("checkpoint is missing model_config")
    allowed = {field.name for field in fields(ModelConfig)}
    model_config = ModelConfig(**{key: value for key, value in raw_config.items() if key in allowed})
    state = checkpoint.get("generator_ema", checkpoint.get("generator"))
    if not isinstance(state, dict):
        raise ValueError("checkpoint is missing generator weights")
    generator = Generator(model_config).to(device).eval().requires_grad_(False)
    generator.load_state_dict(state)
    return generator, checkpoint


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_clip(config: dict[str, Any], device: torch.device) -> tuple[Any, Any]:
    import open_clip

    clip_config = config.get("clip", {})
    model_name = clip_config.get("model_name", "ViT-B-32")
    pretrained = clip_config.get("pretrained", "openai")
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    model.eval().requires_grad_(False)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, tokenizer


def clip_normalize(images: Tensor) -> Tensor:
    images = F.interpolate(images, size=(224, 224), mode="bicubic", align_corners=False)
    images = (images + 1) / 2
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std


@torch.no_grad()
def clipscore_conditioning_fidelity(
    generator: Generator,
    loader: DataLoader,
    model_config: ModelConfig,
    clip_model: Any,
    tokenizer: Any,
    device: torch.device,
    *,
    num_samples: int,
) -> dict[str, Any]:
    scores: list[float] = []
    descriptions: list[str] = []
    seen = 0
    for batch in loader:
        _, context, relationship, emotion, occasion = move_batch(batch, device)
        count = min(context.shape[0], num_samples - seen)
        if count <= 0:
            break
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
        text = list(batch["description"][:count])
        image_features = F.normalize(clip_model.encode_image(clip_normalize(fake)), dim=1)
        text_features = F.normalize(clip_model.encode_text(tokenizer(text).to(device)), dim=1)
        scores.extend((image_features * text_features).sum(dim=1).detach().cpu().tolist())
        descriptions.extend(text)
        seen += count
    if not scores:
        raise ValueError("CLIPScore requires at least one generated sample")
    return {
        "clipscore_mean": float(np.mean(scores)),
        "clipscore_std": float(np.std(scores)),
        "samples": len(scores),
        "description_examples": sorted(set(descriptions))[:5],
    }


def alternate_style_ref(dataset: GANDataset, index: int) -> np.ndarray:
    if len(dataset) < 2:
        return np.zeros(dataset[index]["context"].shape, dtype=np.float32)
    return dataset[(index + 1) % len(dataset)]["context"].numpy()

def lpips_tensor(images: Iterable[Image.Image], device: torch.device) -> Tensor:
    arrays = [
        torch.from_numpy(np.asarray(image.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC), dtype=np.float32).copy())
        .permute(2, 0, 1)
        .unsqueeze(0)
        .div(127.5)
        .sub(1.0)
        for image in images
    ]
    return torch.cat(arrays, dim=0).to(device)


def load_lpips(device: torch.device) -> Any:
    try:
        import lpips

        return lpips.LPIPS(net="alex").to(device).eval()
    except ModuleNotFoundError:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=False).to(device).eval()

        class TorchMetricsLPIPS:
            def __call__(self, first: Tensor, second: Tensor) -> Tensor:
                metric.reset()
                return metric(first, second)

        return TorchMetricsLPIPS()


@torch.no_grad()
def lpips_interpolation_smoothness(
    memory_gan: MemoryGAN,
    dataset: GANDataset,
    metric: Any,
    device: torch.device,
    *,
    num_conditions: int,
    seed: int,
) -> dict[str, Any]:
    condition_reports: list[dict[str, Any]] = []
    for index in range(min(num_conditions, len(dataset))):
        item = dataset[index]
        context = item["context"].numpy()
        human_style_ref = alternate_style_ref(dataset, index)
        emotion_index = int(item["emotion"].argmax().item())
        emotion = EMOTION_TAGS[emotion_index] if emotion_index < len(EMOTION_TAGS) else "other"
        images = [
            memory_gan.generate(
                context,
                relationship_type="other",
                emotion_tag=emotion,
                occasion="other",
                agency_slider=amount,
                human_style_ref=human_style_ref,
                seed=seed + index,
            )
            for amount in STEPS
        ]
        tensors = lpips_tensor(images, device)
        adjacent = [float(metric(tensors[i : i + 1], tensors[i + 1 : i + 2]).item()) for i in range(len(STEPS) - 1)]
        from_start = [float(metric(tensors[0:1], tensors[i : i + 1]).item()) for i in range(len(STEPS))]
        mean_step = float(np.mean(adjacent))
        variation = float(np.std(adjacent) / mean_step) if mean_step else 0.0
        monotonic = all(b + 1e-6 >= a for a, b in zip(from_start, from_start[1:]))
        condition_reports.append(
            {
                "condition_id": item["id"],
                "emotion_tag": emotion,
                "adjacent_lpips": adjacent,
                "lpips_from_t0": from_start,
                "monotonic_from_t0": monotonic,
                "adjacent_coefficient_of_variation": variation,
                "smoothness_pass": monotonic and variation <= 0.5,
            }
        )
    if not condition_reports:
        raise ValueError("LPIPS interpolation requires at least one condition")
    return {
        "conditions": len(condition_reports),
        "mean_adjacent_lpips": float(np.mean([np.mean(row["adjacent_lpips"]) for row in condition_reports])),
        "mean_adjacent_coefficient_of_variation": float(np.mean([row["adjacent_coefficient_of_variation"] for row in condition_reports])),
        "monotonic_pass_rate": float(np.mean([row["monotonic_from_t0"] for row in condition_reports])),
        "smoothness_pass_rate": float(np.mean([row["smoothness_pass"] for row in condition_reports])),
        "t": list(STEPS),
        "per_condition": condition_reports,
    }


@torch.no_grad()
def lpips_intra_condition_diversity(
    memory_gan: MemoryGAN,
    dataset: GANDataset,
    metric: Any,
    device: torch.device,
    *,
    num_conditions: int,
    samples_per_condition: int,
    agency_slider: float,
    seed: int,
) -> dict[str, Any]:
    condition_reports: list[dict[str, Any]] = []
    for index in range(min(num_conditions, len(dataset))):
        item = dataset[index]
        context = item["context"].numpy()
        human_style_ref = alternate_style_ref(dataset, index)
        emotion_index = int(item["emotion"].argmax().item())
        emotion = EMOTION_TAGS[emotion_index] if emotion_index < len(EMOTION_TAGS) else "other"
        images = [
            memory_gan.generate(
                context,
                relationship_type="other",
                emotion_tag=emotion,
                occasion="other",
                agency_slider=agency_slider,
                human_style_ref=alternate_style_ref(dataset, index) if agency_slider < 1.0 else None,
                seed=seed + index * 10_000 + sample,
            )
            for sample in range(samples_per_condition)
        ]
        tensors = lpips_tensor(images, device)
        pairwise = [
            float(metric(tensors[a : a + 1], tensors[b : b + 1]).item())
            for a in range(samples_per_condition)
            for b in range(a + 1, samples_per_condition)
        ]
        condition_reports.append(
            {
                "condition_id": item["id"],
                "emotion_tag": emotion,
                "pairwise_lpips": pairwise,
                "mean_pairwise_lpips": float(np.mean(pairwise)) if pairwise else 0.0,
            }
        )
    if not condition_reports:
        raise ValueError("LPIPS diversity requires at least one condition")
    return {
        "conditions": len(condition_reports),
        "samples_per_condition": samples_per_condition,
        "agency_slider": agency_slider,
        "mean_pairwise_lpips": float(np.mean([row["mean_pairwise_lpips"] for row in condition_reports])),
        "per_condition": condition_reports,
    }


def markdown_table(report: dict[str, Any]) -> str:
    rows = [
        ("FID", report["fid_kid"]["fid"], report["fid_kid"]["feature_backend"]),
        ("KID mean", report["fid_kid"]["kid_mean"], report["fid_kid"]["feature_backend"]),
        ("KID std", report["fid_kid"].get("kid_std", 0.0), report["fid_kid"]["feature_backend"]),
        ("CLIPScore mean", report["clipscore"]["clipscore_mean"], "OpenCLIP image/text cosine"),
        ("CLIPScore std", report["clipscore"]["clipscore_std"], "OpenCLIP image/text cosine"),
        ("Interpolation adjacent LPIPS", report["lpips_interpolation"]["mean_adjacent_lpips"], "Mean over t steps"),
        ("Interpolation smoothness pass", report["lpips_interpolation"]["smoothness_pass_rate"], "Fraction of conditions"),
        ("Intra-condition LPIPS diversity", report["lpips_diversity"]["mean_pairwise_lpips"], "Mean pairwise distance"),
    ]
    lines = [
        "# GAN Metrics Report",
        "",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- Split: `{report['split']}`",
        f"- Samples: `{report['fid_kid']['metric_samples']}` for FID/KID, `{report['clipscore']['samples']}` for CLIPScore",
        f"- Note: {report['note']}",
        "",
        "| Metric | Value | Backend / interpretation |",
        "|---|---:|---|",
    ]
    for metric, value, backend in rows:
        formatted = f"{value:.6f}" if isinstance(value, (float, int)) and math.isfinite(float(value)) else str(value)
        lines.append(f"| {metric} | {formatted} | {backend} |")
    lines.append("")
    return "\n".join(lines)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = args.checkpoint or latest_checkpoint(args.experiments_dir)
    device = resolve_device(args.device)
    generator, checkpoint = load_checkpoint_generator(checkpoint_path, device)
    training_config = checkpoint.get("training_config")
    if not isinstance(training_config, dict):
        config_path = checkpoint_path.parent / "config.json"
        training_config = json.loads(config_path.read_text(encoding="utf-8"))
    model_config = generator.config
    dataset = GANDataset(
        training_config["metadata_path"],
        training_config["embeddings_path"],
        args.split,
        model_config,
        training_config["emotion_vocabulary"],
        max_images=args.max_images,
        seed=int(training_config.get("seed", 2026)),
    )
    loader = DataLoader(dataset, batch_size=min(args.batch_size, max(1, len(dataset))), shuffle=False, num_workers=0)
    torch_home = Path(training_config.get("torch_home", "experiments/.cache/torch"))
    torch_home.mkdir(parents=True, exist_ok=True)
    import os

    os.environ.setdefault("TORCH_HOME", str(torch_home.resolve()))
    clip_model, tokenizer = load_clip(training_config, device)
    fid_config = {
        **training_config,
        "metric_num_samples": min(args.metric_samples, len(dataset)),
        "kid_subset_size": min(int(training_config.get("kid_subset_size", args.metric_samples)), args.metric_samples, len(dataset)),
    }
    fid_kid = evaluate_metrics(
        generator,
        loader,
        fid_config,
        model_config,
        device,
        clip_model if fid_config.get("metric_backend") == "clip" else None,
    )
    clipscore = clipscore_conditioning_fidelity(
        generator,
        loader,
        model_config,
        clip_model,
        tokenizer,
        device,
        num_samples=min(args.metric_samples, len(dataset)),
    )
    memory_gan = MemoryGAN(generator, device=device)
    lpips_metric = load_lpips(device)
    interpolation = lpips_interpolation_smoothness(
        memory_gan,
        dataset,
        lpips_metric,
        device,
        num_conditions=args.lpips_conditions,
        seed=args.seed,
    )
    diversity = lpips_intra_condition_diversity(
        memory_gan,
        dataset,
        lpips_metric,
        device,
        num_conditions=args.lpips_conditions,
        samples_per_condition=args.diversity_samples,
        agency_slider=args.diversity_agency_slider,
        seed=args.seed + 100_000,
    )
    smoke = bool(training_config.get("smoke_test")) or model_config.resolution < 64
    return {
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "device": str(device),
        "model_resolution": model_config.resolution,
        "checkpoint_step": checkpoint.get("step"),
        "note": (
            "Smoke-run diagnostics: latest checkpoint is low-resolution and uses the configured feature backend, not paper-ready canonical Inception results."
            if smoke
            else "Full-resolution evaluation diagnostics."
        ),
        "fid_kid": fid_kid,
        "clipscore": clipscore,
        "lpips_interpolation": interpolation,
        "lpips_diversity": diversity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MemoryGAN generative quality and agency metrics")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--experiments-dir", type=Path, default=Path("experiments"))
    parser.add_argument("--split", choices=["train", "validation"], default="validation")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--metric-samples", type=int, default=256)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--lpips-conditions", type=int, default=4)
    parser.add_argument("--diversity-samples", type=int, default=4)
    parser.add_argument("--diversity-agency-slider", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("experiments/gan_metrics_latest.json"))
    parser.add_argument("--table-output", type=Path, default=Path("experiments/gan_metrics_latest.md"))
    args = parser.parse_args()

    if args.metric_samples < 2:
        raise SystemExit("--metric-samples must be at least 2")
    if args.diversity_samples < 2:
        raise SystemExit("--diversity-samples must be at least 2")
    if not 0.0 <= args.diversity_agency_slider <= 1.0:
        raise SystemExit("--diversity-agency-slider must be between 0 and 1")

    report = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.table_output.parent.mkdir(parents=True, exist_ok=True)
    table = markdown_table(report)
    args.table_output.write_text(table, encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()





