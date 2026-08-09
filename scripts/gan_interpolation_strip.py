from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import Tensor

from src.gan.infer import MemoryGAN
from src.memory_graph.fixtures import load_fixture

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


def _resize_vector(vector: np.ndarray, dimension: int) -> np.ndarray:
    if vector.size > dimension:
        vector = vector[:dimension]
    elif vector.size < dimension:
        vector = np.pad(vector, (0, dimension - vector.size))
    return vector.astype(np.float32)


def _occasion_label(occasion: dict[str, Any]) -> str:
    raw = str(occasion.get("type") or occasion.get("name") or occasion.get("label") or "other").lower()
    for value in ("birthday", "anniversary", "graduation", "housewarming", "promotion", "holiday", "thank-you"):
        if value in raw:
            return value
    return "other"


def fixture_inputs(path: Path, context_dim: int) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    recipient = next(person for person in payload["people"] if person["role"] == "recipient")
    occasion = payload["occasions"][0]
    graph = load_fixture(path)
    context = graph.context_embedding(recipient["id"], occasion["id"])
    style_candidates = [
        np.asarray(memory["embedding"], dtype=np.float32)
        for memory in payload["memories"]
        if recipient["id"] in memory.get("person_ids", [])
    ]
    if not style_candidates:
        raise ValueError(f"Fixture {path} has no recipient memory embedding for {recipient['id']}")
    style = style_candidates[0]
    labels = {
        "relationship_type": payload["relationships"][0].get("type", "other"),
        "emotion_tag": payload["memories"][0].get("emotion_tag", "other"),
        "occasion": _occasion_label(occasion),
    }
    return _resize_vector(context, context_dim), _resize_vector(style, context_dim), labels


def lpips_tensor(images: Iterable[Image.Image], device: torch.device) -> Tensor:
    tensors = [
        torch.from_numpy(
            np.asarray(
                image.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC),
                dtype=np.float32,
            ).copy()
        )
        .permute(2, 0, 1)
        .unsqueeze(0)
        .div(127.5)
        .sub(1.0)
        for image in images
    ]
    return torch.cat(tensors, dim=0).to(device)


def load_lpips(device: torch.device) -> Any:
    try:
        import lpips

        return lpips.LPIPS(net="alex").to(device).eval()
    except ModuleNotFoundError:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=False).to(device).eval()

        class TorchMetricsLPIPS:
            backend = "torchmetrics_alexnet"

            def __call__(self, first: Tensor, second: Tensor) -> Tensor:
                metric.reset()
                return metric(first, second)

        return TorchMetricsLPIPS()


def lpips_report(images: list[Image.Image], device: torch.device) -> dict[str, object]:
    metric = load_lpips(device)
    tensors = lpips_tensor(images, device)
    with torch.inference_mode():
        adjacent = [
            float(metric(tensors[i : i + 1], tensors[i + 1 : i + 2]).item())
            for i in range(len(images) - 1)
        ]
        from_start = [
            float(metric(tensors[0:1], tensors[i : i + 1]).item())
            for i in range(len(images))
        ]
    monotonic = all(b + 1e-6 >= a for a, b in zip(from_start, from_start[1:]))
    mean_step = float(np.mean(adjacent))
    variation = float(np.std(adjacent) / mean_step) if mean_step else 0.0
    return {
        "t": list(STEPS),
        "lpips_backend": getattr(metric, "backend", "lpips_alexnet"),
        "adjacent_lpips": adjacent,
        "mean_adjacent_lpips": mean_step,
        "lpips_from_t0": from_start,
        "monotonic_from_t0": monotonic,
        "adjacent_coefficient_of_variation": variation,
        "smoothness_pass": monotonic and variation <= 0.5,
    }


def save_strip(images: list[Image.Image], output: Path) -> None:
    if len(images) != len(STEPS):
        raise ValueError(f"expected {len(STEPS)} images, got {len(images)}")
    label_height = 34
    strip = Image.new("RGB", (sum(image.width for image in images), images[0].height + label_height), "white")
    draw = ImageDraw.Draw(strip)
    x = 0
    for amount, image in zip(STEPS, images):
        strip.paste(image, (x, label_height))
        draw.text((x + 8, 9), f"t = {amount:g}", fill="black")
        x += image.width
    output.parent.mkdir(parents=True, exist_ok=True)
    strip.save(output)


def render_interpolation(
    checkpoint: Path,
    *,
    fixture: Path,
    output: Path,
    metrics: Path,
    seed: int,
) -> dict[str, Any]:
    os.environ.setdefault("TORCH_HOME", str((Path("experiments") / ".cache" / "torch").resolve()))
    model = MemoryGAN.load(str(checkpoint))
    context, human_style, labels = fixture_inputs(fixture, model.config.context_dim)
    images = [
        model.generate(context, **labels, agency_slider=amount, human_style_ref=human_style, seed=seed)
        for amount in STEPS
    ]
    save_strip(images, output)
    report = {
        "checkpoint": str(checkpoint),
        "fixture": str(fixture),
        "output": str(output),
        "seed": seed,
        "model_resolution": model.config.resolution,
        "labels": labels,
        **lpips_report(images, model.device),
    }
    metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Render and measure a MemoryGAN agency interpolation")
    parser.add_argument("checkpoint", nargs="?", type=Path, help="Checkpoint path. Defaults to the latest experiments/run-*/checkpoint-*.pt")
    parser.add_argument("--experiments-dir", type=Path, default=Path("experiments"))
    parser.add_argument("--fixture", type=Path, default=Path("data/fixtures/close_coworkers.json"))
    parser.add_argument("--output", type=Path, default=Path("experiments/agency_interpolation_strip.png"))
    parser.add_argument("--metrics", type=Path, default=Path("experiments/agency_interpolation_lpips.json"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--strict-smoothness", action="store_true", help="Exit nonzero when the LPIPS smoothness check fails")
    args = parser.parse_args()

    checkpoint = args.checkpoint or latest_checkpoint(args.experiments_dir)
    report = render_interpolation(
        checkpoint,
        fixture=args.fixture,
        output=args.output,
        metrics=args.metrics,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2))
    if args.strict_smoothness and not report["smoothness_pass"]:
        raise SystemExit("LPIPS smoothness check failed")


if __name__ == "__main__":
    main()
