from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .sources import acquire_images


def load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def preprocess_images(records: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    settings = config["preprocessing"]
    resolution = int(settings["resolution"])
    processed_dir = Path(config["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    output: list[dict[str, Any]] = []
    for record in records:
        if "source" not in record:
            record = {**record, "source": "unknown"}
        destination = processed_dir / record["source"] / f"{record['id']}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(record["raw_path"]) as source:
            rgb = source.convert("RGB")
            if settings.get("crop", "center_square") == "center_square":
                processed = ImageOps.fit(
                    rgb,
                    (resolution, resolution),
                    method=Image.Resampling.LANCZOS,
                    centering=tuple(settings.get("centering", [0.5, 0.5])),
                )
            elif settings["crop"] == "pad_square":
                processed = ImageOps.pad(
                    rgb,
                    (resolution, resolution),
                    method=Image.Resampling.LANCZOS,
                    color=tuple(settings.get("pad_color", [255, 255, 255])),
                )
            else:
                raise ValueError(f"Unsupported crop mode: {settings['crop']!r}")
            processed.save(destination, format="PNG", optimize=True)
        content = destination.read_bytes()
        output.append(
            {
                **record,
                "processed_path": destination.as_posix(),
                "processed_sha256": __import__("hashlib").sha256(content).hexdigest(),
                "processed_width": resolution,
                "processed_height": resolution,
                "processed_format": "PNG",
            }
        )
    return output


def encode_clip(records: list[dict[str, Any]], config: dict[str, Any]) -> tuple[np.ndarray, Any, Any, str]:
    import open_clip
    import torch

    clip_config = config["clip"]
    requested_device = clip_config.get("device", "auto")
    device = "cuda" if requested_device == "auto" and torch.cuda.is_available() else "cpu" if requested_device == "auto" else requested_device
    model, _, preprocess = open_clip.create_model_and_transforms(
        clip_config.get("model_name", "ViT-B-32"),
        pretrained=clip_config.get("pretrained", "openai"),
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(clip_config.get("model_name", "ViT-B-32"))
    model.eval().requires_grad_(False)
    vectors: list[np.ndarray] = []
    batch_size = int(clip_config.get("batch_size", 32))
    for start in range(0, len(records), batch_size):
        tensors = []
        for record in records[start : start + batch_size]:
            with Image.open(record["processed_path"]) as image:
                tensors.append(preprocess(image.convert("RGB")))
        batch = torch.stack(tensors).to(device)
        with torch.inference_mode():
            encoded = model.encode_image(batch)
            encoded = encoded / encoded.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        vectors.append(encoded.cpu().numpy().astype(np.float32))
    return np.concatenate(vectors, axis=0), model, tokenizer, device


def kmeans(vectors: np.ndarray, clusters: int, *, seed: int, max_iterations: int) -> tuple[np.ndarray, np.ndarray]:
    if vectors.ndim != 2 or clusters < 1 or len(vectors) < clusters:
        raise ValueError("K-means requires a 2D array with at least one sample per cluster")
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-12)
    rng = np.random.default_rng(seed)
    centroids = np.empty((clusters, vectors.shape[1]), dtype=np.float32)
    centroids[0] = vectors[rng.integers(len(vectors))]
    distances = np.full(len(vectors), np.inf, dtype=np.float64)
    for index in range(1, clusters):
        distances = np.minimum(distances, np.sum((vectors - centroids[index - 1]) ** 2, axis=1))
        total = float(distances.sum())
        if total <= 0 or not np.isfinite(total):
            centroids[index] = vectors[rng.integers(len(vectors))]
        else:
            centroids[index] = vectors[rng.choice(len(vectors), p=distances / total)]
    labels = np.full(len(vectors), -1, dtype=np.int64)
    for _ in range(max_iterations):
        squared = np.sum((vectors[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(squared, axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for index in range(clusters):
            members = vectors[labels == index]
            if len(members):
                centroids[index] = members.mean(axis=0)
            else:
                centroids[index] = vectors[int(np.argmax(np.min(squared, axis=1)))]
        centroids = centroids / np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
    return labels, centroids.astype(np.float32)


def map_clusters_to_emotions(centroids: np.ndarray, model: Any, tokenizer: Any, device: str, prompts: dict[str, str]) -> dict[int, str]:
    import torch

    emotions = list(prompts)
    tokens = tokenizer([prompts[name] for name in emotions]).to(device)
    with torch.inference_mode():
        text_vectors = model.encode_text(tokens)
        text_vectors = text_vectors / text_vectors.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    similarities = centroids @ text_vectors.cpu().numpy().T
    centered = similarities - similarities.mean(axis=0, keepdims=True)
    calibrated = centered / np.maximum(similarities.std(axis=0, keepdims=True), 1e-6)
    return {int(index): emotions[int(np.argmax(row))] for index, row in enumerate(calibrated)}


def assign_stratified_split(records: list[dict[str, Any]], validation_fraction: float, *, seed: int) -> None:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    rng = random.Random(seed)
    groups: dict[int, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[int(record["cluster_id"])].append(index)
    validation_indices: set[int] = set()
    for indices in groups.values():
        rng.shuffle(indices)
        count = min(len(indices) - 1, max(1, round(len(indices) * validation_fraction)))
        if count > 0:
            validation_indices.update(indices[:count])
    if not validation_indices and len(records) > 1:
        validation_indices.add(rng.randrange(len(records)))
    for index, record in enumerate(records):
        record["split"] = "validation" if index in validation_indices else "train"


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def create_contact_sheet(records: list[dict[str, Any]], config: dict[str, Any]) -> None:
    preview = config["preview"]
    sample_count = min(int(preview.get("sample_count", 32)), len(records))
    sample = sorted(records, key=lambda row: (row["cluster_id"], row["split"], row["id"]))[:sample_count]
    columns = int(preview.get("columns", 4))
    size = int(preview.get("thumbnail_size", 192))
    label_height = int(preview.get("label_height", 58))
    rows = (len(sample) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * size, max(rows, 1) * (size + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, record in enumerate(sample):
        x = (index % columns) * size
        y = (index // columns) * (size + label_height)
        with Image.open(record["processed_path"]) as image:
            sheet.paste(image.resize((size, size), Image.Resampling.LANCZOS), (x, y))
        label = f"cluster {record['cluster_id']} | {record['pseudo_emotion_tag']}\n{record['split']} | {record['id']} | {record['source']}"
        draw.multiline_text((x + 4, y + size + 4), label, fill="black", font=font, spacing=3)
    destination = Path(config["paths"]["contact_sheet_path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, quality=92)


def write_data_card(records: list[dict[str, Any]], rejections: list[dict[str, Any]], report: dict[str, Any], config: dict[str, Any]) -> None:
    path = Path(config["paths"].get("data_card_path", "experiments/DATA_CARD.md"))
    source_counts = Counter(record["source"] for record in records)
    license_counts = Counter(record["license"] for record in records)
    label_counts = Counter(record["pseudo_emotion_tag"] for record in records)
    artifact_counts = Counter(record.get("artifact_type", "unknown") for record in records)
    split_counts = Counter(record["split"] for record in records)
    rejection_counts = Counter(row["reason"] for row in rejections)
    lines = [
        "# GMGI GAN Dataset Card",
        "",
        "## Status",
        "",
        "This dataset snapshot was produced by the reproducible GMGI GAN data pipeline. It is intended for training and evaluating the conditional MemoryGAN visual generator.",
        "",
        "## Snapshot Summary",
        "",
        f"- Created at: {report['created_at']}",
        f"- Records retained: {len(records)}",
        f"- Train records: {split_counts.get('train', 0)}",
        f"- Validation records: {split_counts.get('validation', 0)}",
        f"- Resolution: {config['preprocessing']['resolution']} x {config['preprocessing']['resolution']} RGB PNG",
        f"- CLIP model: {report['clip_model']}",
        f"- Clusters: {report['clusters']}",
        f"- Training started by this pipeline: No",
        "",
        "## Sources and Licensing",
        "",
    ]
    for source, count in sorted(source_counts.items()):
        lines.append(f"- {source}: {count} retained images")
    lines.append("")
    for license_name, count in sorted(license_counts.items()):
        lines.append(f"- {license_name}: {count} images")
    lines.extend([
        "",
        "The pipeline loads configured greeting-card image datasets and stores dataset-level provenance, license references, raw checksums, processed checksums, and rejection-independent provenance fields. The first GMGI source is an image-only Hugging Face dataset, so captions and occasion labels are optional and may be absent.",
        "",
        "Users remain responsible for checking the upstream dataset card/license before publication or redistribution.",
        "",
        "## Processing",
        "",
        "1. Load configured greeting-card images from Hugging Face Datasets.",
        "2. Reject rows without decodable image bytes or configured minimum dimensions.",
        "3. Preserve raw images with SHA-256 checksums.",
        "4. Convert to RGB, square-crop or square-pad according to config, resize with Lanczos, and save PNG files.",
        "5. Encode processed images with OpenCLIP.",
        "6. Cluster normalized CLIP embeddings with deterministic k-means++.",
        "7. Map clusters to weak emotion labels using CLIP text prompts.",
        "8. Create a deterministic cluster-stratified train/validation split.",
        "9. Write JSONL metadata, NumPy embeddings, rejection logs, run report, and contact sheet.",
        "",
        "## Label Distribution",
        "",
    ])
    for label, count in sorted(label_counts.items()):
        lines.append(f"- {label}: {count}")
    lines.extend(["", "## Artifact Type Distribution", ""])
    for artifact_type, count in sorted(artifact_counts.items()):
        lines.append(f"- {artifact_type}: {count}")
    lines.extend(["", "## Rejections", ""])
    if rejection_counts:
        for reason, count in sorted(rejection_counts.items()):
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- No rejected candidates were recorded.")
    lines.extend([
        "",
        "## Limitations",
        "",
        "The emotion labels are weak CLIP-derived pseudo-labels, not human annotations. The current greeting-card dataset is image-only, so relationship, occasion, and caption metadata are not guaranteed. Scaling beyond this first source should add explicit labels or additional curated gift datasets, duplicate filtering, visual relevance filtering, and documented rejection reasons.",
        "",
        "## Reproducibility Artifacts",
        "",
        f"- Config: `{report['config_path']}`",
        f"- Manifest: `{config['paths']['metadata_path']}`",
        f"- Embeddings: `{config['paths']['embeddings_path']}`",
        f"- Rejection log: `{config['paths']['rejections_path']}`",
        f"- Run report: `{config['paths']['run_report_path']}`",
        f"- Contact sheet: `{config['paths']['contact_sheet_path']}`",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_pipeline(config_path: str | Path, *, reuse_acquired: bool = False, skip_acquire: bool = False) -> dict[str, Any]:
    config = load_config(config_path)
    metadata_path = Path(config["paths"]["metadata_path"])
    rejections: list[dict[str, Any]] = []
    if (reuse_acquired or skip_acquire) and metadata_path.exists():
        source_records = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        source_records = [record for record in source_records if Path(record["raw_path"]).exists()]
    else:
        source_records, rejections = acquire_images(config)
    records = preprocess_images(source_records, config)
    embeddings, model, tokenizer, device = encode_clip(records, config)
    clustering = config["clustering"]
    labels, centroids = kmeans(
        embeddings,
        min(int(clustering["clusters"]), len(records)),
        seed=int(clustering["seed"]),
        max_iterations=int(clustering["max_iterations"]),
    )
    emotion_map = map_clusters_to_emotions(centroids, model, tokenizer, device, clustering["emotion_prompts"])
    for index, record in enumerate(records):
        cluster_id = int(labels[index])
        record["embedding_index"] = index
        record["cluster_id"] = cluster_id
        record["pseudo_emotion_tag"] = emotion_map[cluster_id]
        record["label_source"] = "openclip_kmeans_pseudo_label"
    assign_stratified_split(records, float(config["split"]["validation_fraction"]), seed=int(config["split"]["seed"]))

    embeddings_path = Path(config["paths"]["embeddings_path"])
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, embeddings)
    write_jsonl(metadata_path, records)
    write_jsonl(config["paths"].get("rejections_path", "data/gan/rejections.jsonl"), rejections)
    create_contact_sheet(records, config)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "records": len(records),
        "train_records": sum(record["split"] == "train" for record in records),
        "validation_records": sum(record["split"] == "validation" for record in records),
        "resolution": config["preprocessing"]["resolution"],
        "clip_model": f"{config['clip'].get('model_name', 'ViT-B-32')}:{config['clip'].get('pretrained', 'openai')}",
        "clip_device": device,
        "clusters": len(set(int(label) for label in labels)),
        "cluster_emotion_map": {str(key): value for key, value in emotion_map.items()},
        "source_counts": dict(Counter(record["source"] for record in records)),
        "label_counts": dict(Counter(record["pseudo_emotion_tag"] for record in records)),
        "artifact_counts": dict(Counter(record.get("artifact_type", "unknown") for record in records)),
        "split_counts": dict(Counter(record["split"] for record in records)),
        "rejection_counts": dict(Counter(row["reason"] for row in rejections)),
        "training_started": False,
    }
    report_path = Path(config["paths"]["run_report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_data_card(records, rejections, report, config)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the licensed GMGI GAN dataset")
    parser.add_argument("--config", default="src/gan/configs/data_pipeline.json", type=Path)
    parser.add_argument("--reuse-acquired", action="store_true", help="Reuse raw images listed in an existing manifest")
    parser.add_argument("--skip-acquire", action="store_true", help="Alias for reuse-acquired; useful in offline notebooks")
    args = parser.parse_args()
    report = run_pipeline(args.config, reuse_acquired=args.reuse_acquired, skip_acquire=args.skip_acquire)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

