from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any, Iterable, Protocol

from PIL import Image

from .visual_taxonomy import artifact_description, normalize_artifact_type


class ImageSource(Protocol):
    def acquire(self, source_config: dict[str, Any], paths: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]: ...


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def verify_image(content: bytes) -> tuple[int, int, str]:
    with Image.open(io.BytesIO(content)) as image:
        image.verify()
    with Image.open(io.BytesIO(content)) as image:
        return image.width, image.height, image.format or "unknown"


def reject(rejections: list[dict[str, Any]], source: str, object_id: Any, reason: str, **extra: Any) -> None:
    rejections.append({"source": source, "object_id": object_id, "reason": reason, **extra})


def _first_present(row: dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return None


def _image_to_bytes(value: Any) -> tuple[bytes, str]:
    if isinstance(value, Image.Image):
        buffer = io.BytesIO()
        image = value.convert("RGB")
        image.save(buffer, format="PNG")
        return buffer.getvalue(), "png"
    if isinstance(value, (bytes, bytearray)):
        return bytes(value), "png"
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return bytes(value["bytes"]), "png"
        if value.get("path"):
            path = Path(str(value["path"]))
            return path.read_bytes(), path.suffix.lstrip(".") or "png"
    if isinstance(value, str):
        path = Path(value)
        if path.exists():
            return path.read_bytes(), path.suffix.lstrip(".") or "png"
    raise ValueError(f"unsupported image value type: {type(value).__name__}")


def _safe_filename(value: str) -> str:
    safe = "".join(character if character.isalnum() or character in ".-_" else "_" for character in value)
    return safe[:180] or "image.png"


def _fallback_description(index: int, object_id: Any, artifact_type: str) -> str:
    base = artifact_description(artifact_type)
    stem = Path(str(object_id)).stem.replace("_", " ").replace("-", " ").strip()
    if stem and not stem.isdigit():
        return f"{base}: {stem}"
    return f"{base} {index}"


class HuggingFaceDatasetSource:
    name = "huggingface_dataset"

    def acquire(self, source_config: dict[str, Any], paths: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        try:
            from datasets import load_dataset
        except ImportError as error:
            raise RuntimeError("Install the `datasets` package to use Hugging Face dataset sources") from error

        raw_dir = Path(paths["raw_dir"])
        dataset_name = source_config["dataset_name"]
        split = source_config.get("split", "train")
        revision = source_config.get("revision")
        max_images = int(source_config.get("max_images", 1000))
        image_columns = source_config.get("image_columns", ["image", "img", "card", "png"])
        text_columns = source_config.get("text_columns", [])
        occasion_columns = source_config.get("occasion_columns", [])
        license_name = source_config.get("license", "Hugging Face dataset license; verify dataset card")
        license_url = source_config.get("license_url", f"https://huggingface.co/datasets/{dataset_name}")
        artifact_type = normalize_artifact_type(source_config.get("artifact_type", "greeting_card"))
        trust_remote_code = bool(source_config.get("trust_remote_code", False))
        keep_in_memory = bool(source_config.get("keep_in_memory", False))
        load_kwargs: dict[str, Any] = {"split": split, "trust_remote_code": trust_remote_code, "keep_in_memory": keep_in_memory}
        if revision:
            load_kwargs["revision"] = revision
        dataset = load_dataset(dataset_name, **load_kwargs)
        available_columns = list(dataset.features) if hasattr(dataset, "features") else list(getattr(dataset, "column_names", []) or [])
        image_column = next((name for name in image_columns if name in available_columns), None)
        if image_column is None:
            raise ValueError(f"No image column found in {dataset_name}; available columns: {available_columns}")

        records: list[dict[str, Any]] = []
        rejections: list[dict[str, Any]] = []
        for index, row in enumerate(dataset):
            if len(records) >= max_images:
                break
            object_id = row.get("id") or row.get("filename") or index
            try:
                image_bytes, suffix = _image_to_bytes(row[image_column])
                checksum = sha256_bytes(image_bytes)
                raw_path = raw_dir / self.name / _safe_filename(f"hf_{index:06d}_{object_id}.{suffix}")
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_bytes(image_bytes)
                width, height, image_format = verify_image(image_bytes)
            except Exception as error:
                reject(rejections, self.name, object_id, "download_or_decode_failed", error=str(error))
                continue
            min_width = int(source_config.get("min_width", 0))
            min_height = int(source_config.get("min_height", 0))
            if width < min_width or height < min_height:
                reject(rejections, self.name, object_id, "image_too_small", width=width, height=height)
                continue
            caption = _first_present(row, text_columns)
            occasion = _first_present(row, occasion_columns)
            description = str(caption or _fallback_description(index, object_id, artifact_type))
            records.append(
                {
                    "id": f"hf-greeting-card-{index:06d}",
                    "source": self.name,
                    "object_id": str(object_id),
                    "title": description[:240],
                    "artist": "Synthetic dataset",
                    "object_date": "Unknown",
                    "medium": "Synthetic greeting card image",
                    "culture": "Synthetic",
                    "department": "Greeting cards",
                    "classification": artifact_type,
                    "artifact_type": artifact_type,
                    "visual_gift_type": artifact_type,
                    "object_url": f"https://huggingface.co/datasets/{dataset_name}",
                    "image_url": str(row.get("image_url") or row.get("url") or ""),
                    "is_public_domain": False,
                    "license": license_name,
                    "license_url": license_url,
                    "raw_path": raw_path.as_posix(),
                    "raw_sha256": checksum,
                    "raw_width": width,
                    "raw_height": height,
                    "raw_format": image_format,
                    "raw_bytes": len(image_bytes),
                    "source_query": str(source_config.get("query_label", dataset_name)),
                    "description": description,
                    "caption": None if caption is None else str(caption),
                    "occasion_label": None if occasion is None else str(occasion),
                    "dataset_name": dataset_name,
                    "dataset_split": str(split),
                }
            )
        return records, rejections


SOURCES: dict[str, ImageSource] = {
    HuggingFaceDatasetSource.name: HuggingFaceDatasetSource(),
}


def acquire_images(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_configs = config.get("sources") or [config["source"]]
    records: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for source_config in source_configs:
        provider = source_config.get("provider") or source_config.get("name", "").lower().replace(" ", "_")
        source = SOURCES.get(provider)
        if source is None:
            raise ValueError(f"Unsupported data source provider: {provider!r}")
        try:
            source_records, source_rejections = source.acquire(source_config, config["paths"])
        except Exception as error:
            reject(rejections, provider, "source", "source_acquisition_failed", error=str(error))
            continue
        rejections.extend(source_rejections)
        for record in source_records:
            if record["id"] in seen_ids:
                reject(rejections, record["source"], record["object_id"], "duplicate_record_id", id=record["id"])
                continue
            seen_ids.add(record["id"])
            records.append(record)
    minimum = int(config.get("quality_gates", {}).get("minimum_records", 1))
    if len(records) < minimum:
        raise RuntimeError(f"Only {len(records)} eligible records acquired; minimum required is {minimum}")
    return records, rejections
