from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from PIL import Image


class ImageSource(Protocol):
    def acquire(self, source_config: dict[str, Any], paths: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]: ...


def fetch_json(url: str, *, timeout: float, user_agent: str) -> dict[str, Any]:
    request = Request(url, headers=_headers(user_agent))
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_bytes(url: str, *, timeout: float, user_agent: str) -> bytes:
    request = Request(url, headers=_headers(user_agent))
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def _headers(user_agent: str) -> dict[str, str]:
    return {
        "User-Agent": f"Mozilla/5.0 {user_agent}",
        "AIC-User-Agent": user_agent,
        "Referer": "https://www.artic.edu/",
    }


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def verify_image(content: bytes) -> tuple[int, int, str]:
    with Image.open(io.BytesIO(content)) as image:
        image.verify()
    with Image.open(io.BytesIO(content)) as image:
        return image.width, image.height, image.format or "unknown"


def save_download(
    *,
    image_url: str,
    raw_path: Path,
    timeout: float,
    user_agent: str,
    reuse_existing: bool,
) -> tuple[str, int, int, str, int]:
    if reuse_existing and raw_path.exists():
        content = raw_path.read_bytes()
    else:
        content = fetch_bytes(image_url, timeout=timeout, user_agent=user_agent)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(content)
    width, height, image_format = verify_image(content)
    return sha256_bytes(content), width, height, image_format, len(content)


def reject(rejections: list[dict[str, Any]], source: str, object_id: Any, reason: str, **extra: Any) -> None:
    rejections.append({"source": source, "object_id": object_id, "reason": reason, **extra})


class ArtInstituteChicagoSource:
    name = "art_institute_chicago"

    def acquire(self, source_config: dict[str, Any], paths: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        raw_dir = Path(paths["raw_dir"])
        base = source_config["api_base_url"].rstrip("/")
        timeout = float(source_config.get("request_timeout_seconds", 30))
        user_agent = source_config.get("user_agent", "GMGI-research-data-pipeline/0.1")
        max_images = int(source_config["max_images"])
        delay = float(source_config.get("download_delay_seconds", 0))
        reuse_existing = bool(source_config.get("reuse_existing_raw", True))
        min_width = int(source_config.get("min_width", 0))
        min_height = int(source_config.get("min_height", 0))
        fields = ",".join(
            [
                "id",
                "title",
                "image_id",
                "is_public_domain",
                "artist_display",
                "date_display",
                "medium_display",
                "department_title",
                "classification_title",
                "style_title",
                "place_of_origin",
            ]
        )
        candidates: list[tuple[str, dict[str, Any], str, str]] = []
        seen: set[int] = set()
        rejections: list[dict[str, Any]] = []
        for query in source_config["queries"]:
            pages = int(source_config.get("pages_per_query", 1))
            for page in range(1, pages + 1):
                params = {
                    "q": query,
                    "query[term][is_public_domain]": "true",
                    "limit": int(source_config.get("page_size", source_config.get("max_candidates_per_query", 100))),
                    "page": page,
                    "fields": fields,
                }
                try:
                    result = fetch_json(f"{base}/artworks/search?{urlencode(params)}", timeout=timeout, user_agent=user_agent)
                except (HTTPError, URLError, TimeoutError, OSError) as error:
                    reject(rejections, self.name, f"{query}:page:{page}", "search_failed", error=str(error))
                    continue
                iiif_url = result["config"]["iiif_url"].rstrip("/")
                website_url = result["config"]["website_url"].rstrip("/")
                for metadata in result.get("data", []):
                    object_id = int(metadata["id"])
                    if object_id not in seen:
                        seen.add(object_id)
                        candidates.append((query, metadata, iiif_url, website_url))

        records: list[dict[str, Any]] = []
        for query, metadata, iiif_url, website_url in candidates:
            if len(records) >= max_images:
                break
            object_id = int(metadata["id"])
            if metadata.get("is_public_domain") is not True:
                reject(rejections, self.name, object_id, "not_public_domain")
                continue
            if not metadata.get("image_id"):
                reject(rejections, self.name, object_id, "missing_image_id")
                continue
            image_url = f"{iiif_url}/{metadata['image_id']}/full/843,/0/default.jpg"
            raw_path = raw_dir / self.name / f"aic_{object_id}.jpg"
            try:
                checksum, width, height, image_format, bytes_count = save_download(
                    image_url=image_url,
                    raw_path=raw_path,
                    timeout=timeout,
                    user_agent=user_agent,
                    reuse_existing=reuse_existing,
                )
            except Exception as error:
                reject(rejections, self.name, object_id, "download_or_decode_failed", error=str(error))
                continue
            if width < min_width or height < min_height:
                reject(rejections, self.name, object_id, "image_too_small", width=width, height=height)
                continue
            records.append(
                {
                    "id": f"aic-{object_id}",
                    "source": self.name,
                    "object_id": object_id,
                    "title": metadata.get("title") or "Untitled",
                    "artist": metadata.get("artist_display") or "Unknown",
                    "object_date": metadata.get("date_display") or "Unknown",
                    "medium": metadata.get("medium_display") or "Unknown",
                    "culture": metadata.get("place_of_origin") or "Unknown",
                    "department": metadata.get("department_title") or "Unknown",
                    "classification": metadata.get("classification_title") or "Unknown",
                    "style": metadata.get("style_title") or "Unknown",
                    "object_url": f"{website_url}/artworks/{object_id}",
                    "image_url": image_url,
                    "is_public_domain": True,
                    "license": "CC0 1.0 / Art Institute of Chicago Open Access",
                    "license_url": "https://www.artic.edu/open-access/open-access-images",
                    "raw_path": raw_path.as_posix(),
                    "raw_sha256": checksum,
                    "raw_width": width,
                    "raw_height": height,
                    "raw_format": image_format,
                    "raw_bytes": bytes_count,
                    "source_query": query,
                }
            )
            if delay:
                time.sleep(delay)
        return records, rejections


class MetMuseumSource:
    name = "met_museum"

    def acquire(self, source_config: dict[str, Any], paths: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        raw_dir = Path(paths["raw_dir"])
        base = source_config["api_base_url"].rstrip("/")
        timeout = float(source_config.get("request_timeout_seconds", 30))
        user_agent = source_config.get("user_agent", "GMGI-research-data-pipeline/0.1")
        max_images = int(source_config["max_images"])
        delay = float(source_config.get("download_delay_seconds", 0))
        reuse_existing = bool(source_config.get("reuse_existing_raw", True))
        min_width = int(source_config.get("min_width", 0))
        min_height = int(source_config.get("min_height", 0))
        rejections: list[dict[str, Any]] = []
        per_query_ids: list[list[int]] = []
        for query in source_config["queries"]:
            params = {"hasImages": "true", "q": query}
            try:
                result = fetch_json(f"{base}/search?{urlencode(params)}", timeout=timeout, user_agent=user_agent)
            except (HTTPError, URLError, TimeoutError, OSError) as error:
                reject(rejections, self.name, query, "search_failed", error=str(error))
                continue
            per_query_ids.append([int(value) for value in (result.get("objectIDs") or [])[: int(source_config.get("max_candidates_per_query", 200))]])
        object_ids: list[int] = []
        seen: set[int] = set()
        for row in zip_longest(*per_query_ids):
            for object_id in row:
                if object_id is not None and object_id not in seen:
                    seen.add(object_id)
                    object_ids.append(object_id)

        records: list[dict[str, Any]] = []
        for object_id in object_ids:
            if len(records) >= max_images:
                break
            try:
                metadata = fetch_json(f"{base}/objects/{object_id}", timeout=timeout, user_agent=user_agent)
            except Exception as error:
                reject(rejections, self.name, object_id, "metadata_failed", error=str(error))
                continue
            if source_config.get("require_public_domain", True) and metadata.get("isPublicDomain") is not True:
                reject(rejections, self.name, object_id, "not_public_domain")
                continue
            image_url = metadata.get("primaryImageSmall") or metadata.get("primaryImage")
            if not image_url:
                reject(rejections, self.name, object_id, "missing_image_url")
                continue
            suffix = Path(str(image_url).split("?")[0]).suffix or ".jpg"
            raw_path = raw_dir / self.name / f"met_{object_id}{suffix}"
            try:
                checksum, width, height, image_format, bytes_count = save_download(
                    image_url=image_url,
                    raw_path=raw_path,
                    timeout=timeout,
                    user_agent=user_agent,
                    reuse_existing=reuse_existing,
                )
            except Exception as error:
                reject(rejections, self.name, object_id, "download_or_decode_failed", error=str(error))
                continue
            if width < min_width or height < min_height:
                reject(rejections, self.name, object_id, "image_too_small", width=width, height=height)
                continue
            records.append(
                {
                    "id": f"met-{object_id}",
                    "source": self.name,
                    "object_id": object_id,
                    "title": metadata.get("title") or "Untitled",
                    "artist": metadata.get("artistDisplayName") or "Unknown",
                    "object_date": metadata.get("objectDate") or "Unknown",
                    "medium": metadata.get("medium") or "Unknown",
                    "culture": metadata.get("culture") or "Unknown",
                    "department": metadata.get("department") or "Unknown",
                    "classification": metadata.get("classification") or "Unknown",
                    "style": "Unknown",
                    "object_url": metadata.get("objectURL"),
                    "image_url": image_url,
                    "is_public_domain": True,
                    "license": "CC0 1.0 / The Met Open Access",
                    "license_url": "https://www.metmuseum.org/about-the-met/policies-and-documents/open-access",
                    "raw_path": raw_path.as_posix(),
                    "raw_sha256": checksum,
                    "raw_width": width,
                    "raw_height": height,
                    "raw_format": image_format,
                    "raw_bytes": bytes_count,
                    "source_query": next((q for q in source_config["queries"] if q.lower() in json.dumps(metadata).lower()), source_config["queries"][0]),
                }
            )
            if delay:
                time.sleep(delay)
        return records, rejections


SOURCES: dict[str, ImageSource] = {
    ArtInstituteChicagoSource.name: ArtInstituteChicagoSource(),
    MetMuseumSource.name: MetMuseumSource(),
}


def acquire_images(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_configs = config.get("sources") or [config["source"]]
    records: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for source_config in source_configs:
        provider = source_config.get("provider") or source_config.get("name", "").lower().replace(" ", "_")
        if provider == "art_institute_of_chicago_open_access_collection":
            provider = "art_institute_chicago"
        source = SOURCES.get(provider)
        if source is None:
            raise ValueError(f"Unsupported data source provider: {provider!r}")
        try:
            source_records, source_rejections = source.acquire(source_config, config["paths"])
        except (HTTPError, URLError, TimeoutError, OSError) as error:
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


# Backward-compatible name used by older scripts/tests.
def acquire_aic_images(config: dict[str, Any]) -> list[dict[str, Any]]:
    source = ArtInstituteChicagoSource()
    records, _ = source.acquire(config["source"], config["paths"])
    return records