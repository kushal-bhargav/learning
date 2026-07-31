import json
from pathlib import Path

import numpy as np

from src.gan.data_pipeline import assign_stratified_split, kmeans, write_data_card, write_jsonl


def test_kmeans_is_deterministic_and_separates_obvious_groups() -> None:
    vectors = np.asarray(
        [[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1]],
        dtype=np.float32,
    )
    first_labels, first_centroids = kmeans(vectors, 2, seed=12, max_iterations=20)
    second_labels, second_centroids = kmeans(vectors, 2, seed=12, max_iterations=20)
    np.testing.assert_array_equal(first_labels, second_labels)
    np.testing.assert_allclose(first_centroids, second_centroids)
    assert first_labels[0] == first_labels[1]
    assert first_labels[2] == first_labels[3]
    assert first_labels[0] != first_labels[2]


def test_kmeans_handles_duplicate_vectors_without_nan() -> None:
    vectors = np.ones((4, 3), dtype=np.float32)
    labels, centroids = kmeans(vectors, 2, seed=2026, max_iterations=5)
    assert labels.shape == (4,)
    assert centroids.shape == (2, 3)
    assert np.isfinite(centroids).all()


def test_cluster_stratified_split_is_deterministic() -> None:
    first = [{"cluster_id": index // 5} for index in range(20)]
    second = [{"cluster_id": index // 5} for index in range(20)]
    assign_stratified_split(first, 0.1, seed=2026)
    assign_stratified_split(second, 0.1, seed=2026)
    assert [row["split"] for row in first] == [row["split"] for row in second]
    assert sum(row["split"] == "validation" for row in first) == 4
    assert all(
        any(row["split"] == "train" for row in first if row["cluster_id"] == cluster)
        for cluster in range(4)
    )


def test_jsonl_and_data_card_are_written() -> None:
    records = [
        {
            "id": "aic-1",
            "source": "art_institute_chicago",
            "object_id": 1,
            "title": "Example",
            "artist": "Unknown",
            "object_date": "1900",
            "medium": "Print",
            "culture": "Unknown",
            "department": "Prints",
            "object_url": "https://example.test/object",
            "image_url": "https://example.test/image.jpg",
            "is_public_domain": True,
            "license": "CC0 1.0 / Test",
            "license_url": "https://example.test/license",
            "raw_path": "raw.jpg",
            "raw_sha256": "abc",
            "raw_width": 512,
            "raw_height": 512,
            "raw_format": "JPEG",
            "raw_bytes": 10,
            "source_query": "postcard",
            "processed_path": "processed.png",
            "processed_sha256": "def",
            "processed_width": 256,
            "processed_height": 256,
            "processed_format": "PNG",
            "embedding_index": 0,
            "cluster_id": 0,
            "pseudo_emotion_tag": "joy",
            "label_source": "openclip_kmeans_pseudo_label",
            "split": "train",
        }
    ]
    tmp_path = Path(".test-tmp/data-pipeline-doc-test")
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest = tmp_path / "metadata.jsonl"
    write_jsonl(manifest, records)
    loaded = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert loaded[0]["processed_sha256"] == "def"

    config = {
        "paths": {
            "data_card_path": str(tmp_path / "DATA_CARD.md"),
            "metadata_path": str(manifest),
            "embeddings_path": str(tmp_path / "clip.npy"),
            "rejections_path": str(tmp_path / "rejections.jsonl"),
            "run_report_path": str(tmp_path / "run.json"),
            "contact_sheet_path": str(tmp_path / "sheet.jpg"),
        },
        "preprocessing": {"resolution": 256},
    }
    report = {
        "created_at": "2026-07-20T00:00:00+00:00",
        "clip_model": "ViT-B-32:openai",
        "clusters": 1,
        "config_path": "config.json",
    }
    write_data_card(records, [{"source": "x", "object_id": 2, "reason": "too_small"}], report, config)
    card = Path(config["paths"]["data_card_path"]).read_text(encoding="utf-8")
    assert "Snapshot Summary" in card
    assert "too_small" in card

