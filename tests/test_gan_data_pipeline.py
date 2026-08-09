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
            "id": "hf-greeting-card-000001",
            "source": "huggingface_dataset",
            "object_id": "1",
            "title": "Synthetic greeting card image 1",
            "artist": "Synthetic dataset",
            "object_date": "Unknown",
            "medium": "Synthetic greeting card image",
            "culture": "Synthetic",
            "department": "Greeting cards",
            "classification": "greeting_card",
            "artifact_type": "greeting_card",
            "visual_gift_type": "greeting_card",
            "object_url": "https://huggingface.co/datasets/gauravs101/synthetic-greeting-cards",
            "image_url": "",
            "is_public_domain": False,
            "license": "Hugging Face dataset license; verify dataset README before publication",
            "license_url": "https://huggingface.co/datasets/gauravs101/synthetic-greeting-cards",
            "raw_path": "raw.jpg",
            "raw_sha256": "abc",
            "raw_width": 512,
            "raw_height": 512,
            "raw_format": "JPEG",
            "raw_bytes": 10,
            "source_query": "synthetic-greeting-cards",
            "description": "Synthetic greeting card image 1",
            "caption": None,
            "occasion_label": None,
            "dataset_name": "gauravs101/synthetic-greeting-cards",
            "dataset_split": "train",
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



def test_acquire_images_skips_blocked_source(monkeypatch) -> None:
    from urllib.error import HTTPError

    from src.gan import sources

    class BlockedSource:
        def acquire(self, source_config, paths):
            raise HTTPError("https://example.test", 403, "Forbidden", hdrs=None, fp=None)

    class WorkingSource:
        def acquire(self, source_config, paths):
            return ([{"id": "ok-1", "source": "working", "object_id": 1}], [])

    monkeypatch.setattr(
        sources,
        "SOURCES",
        {"blocked": BlockedSource(), "working": WorkingSource()},
    )
    records, rejections = sources.acquire_images(
        {
            "sources": [{"provider": "blocked"}, {"provider": "working"}],
            "paths": {},
            "quality_gates": {"minimum_records": 1},
        }
    )
    assert records == [{"id": "ok-1", "source": "working", "object_id": 1}]
    assert rejections[0]["source"] == "blocked"
    assert rejections[0]["reason"] == "source_acquisition_failed"



def test_huggingface_greeting_card_source(monkeypatch) -> None:
    import io
    import sys
    import types

    from PIL import Image

    from src.gan.sources import HuggingFaceDatasetSource

    image = Image.new("RGB", (160, 160), "pink")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    class FakeDataset(list):
        features = {"image": object(), "filename": object()}

    def fake_load_dataset(name, **kwargs):
        assert name == "gauravs101/synthetic-greeting-cards"
        assert kwargs["split"] == "train"
        return FakeDataset([
            {
                "image": {"bytes": buffer.getvalue()},
                "filename": "birthday-card.png",
            }
        ])

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load_dataset))
    root = Path(".test-tmp/hf-source-test")
    source = HuggingFaceDatasetSource()
    records, rejections = source.acquire(
        {
            "dataset_name": "gauravs101/synthetic-greeting-cards",
            "split": "train",
            "max_images": 1,
            "min_width": 128,
            "min_height": 128,
            "license": "test-license",
        },
        {"raw_dir": str(root / "raw")},
    )
    assert rejections == []
    assert len(records) == 1
    assert records[0]["source"] == "huggingface_dataset"
    assert records[0]["classification"] == "greeting_card"
    assert records[0]["artifact_type"] == "greeting_card"
    assert records[0]["description"] == "a synthetic greeting card image for a personalized gift: birthday card"
    assert records[0]["caption"] is None
    assert records[0]["occasion_label"] is None
    assert Path(records[0]["raw_path"]).exists()



def test_synthetic_agent_datasets_load() -> None:
    from src.agents.synthetic_data import load_gift_intent_synthetic, load_relationship_synthetic

    relationship_rows = load_relationship_synthetic()
    intent_rows = load_gift_intent_synthetic()

    assert len(relationship_rows) >= 6
    assert len(intent_rows) >= 6
    assert all({"case_id", "input", "expected_output"} <= set(row) for row in relationship_rows)
    assert all({"case_id", "input", "expected_output"} <= set(row) for row in intent_rows)
    assert {row["expected_output"].get("artifact_type") for row in intent_rows} >= {"greeting_card", "gift_wrap", "keepsake_print", "gift_tag"}
