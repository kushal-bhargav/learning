from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DATASET_DIR = Path(__file__).with_name("datasets")


def load_jsonl_dataset(name: str) -> list[dict[str, Any]]:
    path = DATASET_DIR / name
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for index, row in enumerate(rows, start=1):
        missing = {"case_id", "input", "expected_output"} - set(row)
        if missing:
            raise ValueError(f"{path}:{index} missing required fields: {sorted(missing)}")
    return rows


def load_relationship_synthetic() -> list[dict[str, Any]]:
    return load_jsonl_dataset("relationship_synthetic.jsonl")


def load_gift_intent_synthetic() -> list[dict[str, Any]]:
    return load_jsonl_dataset("gift_intent_synthetic.jsonl")
