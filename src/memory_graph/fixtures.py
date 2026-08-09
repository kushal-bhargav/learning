from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .graph import MemoryGraph, TextEncoder
from .types import EdgeType, NodeType


def load_fixture(
    path: str | Path,
    *,
    text_encoder: TextEncoder | None = None,
) -> MemoryGraph:
    data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    graph = MemoryGraph(text_encoder=text_encoder)

    for person in data["people"]:
        graph.add_node(NodeType.PERSON, **person)
    for relationship in data["relationships"]:
        graph.add_node(NodeType.RELATIONSHIP, **relationship)
        graph.add_edge(
            relationship["person_a"],
            relationship["person_b"],
            EdgeType.RELATES_TO,
            relationship_id=relationship["id"],
        )
    for occasion in data.get("occasions", []):
        graph.add_node(NodeType.OCCASION, **occasion)
    for event in data.get("events", []):
        graph.add_node(NodeType.EVENT, **event)
        for person_id in event.get("participants", []):
            graph.add_edge(person_id, event["id"], EdgeType.PARTICIPATED_IN)
    for memory in data.get("memories", []):
        graph.add_node(NodeType.MEMORY, **memory)
        if event_id := memory.get("event_id"):
            graph.add_edge(memory["id"], event_id, EdgeType.RECALLS)
        for person_id in memory.get("person_ids", []):
            graph.add_edge(memory["id"], person_id, EdgeType.RECALLS)
    for preference in data.get("preferences", []):
        graph.add_node(NodeType.PREFERENCE, **preference)
        graph.add_edge(
            preference["person_id"], preference["id"], EdgeType.PREFERS
        )
    return graph
