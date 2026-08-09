from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import networkx as nx
import numpy as np
from networkx.readwrite import json_graph

from .types import EdgeType, NodeType


class TextEncoder(Protocol):
    def encode(self, text: str) -> np.ndarray: ...


class MemoryGraph:
    """NetworkX-backed memory graph with JSON persistence and context pooling."""

    def __init__(
        self,
        graph: nx.MultiDiGraph | None = None,
        text_encoder: TextEncoder | None = None,
    ) -> None:
        self.graph = graph if graph is not None else nx.MultiDiGraph()
        self.text_encoder = text_encoder

    def add_node(self, node_type: NodeType | str, **attrs: Any) -> str:
        kind = NodeType(node_type)
        node_id = str(attrs.pop("id", f"{kind.value.lower()}-{uuid4()}"))
        if node_id in self.graph:
            raise ValueError(f"Node already exists: {node_id}")
        self.graph.add_node(node_id, node_type=kind.value, **attrs)
        return node_id

    def add_edge(
        self,
        src: str,
        dst: str,
        edge_type: EdgeType | str,
        **attrs: Any,
    ) -> None:
        if src not in self.graph or dst not in self.graph:
            missing = [node for node in (src, dst) if node not in self.graph]
            raise KeyError(f"Unknown edge endpoint(s): {', '.join(missing)}")
        relation = EdgeType(edge_type)
        self.graph.add_edge(src, dst, edge_type=relation.value, **attrs)

    def subgraph_for(
        self,
        person_id: str,
        occasion_id: str | None = None,
    ) -> nx.MultiDiGraph:
        self._require_node(person_id, NodeType.PERSON)
        selected = {person_id}

        for src, dst, data in self.graph.edges(data=True):
            edge_type = data.get("edge_type")
            if edge_type == EdgeType.RELATES_TO and person_id in (src, dst):
                selected.update((src, dst))
                relationship_id = data.get("relationship_id")
                if relationship_id in self.graph:
                    selected.add(relationship_id)
            elif edge_type == EdgeType.PREFERS and src == person_id:
                selected.add(dst)
            elif edge_type == EdgeType.PARTICIPATED_IN and src == person_id:
                selected.add(dst)

        event_ids = {
            node_id
            for node_id in selected
            if self.graph.nodes[node_id].get("node_type") == NodeType.EVENT
        }
        for src, dst, data in self.graph.edges(data=True):
            if data.get("edge_type") == EdgeType.RECALLS and (
                dst == person_id or dst in event_ids
            ):
                selected.update((src, dst))

        if occasion_id is not None:
            self._require_node(occasion_id, NodeType.OCCASION)
            selected.add(occasion_id)
        return self.graph.subgraph(selected).copy()

    def context_embedding(
        self,
        person_id: str,
        occasion_id: str | None = None,
    ) -> np.ndarray:
        relevant = self.subgraph_for(person_id, occasion_id)
        vectors: list[np.ndarray] = []
        for _, attrs in relevant.nodes(data=True):
            node_type = attrs.get("node_type")
            raw_embedding = attrs.get("embedding")
            if raw_embedding is not None and node_type in {
                NodeType.MEMORY,
                NodeType.PREFERENCE,
            }:
                vectors.append(self._as_vector(raw_embedding))
            elif node_type == NodeType.PREFERENCE and self.text_encoder is not None:
                text = f"{attrs.get('category', 'preference')}: {attrs.get('value', '')}"
                vectors.append(self._as_vector(self.text_encoder.encode(text)))

        if not vectors:
            raise ValueError(f"No embeddings available for person {person_id}")
        dimensions = {vector.shape[0] for vector in vectors}
        if len(dimensions) != 1:
            raise ValueError(f"Embedding dimensions do not match: {sorted(dimensions)}")
        return np.mean(np.stack(vectors), axis=0, dtype=np.float32)

    def to_json(self, path: str | Path | None = None, *, indent: int = 2) -> str:
        payload = json_graph.node_link_data(self.graph, edges="edges")
        content = json.dumps(
            payload, indent=indent, sort_keys=True, default=self._json_default
        )
        if path is not None:
            Path(path).write_text(content, encoding="utf-8")
        return content

    @classmethod
    def from_json(
        cls,
        source: str | Path | Mapping[str, Any],
        *,
        text_encoder: TextEncoder | None = None,
    ) -> "MemoryGraph":
        if isinstance(source, Mapping):
            payload = dict(source)
        elif isinstance(source, Path):
            payload = json.loads(source.read_text(encoding="utf-8"))
        else:
            stripped = source.lstrip()
            if stripped.startswith(("{", "[")):
                payload = json.loads(source)
            else:
                payload = json.loads(Path(source).read_text(encoding="utf-8"))
        graph = json_graph.node_link_graph(
            payload, directed=True, multigraph=True, edges="edges"
        )
        return cls(nx.MultiDiGraph(graph), text_encoder=text_encoder)

    def _require_node(self, node_id: str, expected_type: NodeType) -> None:
        if node_id not in self.graph:
            raise KeyError(f"Unknown node: {node_id}")
        actual_type = self.graph.nodes[node_id].get("node_type")
        if actual_type != expected_type:
            raise ValueError(f"Expected {expected_type.value} node, got {actual_type!r}")

    @staticmethod
    def _as_vector(value: Any) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float32)
        if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
            raise ValueError(
                "Embeddings must be non-empty, finite one-dimensional vectors"
            )
        return vector

    @staticmethod
    def _json_default(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(
            f"Object of type {type(value).__name__} is not JSON serializable"
        )
