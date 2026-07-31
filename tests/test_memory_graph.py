from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from src.memory_graph.fixtures import load_fixture
from src.memory_graph.graph import MemoryGraph
from src.memory_graph.types import EdgeType, NodeType


FIXTURES = Path(__file__).parents[1] / "data" / "fixtures"


@pytest.mark.parametrize("fixture_path", sorted(FIXTURES.glob("*.json")))
def test_fixture_builds_a_graph(fixture_path: Path) -> None:
    raw = json.loads(fixture_path.read_text(encoding="utf-8"))
    graph = load_fixture(fixture_path)
    expected_nodes = sum(
        len(raw[key])
        for key in (
            "people",
            "relationships",
            "occasions",
            "events",
            "memories",
            "preferences",
        )
    )
    assert isinstance(graph.graph, nx.MultiDiGraph)
    assert graph.graph.number_of_nodes() == expected_nodes


def test_context_embedding_mean_pools_relevant_fixture_memories() -> None:
    graph = load_fixture(FIXTURES / "long_distance_partners.json")
    result = graph.context_embedding(
        "person-jordan", "occasion-jordan-birthday-2026"
    )
    expected = np.mean(
        np.asarray(
            [
                [0.82, 0.14, 0.61, 0.09, 0.73, 0.31, 0.48, 0.26],
                [0.44, 0.71, 0.28, 0.56, 0.19, 0.87, 0.35, 0.63],
                [0.23, 0.68, 0.75, 0.41, 0.52, 0.16, 0.84, 0.37],
            ],
            dtype=np.float32,
        ),
        axis=0,
    )
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, expected)


def test_subgraph_contains_relevant_nodes_and_excludes_unrelated_nodes() -> None:
    graph = load_fixture(FIXTURES / "parent_teen.json")
    graph.add_node(
        NodeType.PERSON,
        id="person-unrelated",
        display_name="Nobody",
        role="recipient",
    )
    result = graph.subgraph_for(
        "person-lucas", "occasion-lucas-graduation-2026"
    )
    assert "person-lucas" in result
    assert "occasion-lucas-graduation-2026" in result
    assert "memory-robotics-cheer" in result
    assert "preference-lucas-hobby" in result
    assert "person-unrelated" not in result


def test_json_round_trip_preserves_graph_and_embedding() -> None:
    original = load_fixture(FIXTURES / "adult_siblings.json")
    restored = MemoryGraph.from_json(original.to_json())
    assert list(original.graph.nodes(data=True)) == list(
        restored.graph.nodes(data=True)
    )
    assert list(original.graph.edges(data=True)) == list(
        restored.graph.edges(data=True)
    )
    np.testing.assert_allclose(
        original.context_embedding(
            "person-asha", "occasion-asha-housewarming-2026"
        ),
        restored.context_embedding(
            "person-asha", "occasion-asha-housewarming-2026"
        ),
    )


def test_add_edge_rejects_unknown_endpoint() -> None:
    graph = MemoryGraph()
    person_id = graph.add_node(
        NodeType.PERSON,
        id="person-one",
        display_name="One",
        role="giver",
    )
    with pytest.raises(KeyError, match="missing"):
        graph.add_edge(person_id, "missing", EdgeType.RELATES_TO)


def test_context_embedding_rejects_mixed_dimensions() -> None:
    graph = MemoryGraph()
    person_id = graph.add_node(
        NodeType.PERSON,
        id="person-one",
        display_name="One",
        role="recipient",
    )
    first = graph.add_node(
        NodeType.MEMORY, id="memory-one", embedding=[0.1, 0.2]
    )
    second = graph.add_node(
        NodeType.MEMORY, id="memory-two", embedding=[0.1, 0.2, 0.3]
    )
    graph.add_edge(first, person_id, EdgeType.RECALLS)
    graph.add_edge(second, person_id, EdgeType.RECALLS)
    with pytest.raises(ValueError, match="dimensions"):
        graph.context_embedding(person_id)


