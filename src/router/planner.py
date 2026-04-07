"""
Router Planner
--------------
Maps the user query → set of relevant agent nodes.

Flowchart step: User Query → Vector Encoding → Router Planner (Here we decide who will go)

Strategy:
  1. Compute relevance_score for every node (keyword-based + semantic fallback).
  2. Select the top-K agents whose score exceeds the activation threshold.
  3. Always include ResearchNode when live search data is available.
  4. Always run MemoryAgentNode in parallel (cheap personalisation context).
"""
from typing import List, Tuple
from src.agents.nodes import FoodNode, BusinessNode, CodingNode, ResearchNode, MemoryAgentNode

# Singleton node pool — instantiated once, reused across all requests
_NODES = [FoodNode(), BusinessNode(), CodingNode(), ResearchNode()]
_MEMORY_NODE = MemoryAgentNode()

# Minimum relevance score for a node to be activated
ACTIVATION_THRESHOLD = 0.15
# Max specialist nodes to activate concurrently (excluding MemoryNode)
MAX_ACTIVE_NODES = 3


def route(query: str, has_search_data: bool = False) -> Tuple[List, MemoryAgentNode]:
    """
    Returns (active_specialist_nodes, memory_node).
    active_specialist_nodes are sorted by descending relevance.
    """
    scored: List[Tuple[float, object]] = []
    for node in _NODES:
        score = node.relevance_score(query)
        scored.append((score, node))

    # Sort by descending relevance
    scored.sort(key=lambda x: x[0], reverse=True)

    # Filter by activation threshold
    active = [(s, n) for s, n in scored if s >= ACTIVATION_THRESHOLD]

    # If nothing clears the threshold, fall back to top-2
    if not active:
        active = scored[:2]

    # If live search data is available, ensure ResearchNode is included
    if has_search_data:
        research_node = next(n for n in _NODES if isinstance(n, ResearchNode))
        if not any(isinstance(n, ResearchNode) for _, n in active):
            active.append((0.9, research_node))

    # Cap at MAX_ACTIVE_NODES to control API cost
    active = sorted(active, key=lambda x: x[0], reverse=True)[:MAX_ACTIVE_NODES]

    specialist_nodes = [node for _, node in active]
    return specialist_nodes, _MEMORY_NODE


def describe_routing(query: str) -> dict:
    """Returns routing metadata for debugging / UI display."""
    scored = [(node.relevance_score(query), node) for node in _NODES]
    scored.sort(key=lambda x: x[0], reverse=True)
    return {
        "query": query,
        "scores": {node.name: round(score, 3) for score, node in scored}
    }
