
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

MIN_EDGES_FOR_GRAPH = 5
EXTERNAL_PREFIX = "ext:"


@dataclass(frozen=True)
class GraphPayload:
    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    rankings: list[dict[str, Any]] = field(default_factory=list)
    reason: str | None = None
    needed: int | None = None
    have: int | None = None
    unidentified: int = 0
    views: dict[str, Any] = field(default_factory=dict)
    components: dict[str, int] = field(default_factory=dict)
    clusters: dict[str, Any] = field(
        default_factory=lambda: {
            "available": False,
            "reason": "map_not_computed",
            "ids": [],
            "unplaced": 0,
        }
    )

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "nodes": self.nodes,
            "edges": self.edges,
            "rankings": self.rankings,
            "unidentified": self.unidentified,
            "views": self.views,
            "components": self.components,
            "clusters": self.clusters,
        }
        if self.reason is not None:
            payload["views"] = {"available": False, "reason": self.reason}
            payload["reason"] = self.reason
            payload["needed"] = self.needed
            payload["have"] = self.have
        return payload


def _external_id(openalex_id: str | None, title: str | None) -> str | None:
    if openalex_id:
        return f"{EXTERNAL_PREFIX}{openalex_id}"
    if title:
        digest = hashlib.sha256(title.strip().lower().encode("utf-8")).hexdigest()[:16]
        return f"{EXTERNAL_PREFIX}title:{digest}"
    return None


def _paper_nodes(conn: Any, collection_id: str) -> dict[str, dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT id, title, year, citation_count
        FROM papers WHERE collection_id=?
        ORDER BY id
        """,
        (collection_id,),
    ).fetchall()
    return {
        row["id"]: {
            "id": row["id"],
            "title": row["title"],
            "year": row["year"],
            "citation_count": row["citation_count"],
            "resolved": True,
        }
        for row in rows
    }


def cached_clusters(
    conn: Any, collection_id: str
) -> tuple[dict[str, int] | None, dict[str, Any]]:
    row = conn.execute(
        "SELECT payload FROM projections WHERE collection_id=?", (collection_id,)
    ).fetchone()

    def refuse(reason: str) -> tuple[None, dict[str, Any]]:
        return None, {"available": False, "reason": reason, "ids": [], "unplaced": 0}

    if row is None:
        return refuse("map_not_computed")
    try:
        stored = json.loads(row["payload"])
        assignments = {point["paper_id"]: int(point["cluster"]) for point in stored["points"]}
    except (ValueError, KeyError, TypeError):
        return refuse("map_unreadable")
    if not assignments:
        return refuse("map_empty")
    if stored.get("cluster_note") == "clusters_unreliable":
        return refuse("clusters_unreliable")
    known = {
        r["id"]
        for r in conn.execute("SELECT id FROM papers WHERE collection_id=?", (collection_id,))
    }
    if not set(assignments) <= known:
        return refuse("map_stale")
    ids = sorted({cluster for cluster in assignments.values() if cluster >= 0})
    return assignments, {
        "available": True,
        "reason": None,
        "ids": ids,
        "unplaced": len(known - set(assignments)),
    }


def influence_ranking(
    nodes: Sequence[dict[str, Any]], edges: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    citations = {node["id"]: node.get("citation_count") for node in nodes}
    scores: dict[str, float] = {node["id"]: 0.0 for node in nodes}
    complete: dict[str, bool] = {node["id"]: True for node in nodes}

    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        if source == target or target not in scores:
            continue
        weight = citations.get(source)
        if weight is None:
            complete[target] = False
            continue
        scores[target] += float(weight)

    ranked = [
        {"paper_id": paper_id, "score": score, "complete": complete[paper_id]}
        for paper_id, score in scores.items()
    ]
    ranked.sort(key=lambda row: (-row["score"], row["paper_id"]))
    return ranked


TOP_PER_VIEW = 25
PAGERANK_ALPHA = 0.85


def compute_views(
    nodes: dict[str, dict[str, Any]], edges: Sequence[dict[str, Any]], *, top: int = TOP_PER_VIEW
) -> tuple[dict[str, Any], dict[str, int]]:
    try:
        import networkx as nx
    except ImportError:
        return {"available": False, "reason": "networkx_missing"}, {}

    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from((edge["source"], edge["target"]) for edge in edges)

    pagerank = nx.pagerank(graph, alpha=PAGERANK_ALPHA)
    notes: list[str] = []
    hubs = {node: 0.0 for node in graph}
    authorities = dict(hubs)
    if not edges:
        notes.append("no_edges")
    else:
        try:
            hubs, authorities = nx.hits(graph, max_iter=1000, tol=1e-10, normalized=True)
        except nx.PowerIterationFailedConvergence:
            notes.append("hits_did_not_converge")

    def entries(scores: dict[str, float], *, collection_only: bool = False) -> list[dict[str, Any]]:
        rows = [
            {
                "id": node_id,
                "title": nodes[node_id]["title"],
                "in_collection": bool(nodes[node_id]["resolved"]),
                "score": float(score),
            }
            for node_id, score in scores.items()
            if not collection_only or nodes[node_id]["resolved"]
        ]
        rows.sort(key=lambda row: (-row["score"], row["id"]))
        return rows[:top]

    views = {
        "available": True,
        "pagerank": entries(pagerank),
        "authority": entries(authorities),
        "hub": entries(hubs, collection_only=True),
        "scores": {
            node_id: {
                "pagerank": float(pagerank[node_id]),
                "authority": float(authorities[node_id]),
                "hub": float(hubs[node_id]),
            }
            for node_id in nodes
        },
        "notes": notes,
        "method": {
            "pagerank": f"networkx.pagerank alpha={PAGERANK_ALPHA}",
            "hits": "networkx.hits (Kleinberg 1999); can drift toward a densely cited "
            "tangent, so read hub/authority as indicative within this collection",
        },
    }
    internal = [node_id for node_id, node in nodes.items() if node["resolved"]]
    components = {
        "full": nx.number_weakly_connected_components(graph),
        "internal": nx.number_weakly_connected_components(graph.subgraph(internal))
        if internal
        else 0,
    }
    return views, components


def build_graph(
    conn: Any,
    collection_id: str,
    *,
    include_external: bool = False,
    min_edges: int = MIN_EDGES_FOR_GRAPH,
) -> GraphPayload:
    total = conn.execute(
        "SELECT count(*) FROM edges WHERE collection_id=?", (collection_id,)
    ).fetchone()[0]
    if total < min_edges:
        return GraphPayload(
            reason="not_enough_edges", needed=min_edges, have=int(total)
        )

    rows = conn.execute(
        """
        SELECT src_paper_id, dst_paper_id, dst_openalex_id, dst_title
        FROM edges
        WHERE collection_id=?
        ORDER BY src_paper_id, dst_openalex_id, dst_title
        """,
        (collection_id,),
    ).fetchall()

    nodes = _paper_nodes(conn, collection_id)
    edges: list[dict[str, Any]] = []
    unidentified = 0
    seen: set[tuple[str, str]] = set()

    for row in rows:
        target = row["dst_paper_id"] or _external_id(
            row["dst_openalex_id"], row["dst_title"]
        )
        if target is None:
            unidentified += 1
            continue
        resolved = bool(row["dst_paper_id"])
        if not resolved and target not in nodes:
            nodes[target] = {
                "id": target,
                "title": row["dst_title"] or row["dst_openalex_id"],
                "year": None,
                "citation_count": None,
                "resolved": False,
            }
        key = (row["src_paper_id"], target)
        if key in seen:
            continue
        seen.add(key)
        edges.append(
            {"source": row["src_paper_id"], "target": target, "resolved": resolved}
        )

    views, components = compute_views(nodes, edges)
    scores = views.get("scores", {})

    if include_external:
        shown_nodes = dict(nodes)
        shown_edges = edges
    else:
        shown_nodes = {node_id: node for node_id, node in nodes.items() if node["resolved"]}
        shown_edges = [edge for edge in edges if edge["resolved"]]

    assignments, clusters = cached_clusters(conn, collection_id)
    node_list = sorted(shown_nodes.values(), key=lambda node: node["id"])
    for node in node_list:
        node["pagerank"] = scores.get(node["id"], {}).get("pagerank")
        node["cluster"] = (assignments or {}).get(node["id"]) if node["resolved"] else None
    return GraphPayload(
        nodes=node_list,
        edges=shown_edges,
        rankings=influence_ranking(node_list, shown_edges),
        unidentified=unidentified,
        views=views,
        components=components,
        clusters=clusters,
    )
