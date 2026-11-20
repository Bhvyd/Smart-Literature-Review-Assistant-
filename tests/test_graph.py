
import pytest

from store.db import connect, init_db
from store.repo import (
    PaperRow,
    create_collection,
    set_collection_status,
    upsert_edge,
    upsert_paper,
)


@pytest.fixture
def conn(tmp_path):
    db = tmp_path / "graph.sqlite3"
    init_db(db)
    return connect(db)


def _paper(title, arxiv_id, citations):
    return PaperRow(
        arxiv_id=arxiv_id,
        openalex_id=f"W{arxiv_id[-3:]}",
        doi=None,
        title=title,
        abstract="a",
        authors=(),
        year=2024,
        venue=None,
        pdf_url=None,
        citation_count=citations,
        source="arxiv",
        parse_status="ok",
        parse_note=None,
        has_fulltext=1,
    )


def build_collection(conn, papers):
    cid = create_collection(conn, mode="topic", query="q", embedding_model="m")
    ids = {}
    for title, arxiv_id, citations in papers:
        ids[arxiv_id] = upsert_paper(
            conn, collection_id=cid, paper=_paper(title, arxiv_id, citations)
        )
    set_collection_status(conn, cid, "ready")
    return cid, ids


def cite(conn, cid, src_id, *, dst_paper_id=None, dst_openalex_id=None, dst_title=None):
    upsert_edge(
        conn,
        collection_id=cid,
        src_paper_id=src_id,
        dst_paper_id=dst_paper_id,
        dst_openalex_id=dst_openalex_id,
        dst_title=dst_title,
        dst_doi=None,
    )


def test_edge_direction_is_source_cites_target(conn):
    from services.graph import build_graph

    cid, ids = build_collection(
        conn,
        [("A", "2402.001", 100), ("B", "2402.002", 10), ("C", "2402.003", 1)],
    )
    cite(conn, cid, ids["2402.001"], dst_paper_id=ids["2402.002"], dst_openalex_id="W002", dst_title="B")

    payload = build_graph(conn, cid, include_external=False, min_edges=1)
    assert len(payload.edges) == 1
    edge = payload.edges[0]
    assert edge["source"] == ids["2402.001"]
    assert edge["target"] == ids["2402.002"]
    assert edge["resolved"] is True


def test_influence_is_in_degree_weighted_by_the_citing_paper(conn):
    from services.graph import build_graph, influence_ranking

    cid, ids = build_collection(
        conn,
        [("A", "2402.001", 100), ("B", "2402.002", 1), ("C", "2402.003", 0)],
    )
    cite(conn, cid, ids["2402.001"], dst_paper_id=ids["2402.002"], dst_openalex_id="W002", dst_title="B")
    cite(conn, cid, ids["2402.002"], dst_paper_id=ids["2402.003"], dst_openalex_id="W003", dst_title="C")

    payload = build_graph(conn, cid, include_external=False, min_edges=1)
    ranking = {row["paper_id"]: row["score"] for row in influence_ranking(payload.nodes, payload.edges)}
    assert ranking[ids["2402.002"]] == pytest.approx(100.0)
    assert ranking[ids["2402.003"]] == pytest.approx(1.0)
    assert ranking.get(ids["2402.001"], 0.0) == 0.0


def test_influence_ranking_is_sorted_and_deterministic(conn):
    from services.graph import build_graph, influence_ranking

    cid, ids = build_collection(
        conn,
        [("A", "2402.001", 50), ("B", "2402.002", 0), ("C", "2402.003", 0)],
    )
    cite(conn, cid, ids["2402.001"], dst_paper_id=ids["2402.002"], dst_openalex_id="W002", dst_title="B")
    cite(conn, cid, ids["2402.003"], dst_paper_id=ids["2402.002"], dst_openalex_id="W002", dst_title="B")

    payload = build_graph(conn, cid, include_external=False, min_edges=1)
    first = influence_ranking(payload.nodes, payload.edges)
    second = influence_ranking(payload.nodes, payload.edges)
    assert first == second
    assert first[0]["paper_id"] == ids["2402.002"]
    assert first[0]["score"] == pytest.approx(50.0)


def test_a_citer_with_unknown_citations_contributes_zero_and_is_flagged(conn):
    from services.graph import build_graph, influence_ranking

    cid, ids = build_collection(
        conn,
        [("A", "2402.001", None), ("B", "2402.002", 0), ("C", "2402.003", 5)],
    )
    cite(conn, cid, ids["2402.001"], dst_paper_id=ids["2402.002"], dst_openalex_id="W002", dst_title="B")
    cite(conn, cid, ids["2402.003"], dst_paper_id=ids["2402.002"], dst_openalex_id="W002", dst_title="B")

    payload = build_graph(conn, cid, include_external=False, min_edges=1)
    ranking = {row["paper_id"]: row for row in influence_ranking(payload.nodes, payload.edges)}
    b = ranking[ids["2402.002"]]
    assert b["score"] == pytest.approx(5.0)
    assert b["complete"] is False


def test_external_references_are_excluded_by_default(conn):
    from services.graph import build_graph

    cid, ids = build_collection(conn, [("A", "2402.001", 3)])
    cite(conn, cid, ids["2402.001"], dst_openalex_id="W999", dst_title="Some External Work")

    payload = build_graph(conn, cid, include_external=False, min_edges=1)
    assert len(payload.nodes) == 1
    assert payload.edges == []


def test_external_references_can_be_included_with_their_real_title(conn):
    from services.graph import build_graph

    cid, ids = build_collection(conn, [("A", "2402.001", 3)])
    cite(conn, cid, ids["2402.001"], dst_openalex_id="W999", dst_title="Some External Work")

    payload = build_graph(conn, cid, include_external=True, min_edges=1)
    external = [node for node in payload.nodes if not node["resolved"]]
    assert len(external) == 1
    assert external[0]["title"] == "Some External Work"
    assert external[0]["id"] == "ext:W999"
    assert payload.edges[0]["resolved"] is False
    assert payload.edges[0]["target"] == "ext:W999"


def test_an_edge_with_no_identifiable_target_is_dropped_not_invented(conn):
    from services.graph import build_graph

    cid, ids = build_collection(conn, [("A", "2402.001", 3)])
    cite(conn, cid, ids["2402.001"], dst_openalex_id=None, dst_title=None)

    payload = build_graph(conn, cid, include_external=True, min_edges=1)
    assert payload.edges == []
    assert payload.unidentified == 1


def test_too_few_edges_returns_an_empty_payload_with_a_reason(conn):
    from services.graph import MIN_EDGES_FOR_GRAPH, build_graph

    cid, ids = build_collection(conn, [("A", "2402.001", 3), ("B", "2402.002", 1)])
    cite(conn, cid, ids["2402.001"], dst_paper_id=ids["2402.002"], dst_openalex_id="W002", dst_title="B")

    payload = build_graph(conn, cid, include_external=False)
    assert payload.reason == "not_enough_edges"
    assert payload.needed == MIN_EDGES_FOR_GRAPH
    assert payload.have == 1
    assert payload.nodes == []
    assert payload.edges == []


def test_enough_edges_produces_no_reason(conn):
    from services.graph import build_graph

    cid, ids = build_collection(
        conn,
        [(f"P{i}", f"2402.00{i}", i) for i in range(7)],
    )
    for i in range(6):
        cite(
            conn,
            cid,
            ids[f"2402.00{i}"],
            dst_paper_id=ids[f"2402.00{i + 1}"],
            dst_openalex_id=f"W00{i + 1}",
            dst_title=f"P{i + 1}",
        )

    payload = build_graph(conn, cid, include_external=False)
    assert payload.reason is None
    assert len(payload.edges) == 6
    assert len(payload.nodes) == 7


def test_nodes_carry_stored_metadata_only(conn):
    from services.graph import build_graph

    cid, ids = build_collection(conn, [(f"P{i}", f"2402.00{i}", i * 10) for i in range(7)])
    for i in range(6):
        cite(
            conn,
            cid,
            ids[f"2402.00{i}"],
            dst_paper_id=ids[f"2402.00{i + 1}"],
            dst_openalex_id=f"W00{i + 1}",
            dst_title=f"P{i + 1}",
        )

    payload = build_graph(conn, cid, include_external=False)
    node = next(n for n in payload.nodes if n["id"] == ids["2402.001"])
    assert node["title"] == "P1"
    assert node["citation_count"] == 10
    assert node["year"] == 2024
    assert node["resolved"] is True


def test_views_rank_a_widely_cited_outside_work_as_the_foundational_one(conn):
    from services.graph import build_graph

    cid, ids = build_collection(conn, [("A", "2402.001", 5), ("B", "2402.002", 5), ("C", "2402.003", 5)])
    for src in ("2402.001", "2402.002"):
        cite(conn, cid, ids[src], dst_openalex_id="W9000", dst_title="Foundational Work")
    cite(conn, cid, ids["2402.001"], dst_openalex_id="W9001", dst_title="Niche Work")
    cite(conn, cid, ids["2402.003"], dst_paper_id=ids["2402.001"], dst_openalex_id="W001", dst_title="A")

    payload = build_graph(conn, cid, include_external=False, min_edges=1)
    views = payload.views
    assert views["available"] is True
    assert views["authority"][0]["title"] == "Foundational Work"
    assert views["authority"][0]["in_collection"] is False
    assert views["pagerank"][0]["title"] == "Foundational Work"
    assert sorted(node["title"] for node in payload.nodes) == ["A", "B", "C"]


def test_hubs_are_collection_papers_that_cite_many_authorities(conn):
    from services.graph import build_graph

    cid, ids = build_collection(conn, [("Survey", "2402.001", 5), ("Narrow", "2402.002", 5)])
    for work_id in ("W9000", "W9001", "W9002"):
        cite(conn, cid, ids["2402.001"], dst_openalex_id=work_id, dst_title=work_id)
    cite(conn, cid, ids["2402.002"], dst_openalex_id="W9000", dst_title="W9000")

    hub = build_graph(conn, cid, include_external=False, min_edges=1).views["hub"]
    assert [row["title"] for row in hub] == ["Survey", "Narrow"]
    assert all(row["in_collection"] for row in hub)


def test_direction_is_preserved_in_the_views(conn):
    from services.graph import build_graph

    cid, ids = build_collection(conn, [("Citing", "2402.001", 5), ("Cited", "2402.002", 5)])
    cite(conn, cid, ids["2402.001"], dst_paper_id=ids["2402.002"], dst_openalex_id="W002", dst_title="Cited")

    views = build_graph(conn, cid, include_external=False, min_edges=1).views
    assert views["authority"][0]["title"] == "Cited"
    assert views["hub"][0]["title"] == "Citing"


def test_a_graph_without_edges_yields_zero_scores_not_nan(conn):
    from services.graph import compute_views

    nodes = {"p1": {"id": "p1", "title": "P1", "resolved": True}}
    views, components = compute_views(nodes, [])
    assert views["scores"]["p1"] == {"pagerank": 1.0, "authority": 0.0, "hub": 0.0}
    assert "no_edges" in views["notes"]
    assert components == {"full": 1, "internal": 1}


def test_component_counts_are_reported_not_hidden(conn):
    from services.graph import compute_views

    nodes = {n: {"id": n, "title": n, "resolved": True} for n in ("a", "b", "c", "d")}
    edges = [{"source": "a", "target": "b", "resolved": True}]
    _, components = compute_views(nodes, edges)
    assert components == {"full": 3, "internal": 3}


def test_a_refused_graph_says_why_in_views_too(conn):
    from services.graph import build_graph

    cid, _ = build_collection(conn, [("A", "2402.001", 1)])
    payload = build_graph(conn, cid, min_edges=5).as_dict()
    assert payload["views"] == {"available": False, "reason": "not_enough_edges"}


def _store_projection(conn, cid, assignments, *, cluster_note=None):
    import json

    points = [
        {"paper_id": pid, "x": float(i), "y": 0.0, "cluster": cluster}
        for i, (pid, cluster) in enumerate(assignments.items())
    ]
    payload = {"points": points, "method": "umap-hdbscan", "params_hash": "h", "excluded": 0,
               "cluster_note": cluster_note, "vector_source": "title_abstract"}
    with conn:
        conn.execute(
            "INSERT INTO projections (collection_id, method, params_hash, payload, created_at) "
            "VALUES (?, 'umap-hdbscan', 'h', ?, 'now')",
            (cid, json.dumps(payload)),
        )


def _clustered_world(conn):
    cid, ids = build_collection(
        conn, [("A", "2402.001", 3), ("B", "2402.002", 2), ("C", "2402.003", 1)]
    )
    cite(conn, cid, ids["2402.001"], dst_paper_id=ids["2402.002"], dst_openalex_id="W002", dst_title="B")
    cite(conn, cid, ids["2402.001"], dst_openalex_id="W999", dst_title="Outside")
    return cid, ids


def test_nodes_take_their_cluster_from_the_cached_map_without_computing_one(conn, monkeypatch):
    import services.map as map_module
    from services.graph import build_graph

    def forbidden(*args, **kwargs):
        raise AssertionError("the graph path must never trigger the UMAP/HDBSCAN computation")

    monkeypatch.setattr(map_module, "build_map", forbidden)
    cid, ids = _clustered_world(conn)
    _store_projection(conn, cid, {ids["2402.001"]: 1, ids["2402.002"]: 0, ids["2402.003"]: -1})

    payload = build_graph(conn, cid, include_external=True, min_edges=1).as_dict()
    by_id = {node["id"]: node for node in payload["nodes"]}
    assert by_id[ids["2402.001"]]["cluster"] == 1
    assert by_id[ids["2402.002"]]["cluster"] == 0
    assert by_id[ids["2402.003"]]["cluster"] == -1
    assert by_id["ext:W999"]["cluster"] is None
    assert payload["clusters"] == {"available": True, "reason": None, "ids": [0, 1], "unplaced": 0}


def test_without_a_cached_map_the_graph_says_why_nodes_are_uncoloured(conn):
    from services.graph import build_graph

    cid, ids = _clustered_world(conn)
    payload = build_graph(conn, cid, include_external=False, min_edges=1).as_dict()
    assert payload["clusters"] == {
        "available": False, "reason": "map_not_computed", "ids": [], "unplaced": 0,
    }
    assert all(node["cluster"] is None for node in payload["nodes"])


def test_an_unreliable_or_stale_map_is_not_used_for_colour(conn):
    from services.graph import build_graph

    cid, ids = _clustered_world(conn)
    _store_projection(conn, cid, {ids["2402.001"]: -1, ids["2402.002"]: -1}, cluster_note="clusters_unreliable")
    assert build_graph(conn, cid, min_edges=1).as_dict()["clusters"]["reason"] == "clusters_unreliable"

    with conn:
        conn.execute("DELETE FROM projections")
    _store_projection(conn, cid, {ids["2402.001"]: 0, "a-paper-that-left-the-collection": 1})
    payload = build_graph(conn, cid, min_edges=1).as_dict()
    assert payload["clusters"]["reason"] == "map_stale"
    assert all(node["cluster"] is None for node in payload["nodes"])


def test_papers_missing_from_the_cached_map_are_counted_and_carry_no_cluster(conn):
    from services.graph import build_graph

    cid, ids = _clustered_world(conn)
    _store_projection(conn, cid, {ids["2402.001"]: 0, ids["2402.002"]: 1})
    payload = build_graph(conn, cid, include_external=False, min_edges=1).as_dict()
    by_id = {node["id"]: node for node in payload["nodes"]}
    assert payload["clusters"] == {"available": True, "reason": None, "ids": [0, 1], "unplaced": 1}
    assert by_id[ids["2402.003"]]["cluster"] is None
    assert by_id[ids["2402.001"]]["cluster"] == 0


def test_a_map_with_no_points_is_unavailable_not_an_empty_legend(conn):
    from services.graph import build_graph

    cid, _ = _clustered_world(conn)
    _store_projection(conn, cid, {})
    clusters = build_graph(conn, cid, min_edges=1).as_dict()["clusters"]
    assert clusters == {"available": False, "reason": "map_empty", "ids": [], "unplaced": 0}


def test_only_this_collections_papers_count_when_judging_a_cached_map(conn):
    from services.graph import build_graph

    cid, ids = _clustered_world(conn)
    other_cid, other_ids = build_collection(conn, [("Elsewhere", "2402.900", 1)])
    _store_projection(conn, cid, {ids["2402.001"]: 0, other_ids["2402.900"]: 1})
    payload = build_graph(conn, cid, min_edges=1).as_dict()
    assert payload["clusters"]["reason"] == "map_stale"
    with conn:
        conn.execute("DELETE FROM projections")
    _store_projection(conn, cid, {ids["2402.001"]: 0, ids["2402.002"]: 0, ids["2402.003"]: 0})
    assert build_graph(conn, cid, min_edges=1).as_dict()["clusters"]["unplaced"] == 0


def test_the_graph_module_does_not_import_the_map_module_at_all():
    import ast
    from pathlib import Path

    import services.graph as graph_module

    tree = ast.parse(Path(graph_module.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported.update(f"{node.module}.{alias.name}" for alias in node.names)
    assert not {name for name in imported if name.endswith(("services.map", ".map")) or name == "map"}
