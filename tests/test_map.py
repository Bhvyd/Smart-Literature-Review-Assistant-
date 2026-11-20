
import math

import pytest

from ingest.chunk import Chunk
from store.db import connect, init_db
from store.repo import (
    PaperRow,
    create_collection,
    replace_chunks,
    save_embedding,
    upsert_paper,
)


@pytest.fixture
def conn(tmp_path):
    db = tmp_path / "map.sqlite3"
    init_db(db)
    return connect(db)


def _paper(title, arxiv_id):
    return PaperRow(
        arxiv_id=arxiv_id,
        openalex_id=None,
        doi=None,
        title=title,
        abstract="a",
        authors=(),
        year=2024,
        venue=None,
        pdf_url=None,
        citation_count=0,
        source="arxiv",
        parse_status="ok",
        parse_note=None,
        has_fulltext=1,
    )


def add_paper(conn, cid, title, arxiv_id, vectors):
    pid = upsert_paper(conn, collection_id=cid, paper=_paper(title, arxiv_id))
    chunks = [Chunk(seq=i, text=f"{title} chunk {i}", n_tokens=2) for i in range(len(vectors))]
    replace_chunks(conn, pid, chunks)
    ids = [row[0] for row in conn.execute("SELECT id FROM chunks WHERE paper_id=? ORDER BY seq", (pid,))]
    for chunk_id, vec in zip(ids, vectors):
        save_embedding(conn, owner_type="chunk", owner_id=chunk_id, vec=list(vec), model="m")
    return pid


def test_paper_vector_is_the_unit_mean_of_its_chunks(conn):
    from services.map import paper_vectors

    cid = create_collection(conn, mode="topic", query="q", embedding_model="m")
    pid = add_paper(conn, cid, "A", "2402.00001", [(1.0, 0.0), (0.0, 1.0)])

    ids, vectors, excluded = paper_vectors(conn, cid)
    assert ids == [pid]
    assert excluded == 0
    assert vectors[0] == pytest.approx([1 / math.sqrt(2), 1 / math.sqrt(2)], abs=1e-6)
    assert math.sqrt(sum(v * v for v in vectors[0])) == pytest.approx(1.0, abs=1e-6)


def test_a_single_chunk_vector_keeps_its_direction(conn):
    from services.map import paper_vectors

    cid = create_collection(conn, mode="topic", query="q", embedding_model="m")
    add_paper(conn, cid, "A", "2402.00001", [(3.0, 4.0)])

    _, vectors, _ = paper_vectors(conn, cid)
    assert vectors[0] == pytest.approx([0.6, 0.8], abs=1e-6)


def test_a_paper_without_vectors_is_excluded_and_counted(conn):
    from services.map import paper_vectors

    cid = create_collection(conn, mode="topic", query="q", embedding_model="m")
    add_paper(conn, cid, "HasVector", "2402.00001", [(1.0, 0.0)])
    upsert_paper(conn, collection_id=cid, paper=_paper("NoChunks", "2402.00002"))

    ids, vectors, excluded = paper_vectors(conn, cid)
    assert len(ids) == len(vectors) == 1
    assert excluded == 1


def test_vectors_that_cancel_to_zero_are_excluded(conn):
    from services.map import paper_vectors

    cid = create_collection(conn, mode="topic", query="q", embedding_model="m")
    add_paper(conn, cid, "Cancels", "2402.00001", [(1.0, 0.0), (-1.0, 0.0)])

    ids, vectors, excluded = paper_vectors(conn, cid)
    assert ids == [] and vectors == []
    assert excluded == 1


def test_paper_ids_are_deterministic_and_sorted(conn):
    from services.map import paper_vectors

    cid = create_collection(conn, mode="topic", query="q", embedding_model="m")
    for i in range(4):
        add_paper(conn, cid, f"P{i}", f"2402.0000{i}", [(1.0, float(i + 1))])

    ids, vectors, _ = paper_vectors(conn, cid)
    assert ids == sorted(ids)
    assert len(ids) == len(vectors) == 4


def test_an_empty_collection_yields_nothing(conn):
    from services.map import paper_vectors

    cid = create_collection(conn, mode="topic", query="q", embedding_model="m")
    assert paper_vectors(conn, cid) == ([], [], 0)


def _seeded_collection(conn, n=20, dim=4):
    cid = create_collection(conn, mode="topic", query="q", embedding_model="m")
    ids = []
    for i in range(n):
        vec = [((i + 1) * (j + 1)) % 7 + 1.0 for j in range(dim)]
        ids.append(add_paper(conn, cid, f"P{i}", f"2402.000{i:02d}", [vec]))
    return cid, ids


def fake_projector(vectors, *, method, seed, n_neighbors):
    return [(float(index), float(index % 3)) for index in range(len(vectors))]


def fake_clusterer(points, *, min_cluster_size):
    return [index % 2 for index in range(len(points))]


def test_params_hash_changes_with_papers_vectors_excluded_and_params():
    from services.map import params_hash

    ids = ["a", "b"]
    vectors = [[1.0, 0.0], [0.0, 1.0]]
    base = params_hash(ids, vectors)
    assert params_hash(ids, vectors) == base
    assert params_hash(ids, vectors, seed=7) != base
    assert params_hash(ids, vectors, method="pca") != base
    assert params_hash(ids, [[1.0, 0.0], [1.0, 0.0]]) != base
    assert params_hash(["b", "a"], vectors) != base
    assert params_hash(ids, vectors, excluded=1) != base


def test_build_map_returns_one_finite_point_per_paper(conn):
    from services.map import build_map

    cid, ids = _seeded_collection(conn, n=20)
    payload = build_map(conn, cid, projector=fake_projector, clusterer=fake_clusterer)

    assert [point.paper_id for point in payload.points] == sorted(ids)
    assert all(math.isfinite(point.x) and math.isfinite(point.y) for point in payload.points)
    assert payload.method == "umap-hdbscan"
    assert payload.excluded == 0


def test_the_second_call_returns_identical_coordinates_from_cache(conn, monkeypatch):
    from services import map as map_service

    cid, _ = _seeded_collection(conn, n=20)
    first = map_service.build_map(conn, cid, projector=fake_projector, clusterer=fake_clusterer)

    def must_not_run(*args, **kwargs):
        raise AssertionError("cached payload should have been reused")

    second = map_service.build_map(conn, cid, projector=must_not_run, clusterer=must_not_run)
    assert [(p.x, p.y, p.cluster) for p in first.points] == [
        (p.x, p.y, p.cluster) for p in second.points
    ]
    assert first.params_hash == second.params_hash


def test_changing_the_seed_invalidates_the_cache(conn):
    from services.map import build_map

    cid, _ = _seeded_collection(conn, n=20)
    first = build_map(conn, cid, seed=42, projector=fake_projector, clusterer=fake_clusterer)
    second = build_map(conn, cid, seed=7, projector=fake_projector, clusterer=fake_clusterer)
    assert first.params_hash != second.params_hash
    assert conn.execute("SELECT count(*) FROM projections").fetchone()[0] == 1


def test_a_paper_added_without_vectors_invalidates_the_cached_map(conn):
    from services.map import build_map

    cid, _ = _seeded_collection(conn, n=20)
    first = build_map(conn, cid, projector=fake_projector, clusterer=fake_clusterer)
    assert first.excluded == 0

    upsert_paper(conn, collection_id=cid, paper=_paper("NoVectors", "2402.99999"))

    second = build_map(conn, cid, projector=fake_projector, clusterer=fake_clusterer)
    assert second.excluded == 1
    assert len(second.points) == len(first.points)


def test_too_few_papers_is_a_reasoned_refusal(conn):
    from services.map import MapUnavailable, build_map

    cid, _ = _seeded_collection(conn, n=3)
    with pytest.raises(MapUnavailable) as excinfo:
        build_map(conn, cid, min_papers=15, projector=fake_projector, clusterer=fake_clusterer)
    assert excinfo.value.reason == "too_few_papers"
    assert excinfo.value.needed == 15
    assert excinfo.value.have == 3


def test_papers_without_vectors_are_reported_as_excluded(conn):
    from services.map import build_map

    cid, _ = _seeded_collection(conn, n=20)
    upsert_paper(conn, collection_id=cid, paper=_paper("NoVector", "2402.99999"))
    payload = build_map(conn, cid, projector=fake_projector, clusterer=fake_clusterer)
    assert payload.excluded == 1
    assert len(payload.points) == 20


def test_a_clusterer_that_calls_everything_noise_is_retried_once(conn):
    from services.map import build_map

    cid, _ = _seeded_collection(conn, n=20)
    sizes = []

    def noisy_then_fine(points, *, min_cluster_size):
        sizes.append(min_cluster_size)
        if len(sizes) == 1:
            return [-1] * len(points)
        return [index % 2 for index in range(len(points))]

    payload = build_map(conn, cid, projector=fake_projector, clusterer=noisy_then_fine)
    assert sizes == [max(5, 20 // 50), max(3, max(5, 20 // 50) // 2)]
    assert {point.cluster for point in payload.points} == {0, 1}
    assert payload.cluster_note is None


def test_still_all_noise_after_the_retry_returns_unclustered_with_a_note(conn):
    from services.map import build_map

    cid, _ = _seeded_collection(conn, n=20)

    def always_noise(points, *, min_cluster_size):
        return [-1] * len(points)

    payload = build_map(conn, cid, projector=fake_projector, clusterer=always_noise)
    assert payload.cluster_note == "clusters_unreliable"
    assert {point.cluster for point in payload.points} == {-1}


def test_a_noisy_first_pass_is_retried_and_the_better_result_kept(conn):
    from services.map import build_map

    cid, _ = _seeded_collection(conn, n=20)
    calls = []

    def improving(points, *, min_cluster_size):
        calls.append(min_cluster_size)
        labels = [-1] * len(points)
        if len(calls) == 1:
            labels[0] = labels[1] = labels[2] = labels[3] = labels[4] = labels[5] = 0
            return labels
        for index in range(12):
            labels[index] = index % 2
        return labels

    payload = build_map(conn, cid, projector=fake_projector, clusterer=improving)
    assert len(calls) == 2
    assert payload.cluster_note is None
    assert {point.cluster for point in payload.points} == {0, 1, -1}

import os

RUN_MAP = pytest.mark.skipif(
    os.environ.get("RUN_MAP_TESTS") != "1", reason="set RUN_MAP_TESTS=1"
)


@RUN_MAP
def test_real_umap_projection_is_reproducible(conn):
    from services.map import build_map, project_2d

    cid, _ = _seeded_collection(conn, n=24, dim=8)
    ids, vectors, _ = __import__("services.map", fromlist=["paper_vectors"]).paper_vectors(conn, cid)

    first = project_2d(vectors, method="umap", seed=42)
    second = project_2d(vectors, method="umap", seed=42)
    assert len(first) == len(ids) == 24
    assert all(math.isfinite(x) and math.isfinite(y) for x, y in first)
    flat = [value for pair in first for value in pair]
    flat_again = [value for pair in second for value in pair]
    assert flat == pytest.approx(flat_again, abs=1e-9)

    payload = build_map(conn, cid)
    assert len(payload.points) == 24
    assert all(point.cluster >= -1 for point in payload.points)
