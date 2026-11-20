
import json
import sys

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from ingest.arxiv import build_query
from ingest.cache import Cache
from ingest.chunk import Chunk
from ingest.http import HttpClient, RateLimited
from ingest.openalex import titles_for
from ingest.pdf import drop_references
from rag_core import Retriever, cosine_search, document_norms
from rag_core.index import InvertedIndex
from rag_core.lexical import tfidf_cosine_search
from services import retrieval
from store.db import connect, init_db
from store.repo import (
    PaperRow,
    create_collection,
    replace_chunks,
    save_embedding,
    upsert_paper,
)


def test_every_word_is_required():
    assert build_query("retrieval augmented generation") == (
        "all:retrieval AND all:augmented AND all:generation"
    )


def test_quoted_phrases_stay_intact():
    assert build_query('"retrieval augmented" survey') == 'all:"retrieval augmented" AND all:survey'


def test_operators_and_punctuation_cannot_inject_query_syntax():
    assert build_query("a AND b (c)") == "all:a AND all:b AND all:c"
    with pytest.raises(ValueError):
        build_query("   ")


def test_text_is_cut_at_the_reference_heading_and_appendices_dropped():
    pages = [
        "Intro.",
        "Method.",
        "Conclusion.\nReferences\n[1] A. arXiv preprint arXiv:2501.1 (2025).",
        "[2] B. arXiv preprint (2024).",
    ]
    kept, removed = drop_references(pages)
    assert removed is True
    assert "arXiv preprint" not in " ".join(kept)
    assert kept[-1].strip() == "Conclusion."


def test_a_mid_sentence_mention_of_references_does_not_cut_the_paper():
    pages = ["a", "b", "References are useful here, as we argue.", "d"]
    assert drop_references(pages) == (pages, False)


def test_a_references_heading_in_the_front_of_the_paper_is_ignored():
    pages = ["References", "body", "body", "body", "body"]
    assert drop_references(pages)[1] is False


def test_an_out_of_vocabulary_query_word_does_not_change_scores():
    index = InvertedIndex()
    index.add("d1", "beta beta")
    index.add("d2", "beta gamma")
    index.finalize()
    plain = dict(tfidf_cosine_search(index, "beta"))
    with_unseen = dict(tfidf_cosine_search(index, "beta zzqxv"))
    assert with_unseen == pytest.approx(plain)
    assert tfidf_cosine_search(index, "zzqxv") == []


def test_precomputed_norms_give_identical_rankings():
    docs = {f"d{i}": [float(i % 5 + 1), float(i % 3), 1.0, float(i % 7)] for i in range(40)}
    query = [0.3, 1.0, 0.5, 2.0]
    assert cosine_search(query, docs, 10, norms=document_norms(docs)) == cosine_search(
        query, docs, 10
    )


def test_zero_norm_documents_are_skipped_with_precomputed_norms():
    docs = {"zero": [0.0, 0.0], "ok": [1.0, 0.0]}
    out = cosine_search([1.0, 0.0], docs, 5, norms=document_norms(docs))
    assert [doc for doc, _ in out] == ["ok"]


def test_retriever_reuses_one_norm_table_across_queries():
    retriever = Retriever.build(
        [("a", "alpha"), ("b", "beta")], {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    )
    retriever.search("alpha", mode="dense", query_vector=[1.0, 0.0])
    first = retriever._norms
    retriever.search("beta", mode="hybrid", query_vector=[0.0, 1.0])
    assert retriever._norms is first


def _paper(title, arxiv_id):
    return PaperRow(
        arxiv_id=arxiv_id, openalex_id=None, doi=None, title=title, abstract="abs",
        authors=(), year=2024, venue=None, pdf_url=None, citation_count=1, source="arxiv",
        parse_status="ok", parse_note=None, has_fulltext=1,
    )


@pytest.fixture
def conn(tmp_path):
    db = tmp_path / "t.sqlite3"
    init_db(db)
    return connect(db)


def _add(conn, cid, title, arxiv_id, text):
    pid = upsert_paper(conn, collection_id=cid, paper=_paper(title, arxiv_id))
    replace_chunks(conn, pid, [Chunk(seq=0, text=text, n_tokens=len(text.split()))])
    chunk_id = conn.execute("SELECT id FROM chunks WHERE paper_id=?", (pid,)).fetchone()[0]
    save_embedding(conn, owner_type="chunk", owner_id=chunk_id, vec=[1.0, 0.0], model="m")


def test_the_index_is_built_once_and_reused(conn):
    cid = create_collection(conn, mode="topic", query="q", embedding_model="m")
    _add(conn, cid, "Alpha", "2401.00001", "alpha retrieval")
    first = retrieval.cached_index(conn, cid)
    assert retrieval.cached_index(conn, cid) is first


def test_a_changed_collection_is_never_served_from_a_stale_index(conn):
    cid = create_collection(conn, mode="topic", query="q", embedding_model="m")
    _add(conn, cid, "Alpha", "2401.00001", "alpha retrieval")
    before, _ = retrieval.cached_index(conn, cid)
    _add(conn, cid, "Beta", "2401.00002", "beta generation")
    after, _ = retrieval.cached_index(conn, cid)
    assert after is not before
    assert len(after.papers) == 2


def test_a_fresh_database_serves_requests_instead_of_returning_500(tmp_path):
    settings = Settings(db_path=tmp_path / "fresh" / "x.sqlite3", data_dir=tmp_path)
    app = create_app(settings, warm_query_embedder=False)
    with TestClient(app) as client:
        response = client.get("/api/collections")
    assert response.status_code == 200
    assert response.json() == {"collections": []}


def test_an_unexpected_error_returns_the_json_envelope_with_a_correlation_id(tmp_path):
    settings = Settings(db_path=tmp_path / "x.sqlite3", data_dir=tmp_path)
    app = create_app(settings, warm_query_embedder=False)

    @app.get("/boom")
    def boom():
        raise RuntimeError("secret internals")

    app.router.routes.insert(0, app.router.routes.pop())

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/boom")
    body = response.json()
    assert response.status_code == 500
    assert body["error"]["code"] == "internal_error"
    assert body["error"]["detail"]["correlation_id"] in body["error"]["message"]
    assert "secret internals" not in response.text


def test_a_missing_umap_is_reported_as_a_dependency_problem(monkeypatch):
    from services.map import MapDependencyMissing, project_2d

    monkeypatch.setitem(sys.modules, "umap", None)
    with pytest.raises(MapDependencyMissing, match="umap-learn"):
        project_2d([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])


def test_titles_from_earlier_batches_survive_a_later_failure(tmp_path):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            payload = {"results": [{"id": "https://openalex.org/W1", "display_name": "One"}]}
            return httpx.Response(200, content=json.dumps(payload).encode())
        return httpx.Response(429, headers={"Retry-After": "1"})

    client = HttpClient(Cache(tmp_path), user_agent="t", transport=httpx.MockTransport(handler))
    errors = []
    ids = [f"W{i}" for i in range(1, 120)]
    titles = titles_for(client, ids, on_error=errors.append)
    assert titles == {"W1": "One"}
    assert len(errors) == 1 and isinstance(errors[0], RateLimited)


def test_only_one_build_runs_at_a_time():
    from services.build_jobs import _BUILD_SLOT

    assert _BUILD_SLOT.acquire(blocking=False)
    try:
        assert not _BUILD_SLOT.acquire(blocking=False)
    finally:
        _BUILD_SLOT.release()


def test_related_work_on_a_tiny_collection_is_409_like_the_map_not_500(tmp_path):
    from store.repo import set_collection_status

    settings = Settings(db_path=tmp_path / "x.sqlite3", data_dir=tmp_path)
    with TestClient(create_app(settings, warm_query_embedder=False)) as client:
        conn = connect(settings.db_path)
        cid = create_collection(conn, mode="topic", query="q", embedding_model="m")
        set_collection_status(conn, cid, "ready")
        response = client.post(f"/api/collections/{cid}/related-work", json={"cluster_id": 0})
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "map_needs_more_papers"


def test_an_oversized_upload_is_rejected_while_reading(tmp_path, monkeypatch):
    import app.main as main_module

    monkeypatch.setattr(main_module, "MAX_FILE_BYTES", 1024)
    settings = Settings(db_path=tmp_path / "x.sqlite3", data_dir=tmp_path)
    app = create_app(settings, warm_query_embedder=False)
    app.state.upload_starter = lambda *a, **k: None
    with TestClient(app) as client:
        response = client.post(
            "/api/collections/upload",
            files=[("files", ("big.pdf", b"%PDF-" + b"x" * 5000, "application/pdf"))],
        )
    assert response.status_code == 413 and "larger than" in response.json()["error"]["message"]


def test_two_embedder_instances_share_one_load_lock():
    from embed.specter2 import Specter2Embedder

    assert Specter2Embedder()._load_lock is Specter2Embedder()._load_lock


def test_the_database_uses_wal(tmp_path):
    init_db(tmp_path / "w.sqlite3")
    conn = connect(tmp_path / "w.sqlite3")
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_the_index_cache_is_invalidated_by_a_change_that_keeps_the_counts(conn):
    cid = create_collection(conn, mode="topic", query="q", embedding_model="m")
    _add(conn, cid, "Alpha", "2401.00001", "alpha retrieval")
    before, _ = retrieval.cached_index(conn, cid)
    conn.execute("UPDATE chunks SET text='completely different text here'")
    conn.commit()
    after, _ = retrieval.cached_index(conn, cid)
    assert after is not before and "different" in after.texts[0]


def test_dense_scores_are_value_checked_against_hand_computed_cosines():
    docs = {"a": [1.0, 1.0], "b": [0.0, 1.0], "c": [2.0, 0.0]}
    out = dict(cosine_search([1.0, 0.0], docs, 5, norms=document_norms(docs)))
    assert out == pytest.approx({"c": 1.0, "a": 2 ** -0.5})
    assert dict(cosine_search([10.0, 0.0], docs, 5)) == pytest.approx(out)


def test_the_map_never_mixes_title_abstract_vectors_with_chunk_means(conn):
    from services.map import select_vectors

    cid = create_collection(conn, mode="topic", query="q", embedding_model="m")
    pids = []
    for n, title in enumerate(["Alpha", "Beta"]):
        pids.append(upsert_paper(conn, collection_id=cid, paper=_paper(title, f"2401.0000{n}")))
        replace_chunks(conn, pids[-1], [Chunk(seq=0, text="text " + title, n_tokens=2)])
        (chunk_id,) = conn.execute("SELECT id FROM chunks WHERE paper_id=?", (pids[-1],)).fetchone()
        save_embedding(conn, owner_type="chunk", owner_id=chunk_id, vec=[1.0, float(n)], model="m")

    save_embedding(conn, owner_type="paper", owner_id=pids[0], vec=[0.0, 1.0], model="m")
    _, _, _, source = select_vectors(conn, cid)
    assert source == "chunk_mean"

    save_embedding(conn, owner_type="paper", owner_id=pids[1], vec=[1.0, 1.0], model="m")
    ids, _, excluded, source = select_vectors(conn, cid)
    assert source == "title_abstract" and excluded == 0 and sorted(ids) == sorted(pids)
