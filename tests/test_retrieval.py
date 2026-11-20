
import pytest

from ingest.chunk import Chunk
from services.retrieval import CollectionIndex, load_chunks, load_papers, rank_papers
from store.db import connect
from store.repo import (
    PaperRow,
    create_collection,
    replace_chunks,
    save_embedding,
    upsert_paper,
)


@pytest.fixture
def conn(tmp_path):
    from store.db import init_db

    db = tmp_path / "t.sqlite3"
    init_db(db)
    return connect(db)


def _paper(title, abstract, arxiv_id):
    return PaperRow(
        arxiv_id=arxiv_id,
        openalex_id=None,
        doi=None,
        title=title,
        abstract=abstract,
        authors=("Ada",),
        year=2024,
        venue=None,
        pdf_url=None,
        citation_count=10,
        source="arxiv",
        parse_status="ok",
        parse_note=None,
        has_fulltext=1,
    )


def _collection(conn, papers):
    cid = create_collection(conn, mode="topic", query="q", embedding_model="m")
    for title, abstract, arxiv_id, texts in papers:
        pid = upsert_paper(conn, collection_id=cid, paper=_paper(title, abstract, arxiv_id))
        chunks = [Chunk(seq=i, text=text, n_tokens=len(text.split())) for i, text in enumerate(texts)]
        replace_chunks(conn, pid, chunks)
        ids = [row[0] for row in conn.execute("SELECT id FROM chunks WHERE paper_id=? ORDER BY seq", (pid,))]
        for chunk_id in ids:
            save_embedding(conn, owner_type="chunk", owner_id=chunk_id, vec=[1.0, 0.0], model="m")
    return cid


def test_load_papers_reads_stored_rows(conn):
    cid = _collection(conn, [("Attention", "about attention", "2402.00001", ["attention text"])])
    papers = load_papers(conn, cid)
    assert len(papers) == 1
    assert papers[0].title == "Attention"
    assert papers[0].citation_count == 10


def test_load_chunks_is_ordered_and_attributed(conn):
    cid = _collection(
        conn,
        [("A", "abstract a", "2402.00001", ["a one", "a two"]), ("B", "abstract b", "2402.00002", ["b one"])],
    )
    chunks = load_chunks(conn, cid)
    assert len(chunks) == 3

    assert [c.text for c in chunks if c.text.startswith("a")] == ["a one", "a two"]
    assert [c.text for c in chunks if c.text.startswith("b")] == ["b one"]
    paper_sequence = [c.paper_id for c in chunks]
    assert paper_sequence == sorted(paper_sequence)
    assert len({c.paper_id for c in chunks}) == 2


def test_index_aligns_chunk_ids_papers_and_vectors(conn):
    cid = _collection(conn, [("A", "abstract a", "2402.00001", ["a one", "a two"])])
    index = CollectionIndex.build(conn, cid)
    assert len(index.chunk_ids) == 2
    assert len(index.texts) == 2
    assert len(index.chunk_paper) == 2
    assert set(index.dense_vectors) == set(index.chunk_ids)
    assert index.missing_vectors == 0


def test_a_chunk_without_a_vector_is_still_searchable_lexically(conn):
    cid = _collection(conn, [("A", "abstract a", "2402.00001", ["unique zebra phrase"])])
    conn.execute("DELETE FROM embeddings")
    conn.commit()

    index = CollectionIndex.build(conn, cid)
    assert index.missing_vectors == 1
    result = index.retriever().search("zebra", mode="lexical")
    assert [hit.doc_id for hit in result.hits] == index.chunk_ids


def test_a_paper_with_no_chunks_still_gets_an_abstract_document(conn):
    cid = create_collection(conn, mode="topic", query="q", embedding_model="m")
    upsert_paper(
        conn,
        collection_id=cid,
        paper=_paper("Orphan Title", "an abstract about giraffes", "2402.00009"),
    )
    index = CollectionIndex.build(conn, cid)
    assert len(index.chunk_ids) == 0
    assert list(index.abstract_documents.values())[0].startswith("Orphan Title")

    hits = index.retriever().search("giraffes", mode="lexical")
    assert hits.hits
    assert index.paper_for_document(hits.hits[0].doc_id) is not None


def test_retriever_answers_lexical_queries_over_real_rows(conn):
    cid = _collection(
        conn,
        [
            ("Attention", "x", "2402.00001", ["self attention transformer"]),
            ("Cooking", "y", "2402.00002", ["sourdough starter recipe"]),
        ],
    )
    index = CollectionIndex.build(conn, cid)
    hits = index.retriever().search("attention", mode="lexical")
    assert hits.hits
    top_paper = index.paper_for_document(hits.hits[0].doc_id)
    assert index.paper_by_id[top_paper].title == "Attention"


def test_empty_collection_builds_an_empty_index(conn):
    cid = create_collection(conn, mode="topic", query="q", embedding_model="m")
    index = CollectionIndex.build(conn, cid)
    assert index.chunk_ids == []
    assert index.documents() == []


def test_ranking_puts_the_matching_paper_first(conn):
    cid = _collection(
        conn,
        [
            ("Attention", "x", "2402.00001", ["self attention transformer"]),
            ("Cooking", "y", "2402.00002", ["sourdough starter recipe"]),
        ],
    )
    result = rank_papers(conn, cid, query="attention", mode="lexical", top_k=5)
    assert [paper.title for paper in result.papers][:1] == ["Attention"]
    assert result.papers[0].score is not None
    assert result.mode == "lexical"


def test_ranking_collapses_chunks_to_one_entry_per_paper(conn):
    cid = _collection(
        conn,
        [("A", "x", "2402.00001", ["attention one", "attention two", "attention three"])],
    )
    result = rank_papers(conn, cid, query="attention", mode="lexical", top_k=10)
    assert len(result.papers) == 1
    assert len({paper.paper_id for paper in result.papers}) == 1


def test_ranking_without_a_query_orders_by_citations_with_nulls_last(conn):
    cid = create_collection(conn, mode="topic", query="q", embedding_model="m")
    for title, arxiv_id, cites in (("Low", "2402.00001", 1), ("None", "2402.00002", None), ("High", "2402.00003", 99)):
        upsert_paper(
            conn,
            collection_id=cid,
            paper=PaperRow(
                arxiv_id=arxiv_id, openalex_id=None, doi=None, title=title, abstract="a",
                authors=(), year=2024, venue=None, pdf_url=None, citation_count=cites,
                source="arxiv", parse_status="ok", parse_note=None, has_fulltext=1,
            ),
        )
    result = rank_papers(conn, cid, query=None, top_k=10)
    assert [paper.title for paper in result.papers] == ["High", "Low", "None"]
    assert result.mode == "influence"


def test_ranking_is_deterministic_across_calls(conn):
    cid = _collection(
        conn,
        [
            ("A", "x", "2402.00001", ["attention attention"]),
            ("B", "y", "2402.00002", ["attention attention"]),
        ],
    )
    first = rank_papers(conn, cid, query="attention", mode="lexical", top_k=5)
    second = rank_papers(conn, cid, query="attention", mode="lexical", top_k=5)
    assert [p.paper_id for p in first.papers] == [p.paper_id for p in second.papers]


def test_dense_mode_without_a_query_vector_degrades_to_lexical(conn):
    cid = _collection(conn, [("A", "x", "2402.00001", ["attention text"])])
    result = rank_papers(conn, cid, query="attention", mode="hybrid", top_k=5)
    assert result.mode == "lexical"
    assert "query_vector_missing" in result.degraded


def test_dense_ranking_uses_the_stored_vectors(conn):
    cid = create_collection(conn, mode="topic", query="q", embedding_model="m")
    pids = []
    for title, arxiv_id, vec in (
        ("Near", "2402.00001", [1.0, 0.0]),
        ("Far", "2402.00002", [0.0, 1.0]),
    ):
        pid = upsert_paper(
            conn,
            collection_id=cid,
            paper=PaperRow(
                arxiv_id=arxiv_id, openalex_id=None, doi=None, title=title, abstract="a",
                authors=(), year=2024, venue=None, pdf_url=None, citation_count=0,
                source="arxiv", parse_status="ok", parse_note=None, has_fulltext=1,
            ),
        )
        replace_chunks(conn, pid, [Chunk(seq=0, text=f"{title} text", n_tokens=2)])
        chunk_id = conn.execute("SELECT id FROM chunks WHERE paper_id=?", (pid,)).fetchone()[0]
        save_embedding(conn, owner_type="chunk", owner_id=chunk_id, vec=vec, model="m")
        pids.append(pid)

    result = rank_papers(conn, cid, query="anything", mode="dense", top_k=5, query_vector=[1.0, 0.0])
    assert result.mode == "dense"
    assert result.papers[0].title == "Near"


def test_hard_query_limit_is_respected(conn):
    cid = _collection(
        conn,
        [(f"P{i}", "x", f"2402.0000{i}", [f"shared keyword {i}"]) for i in range(10)],
    )
    result = rank_papers(conn, cid, query="shared keyword", mode="lexical", top_k=3)
    assert len(result.papers) == 3


def test_paging_slices_the_full_deterministic_order(conn):
    cid = _collection(
        conn,
        [(f"P{i}", "x", f"2402.0000{i}", [f"shared keyword {i}"]) for i in range(10)],
    )
    full = rank_papers(conn, cid, query="shared keyword", mode="lexical", top_k=None)
    page = rank_papers(
        conn, cid, query="shared keyword", mode="lexical", top_k=3, offset=3
    )

    assert full.total == 10
    assert page.total == 10
    assert [paper.paper_id for paper in page.papers] == [
        paper.paper_id for paper in full.papers
    ][3:6]


def test_paging_metadata_order_reports_the_whole_collection(conn):
    cid = _collection(
        conn, [(f"P{i}", "x", f"2402.0000{i}", ["x"]) for i in range(10)]
    )
    page = rank_papers(conn, cid, query=None, sort="influence", top_k=4, offset=8)
    assert page.total == 10
    assert len(page.papers) == 2


def test_unknown_collection_returns_nothing(conn):
    result = rank_papers(conn, "no-such-id", query="attention", top_k=5)
    assert result.papers == []
