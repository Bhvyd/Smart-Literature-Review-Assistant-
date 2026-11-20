
import json

import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from evaluation import harness
from ingest.cache import Cache
from ingest.chunk import Chunk
from ingest.ids import arxiv_pdf_url
from store import repo
from store.db import connect, init_db

TOPICS = {
    "Retrieval Reduces Hallucination": ("2401.00001", "retrieval grounds generation hallucination evidence", [1.0, 0.0]),
    "Dense Passage Search": ("2401.00002", "dense passage search semantic similarity ranking", [0.7, 0.3]),
    "Graph Neural Networks": ("2401.00003", "graph message passing node embeddings", [0.0, 1.0]),
    "Protein Folding Models": ("2401.00004", "protein structure folding prediction", [0.1, 0.9]),
}


@pytest.fixture
def world(tmp_path):
    db = tmp_path / "e.sqlite3"
    init_db(db)
    conn = connect(db)
    cid = repo.create_collection(conn, mode="topic", query="q", embedding_model="m")
    ids = {}
    for title, (arxiv_id, text, vec) in TOPICS.items():
        pid = repo.upsert_paper(conn, collection_id=cid, paper=repo.PaperRow(
            arxiv_id=arxiv_id, openalex_id=None, doi=None, title=title, abstract=text, authors=("A B",),
            year=2024, venue=None, pdf_url=None, citation_count=1, source="arxiv",
            parse_status="ok", parse_note=None, has_fulltext=1))
        ids[title] = pid
        repo.replace_chunks(conn, pid, [Chunk(seq=0, text=text, n_tokens=len(text.split()))])
        (chunk_id,) = conn.execute("SELECT id FROM chunks WHERE paper_id=?", (pid,)).fetchone()
        repo.save_embedding(conn, owner_type="chunk", owner_id=chunk_id, vec=vec, model="m")
    repo.set_collection_status(conn, cid, "ready")
    return conn, cid, ids, tmp_path


def vector_fn(text):
    return [1.0, 0.0] if "retriev" in text.lower() or "dense" in text.lower() else [0.0, 1.0]


def test_title_queries_are_real_titles_and_deterministic(world):
    conn, cid, *_ = world
    first = harness.title_queries(conn, cid, n=4)
    assert first == harness.title_queries(conn, cid, n=4)
    assert set(first) <= {" ".join(t.split()[:8]) for t in TOPICS}


def test_latency_reports_every_mode_with_real_timings(world):
    conn, cid, *_ = world
    result = harness.latency(conn, cid, vector_fn)
    assert result["n_queries"] == 4 and result["n_chunks"] == 4
    for mode in ("lexical", "tfidf", "dense", "hybrid"):
        stats = result["modes"][mode]
        assert stats["n"] == 4 and 0 <= stats["p50_ms"] <= stats["p95_ms"] <= stats["max_ms"]


def test_dense_modes_are_skipped_not_faked_when_no_query_vector_exists(world):
    conn, cid, *_ = world
    result = harness.latency(conn, cid, lambda text: None)
    assert result["modes"]["dense"] == {"n": 0, "skipped_no_query_vector": 4}
    assert result["modes"]["lexical"]["n"] == 4


def test_the_faiss_oracle_agrees_with_the_hand_written_scorer(world):
    pytest.importorskip("faiss")
    conn, cid, *_ = world
    result = harness.faiss_oracle(conn, cid, vector_fn, k=3)
    assert result["identical_top_k_order"] == "4/4"
    assert result["max_abs_score_difference"] < 1e-5


def _pdf_with_references(*cited_titles):
    doc = pymupdf.open()
    for text in ("Body text of the paper. " * 20, "More body. " * 20):
        page = doc.new_page()
        page.insert_text((40, 60), text[:90], fontsize=9)
    last = doc.new_page()
    last.insert_text((40, 60), "References", fontsize=12)
    y = 80
    for i, title in enumerate(cited_titles, start=1):
        last.insert_text((40, y), f"[{i}] A. Author. {title}. arXiv preprint (2024).", fontsize=8)
        y += 14
    return doc.tobytes()


def test_an_edge_is_corroborated_only_when_the_citing_pdf_lists_the_cited_title(world):
    conn, cid, ids, tmp_path = world
    citing = ids["Retrieval Reduces Hallucination"]
    repo.upsert_edge(conn, collection_id=cid, src_paper_id=citing, dst_paper_id=None,
                     dst_openalex_id="W1", dst_title="Dense Passage Retrieval For Open Domain Question Answering", dst_doi=None)
    repo.upsert_edge(conn, collection_id=cid, src_paper_id=citing, dst_paper_id=None,
                     dst_openalex_id="W2", dst_title="A Paper That Is Definitely Not Listed Anywhere", dst_doi=None)
    cache = tmp_path / "cache"
    Cache(cache).put(arxiv_pdf_url("2401.00001"), _pdf_with_references("Dense  Passage-Retrieval for Open-Domain Question Answering"))

    result = harness.graph_corroboration(conn, cid, cache)
    assert result["edges_checked"] == 2 and result["corroborated"] == 1
    assert result["corroboration_rate"] == 0.5
    assert result["unmatched_examples"][0]["cited"] == "A Paper That Is Definitely Not Listed Anywhere"


def test_edges_whose_pdf_is_not_cached_are_skipped_not_counted(world):
    conn, cid, ids, tmp_path = world
    repo.upsert_edge(conn, collection_id=cid, src_paper_id=ids["Graph Neural Networks"],
                     dst_paper_id=None, dst_openalex_id="W1", dst_title="Anything", dst_doi=None)
    result = harness.graph_corroboration(conn, cid, tmp_path / "empty-cache")
    assert result["edges_checked"] == 0 and result["skipped_pdf_not_cached"] == 1
    assert result["corroboration_rate"] is None


def test_the_pool_is_blind_and_unjudged(world):
    conn, cid, *_ = world
    pool = harness.retrieval_pool(conn, cid, ["retrieval hallucination"], vector_fn, per_mode=3)
    entry = pool["queries"][0]
    assert entry["judged"] is False and entry["relevant"] == []
    assert "mode" not in json.dumps(entry["candidates"]) and "score" not in json.dumps(entry["candidates"])
    assert len({c["paper_id"] for c in entry["candidates"]}) == len(entry["candidates"])


def test_metrics_refuse_to_run_without_judgements(world):
    conn, cid, *_ = world
    pool = harness.retrieval_pool(conn, cid, ["retrieval hallucination"], vector_fn)
    with pytest.raises(harness.NoLabels):
        harness.retrieval_metrics(conn, cid, pool, vector_fn)


def test_metrics_are_computed_from_the_judged_queries_only(world):
    conn, cid, ids, _ = world
    labels = {"judged_by": "a person", "queries": [
        {"query": "retrieval hallucination evidence", "judged": True,
         "relevant": [ids["Retrieval Reduces Hallucination"]]},
        {"query": "protein", "judged": False, "relevant": []},
    ]}
    result = harness.retrieval_metrics(conn, cid, labels, vector_fn)
    assert result["n_judged_queries"] == 1 and result["n_unjudged_queries"] == 1
    assert result["modes"]["lexical"]["mrr"] == 1.0
    assert result["modes"]["lexical"]["p@5"] == pytest.approx(0.2)
    assert "pooling bias" in result["caveat"]


def test_tau_is_suggested_from_labelled_questions(world):
    conn, cid, *_ = world
    labelled = [
        {"question": "retrieval hallucination evidence", "answerable": True},
        {"question": "dense passage search ranking", "answerable": True},
        {"question": "quantum chromodynamics lattice", "answerable": False},
    ]
    result = harness.calibrate_tau(conn, cid, labelled, vector_fn, mode="lexical")
    assert result["suggested"]["f1"] == 1.0
    with pytest.raises(harness.NoLabels):
        harness.calibrate_tau(conn, cid, labelled[:2], vector_fn, mode="lexical")


def test_faithfulness_is_sampled_from_stored_answers_and_scored_from_adjudication(world):
    conn, cid, *_ = world
    with pytest.raises(harness.NoLabels):
        harness.faithfulness_pool(conn, cid)
    repo.save_answer(conn, collection_id=cid, question="q", answer="Retrieval helps [1]. Nothing else.",
                     citations=[{"n": 1, "title": "T", "quote": "evidence"}], refused=False,
                     model="m", latency_ms=1, retrieval_snapshot={})
    pool = harness.faithfulness_pool(conn, cid)
    assert [c["claim"] for c in pool["claims"]] == ["Retrieval helps [1]."]
    assert pool["claims"][0]["supported"] is None
    with pytest.raises(harness.NoLabels):
        harness.faithfulness_score(pool)
    pool["claims"][0]["supported"] = False
    assert harness.faithfulness_score(pool)["unsupported_rate"] == 1.0


def test_sections_accumulate_in_one_report(tmp_path):
    harness.save_section(tmp_path, "c1", "latency", {"x": 1})
    report = harness.save_section(tmp_path, "c1", "faiss", {"y": 2})
    assert set(report["sections"]) == {"latency", "faiss"}
    assert harness.load_report(tmp_path, "c1")["sections"]["latency"]["x"] == 1
    assert harness.load_report(tmp_path, "other") is None


def test_the_api_serves_only_stored_measurements(world):
    conn, cid, _, tmp_path = world
    settings = Settings(db_path=tmp_path / "e.sqlite3", data_dir=tmp_path)
    with TestClient(create_app(settings, warm_query_embedder=False)) as client:
        missing = client.get(f"/api/collections/{cid}/evaluation")
        harness.save_section(tmp_path, cid, "latency", {"n_queries": 4})
        found = client.get(f"/api/collections/{cid}/evaluation")
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "no_evaluation"
    assert found.json()["sections"]["latency"]["n_queries"] == 4


def test_unlabelled_questions_are_ignored_not_counted_as_unanswerable(world):
    conn, cid, *_ = world
    labelled = [
        {"question": "retrieval hallucination evidence", "answerable": True},
        {"question": "dense passage search ranking", "answerable": None},
        {"question": "quantum chromodynamics lattice", "answerable": False},
    ]
    result = harness.calibrate_tau(conn, cid, labelled, vector_fn, mode="lexical")
    assert result["n_questions"] == 2


def test_a_generic_short_title_is_skipped_not_matched_against_any_reference(world):
    conn, cid, ids, tmp_path = world
    repo.upsert_edge(conn, collection_id=cid, src_paper_id=ids["Retrieval Reduces Hallucination"],
                     dst_paper_id=None, dst_openalex_id="W1", dst_title="Deep Learning", dst_doi=None)
    cache = tmp_path / "cache"
    Cache(cache).put(arxiv_pdf_url("2401.00001"), _pdf_with_references("Deep Learning for Computer Vision"))
    result = harness.graph_corroboration(conn, cid, cache)
    assert result["edges_checked"] == 0 and result["skipped_title_too_short"] == 1


def test_a_query_judged_to_have_no_relevant_paper_lowers_precision_instead_of_vanishing(world):
    conn, cid, ids, _ = world
    base = {"query": "retrieval hallucination evidence", "judged": True,
            "relevant": [ids["Retrieval Reduces Hallucination"]]}
    only_hit = harness.retrieval_metrics(conn, cid, {"queries": [base]}, vector_fn)
    with_empty = harness.retrieval_metrics(
        conn, cid, {"queries": [base, {"query": "quantum lattice", "judged": True, "relevant": []}]}, vector_fn
    )
    assert with_empty["n_judged_with_no_relevant_paper"] == 1
    assert with_empty["modes"]["lexical"]["mrr"] < only_hit["modes"]["lexical"]["mrr"]
    assert with_empty["queries_per_mode"]["lexical"] == 2
    assert only_hit["modes"]["lexical"]["r@10"] == with_empty["modes"]["lexical"]["r@10"]


def test_asking_the_same_question_twice_does_not_double_count_claims(world):
    conn, cid, *_ = world
    for _ in range(2):
        repo.save_answer(conn, collection_id=cid, question="q", answer="Same claim [1].",
                         citations=[{"n": 1, "title": "T", "quote": "x"}], refused=False, model="m",
                         latency_ms=1, retrieval_snapshot={})
    assert len(harness.faithfulness_pool(conn, cid)["claims"]) == 1


def test_the_faithfulness_pool_carries_the_full_cited_passage_not_just_the_excerpt(world):
    conn, cid, ids, _ = world
    (chunk_id, text) = conn.execute("SELECT id, text FROM chunks LIMIT 1").fetchone()
    repo.save_answer(conn, collection_id=cid, question="q", answer="A claim [1].",
                     citations=[{"n": 1, "title": "T", "quote": text[:5], "chunk_id": chunk_id}],
                     refused=False, model="m", latency_ms=1, retrieval_snapshot={})
    cited = harness.faithfulness_pool(conn, cid)["claims"][0]["cited"][0]
    assert cited["quote"] == text[:5] and cited["passage"] == text
