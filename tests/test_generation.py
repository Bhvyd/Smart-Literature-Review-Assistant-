
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from ingest.chunk import Chunk
from services.answer import ask, citation_coverage, parse_citations
from services.llm import LLMClient, LLMUnavailable
from services.related_work import RelatedWorkError, draft_related_work, unique_keys
from services.summarize import paper_summary, unsupported_numbers
from store.db import connect, init_db
from store.repo import (
    PaperRow,
    create_collection,
    replace_chunks,
    save_embedding,
    set_collection_status,
    upsert_paper,
)


class Provider:
    def __init__(self, reply="ok", status=200, error=None):
        self.reply = reply
        self.status = status
        self.error = error
        self.requests = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append({"body": body, "auth": request.headers.get("authorization")})
        if self.status != 200:
            return httpx.Response(self.status, json={"error": self.error or {}})
        text = self.reply(body) if callable(self.reply) else self.reply
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                "usage": {"total_tokens": 10},
            },
        )

    @property
    def transport(self):
        return httpx.MockTransport(self)


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "gen.sqlite3"
    init_db(path)
    return path


def make_llm(db_path, provider, api_key="sk-test-key", model="gpt-5-nano"):
    return LLMClient(
        api_key=api_key, model=model, db_path=db_path, transport=provider.transport
    )


def test_identical_calls_are_paid_for_once(db_path):
    provider = Provider("hello")
    llm = make_llm(db_path, provider)
    first = llm.complete("sys", "user")
    second = llm.complete("sys", "user")
    assert (first.text, first.cached) == ("hello", False)
    assert (second.text, second.cached) == ("hello", True)
    assert len(provider.requests) == 1


def test_cached_answers_work_with_no_key_and_no_network(db_path):
    make_llm(db_path, Provider("remembered")).complete("sys", "user")
    offline = LLMClient(api_key=None, model="gpt-5-nano", db_path=db_path)
    assert offline.complete("sys", "user").text == "remembered"


def test_no_key_and_no_cache_is_an_explicit_error(db_path):
    llm = LLMClient(api_key=None, model="gpt-5-nano", db_path=db_path)
    with pytest.raises(LLMUnavailable) as caught:
        llm.complete("sys", "user")
    assert caught.value.code == "no_api_key"


def test_an_exhausted_account_is_reported_as_insufficient_quota_and_not_cached(db_path):
    provider = Provider(
        status=429, error={"code": "credit_balance_exhausted", "message": "You have no credits"}
    )
    llm = make_llm(db_path, provider)
    with pytest.raises(LLMUnavailable) as caught:
        llm.complete("sys", "user")
    assert caught.value.code == "insufficient_quota"
    assert llm.last_error["code"] == "insufficient_quota"
    count = connect(db_path).execute("SELECT count(*) FROM llm_cache").fetchone()[0]
    assert count == 0


def test_other_429s_and_bad_keys_have_their_own_codes(db_path):
    with pytest.raises(LLMUnavailable) as limited:
        make_llm(db_path, Provider(status=429, error={"code": "rate_limit_exceeded"})).complete("a", "b")
    assert limited.value.code == "rate_limited"
    with pytest.raises(LLMUnavailable) as rejected:
        make_llm(db_path, Provider(status=401, error={"code": "invalid_api_key"})).complete("a", "b")
    assert rejected.value.code == "invalid_api_key"


def test_an_empty_completion_is_an_error_and_is_not_cached(db_path):
    llm = make_llm(db_path, Provider(""))
    with pytest.raises(LLMUnavailable) as caught:
        llm.complete("sys", "user")
    assert caught.value.code == "empty_response"
    assert connect(db_path).execute("SELECT count(*) FROM llm_cache").fetchone()[0] == 0


def test_gpt5_requests_use_max_completion_tokens_and_minimal_reasoning(db_path):
    provider = Provider("x")
    make_llm(db_path, provider).complete("sys", "user", max_output_tokens=777)
    body = provider.requests[0]["body"]
    assert body["max_completion_tokens"] == 777
    assert body["reasoning_effort"] == "minimal"
    assert "temperature" not in body and "max_tokens" not in body
    assert provider.requests[0]["auth"] == "Bearer sk-test-key"


def test_other_models_do_not_get_a_reasoning_parameter(db_path):
    provider = Provider("x")
    make_llm(db_path, provider, model="gpt-4.1-mini").complete("sys", "user")
    assert "reasoning_effort" not in provider.requests[0]["body"]


def test_the_key_never_reaches_the_cache(db_path):
    make_llm(db_path, Provider("x")).complete("sys", "user")
    rows = connect(db_path).execute("SELECT * FROM llm_cache").fetchall()
    assert "sk-test-key" not in json.dumps([dict(row) for row in rows])


def test_citations_are_validated_against_the_slots_that_exist():
    clean, valid, invalid = parse_citations("A is true [1]. B is true [2, 3]. C [9].", slots=3)
    assert valid == [1, 2, 3]
    assert invalid == [9]
    assert "[9]" not in clean and "[2][3]" in clean


def test_coverage_is_the_share_of_cited_sentences():
    assert citation_coverage("One [1]. Two. Three [2].") == pytest.approx(2 / 3)
    assert citation_coverage("") == 0.0


def _paper(title, arxiv_id, abstract="An abstract about retrieval.", year=2023, authors=("Ada Smith",), fulltext=1):
    return PaperRow(
        arxiv_id=arxiv_id, openalex_id=None, doi=None, title=title, abstract=abstract,
        authors=authors, year=year, venue=None, pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        citation_count=3, source="arxiv", parse_status="ok", parse_note=None, has_fulltext=fulltext,
    )


PAPERS = [
    ("Retrieval Reduces Hallucination", "2401.00001", ["Retrieval grounds generation in evidence and reduces hallucination in language models.", "Experiments show a 12% drop in unsupported claims."], [1.0, 0.0]),
    ("Dense Passage Search", "2401.00002", ["Dense vectors rank passages by semantic similarity for question answering."], [0.8, 0.2]),
    ("Graph Neural Networks", "2401.00003", ["Message passing over graphs learns node embeddings."], [0.0, 1.0]),
]


@pytest.fixture
def collection(db_path):
    conn = connect(db_path)
    cid = create_collection(conn, mode="topic", query="rag", embedding_model="m")
    ids = {}
    for title, arxiv_id, texts, vec in PAPERS:
        pid = upsert_paper(conn, collection_id=cid, paper=_paper(title, arxiv_id))
        ids[arxiv_id] = pid
        replace_chunks(conn, pid, [Chunk(seq=i, text=t, n_tokens=len(t.split())) for i, t in enumerate(texts)])
        for (chunk_id,) in conn.execute("SELECT id FROM chunks WHERE paper_id=? ORDER BY seq", (pid,)):
            save_embedding(conn, owner_type="chunk", owner_id=chunk_id, vec=vec, model="m")
    set_collection_status(conn, cid, "ready")
    return conn, cid, ids


def test_a_cited_answer_resolves_to_real_papers_and_is_persisted(collection, db_path):
    conn, cid, ids = collection
    provider = Provider("Retrieval reduces hallucination [1]. Dense search ranks passages [2][9].")
    result = ask(conn, cid, "does retrieval reduce hallucination", llm=make_llm(db_path, provider),
                 query_vector=[1.0, 0.0])

    assert result["refused"] is False
    assert result["citation_anomaly"] == [9]
    assert "[9]" not in result["answer"]
    cited_titles = {row["title"] for row in result["citations"]}
    assert cited_titles <= {title for title, *_ in PAPERS}
    assert all(row["url"].startswith("https://arxiv.org/abs/") for row in result["citations"])
    assert result["confidence"] == 1.0 and result["confidence_basis"] == "citation_coverage"
    assert result["retrieval"]["tau_calibrated"] is False

    row = conn.execute("SELECT * FROM answers WHERE id=?", (result["answer_id"],)).fetchone()
    snapshot = json.loads(row["retrieval_snapshot"])
    assert snapshot["passages"] and row["refused"] == 0


def test_the_prompt_sent_to_the_model_contains_numbered_real_passages(collection, db_path):
    conn, cid, _ = collection
    provider = Provider("Yes [1].")
    ask(conn, cid, "hallucination", llm=make_llm(db_path, provider), query_vector=[1.0, 0.0])
    user_prompt = provider.requests[0]["body"]["messages"][1]["content"]
    assert "[1] (" in user_prompt and "Question: hallucination" in user_prompt
    assert "Retrieval grounds generation" in user_prompt
    assert "ONLY the numbered" in provider.requests[0]["body"]["messages"][0]["content"]


def test_the_model_can_decline_and_the_refusal_text_is_fixed(collection, db_path):
    conn, cid, _ = collection
    result = ask(conn, cid, "hallucination", llm=make_llm(db_path, Provider("INSUFFICIENT_CONTEXT")),
                 query_vector=[1.0, 0.0])
    assert result["refused"] is True
    assert result["refusal_reason"] == "model_declined"
    assert result["answer"].startswith("The collection does not contain enough information")
    assert result["citations"] == []


def test_an_answer_with_no_valid_citation_is_withheld(collection, db_path):
    conn, cid, _ = collection
    result = ask(conn, cid, "hallucination", llm=make_llm(db_path, Provider("It is true, trust me.")),
                 query_vector=[1.0, 0.0])
    assert result["refused"] is True
    assert result["refusal_reason"] == "answer_had_no_valid_citation"
    assert result["answer"] is None


def test_no_retrieval_hits_refuses_without_calling_the_model(collection, db_path):
    conn, cid, _ = collection
    provider = Provider("should never be asked")
    result = ask(conn, cid, "zzqxv wibble", llm=make_llm(db_path, provider), query_vector=None,
                 mode="lexical")
    assert result["refused"] is True and result["refusal_reason"] == "no_retrieval_hits"
    assert provider.requests == []


def test_a_calibrated_threshold_refuses_weak_retrieval_without_calling_the_model(collection, db_path):
    conn, cid, _ = collection
    provider = Provider("should never be asked")
    result = ask(conn, cid, "hallucination", llm=make_llm(db_path, provider), query_vector=[1.0, 0.0],
                 mode="lexical", refusal_tau=1e9)
    assert result["refusal_reason"] == "below_threshold"
    assert result["retrieval"]["tau_calibrated"] is True
    assert provider.requests == []


def test_when_generation_is_unavailable_the_evidence_is_still_returned(collection, db_path):
    conn, cid, _ = collection
    provider = Provider(status=429, error={"code": "credit_balance_exhausted", "message": "no credits"})
    result = ask(conn, cid, "hallucination", llm=make_llm(db_path, provider), query_vector=[1.0, 0.0])
    assert result["answer"] is None and result["refused"] is False
    assert result["answer_unavailable"]["code"] == "insufficient_quota"
    assert result["citations"], "the passages an answer would have used are still shown"


def test_one_paper_cannot_fill_the_whole_context(collection, db_path):
    conn, cid, _ = collection
    provider = Provider("x [1]")
    result = ask(conn, cid, "retrieval hallucination dense passage graph", llm=make_llm(db_path, provider),
                 query_vector=[1.0, 0.0], mode="lexical")
    per_paper = {}
    for row in result["citations"]:
        per_paper[row["paper_id"]] = per_paper.get(row["paper_id"], 0) + 1
    assert max(per_paper.values()) <= 2


def test_numbers_missing_from_the_source_are_flagged():
    source = "We improve accuracy by 12.5% over 3 baselines."
    assert unsupported_numbers("It gains 12.5% on 3 baselines.", source) == []
    assert unsupported_numbers("It gains 40% on 3 baselines.", source) == ["40"]


def test_summary_uses_a_fixed_template_and_only_stored_text(collection, db_path):
    conn, _, ids = collection
    provider = Provider("Problem: x\nMethod: y\nResult: a 99 percent gain")
    result = paper_summary(conn, ids["2401.00001"], llm=make_llm(db_path, provider))
    prompt = provider.requests[0]["body"]["messages"][1]["content"]
    assert "Title: Retrieval Reduces Hallucination" in prompt
    assert "An abstract about retrieval." in prompt
    assert "Introduction excerpt: Experiments show a 12% drop" in prompt
    assert "Conclusion excerpt" not in prompt
    assert "Graph Neural" not in prompt
    assert result["unsupported_numbers"] == ["99"]
    assert result["ai_generated"] is True


def test_summary_of_an_unknown_paper_is_a_lookup_error(collection, db_path):
    from services.summarize import PaperNotFound

    conn, *_ = collection
    with pytest.raises(PaperNotFound):
        paper_summary(conn, "nope", llm=make_llm(db_path, Provider("x")))


def test_keys_are_unique_when_authors_and_years_clash():
    rows = [{"authors": ["A Smith"], "year": 2023}, {"authors": ["B Smith"], "year": 2023},
            {"authors": ["C Lee"], "year": 2024}]
    assert unique_keys(rows) == ["Smith, 2023a", "Smith, 2023b", "Lee, 2024"]


def test_related_work_is_checked_for_unknown_and_missing_citations(collection, db_path):
    conn, cid, ids = collection
    provider = Provider("Retrieval helps [Smith, 2023a]. Others agree [Nobody, 1999].")
    result = draft_related_work(conn, cid, [ids["2401.00001"], ids["2401.00002"]],
                                llm=make_llm(db_path, provider))
    assert result["ai_assisted_draft"] is True and "verify" in result["label"].lower()
    assert result["unknown_citations"] == ["Nobody, 1999"]
    assert result["uncited_papers"] == ["Smith, 2023b"]
    assert result["word_count"] > 0
    prompt = provider.requests[0]["body"]["messages"][1]["content"]
    assert "[Smith, 2023a]" in prompt and "[Smith, 2023b]" in prompt


def test_related_work_rejects_too_few_or_foreign_papers(collection, db_path):
    conn, cid, ids = collection
    llm = make_llm(db_path, Provider("x"))
    with pytest.raises(RelatedWorkError):
        draft_related_work(conn, cid, [ids["2401.00001"]], llm=llm)
    with pytest.raises(RelatedWorkError):
        draft_related_work(conn, cid, [ids["2401.00001"], "not-a-paper"], llm=llm)


class StubQueryEmbedder:
    def vector(self, text):
        return [1.0, 0.0]


def api(db_path, provider, key="sk-test-key"):
    settings = Settings(db_path=db_path, data_dir=db_path.parent, llm_api_key=key)
    app = create_app(settings, warm_query_embedder=False, llm_transport=provider.transport)
    app.state.query_embedder = StubQueryEmbedder()
    return TestClient(app)


def test_ask_over_http_returns_a_cited_answer(collection, db_path):
    _, cid, _ = collection
    with api(db_path, Provider("Retrieval reduces hallucination [1].")) as client:
        body = client.post(f"/api/collections/{cid}/ask", json={"question": "does retrieval help?"}).json()
    assert body["refused"] is False and body["citations"][0]["n"] == 1


def test_ask_validates_its_input(collection, db_path):
    _, cid, _ = collection
    with api(db_path, Provider("x")) as client:
        assert client.post(f"/api/collections/{cid}/ask", json={"question": "hi"}).status_code == 422
        assert client.post(f"/api/collections/{cid}/ask", json={"question": "valid question", "mode": "zz"}).status_code == 422
        assert client.post("/api/collections/nope/ask", json={"question": "valid question"}).status_code == 404


def test_an_exhausted_account_degrades_ask_to_evidence_and_health_reports_it(collection, db_path):
    _, cid, _ = collection
    provider = Provider(status=429, error={"code": "credit_balance_exhausted", "message": "no credits"})
    with api(db_path, provider) as client:
        body = client.post(f"/api/collections/{cid}/ask", json={"question": "does retrieval help?"}).json()
        health = client.get("/api/health").json()
    assert body["answer"] is None and body["answer_unavailable"]["code"] == "insufficient_quota"
    assert health["generation_last_error"]["code"] == "insufficient_quota"
    assert "sk-test-key" not in json.dumps(health)


def test_summary_endpoint_is_503_with_a_reason_when_generation_fails(collection, db_path):
    _, _, ids = collection
    provider = Provider(status=429, error={"code": "credit_balance_exhausted", "message": "no credits"})
    with api(db_path, provider) as client:
        response = client.post(f"/api/papers/{ids['2401.00001']}/summary")
        missing = client.post("/api/papers/nope/summary")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "generation_unavailable"
    assert response.json()["error"]["detail"]["reason"] == "insufficient_quota"
    assert missing.status_code == 404


def test_summary_endpoint_returns_a_summary(collection, db_path):
    _, _, ids = collection
    with api(db_path, Provider("Problem: p\nMethod: m\nResult: r")) as client:
        body = client.post(f"/api/papers/{ids['2401.00001']}/summary").json()
    assert body["summary"].startswith("Problem:") and body["ai_generated"] is True


def test_related_work_endpoint_validates_and_drafts(collection, db_path):
    _, cid, ids = collection
    reply = "Early work [Smith, 2023a] and later [Smith, 2023b]."
    with api(db_path, Provider(reply)) as client:
        ok = client.post(f"/api/collections/{cid}/related-work",
                         json={"paper_ids": [ids["2401.00001"], ids["2401.00002"]]})
        both = client.post(f"/api/collections/{cid}/related-work",
                           json={"paper_ids": [ids["2401.00001"]], "cluster_id": 0})
        one = client.post(f"/api/collections/{cid}/related-work", json={"paper_ids": [ids["2401.00001"]]})
    assert ok.status_code == 200 and ok.json()["ai_assisted_draft"] is True
    assert both.status_code == 422 and one.status_code == 422


def test_the_reported_passage_count_is_what_actually_fit_the_prompt(collection, db_path, monkeypatch):
    import services.answer as answer_module

    conn, cid, _ = collection
    monkeypatch.setattr(answer_module, "PROMPT_MAX_CHARS", 330)
    provider = Provider("Only one source [1].")
    result = ask(conn, cid, "retrieval hallucination dense graph", llm=make_llm(db_path, provider),
                 query_vector=[1.0, 0.0], mode="lexical")
    assert result["retrieval"]["n_passages"] == len(result["citations"]) <= 2
    snapshot = json.loads(
        conn.execute("SELECT retrieval_snapshot FROM answers WHERE id=?", (result["answer_id"],)).fetchone()[0]
    )
    assert len(snapshot["passages"]) == result["retrieval"]["n_passages"]


def test_an_uploaded_paper_is_cited_with_a_link_a_reader_can_open(db_path):
    conn = connect(db_path)
    cid = create_collection(conn, mode="upload", query="u", embedding_model="m")
    ids = []
    for n, title in enumerate(["Uploaded Paper On Retrieval", "Second Uploaded Paper"]):
        base = _paper(title, f"2401.9999{n}")
        row = PaperRow(**{**base.__dict__, "arxiv_id": None, "source": "upload", "pdf_url": "upload://" + str(n) * 64})
        pid = upsert_paper(conn, collection_id=cid, paper=row)
        ids.append(pid)
        replace_chunks(conn, pid, [Chunk(seq=0, text="retrieval grounds generation", n_tokens=3)])
        (chunk_id,) = conn.execute("SELECT id FROM chunks WHERE paper_id=?", (pid,)).fetchone()
        save_embedding(conn, owner_type="chunk", owner_id=chunk_id, vec=[1.0, 0.0], model="m")
    set_collection_status(conn, cid, "ready")

    answer = ask(conn, cid, "retrieval", llm=make_llm(db_path, Provider("It does [1].")), query_vector=[1.0, 0.0])
    assert answer["citations"][0]["url"] == f"/api/papers/{answer['citations'][0]['paper_id']}/pdf"

    draft = draft_related_work(conn, cid, ids, llm=make_llm(db_path, Provider("Text [Smith, 2023a].")))
    assert {c["url"] for c in draft["citations"]} == {f"/api/papers/{pid}/pdf" for pid in ids}
    assert all("upload://" not in (c["url"] or "") for c in draft["citations"])


def test_quotes_are_cut_at_a_word_boundary_not_mid_word():
    from services.answer import _excerpt

    text = "word " * 200
    cut = _excerpt(text, limit=42)
    assert cut.endswith("…") and "wor…" not in cut and len(cut) <= 43
    assert _excerpt("short") == "short"


def test_ranges_and_semicolon_lists_are_expanded_validated_and_reported():
    clean, valid, invalid = parse_citations("A [1]. B [1-9]. C [2; 8].", slots=3)
    assert valid == [1, 2, 3] and invalid == [4, 5, 6, 7, 8, 9]
    assert "[1-9]" not in clean and "[2; 8]" not in clean and "[2]" in clean


def test_a_bracketed_year_or_figure_reference_is_text_not_a_citation():
    clean, valid, invalid = parse_citations("Introduced in [2023]. See [Fig. 2]. Fact [1].", slots=3)
    assert "[2023]" in clean and "[Fig. 2]" in clean
    assert valid == [1] and invalid == []


def test_the_threshold_is_judged_on_its_own_scale_whatever_mode_retrieved(collection, db_path):
    conn, cid, _ = collection
    provider = Provider("Yes [1].")
    skipped = ask(conn, cid, "retrieval hallucination", llm=make_llm(db_path, provider),
                  query_vector=None, mode="hybrid", refusal_tau=0.99, tau_mode="dense")
    assert skipped["retrieval"]["tau_skipped"] == "query_vector_missing"
    assert skipped["refused"] is False and skipped["retrieval"]["mode"] == "lexical"
    gated = ask(conn, cid, "retrieval hallucination", llm=make_llm(db_path, Provider("never asked")),
                query_vector=[1.0, 1.0], mode="hybrid", refusal_tau=0.99, tau_mode="dense")
    assert gated["refusal_reason"] == "below_threshold"
    assert gated["retrieval"]["tau_score"] == pytest.approx(1.0 / (2 ** 0.5 * (0.64 + 0.04) ** 0.5))


def test_one_paper_cannot_fill_the_context_even_when_every_slot_is_cited(db_path):
    conn = connect(db_path)
    cid = create_collection(conn, mode="topic", query="q", embedding_model="m")
    for title, arxiv_id, texts in [
        ("Long Repetitive Paper", "2401.10001", [f"alpha retrieval chunk number {i}" for i in range(5)]),
        ("Other Paper", "2401.10002", ["alpha retrieval from another source"]),
    ]:
        pid = upsert_paper(conn, collection_id=cid, paper=_paper(title, arxiv_id))
        replace_chunks(conn, pid, [Chunk(seq=i, text=t, n_tokens=4) for i, t in enumerate(texts)])
    set_collection_status(conn, cid, "ready")

    provider = Provider("A [1]. B [2]. C [3]. D [4]. E [5]. F [6].")
    result = ask(conn, cid, "alpha retrieval", llm=make_llm(db_path, provider), query_vector=None,
                 mode="lexical")
    titles = [row["title"] for row in result["citations"]]
    assert titles.count("Long Repetitive Paper") == 2
    assert titles.count("Other Paper") == 1


def test_a_blank_question_is_rejected_by_the_api(collection, db_path):
    _, cid, _ = collection
    with api(db_path, Provider("x")) as client:
        assert client.post(f"/api/collections/{cid}/ask", json={"question": "     "}).status_code == 422


def test_settings_never_print_the_api_key_and_a_bad_tau_fails_loudly(monkeypatch):
    assert "sk-secret" not in repr(Settings(llm_api_key="sk-secret"))
    monkeypatch.setenv("REFUSAL_TAU", "not-a-number")
    with pytest.raises(ValueError, match="REFUSAL_TAU"):
        Settings.from_env(env_path=None)


def test_a_passages_own_reference_numbers_are_neutralised_before_the_model_sees_them():
    from services.answer import neutralise_inline_references

    assert neutralise_inline_references("as shown in [4] and [3, 5-7] and [56].") == (
        "as shown in (ref. 4) and (ref. 3, 5-7) and (ref. 56).")
    assert neutralise_inline_references("a [Fig. 2] panel") == "a [Fig. 2] panel"


def test_the_model_never_sees_bracketed_reference_numbers_but_the_reader_still_gets_the_original_quote(db_path):
    conn = connect(db_path)
    cid = create_collection(conn, mode="topic", query="q", embedding_model="m")
    pid = upsert_paper(conn, collection_id=cid, paper=_paper("Sparse Attention Paper", "2401.77777"))
    text = "Sparse patterns reduce cost, as shown in [4] and [56], for long sequences."
    replace_chunks(conn, pid, [Chunk(seq=0, text=text, n_tokens=12)])
    set_collection_status(conn, cid, "ready")
    provider = Provider("It reduces cost [1].")
    result = ask(conn, cid, "sparse patterns cost", llm=make_llm(db_path, provider), query_vector=None, mode="lexical")
    sent = provider.requests[0]["body"]["messages"][1]["content"]
    assert "(ref. 4)" in sent and "(ref. 56)" in sent and "[4]" not in sent and "[56]" not in sent
    assert "[4] and [56]" in result["citations"][0]["quote"]
    system = provider.requests[0]["body"]["messages"][0]["content"]
    assert "(ref. 56)" in system and "by the title" in system
