
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from services.contradiction import choose_pairs, compare_papers, judge, parse_verdict, split_sentences
from services.llm import LLMClient
from store.db import connect, init_db
from store.repo import PaperRow, create_collection, save_embedding, set_collection_status, upsert_paper

ABSTRACTS = {
    "Sparse Retrieval Wins": (2021, [1.0, 0.0],
        "We study lexical and dense retrievers on five benchmarks. Our results show that BM25 outperforms dense retrieval on out-of-domain queries. "
        "We release all code and data."),
    "Dense Retrieval Wins": (2023, [0.9, 0.1],
        "We revisit retrieval with larger encoders. Our results show that dense retrieval outperforms BM25 on out-of-domain queries. "
        "Training takes two days on one GPU."),
    "Graph Methods": (2022, [0.0, 1.0],
        "We propose a graph neural network for molecule property prediction. The model reaches state of the art accuracy on three datasets."),
}


class Provider:
    def __init__(self, *replies):
        self.replies, self.requests = list(replies), []

    def __call__(self, request):
        self.requests.append(json.loads(request.content))
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, httpx.Response):
            return reply
        return httpx.Response(200, json={"choices": [{"message": {"content": reply}}]})


CONFIRM_CONFLICT = json.dumps({"subject_a": "x", "subject_b": "y", "kind_a": "result", "kind_b": "result", "relation": "conflict"})


@pytest.fixture
def world(tmp_path):
    db = tmp_path / "c.sqlite3"
    init_db(db)
    conn = connect(db)
    cid = create_collection(conn, mode="topic", query="q", embedding_model="m")
    ids = {}
    for n, (title, (year, vec, abstract)) in enumerate(ABSTRACTS.items()):
        ids[title] = upsert_paper(conn, collection_id=cid, paper=PaperRow(
            arxiv_id=f"2401.0000{n}", openalex_id=None, doi=None, title=title, abstract=abstract, authors=("A B",),
            year=year, venue=None, pdf_url=None, citation_count=1, source="arxiv", parse_status="ok",
            parse_note=None, has_fulltext=0))
        save_embedding(conn, owner_type="paper", owner_id=ids[title], vec=vec, model="m")
    set_collection_status(conn, cid, "ready")
    return conn, cid, ids, db


def llm_for(db, provider):
    return LLMClient(api_key="sk-test", model="gpt-5-nano", db_path=db, provider="openai",
                     transport=httpx.MockTransport(provider))


def paper(conn, cid, ids, title):
    from services.retrieval import paper_details
    row = conn.execute("SELECT title, year FROM papers WHERE id=?", (ids[title],)).fetchone()
    d = paper_details(conn, [ids[title]])[ids[title]]
    return {"paper_id": ids[title], "title": row["title"], "year": row["year"], "abstract": d["abstract"], "source_url": d["source_url"]}


def test_sentences_are_split_and_short_fragments_dropped():
    assert split_sentences("We study retrieval in great depth. Ok. Results are strong on three benchmarks.") == [
        "We study retrieval in great depth.", "Results are strong on three benchmarks."]
    assert split_sentences("") == []


def test_replies_are_parsed_from_json_even_with_fences_and_chatter():
    assert parse_verdict('```json\n{"verdict": "agree", "a": 1, "b": 2}\n```')["verdict"] == "agree"
    assert parse_verdict('Here you go: {"verdict": "none", "a": null, "b": null} done')["verdict"] == "none"
    assert parse_verdict("no json at all") is None
    assert parse_verdict("{broken json") is None


def test_the_shown_quotes_are_the_exact_stored_sentences_not_model_text(world):
    conn, cid, ids, db = world
    a, b = paper(conn, cid, ids, "Sparse Retrieval Wins"), paper(conn, cid, ids, "Dense Retrieval Wins")
    reply = json.dumps({"kind_a": "result", "kind_b": "result", "verdict": "conflict", "a": 2, "b": 2, "reason": "They report opposite winners."})
    result = judge(llm_for(db, Provider(reply, CONFIRM_CONFLICT)), a, b)
    assert result["verdict"] == "conflict"
    assert result["quote_a"] in a["abstract"] and result["quote_b"] in b["abstract"]
    assert result["quote_a"] == "Our results show that BM25 outperforms dense retrieval on out-of-domain queries."
    assert result["paper_a"]["title"] == "Sparse Retrieval Wins" and result["paper_b"]["url"]


def test_a_model_that_tries_to_smuggle_in_its_own_quote_changes_nothing(world):
    conn, cid, ids, db = world
    a, b = paper(conn, cid, ids, "Sparse Retrieval Wins"), paper(conn, cid, ids, "Dense Retrieval Wins")
    reply = json.dumps({"kind_a": "result", "kind_b": "result", "verdict": "conflict", "a": 2, "b": 2, "quote_a": "INVENTED TEXT", "quote_b": "ALSO INVENTED"})
    result = judge(llm_for(db, Provider(reply, CONFIRM_CONFLICT)), a, b)
    assert "INVENTED" not in json.dumps(result)


@pytest.mark.parametrize("reply", [
    {"verdict": "conflict", "a": 99, "b": 2},
    {"verdict": "conflict", "a": 2, "b": None},
    {"verdict": "conflict", "a": 0, "b": 1},
    {"verdict": "conflict", "a": "x", "b": 1},
    {"verdict": "maybe", "a": 1, "b": 1},
    {"verdict": "none", "a": None, "b": None},
])
def test_a_verdict_without_two_real_sentences_is_withheld(world, reply):
    conn, cid, ids, db = world
    result = judge(llm_for(db, Provider(json.dumps(reply))), paper(conn, cid, ids, "Sparse Retrieval Wins"),
                   paper(conn, cid, ids, "Dense Retrieval Wins"))
    assert result["verdict"] == "no_comparable_claims" and result["quote_a"] is None and result["quote_b"] is None


def test_non_json_replies_are_withheld_not_guessed(world):
    conn, cid, ids, db = world
    result = judge(llm_for(db, Provider("They clearly disagree.")), paper(conn, cid, ids, "Sparse Retrieval Wins"),
                   paper(conn, cid, ids, "Dense Retrieval Wins"))
    assert result["verdict"] == "no_comparable_claims" and "not valid JSON" in result["note"]


def test_superseded_needs_two_different_known_years(world):
    conn, cid, ids, db = world
    a, b = paper(conn, cid, ids, "Sparse Retrieval Wins"), paper(conn, cid, ids, "Dense Retrieval Wins")
    reply = json.dumps({"kind_a": "result", "kind_b": "result", "verdict": "superseded", "a": 2, "b": 2})
    assert judge(llm_for(db, Provider(reply, CONFIRM_CONFLICT)), a, b)["verdict"] == "superseded"
    assert judge(llm_for(db, Provider(reply, CONFIRM_CONFLICT)), a, {**b, "year": None})["verdict"] == "no_comparable_claims"
    assert judge(llm_for(db, Provider(reply, CONFIRM_CONFLICT)), a, {**b, "year": a["year"]})["verdict"] == "no_comparable_claims"


def test_a_paper_without_an_abstract_cannot_be_compared(world):
    conn, cid, ids, db = world
    a = paper(conn, cid, ids, "Sparse Retrieval Wins")
    result = judge(llm_for(db, Provider("{}")), a, {**a, "abstract": ""})
    assert result["verdict"] == "no_comparable_claims"


def test_pairs_are_chosen_by_similarity_and_capped(world):
    conn, cid, ids, _ = world
    everyone = list(ids.values())
    top = choose_pairs(conn, cid, everyone, 1)
    assert set(top[0]) == {ids["Sparse Retrieval Wins"], ids["Dense Retrieval Wins"]}
    assert len(choose_pairs(conn, cid, everyone, 10)) == 3


def test_the_prompt_contains_only_numbered_stored_sentences(world):
    conn, cid, ids, db = world
    provider = Provider(json.dumps({"verdict": "none", "a": None, "b": None}))
    compare_papers(conn, cid, [ids["Sparse Retrieval Wins"], ids["Dense Retrieval Wins"]], llm=llm_for(db, provider))
    prompt = provider.requests[0]["messages"][1]["content"]
    assert "Our results show that BM25 outperforms dense retrieval on out-of-domain queries." in prompt
    assert "Our results show that dense retrieval outperforms BM25 on out-of-domain queries." in prompt
    assert "A1. " in prompt and "B1. " in prompt and "A3. " in prompt and "B3. " in prompt
    assert "Graph neural network" not in prompt and "molecule" not in prompt


def test_generation_failure_stops_the_run_and_says_why(world):
    conn, cid, ids, db = world
    quota = httpx.Response(429, json={"error": {"code": "insufficient_quota", "message": "empty"}})
    provider = Provider(quota)
    out = compare_papers(conn, cid, list(ids.values()), llm=llm_for(db, provider), max_pairs=3)
    assert out["comparisons"] == [] and out["unavailable"]["code"] == "insufficient_quota"
    assert len(provider.requests) == 1


def test_over_http_the_papers_are_validated_and_results_are_labelled(world, tmp_path):
    conn, cid, ids, db = world
    reply = json.dumps({"kind_a": "result", "kind_b": "result", "verdict": "conflict", "a": 2, "b": 2, "reason": "Opposite winners."})
    settings = Settings(db_path=db, data_dir=tmp_path, llm_api_key="sk-test", llm_provider="openai", llm_model="gpt-5-nano")
    app = create_app(settings, warm_query_embedder=False, llm_transport=httpx.MockTransport(Provider(reply, CONFIRM_CONFLICT)))
    with TestClient(app) as client:
        ok = client.post(f"/api/collections/{cid}/contradictions",
                         json={"paper_ids": [ids["Sparse Retrieval Wins"], ids["Dense Retrieval Wins"]], "max_pairs": 1})
        one = client.post(f"/api/collections/{cid}/contradictions", json={"paper_ids": [ids["Graph Methods"]]})
        both = client.post(f"/api/collections/{cid}/contradictions", json={"paper_ids": [ids["Graph Methods"]], "cluster_id": 0})
        foreign = client.post(f"/api/collections/{cid}/contradictions", json={"paper_ids": [ids["Graph Methods"], "nope"]})
        tiny_map = client.post(f"/api/collections/{cid}/contradictions", json={"cluster_id": 0})
    body = ok.json()
    assert ok.status_code == 200 and body["ai_generated"] is True and body["comparisons"][0]["verdict"] == "conflict"
    assert "verbatim" in body["method"] and "lead to check" in body["note"]
    assert one.status_code == 422 and both.status_code == 422 and foreign.status_code == 422
    assert tiny_map.status_code == 409


def test_the_number_of_possible_pairs_is_reported_separately_from_the_number_requested(world):
    conn, cid, ids, db = world
    provider = Provider(json.dumps({"verdict": "none", "a": None, "b": None}))
    out = compare_papers(conn, cid, list(ids.values()), llm=llm_for(db, provider), max_pairs=10)
    assert out["n_pairs_requested"] == 10 and out["n_pairs_possible"] == 3 and len(out["comparisons"]) == 3


@pytest.mark.parametrize("a, b", [("A2", "B2"), ("a2", "b2"), (" A2 ", "B2."), ("A 2", "B 2")])
def test_sentence_numbers_given_with_their_label_are_accepted(world, a, b):
    conn, cid, ids, db = world
    pa, pb = paper(conn, cid, ids, "Sparse Retrieval Wins"), paper(conn, cid, ids, "Dense Retrieval Wins")
    reply = json.dumps({"kind_a": "result", "kind_b": "result", "verdict": "conflict", "a": a, "b": b, "reason": "Opposite winners."})
    result = judge(llm_for(db, Provider(reply, CONFIRM_CONFLICT)), pa, pb)
    assert result["verdict"] == "conflict"
    assert result["quote_a"] == "Our results show that BM25 outperforms dense retrieval on out-of-domain queries."
    assert result["quote_b"] == "Our results show that dense retrieval outperforms BM25 on out-of-domain queries."


@pytest.mark.parametrize("a, b", [("B2", "B2"), ("A2", "A2"), ("C2", "B2"), ("A", "B2"), ("A2B", "B2")])
def test_a_label_for_the_other_paper_or_garbage_is_still_withheld(world, a, b):
    conn, cid, ids, db = world
    reply = json.dumps({"kind_a": "result", "kind_b": "result", "verdict": "conflict", "a": a, "b": b})
    result = judge(llm_for(db, Provider(reply, CONFIRM_CONFLICT)), paper(conn, cid, ids, "Sparse Retrieval Wins"),
                   paper(conn, cid, ids, "Dense Retrieval Wins"))
    assert result["verdict"] == "no_comparable_claims" and result["quote_a"] is None


CHUNKS = [
    "Sparse Retrieval Wins Some Author University of Somewhere author@example.org",
    "We describe the setup. Our experiments show that BM25 beats the dense retriever by a wide margin on "
    "every out-of-domain test set. Table 3 lists the exact scores for all systems. Smith et al. (2020) showed "
    "that dense retrieval is better in domain. 0.31 0.22 0.19 0.44 0.52 0.12 0.31 0.22 0.19 0.44 0.52 0.12 end.",
    "ould be extended. We conclude that lexical matching remains a strong baseline for new domains today. "
    "The learning rate was set to a fixed value for all of the experiments in this paper",
]


def _with_chunks(conn, ids, title, texts):
    from ingest.chunk import Chunk
    from store.repo import replace_chunks
    replace_chunks(conn, ids[title], [Chunk(seq=i, text=t, n_tokens=len(t.split())) for i, t in enumerate(texts)])


def test_body_sentences_keep_own_findings_and_drop_debris():
    from services.contradiction import body_sentences
    got = body_sentences(CHUNKS[1:])
    assert got == [
        "Our experiments show that BM25 beats the dense retriever by a wide margin on every out-of-domain test set.",
        "We conclude that lexical matching remains a strong baseline for new domains today.",
    ]
    assert body_sentences([]) == [] and body_sentences(["No finding words here at all, just a long plain sentence."]) == []


def test_body_sentences_are_verbatim_and_capped_and_deduplicated():
    from services.contradiction import body_sentences
    sentence = "Our results show that the larger model is better than the smaller one on this task."
    many = [f"{sentence.replace('larger', f'larger{n}')} " for n in range(30)]
    assert len(body_sentences(many, limit=5)) == 5
    assert body_sentences([sentence, sentence]) == [sentence]
    assert all(s in " ".join(many) for s in body_sentences(many))


def test_full_text_sentences_are_offered_numbered_after_the_abstract_and_quoted_verbatim(world):
    conn, cid, ids, db = world
    _with_chunks(conn, ids, "Sparse Retrieval Wins", CHUNKS)
    a, b = paper_with_body(conn, cid, ids, "Sparse Retrieval Wins"), paper_with_body(conn, cid, ids, "Dense Retrieval Wins")
    assert a["body"] and b["body"] == []
    provider = Provider(json.dumps({"kind_a": "result", "kind_b": "result", "verdict": "conflict", "a": "A4", "b": "B2", "reason": "Opposite winners."}), CONFIRM_CONFLICT)
    result = judge(llm_for(db, provider), a, b)
    prompt = provider.requests[0]["messages"][1]["content"]
    assert "A4. Our experiments show that BM25 beats the dense retriever" in prompt
    assert "title page" not in prompt and "author@example.org" not in prompt and "Smith et al" not in prompt
    assert result["verdict"] == "conflict" and result["source_a"] == "full text" and result["source_b"] == "abstract"
    assert result["quote_a"] == a["body"][0]
    stored = " ".join(" ".join(t.split()) for t in CHUNKS)
    assert result["quote_a"] in stored


def test_a_paper_whose_abstract_is_missing_can_still_be_compared_through_its_full_text(world):
    conn, cid, ids, db = world
    _with_chunks(conn, ids, "Sparse Retrieval Wins", CHUNKS)
    a = paper_with_body(conn, cid, ids, "Sparse Retrieval Wins")
    result = judge(llm_for(db, Provider(json.dumps({"verdict": "none"}))), {**a, "abstract": ""},
                   paper_with_body(conn, cid, ids, "Dense Retrieval Wins"))
    assert result["verdict"] == "no_comparable_claims" and "usable sentences" not in (result.get("note") or "")


def paper_with_body(conn, cid, ids, title):
    from services.contradiction import load_papers
    return load_papers(conn, cid, [ids[title]])[ids[title]]


def test_load_papers_rejects_papers_from_another_collection(world):
    from services.contradiction import ContradictionError, load_papers
    conn, cid, ids, _ = world
    with pytest.raises(ContradictionError):
        load_papers(conn, cid, [ids["Graph Methods"], "nope"])


def _judge_with_confirmation(world, verdict, relation, years=(None, None)):
    conn, cid, ids, db = world
    a, b = paper(conn, cid, ids, "Sparse Retrieval Wins"), paper(conn, cid, ids, "Dense Retrieval Wins")
    first = json.dumps({"kind_a": "result", "kind_b": "result", "verdict": verdict, "a": 2, "b": 2, "reason": "r"})
    second = json.dumps({"subject_a": "x", "subject_b": "y", "kind_a": "result", "kind_b": "result", "relation": relation}) if relation else "not json"
    provider = Provider(first, second)
    return judge(llm_for(db, provider), a, b), provider


@pytest.mark.parametrize("verdict, relation", [("conflict", "conflict"), ("agree", "agree"), ("superseded", "conflict")])
def test_a_confirmed_verdict_is_shown(world, verdict, relation):
    result, provider = _judge_with_confirmation(world, verdict, relation)
    assert result["verdict"] == verdict and result["quote_a"] and result["quote_b"]
    assert len(provider.requests) == 2


@pytest.mark.parametrize("verdict, relation", [
    ("conflict", "compatible"),
    ("conflict", "unrelated"),
    ("conflict", "agree"),
    ("agree", "compatible"),
    ("agree", "conflict"),
    ("superseded", "agree"),
    ("conflict", None),
])
def test_a_verdict_the_second_reading_does_not_confirm_is_withheld_with_the_reason(world, verdict, relation):
    result, _ = _judge_with_confirmation(world, verdict, relation)
    assert result["verdict"] == "no_comparable_claims" and result["quote_a"] is None and result["quote_b"] is None
    assert "second reading" in result["note"] and (relation or "unreadable") in result["note"]


def test_the_second_reading_sees_only_the_two_quotes_not_the_proposed_verdict(world):
    _, provider = _judge_with_confirmation(world, "conflict", "conflict")
    second = provider.requests[1]["messages"][1]["content"]
    assert second == ("A: Our results show that BM25 outperforms dense retrieval on out-of-domain queries.\n"
                      "B: Our results show that dense retrieval outperforms BM25 on out-of-domain queries.") or \
           second == ("A: Our results show that dense retrieval outperforms BM25 on out-of-domain queries.\n"
                      "B: Our results show that BM25 outperforms dense retrieval on out-of-domain queries.")
    assert "conflict" not in second.lower() and "reason" not in second.lower()


def test_no_second_call_is_spent_when_the_first_found_nothing(world):
    result, provider = _judge_with_confirmation(world, "none", "conflict")
    assert result["verdict"] == "no_comparable_claims" and len(provider.requests) == 1


from services.contradiction import CONFIRM_PROMPT, CONFIRM_TOKENS, JUDGE_TOKENS, MAX_PAIRS, SYSTEM_PROMPT, body_sentences


def _verdict_reply(**overrides):
    return json.dumps({"kind_a": "result", "kind_b": "result", "verdict": "conflict", "a": 2, "b": 2, "reason": "r",
                       **overrides})


def _confirm_reply(**overrides):
    return json.dumps({"subject_a": "x", "subject_b": "y", "kind_a": "result", "kind_b": "result",
                       "relation": "conflict", **overrides})


def _judge(world, *replies):
    conn, cid, ids, db = world
    provider = Provider(*replies)
    result = judge(llm_for(db, provider), paper(conn, cid, ids, "Sparse Retrieval Wins"),
                   paper(conn, cid, ids, "Dense Retrieval Wins"))
    return result, provider


@pytest.mark.parametrize("a, b", [("A0", 2), (0, 2), ("A4", 2), (4, 2), (2, "B0"), (2, 0), (2, "B4"), (2, 4)])
def test_sentence_numbers_outside_the_shown_range_are_withheld(world, a, b):
    result, provider = _judge(world, _verdict_reply(a=a, b=b), _confirm_reply())
    assert result["verdict"] == "no_comparable_claims" and "no valid sentence" in result["note"]
    assert len(provider.requests) == 1


def test_the_last_numbered_sentence_is_accepted(world):
    result, _ = _judge(world, _verdict_reply(a="A3", b="B3"), _confirm_reply())
    assert result["verdict"] == "conflict" and result["quote_a"] == "We release all code and data."


@pytest.mark.parametrize("value", ["٢", "A٢", "２"])
def test_only_ascii_digits_are_sentence_numbers(world, value):
    result, _ = _judge(world, _verdict_reply(a=value), _confirm_reply())
    assert result["verdict"] == "no_comparable_claims" and result["quote_a"] is None


def test_a_null_verdict_is_reported_as_no_verdict_not_as_a_search_that_found_nothing(world):
    result, _ = _judge(world, json.dumps({"verdict": None, "a": None, "b": None}))
    assert result["verdict"] == "no_comparable_claims" and result["note"] == "the model gave no verdict"


@pytest.mark.parametrize("kinds", [
    {"kind_a": "background"}, {"kind_b": "background"}, {"kind_a": "opinion"}, {"kind_b": "opinion"},
    {"kind_a": None}, {"kind_b": 3}, {"kind_a": ""},
])
def test_a_verdict_resting_on_a_non_result_sentence_is_withheld_before_any_confirmation(world, kinds):
    result, provider = _judge(world, _verdict_reply(**kinds), _confirm_reply())
    assert result["verdict"] == "no_comparable_claims" and "not a finding" in result["note"]
    assert result["quote_a"] is None and len(provider.requests) == 1


def test_missing_kind_fields_withhold_the_verdict(world):
    reply = json.dumps({"verdict": "conflict", "a": 2, "b": 2})
    assert _judge(world, reply, _confirm_reply())[0]["verdict"] == "no_comparable_claims"


def test_labels_are_read_case_insensitively(world):
    result, _ = _judge(world, _verdict_reply(kind_a=" RESULT ", kind_b="Result", verdict="CONFLICT"),
                       _confirm_reply(relation=" CONFLICT ", kind_a="Result", kind_b=" RESULT"))
    assert result["verdict"] == "conflict"


@pytest.mark.parametrize("kinds", [{"kind_a": "background"}, {"kind_b": "background"}, {"kind_a": None}, {"kind_b": None}])
def test_the_second_reading_must_also_find_a_result_in_both_sentences(world, kinds):
    result, provider = _judge(world, _verdict_reply(), _confirm_reply(**kinds))
    assert result["verdict"] == "no_comparable_claims" and "did not find a finding" in result["note"]
    assert result["quote_a"] is None and len(provider.requests) == 2


def test_the_second_reading_gets_quote_a_then_quote_b_and_its_own_token_budget(world):
    result, provider = _judge(world, _verdict_reply(a=2, b=3), _confirm_reply())
    assert provider.requests[1]["messages"][1]["content"] == (
        "A: Our results show that BM25 outperforms dense retrieval on out-of-domain queries.\n"
        "B: Training takes two days on one GPU.")
    assert provider.requests[0]["max_completion_tokens"] == JUDGE_TOKENS >= 800
    assert provider.requests[1]["max_completion_tokens"] == CONFIRM_TOKENS >= 500
    assert provider.requests[1]["messages"][0]["content"] == CONFIRM_PROMPT
    assert result["verdict"] == "conflict"


def test_both_prompts_end_with_the_untrusted_text_rule_and_ask_for_the_kind_of_sentence():
    from services.llm import UNTRUSTED_TEXT_RULE
    assert SYSTEM_PROMPT.endswith(UNTRUSTED_TEXT_RULE) and CONFIRM_PROMPT.endswith(UNTRUSTED_TEXT_RULE)
    assert '"kind_a"' in SYSTEM_PROMPT and '"kind_a"' in CONFIRM_PROMPT


def test_the_b_side_source_label_follows_the_b_sentence(world):
    conn, cid, ids, db = world
    _with_chunks(conn, ids, "Dense Retrieval Wins", CHUNKS)
    a, b = paper_with_body(conn, cid, ids, "Sparse Retrieval Wins"), paper_with_body(conn, cid, ids, "Dense Retrieval Wins")
    assert b["body"]
    result = judge(llm_for(db, Provider(_verdict_reply(a=2, b=4), _confirm_reply())), a, b)
    assert (result["source_a"], result["source_b"]) == ("abstract", "full text")
    assert result["quote_b"] == b["body"][0]


def test_the_title_page_chunk_is_never_a_source_of_sentences(world):
    conn, cid, ids, _ = world
    finding = "Our results show that the proposed method is better than every baseline on all benchmarks."
    _with_chunks(conn, ids, "Sparse Retrieval Wins", [finding + " Some Author University", "Plain text without any claim at all."])
    assert paper_with_body(conn, cid, ids, "Sparse Retrieval Wins")["body"] == []
    _with_chunks(conn, ids, "Sparse Retrieval Wins", ["Title page", finding])
    assert paper_with_body(conn, cid, ids, "Sparse Retrieval Wins")["body"] == [finding]


def _body(*sentences, limit=8):
    return body_sentences([" ".join(sentences)], limit=limit)


GOOD = "Our results show that the proposed method is better than the baseline {} on every single benchmark we tried."
NOT_OWN = ["Smith et al", "[12]", "[3, 4]", "[C,H,J,K,L]", "[C]", "(Smith, 2020)", "(Smith Jones, 2020a)", "arXiv", "Proceedings",
           "pp. 12", "doi", "https://example.org", "a@b.org", "Table 3", "Tables 3", "Table V", "Fig. 3", "Figure 4",
           "Figs 2", "Figure IV"]
DEBRIS = ["thank", "grateful", "supported by", "grant", "acknowledgements", "funded", "funding", "fellowship", "scholarship",
          "sponsored", "Springer Nature", "LATEX template", "IEEE TRANSACTIONS", "published as a conference", "under review",
          "preprint submitted", "©", "copyright", "vol. 12", "NEURAL NETWORKS AND LEARNING"]


def test_the_control_sentence_is_kept():
    sentence = " ".join(GOOD.format("").split())
    assert _body(GOOD.format("")) == [sentence]


@pytest.mark.parametrize("marker", NOT_OWN + DEBRIS)
def test_sentences_about_other_work_or_made_of_debris_are_dropped(marker):
    assert _body(GOOD.format(marker)) == []


@pytest.mark.parametrize("sentence", [
    "Our results show that the model is better on 1234567 of 9876543 cases overall in total.",
    "Our results show that the model is ======= ----- ///// ##### &&&&& ***** $$$$$ %%%%% ~~~~~ ^^^^^ ;;;;; ::::: +++++ done.",
    "Our results show that the proposed method is better than the baseline on every benchmark we tried",
    "our results show that the proposed method is better than the baseline on every benchmark we tried.",
    "The learning rate was set to a fixed value for all of the experiments in this paper and no more.",
    "Our results show that the method is better. it also wins on every benchmark we tried here today.",
])
def test_each_body_filter_drops_its_own_kind_of_sentence(sentence):
    assert body_sentences([sentence]) == []


def test_the_digit_share_boundary_is_ten_percent():
    ok = "Our results show that the proposed method is better than the baseline on all 12 benchmarks we tried."
    assert _body(ok) == [ok]
    exactly_ten = _of_length(100)[:-12] + "1" * 10 + "a."
    assert len(exactly_ten) == 100 and sum(c.isdigit() for c in exactly_ten) == 10 and _body(exactly_ten) == [exactly_ten]
    assert _body(exactly_ten.replace("a.", "1.")) == []


def test_the_letter_share_boundary_is_three_quarters_of_the_non_space_characters():
    base = "Our results show that the model is better than the baseline"

    def sentence(n):
        return base + " " + "=" * n + " ok."

    def share(text):
        return sum(c.isalpha() for c in text) / len(text.replace(" ", ""))

    shares = {n: share(sentence(n)) for n in range(0, 60)}
    last_kept = max(n for n, v in shares.items() if v >= 0.75)
    first_dropped = last_kept + 1
    assert 0.72 < shares[first_dropped] < 0.75 and 0.75 <= shares[last_kept] < 0.78
    assert _body(sentence(last_kept)) == [sentence(last_kept)]
    assert _body(sentence(first_dropped)) == []


def _of_length(n):
    head = "Our results show that the new method is better than "
    return head + "a" * (n - len(head) - 1) + "."


def test_length_limits_are_inclusive_at_60_and_400_characters():
    assert [len(s) for s in _body(_of_length(400))] == [400] and _body(_of_length(401)) == []
    head = "We show that it is better "
    sixty = head + "x" * (59 - len(head)) + "."
    assert len(sixty) == 59 + 1 == 60 and _body(sixty) == [sixty]
    assert _body(sixty[:-2] + ".") == []


def test_a_sentence_with_an_abbreviation_inside_is_kept():
    kept = "Our results show that the method is better, e.g. it wins on every benchmark we tried."
    assert _body(kept) == [kept]


@pytest.mark.parametrize("text", [
    "Our results show that the model is better than the GNN of Ref. 5 on every benchmark we tried here.",
    "Our results show that BM25 is better vs. dense retrieval on every benchmark we tried here.",
    "Our results show that, i.e. in practice, the model is better than the baseline on every benchmark.",
])
def test_abbreviations_do_not_end_a_sentence(text):
    tail = "We conclude that this holds for all tasks we tried in our study."
    assert _body(text + " " + tail) == [text, tail]


def test_a_numbered_list_marker_does_not_cut_a_sentence():
    pieces = split_sentences("We list three findings in order of importance 1. Retrieval improves accuracy on all tasks. Next sentence follows here.")
    assert pieces == ["We list three findings in order of importance 1. Retrieval improves accuracy on all tasks.",
                      "Next sentence follows here."]


def test_sentences_about_earlier_work_need_a_first_person_marker():
    about_others = "Existing methods show that retrieval is better than generation alone on many benchmarks."
    assert _body(about_others) == []
    for own in ("We show that, unlike existing methods, retrieval is better than generation alone on benchmarks.",
                "This paper shows that, unlike existing methods, retrieval is better than generation alone on benchmarks.",
                "Our analysis shows that previous methods are worse than retrieval alone on many benchmarks."):
        assert _body(own) == [own]


def test_own_work_sentences_outrank_equally_cued_ones_and_later_text_wins_ties():
    plain_one = "The proposed method is better than the baseline on every single benchmark tried here."
    plain_two = "The proposed method is better than the baseline on every single dataset tried here."
    own = "We saw the proposed method better than the baseline on every single benchmark tried here."
    assert _body(plain_one, plain_two, limit=1) == [plain_two]
    assert _body(own, plain_one, limit=1) == [own]
    assert _body(plain_one, own, plain_two, limit=2) == [own, plain_two]


def test_more_cues_outrank_fewer():
    one = "The proposed method is better than the baseline on every single benchmark tried here."
    two = "The proposed method clearly outperforms the baseline and improves results on every benchmark tried."
    assert _body(two, one, limit=1) == [two] and _body(one, two, limit=1) == [two]


def test_the_number_of_pairs_is_capped(world):
    conn, cid, ids, db = world
    for n in range(3, 6):
        pid = upsert_paper(conn, collection_id=cid, paper=PaperRow(
            arxiv_id=f"2401.1000{n}", openalex_id=None, doi=None, title=f"Extra {n}", abstract="We study something quite specific here.",
            authors=("A B",), year=2022, venue=None, pdf_url=None, citation_count=1, source="arxiv", parse_status="ok",
            parse_note=None, has_fulltext=0))
        save_embedding(conn, owner_type="paper", owner_id=pid, vec=[0.5, 0.5], model="m")
        ids[f"Extra {n}"] = pid
    provider = Provider(json.dumps({"verdict": "none"}))
    out = compare_papers(conn, cid, list(ids.values()), llm=llm_for(db, provider), max_pairs=50)
    assert MAX_PAIRS == 10 and out["n_pairs_requested"] == 10 and len(out["comparisons"]) == 10
    assert out["n_pairs_possible"] == 15 and len(provider.requests) == 10
    assert compare_papers(conn, cid, list(ids.values()), llm=llm_for(db, provider), max_pairs=0)["n_pairs_requested"] == 1


def test_when_the_provider_gives_out_the_pair_that_was_being_compared_is_named(world):
    conn, cid, ids, db = world
    quota = httpx.Response(429, json={"error": {"code": "insufficient_quota", "message": "empty"}})
    pair = [ids["Sparse Retrieval Wins"], ids["Dense Retrieval Wins"]]
    first_call = compare_papers(conn, cid, pair, llm=llm_for(db, Provider(quota)), max_pairs=1)
    assert first_call["comparisons"] == [] and first_call["unavailable"]["code"] == "insufficient_quota"
    titles = {first_call["incomplete_pair"]["paper_a"]["title"], first_call["incomplete_pair"]["paper_b"]["title"]}
    assert titles == {"Sparse Retrieval Wins", "Dense Retrieval Wins"}
    assert all(f'"{t}"' in first_call["note"] for t in titles) and "no result" in first_call["note"]
    mid = compare_papers(conn, cid, pair, llm=llm_for(db, Provider(_verdict_reply(), quota)), max_pairs=1)
    assert mid["comparisons"] == [] and mid["incomplete_pair"]["paper_a"]["title"] in mid["note"]
    fine = compare_papers(conn, cid, pair, llm=llm_for(db, Provider(json.dumps({"verdict": "none"}))), max_pairs=1)
    assert fine["incomplete_pair"] is None and "stopped" not in fine["note"]
