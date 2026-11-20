
import json

import pytest

import cli
from evaluation import harness
from evaluation.metrics import paired_bootstrap
from ingest.chunk import Chunk
from services.llm import LLMResult, LLMUnavailable
from store import repo
from store.db import connect, init_db

TOPICS = {
    "Retrieval Reduces Hallucination": ("2401.00001", "retrieval grounds generation hallucination evidence", [1.0, 0.0]),
    "Dense Passage Search": ("2401.00002", "dense passage search semantic similarity ranking", [0.7, 0.3]),
    "Graph Neural Networks": ("2401.00003", "graph message passing node embeddings", [0.0, 1.0]),
    "Protein Folding Models": ("2401.00004", "protein structure folding prediction", [0.1, 0.9]),
}
USAGE = {"promptTokenCount": 1000, "candidatesTokenCount": 100, "thoughtsTokenCount": 20, "latency_ms": 500}


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
        ids[arxiv_id] = pid
        repo.replace_chunks(conn, pid, [Chunk(seq=0, text=text, n_tokens=len(text.split()))])
        (chunk_id,) = conn.execute("SELECT id FROM chunks WHERE paper_id=?", (pid,)).fetchone()
        repo.save_embedding(conn, owner_type="chunk", owner_id=chunk_id, vec=vec, model="m")
    repo.set_collection_status(conn, cid, "ready")
    return conn, cid, ids, db, tmp_path


def vector_fn(text):
    return [1.0, 0.0] if "retriev" in text.lower() or "dense" in text.lower() else [0.0, 1.0]


class FakeLLM:
    provider = "gemini"
    model = "gemini-3.1-flash-lite"

    def __init__(self, usage=USAGE, cached=False):
        self.usage, self.cached, self.calls = usage, cached, 0

    def complete(self, system, user, *, max_output_tokens=1500):
        self.calls += 1
        if "Abstract:" in user:
            return LLMResult("Problem: retrieval.\nMethod: search.\nResult: better.", self.model, self.cached, self.usage)
        if "Protein" in user.split("Question:")[-1]:
            return LLMResult("INSUFFICIENT_CONTEXT", self.model, self.cached, self.usage)
        return LLMResult("Retrieval grounds generation [1].", self.model, self.cached, self.usage)


class DownLLM:
    provider, model = "gemini", "m"

    def complete(self, *a, **k):
        raise LLMUnavailable("insufficient_quota", "out of quota")


QUESTIONS = [
    {"question": "How does retrieval reduce hallucination?", "answerable": True},
    {"question": "How does dense passage search rank?", "answerable": True},
    {"question": "How does Protein folding work?", "answerable": False},
    {"question": "unlabelled one", "answerable": None},
]


def test_paired_bootstrap_is_deterministic_and_exact_on_a_constant_difference():
    a, b = [0.9, 0.8, 0.7, 0.6], [0.4, 0.3, 0.2, 0.1]
    first = paired_bootstrap(a, b, resamples=200, seed=1)
    assert first == paired_bootstrap(a, b, resamples=200, seed=1)
    assert first["mean_diff"] == 0.5 and first["ci95"] == [0.5, 0.5] and first["excludes_zero"] is True
    same = paired_bootstrap(a, a, resamples=200, seed=1)
    assert same["ci95"] == [0.0, 0.0] and same["excludes_zero"] is False


def test_retrieval_bootstrap_uses_only_queries_with_a_relevant_paper(world):
    conn, cid, ids, *_ = world
    labels = {"queries": [
        {"query": "retrieval hallucination", "judged": True, "relevant": ["2401.00001"]},
        {"query": "dense passage search", "judged": True, "relevant": ["2401.00002"]},
        {"query": "nothing relevant", "judged": True, "relevant": []},
        {"query": "unjudged", "judged": False, "relevant": ["2401.00003"]},
    ]}
    out = harness.retrieval_bootstrap(conn, cid, labels, vector_fn, resamples=100)
    assert out["n_queries_with_a_relevant_paper"] == 2
    assert out["resamples"] == 100 and out["seed"] == 0
    assert set(out["comparisons"]) == {f"{p} ({m})" for p in ("hybrid - dense", "hybrid - lexical", "lexical - dense")
                                       for m in ("ndcg", "mrr")}
    with pytest.raises(harness.NoLabels):
        harness.retrieval_bootstrap(conn, cid, {"queries": []}, vector_fn)


def fake_bert(candidates, references, **kw):
    n = len(candidates)
    return {"raw_f1": [0.9] * n, "rescaled_f1": [0.5] * n, "baseline_f1": 0.8, "model_type": "fake-bert",
            "num_layers": 3, "versions": {"bert_score": "x"}}


def test_summaries_report_rouge_and_bertscore_against_the_stored_abstract(world):
    conn, cid, *_ = world
    out = harness.summary_scores(conn, cid, FakeLLM(), n=3, bert_fn=fake_bert)
    assert out["n"] == 3 and out["n_requested"] == 3
    assert 0 <= out["rougeL_f1_mean"] <= 1
    assert out["bertscore"]["f1_rescaled_mean"] == 0.5 and out["bertscore"]["f1_raw_mean"] == 0.9
    assert out["bertscore"]["model_type"] == "fake-bert" and out["bertscore"]["baseline_f1"] == 0.8
    assert "floor" in out["caveat"] and "abstract is part of" in out["caveat"]
    assert [p["paper_id"] for p in out["per_paper"]] == sorted(p["paper_id"] for p in out["per_paper"])
    assert out["versions"]["rouge-score"]
    again = harness.summary_scores(conn, cid, FakeLLM(), n=3, bert_fn=fake_bert)
    assert [p["paper_id"] for p in again["per_paper"]] == [p["paper_id"] for p in out["per_paper"]]


def test_summaries_stop_on_a_provider_failure_instead_of_scoring_nothing(world):
    conn, cid, *_ = world
    out = harness.summary_scores(conn, cid, DownLLM(), n=3, bert_fn=fake_bert)
    assert out["n"] == 0 and out["stopped_early"] == "insufficient_quota"
    assert out["rougeL_f1_mean"] is None and out["bertscore"] is None


def test_refusal_end_to_end_counts_against_the_labels(world):
    conn, cid, *_ = world
    out = harness.refusal_end_to_end(conn, cid, {"judged_by": "x", "questions": QUESTIONS}, FakeLLM(), vector_fn)
    assert out["n_questions"] == 3 and out["n_skipped_unlabelled"] == 1
    assert out["unanswerable_refused"] == "1/1" and out["answerable_wrongly_refused"] == "0/2"
    assert out["accuracy_as_labelled"] == 1.0
    assert out["mean_citation_coverage_when_answered"] == 1.0
    assert out["unanswerable_that_were_answered"] == []
    assert out["refusal_reasons"] == {"model_declined": 1}
    assert out["judged_by"] == "x" and out["model"] == "gemini-3.1-flash-lite"
    row = out["per_question"][0]
    assert set(row) >= {"question", "answerable", "refused", "citation_coverage", "latency_ms", "cached"}
    assert "would_falsely_refuse_answerable" in out["score_threshold_alone"]


def test_a_provider_outage_is_unavailable_not_refused(world):
    conn, cid, *_ = world
    out = harness.refusal_end_to_end(conn, cid, {"questions": QUESTIONS}, DownLLM(), vector_fn)
    assert out["n_unavailable"] == 1 and out["stopped_early"] == "insufficient_quota"
    assert out["accuracy_as_labelled"] is None
    assert out["unanswerable_refused"] == "0/0" and out["answerable_wrongly_refused"] == "0/0"


def test_a_missed_unanswerable_question_is_listed(world):
    conn, cid, *_ = world
    qs = [{"question": "How does retrieval reduce hallucination?", "answerable": False}]
    out = harness.refusal_end_to_end(conn, cid, {"questions": qs}, FakeLLM(), vector_fn)
    assert out["unanswerable_refused"] == "0/1"
    assert out["unanswerable_that_were_answered"] == [qs[0]["question"]]


def test_cost_counts_provider_tokens_and_prices_only_a_known_model(world):
    conn, cid, *_ = world
    out = harness.cost_per_query(conn, cid, {"questions": QUESTIONS}, FakeLLM(), vector_fn, n_answers=2, n_summaries=2)
    answered = out["per_answered_question"]
    assert answered["calls"] == 2 and answered["prompt_tokens_mean"] == 1000
    assert answered["output_tokens_mean"] == 100 and answered["thinking_tokens_mean"] == 20
    assert out["per_summary"]["calls"] == 2
    price = out["price_used_usd_per_million_tokens"]
    assert price["input"] == 0.25 and price["output_including_thinking"] == 1.5 and price["source"]
    assert out["usd_per_answered_question_if_billed"] == round((1000 * 0.25 + 120 * 1.5) / 1e6, 6)


def test_cost_without_a_confirmed_price_reports_tokens_only(world):
    conn, cid, *_ = world
    llm = FakeLLM()
    llm.model = "some-unpriced-model"
    out = harness.cost_per_query(conn, cid, {"questions": QUESTIONS}, llm, vector_fn, n_answers=1, n_summaries=1)
    assert out["price_used_usd_per_million_tokens"] is None
    assert "usd_per_answered_question_if_billed" not in out
    assert out["per_answered_question"]["prompt_tokens_total"] == 1000


def test_cached_and_usage_less_calls_are_excluded_from_cost(world):
    conn, cid, *_ = world
    cached = harness.cost_per_query(conn, cid, {"questions": QUESTIONS}, FakeLLM(cached=True), vector_fn,
                                    n_answers=2, n_summaries=2)
    assert cached["per_answered_question"]["calls"] == 0 and cached["excluded_cached_calls"] == 4
    nousage = harness.cost_per_query(conn, cid, {"questions": QUESTIONS}, FakeLLM(usage=None), vector_fn,
                                     n_answers=2, n_summaries=2)
    assert nousage["per_summary"]["calls"] == 0 and nousage["excluded_calls_without_usage"] == 4


def test_openai_usage_is_normalised():
    got = harness.normalise_usage({"prompt_tokens": 50, "completion_tokens": 30,
                                   "completion_tokens_details": {"reasoning_tokens": 10}})
    assert got == {"prompt": 50, "output": 20, "thinking": 10, "latency_ms": None}


@pytest.fixture
def cli_env(world, monkeypatch):
    conn, cid, ids, db, tmp = world
    conn.close()
    monkeypatch.setenv("DATA_DIR", str(tmp / "data"))
    monkeypatch.setenv("DB_PATH", str(db))
    monkeypatch.setattr("services.query_embedder.QueryEmbedder.vector", lambda self, text: vector_fn(text))
    monkeypatch.setattr(cli, "_make_llm", lambda settings, db_path, **kw: FakeLLM())
    monkeypatch.setattr(harness, "bertscore_f1", fake_bert)
    labels = tmp / "q.json"
    labels.write_text(json.dumps({"judged_by": "t", "questions": QUESTIONS}), encoding="utf-8")
    rel = tmp / "rel.json"
    rel.write_text(json.dumps({"queries": [
        {"query": "retrieval hallucination", "judged": True, "relevant": ["2401.00001"]},
        {"query": "dense passage search", "judged": True, "relevant": ["2401.00002"]}]}), encoding="utf-8")
    return cid, tmp, labels, rel


def test_cli_writes_the_same_section_names_as_the_stored_report(cli_env):
    cid, tmp, labels, rel = cli_env
    base = ["--db", str(tmp / "e.sqlite3"), "eval"]
    for argv in (["refusal", "--labels", str(labels)], ["summaries", "--n", "2"],
                 ["cost", "--labels", str(labels), "--n", "1"], ["bootstrap", "--labels", str(rel)]):
        assert cli.main(base + [argv[0], "--collection", cid] + argv[1:]) == 0
    sections = harness.load_report(tmp / "data", cid)["sections"]
    assert {"refusal_end_to_end", "summary_rouge", "cost_per_query", "retrieval_bootstrap"} <= set(sections)
    assert sections["summary_rouge"]["bertscore"]["f1_raw_mean"] == 0.9


import sqlite3


def count(db, table):
    c = sqlite3.connect(db)
    try:
        return c.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    finally:
        c.close()


class FailingLLM:
    provider, model = "gemini", "gemini-3.1-flash-lite"

    def __init__(self, ok=0, code="insufficient_quota"):
        self.ok, self.code, self.calls = ok, code, 0
        self.good = FakeLLM()

    def complete(self, system, user, **kw):
        self.calls += 1
        if self.calls > self.ok:
            raise LLMUnavailable(self.code, "down")
        return self.good.complete(system, user, **kw)


class BoomLLM(FakeLLM):
    def complete(self, system, user, **kw):
        if self.calls >= 1:
            raise RuntimeError("boom")
        return super().complete(system, user, **kw)


def test_save_section_keeps_hand_written_keys_and_moves_replaced_values_to_previous(tmp_path):
    harness.save_section(tmp_path, "c", "refusal_end_to_end", {"n_questions": 40, "accuracy_as_labelled": 0.95})
    path = harness.report_path(tmp_path, "c")
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["sections"]["refusal_end_to_end"].update({"decision": "keep tau unset", "reading_of_those": "borderline"})
    old_stamp = stored["sections"]["refusal_end_to_end"]["generated_at"]
    path.write_text(json.dumps(stored), encoding="utf-8")

    report = harness.save_section(tmp_path, "c", "refusal_end_to_end", {"n_questions": 39, "model": "m"})
    section = report["sections"]["refusal_end_to_end"]
    assert section["n_questions"] == 39 and section["model"] == "m"
    assert section["decision"] == "keep tau unset" and section["reading_of_those"] == "borderline"
    assert section["preserved_from_earlier_section"] == ["accuracy_as_labelled", "decision", "reading_of_those"]
    assert section["previous"] == {"n_questions": 40, "generated_at": old_stamp}
    again = harness.save_section(tmp_path, "c", "refusal_end_to_end", {"n_questions": 38})
    assert "previous" not in again["sections"]["refusal_end_to_end"]["previous"]
    assert again["sections"]["refusal_end_to_end"]["previous"]["n_questions"] == 39


def test_unusable_reason_flags_every_empty_run_and_passes_a_real_one():
    assert harness.unusable_reason("summary_rouge", {"n": 0})
    assert harness.unusable_reason("summary_rouge", {"n": 3, "stopped_early": "insufficient_quota"})
    assert harness.unusable_reason("summary_rouge", {"n": 3}) is None
    assert harness.unusable_reason("refusal_end_to_end", {"n_questions": 0, "n_unavailable": 0})
    assert harness.unusable_reason("refusal_end_to_end", {"n_questions": 4, "n_unavailable": 4})
    assert harness.unusable_reason("refusal_end_to_end", {"n_questions": 4, "n_unavailable": 1}) is None
    assert harness.unusable_reason("refusal_end_to_end", {"n_questions": 4, "n_unavailable": 1,
                                                          "stopped_early": "invalid_api_key"})
    ok = {"per_answered_question": {"calls": 2}, "per_summary": {"calls": 2}}
    assert harness.unusable_reason("cost_per_query", ok) is None
    assert harness.unusable_reason("cost_per_query", {**ok, "per_summary": {"calls": 0}})
    assert harness.unusable_reason("cost_per_query", {**ok, "per_answered_question": {"calls": 0}})
    assert harness.unusable_reason("latency", {}) is None


def stored(tmp, cid):
    return harness.report_path(tmp / "data", cid)


def seed_report(tmp, cid):
    harness.save_section(tmp / "data", cid, "refusal_end_to_end", {"n_questions": 40, "unanswerable_refused": "14/16"})
    harness.save_section(tmp / "data", cid, "summary_rouge", {"n": 20, "rougeL_f1_mean": 0.32})
    harness.save_section(tmp / "data", cid, "cost_per_query", {"per_answered_question": {"calls": 12},
                                                              "per_summary": {"calls": 20}})
    return stored(tmp, cid).read_bytes()


@pytest.mark.parametrize("argv", [["refusal", "--labels", "Q"], ["summaries", "--n", "2"], ["cost", "--labels", "Q"]])
def test_a_run_during_an_outage_exits_nonzero_and_leaves_the_stored_report_untouched(cli_env, monkeypatch, argv, capsys):
    cid, tmp, labels, rel = cli_env
    monkeypatch.setattr(cli, "_make_llm", lambda settings, db_path, **kw: FailingLLM(ok=0))
    before = seed_report(tmp, cid)
    argv = [str(labels) if a == "Q" else a for a in argv]
    code = cli.main(["--db", str(tmp / "e.sqlite3"), "eval", argv[0], "--collection", cid] + argv[1:])
    assert code == 3
    assert stored(tmp, cid).read_bytes() == before
    err = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert err["stored_report"] == "left unchanged" and err["error"]


def test_a_good_run_keeps_the_hand_written_keys_of_the_stored_section(cli_env):
    cid, tmp, labels, rel = cli_env
    harness.save_section(tmp / "data", cid, "refusal_end_to_end",
                         {"n_questions": 40, "decision": "REFUSAL_TAU stays unset"})
    assert cli.main(["--db", str(tmp / "e.sqlite3"), "eval", "refusal", "--collection", cid,
                     "--labels", str(labels)]) == 0
    section = harness.load_report(tmp / "data", cid)["sections"]["refusal_end_to_end"]
    assert section["decision"] == "REFUSAL_TAU stays unset"
    assert section["previous"]["n_questions"] == 40 and section["n_questions"] == 3


def test_refusal_and_cost_leave_the_answers_table_as_they_found_it(world):
    conn, cid, ids, db, tmp = world
    before = count(db, "answers")
    out = harness.refusal_end_to_end(conn, cid, {"questions": QUESTIONS}, FakeLLM(), vector_fn)
    assert out["answers_rows_created_and_deleted"] == 3 and count(db, "answers") == before
    out = harness.cost_per_query(conn, cid, {"questions": QUESTIONS}, FakeLLM(), vector_fn, n_answers=2, n_summaries=1)
    assert out["answers_rows_created_and_deleted"] == 2 and count(db, "answers") == before


def test_answers_are_deleted_even_when_the_run_crashes(world):
    conn, cid, ids, db, tmp = world
    before = count(db, "answers")
    with pytest.raises(RuntimeError):
        harness.refusal_end_to_end(conn, cid, {"questions": QUESTIONS}, BoomLLM(), vector_fn)
    assert count(db, "answers") == before
    with pytest.raises(RuntimeError):
        harness.cost_per_query(conn, cid, {"questions": QUESTIONS}, BoomLLM(), vector_fn, n_answers=2, n_summaries=1)
    assert count(db, "answers") == before


def test_cli_leaves_the_answers_table_unchanged(cli_env):
    cid, tmp, labels, rel = cli_env
    db = tmp / "e.sqlite3"
    before = count(db, "answers")
    for argv in (["refusal"], ["cost", "--n", "1"]):
        assert cli.main(["--db", str(db), "eval", argv[0], "--collection", cid, "--labels", str(labels)] + argv[1:]) == 0
    assert count(db, "answers") == before


def test_a_quota_failure_stops_the_refusal_and_cost_loops_at_the_first_call(world):
    conn, cid, *_ = world
    llm = FailingLLM(ok=0)
    out = harness.refusal_end_to_end(conn, cid, {"questions": QUESTIONS}, llm, vector_fn)
    assert llm.calls == 1 and out["stopped_early"] == "insufficient_quota"
    assert out["n_questions"] == 1 and out["n_questions_requested"] == 3
    llm = FailingLLM(ok=0, code="invalid_api_key")
    out = harness.cost_per_query(conn, cid, {"questions": QUESTIONS}, llm, vector_fn, n_answers=2, n_summaries=2)
    assert llm.calls == 1 and out["stopped_early"] == "invalid_api_key"


def test_a_quota_failure_during_the_summaries_stops_the_cost_run(world):
    conn, cid, *_ = world
    llm = FailingLLM(ok=1)
    out = harness.cost_per_query(conn, cid, {"questions": QUESTIONS}, llm, vector_fn, n_answers=1, n_summaries=3)
    assert llm.calls == 2 and out["stopped_early"] == "insufficient_quota"
    assert out["per_answered_question"]["calls"] == 1 and out["per_summary"]["calls"] == 0


def test_a_transient_failure_does_not_stop_the_run(world):
    conn, cid, *_ = world
    llm = FailingLLM(ok=0, code="rate_limited")
    out = harness.refusal_end_to_end(conn, cid, {"questions": QUESTIONS}, llm, vector_fn)
    assert llm.calls == 3 and out["stopped_early"] is None and out["n_unavailable"] == 3
    assert out["accuracy_as_labelled"] is None


def test_the_seeded_bootstrap_gives_fixed_intervals_and_the_sign_follows_the_argument_order():
    a, b = [0.9, 0.5, 0.7, 0.2, 0.6], [0.4, 0.5, 0.3, 0.1, 0.8]
    assert paired_bootstrap(a, b, resamples=1000, seed=0) == {
        "mean_diff": 0.16, "ci95": [-0.04, 0.38], "excludes_zero": False}
    assert paired_bootstrap(b, a, resamples=1000, seed=0) == {
        "mean_diff": -0.16, "ci95": [-0.38, 0.04], "excludes_zero": False}
    assert paired_bootstrap(a, b, resamples=1000, seed=0)["ci95"] != paired_bootstrap(a, b, resamples=1000, seed=1)["ci95"]
    assert paired_bootstrap([1.0, 0.9, 0.8], [0.1, 0.2, 0.1], resamples=200, seed=0)["excludes_zero"] is True
    with pytest.raises(ValueError):
        paired_bootstrap([1.0], [1.0, 2.0])


def test_bootstrap_output_is_labelled_descriptive_with_scope_and_wilcoxon(world):
    conn, cid, *_ = world
    labels = {"queries": [
        {"query": "retrieval hallucination", "judged": True, "relevant": ["2401.00001"]},
        {"query": "dense passage search", "judged": True, "relevant": ["2401.00002"]},
        {"query": "nothing relevant", "judged": True, "relevant": []}]}
    out = harness.retrieval_bootstrap(conn, cid, labels, vector_fn, resamples=50)
    assert out["n_queries_with_a_relevant_paper"] == 2 and out["n_judged_queries"] == 3
    assert "2 queries with a relevant paper" in out["mrr_scope"] and "all 3 judged" in out["mrr_scope"]
    for word in ("6 comparisons", "no multiple-comparison correction", "undercover", "same retrieval", "not significance"):
        assert word in out["caveat"]
    for cmp in out["comparisons"].values():
        assert "wilcoxon_p_two_sided" in cmp


def test_wilcoxon_is_none_when_undefined():
    assert harness._wilcoxon_p([0.5, 0.5], [0.5, 0.5]) == 1.0
    assert harness._wilcoxon_p([0.5, 0.6, 0.7], [0.5, 0.6]) is None
    p = harness._wilcoxon_p([0.9, 0.8, 0.7, 0.9, 0.8, 0.6], [0.1, 0.2, 0.3, 0.2, 0.1, 0.3])
    assert 0 < p <= 1


class AbstractEcho:
    provider, model = "gemini", "m"

    def __init__(self, transform):
        self.transform = transform

    def complete(self, system, user, **kw):
        abstract = user.split("Abstract: ")[1].split("\n\n")[0]
        return LLMResult(self.transform(abstract), "m", False, None)


def test_rouge_l_is_the_f_measure_against_the_abstract_not_the_title(world):
    conn, cid, *_ = world
    out = harness.summary_scores(conn, cid, AbstractEcho(lambda a: "Problem: " + a), n=4, bert_fn=fake_bert)
    abstracts = {r["id"]: r["abstract"] for r in conn.execute("SELECT id, abstract FROM papers")}
    assert len(out["per_paper"]) == 4
    for row in out["per_paper"]:
        n = len(abstracts[row["paper_id"]].split())
        assert row["rougeL_f1"] == round(2 * n / (2 * n + 1), 4)
    assert out["rougeL_f1_mean"] == round(sum(r["rougeL_f1"] for r in out["per_paper"]) / 4, 4)


def test_rouge_l_stems_words(world):
    conn, cid, *_ = world
    inflect = lambda a: a.replace("grounds", "grounding").replace("generation", "generations")
    out = harness.summary_scores(conn, cid, AbstractEcho(inflect), n=4, bert_fn=fake_bert)
    assert [r["rougeL_f1"] for r in out["per_paper"]] == [1.0] * 4


def test_bertscore_rescaling_uses_the_f_baseline_and_the_documented_formula(monkeypatch):
    pytest.importorskip("bert_score")
    import bert_score

    class FakeScorer:
        num_layers = 7
        made = {}

        def __init__(self, **kw):
            FakeScorer.made = kw

        baseline_vals = [0.1, 0.2, 0.6]

        def score(self, cands, refs):
            return [0.0, 0.0], [0.0, 0.0], [0.9, 0.75]

    monkeypatch.setattr(bert_score, "BERTScorer", FakeScorer)
    out = harness.bertscore_f1(["a", "b"], ["c", "d"], model_type="some-model")
    assert out["raw_f1"] == [0.9, 0.75]
    assert out["rescaled_f1"] == pytest.approx([0.75, 0.375])
    assert out["baseline_f1"] == 0.6 and out["num_layers"] == 7 and out["model_type"] == "some-model"
    assert FakeScorer.made["model_type"] == "some-model" and FakeScorer.made["rescale_with_baseline"] is False
    assert FakeScorer.made["device"] == "cpu" and FakeScorer.made["lang"] == "en"


@pytest.mark.parametrize("error", [ValueError("Baseline not Found"), AssertionError(), FileNotFoundError()])
def test_bertscore_without_a_baseline_reports_raw_only(monkeypatch, error):
    pytest.importorskip("bert_score")
    import bert_score

    class NoBaseline:
        num_layers = 1

        def __init__(self, **kw):
            pass

        @property
        def baseline_vals(self):
            raise error

        def score(self, cands, refs):
            return [0.0], [0.0], [0.9]

    monkeypatch.setattr(bert_score, "BERTScorer", NoBaseline)
    out = harness.bertscore_f1(["a"], ["b"])
    assert out["raw_f1"] == [0.9] and out["rescaled_f1"] is None and out["baseline_f1"] is None


class HalfCited(FakeLLM):
    def complete(self, system, user, **kw):
        if "Protein" in user.split("Question:")[-1]:
            return LLMResult("INSUFFICIENT_CONTEXT", self.model, False, self.usage)
        return LLMResult("Retrieval grounds generation [1]. It is unclear.", self.model, False, self.usage)


def test_citation_coverage_is_averaged_over_answered_questions_only(world):
    conn, cid, *_ = world
    out = harness.refusal_end_to_end(conn, cid, {"questions": QUESTIONS}, HalfCited(), vector_fn)
    assert out["unanswerable_refused"] == "1/1"
    assert out["mean_citation_coverage_when_answered"] == 0.5
    assert [r["citation_coverage"] for r in out["per_question"]] == [0.5, 0.5, None]


def test_the_report_records_tau_and_its_mode_and_that_it_is_a_single_run(world):
    conn, cid, *_ = world
    out = harness.refusal_end_to_end(conn, cid, {"questions": QUESTIONS}, FakeLLM(), vector_fn,
                                     refusal_tau=1.5, tau_mode="dense")
    assert out["refusal_tau"] == 1.5 and out["refusal_tau_mode"] == "dense"
    assert out["refusal_reasons"] == {"below_threshold": 3}
    assert "single stochastic run" in out["run"] and "client retries" in out["latency_ms_uncached"]["note"]


def test_usage_in_neither_providers_shape_is_excluded_not_counted_as_zero_tokens(world):
    assert harness.normalise_usage({"totalTokenCount": 5}) is None
    assert harness.normalise_usage({}) is None
    conn, cid, *_ = world
    out = harness.cost_per_query(conn, cid, {"questions": QUESTIONS}, FakeLLM(usage={"totalTokenCount": 5}),
                                 vector_fn, n_answers=2, n_summaries=2)
    assert out["per_answered_question"]["calls"] == 0 and out["per_summary"]["calls"] == 0
    assert out["excluded_calls_without_usage"] == 4
    assert harness.unusable_reason("cost_per_query", out)


def test_cost_labels_usd_as_an_upper_bound_and_a_single_sample(world):
    conn, cid, *_ = world
    out = harness.cost_per_query(conn, cid, {"questions": QUESTIONS}, FakeLLM(), vector_fn, n_answers=1, n_summaries=1)
    assert "upper bound" in out["usd_note"] and "implicit caching" in out["usd_note"]
    assert "client retries" in out["note"] and "single stochastic run" in out["note"]


@pytest.fixture
def real_llm_env(world, monkeypatch):
    conn, cid, ids, db, tmp = world
    conn.close()
    monkeypatch.setenv("DATA_DIR", str(tmp / "data"))
    monkeypatch.setenv("DB_PATH", str(db))
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_MODEL", "gemini-3.1-flash-lite")
    monkeypatch.setenv("REFUSAL_TAU", "")
    monkeypatch.setattr("services.query_embedder.QueryEmbedder.vector", lambda self, text: vector_fn(text))
    calls = []

    def generate(self, system, user, max_output_tokens):
        calls.append(user)
        text = "Problem: p\nMethod: m\nResult: r" if "Abstract:" in user else "Retrieval grounds generation [1]."
        return text, {"promptTokenCount": 100, "candidatesTokenCount": 10}

    monkeypatch.setattr("services.llm.LLMClient._generate", generate)
    labels = tmp / "q.json"
    labels.write_text(json.dumps({"questions": QUESTIONS}), encoding="utf-8")
    return cid, db, tmp, labels, calls


def test_eval_cost_through_the_real_client_never_writes_the_real_llm_cache(real_llm_env):
    cid, db, tmp, labels, calls = real_llm_env
    before_cache, before_answers = count(db, "llm_cache"), count(db, "answers")
    assert cli.main(["--db", str(db), "eval", "cost", "--collection", cid, "--labels", str(labels), "--n", "2"]) == 0
    assert len(calls) == 4
    assert count(db, "llm_cache") == before_cache and count(db, "answers") == before_answers
    section = harness.load_report(tmp / "data", cid)["sections"]["cost_per_query"]
    assert section["per_answered_question"]["calls"] == 2 and section["per_summary"]["calls"] == 2
    assert section["excluded_cached_calls"] == 0


def test_eval_cost_does_not_reuse_a_response_already_in_the_real_cache(real_llm_env):
    from services.llm import LLMClient
    from store.db import connect as open_db

    cid, db, tmp, labels, calls = real_llm_env
    conn = open_db(db)
    harness.summary_scores(conn, cid, LLMClient(api_key="k", model="gemini-3.1-flash-lite", db_path=db,
                                                provider="gemini"), n=2, bert_fn=fake_bert)
    conn.close()
    assert len(calls) == 2 and count(db, "llm_cache") == 2
    n_before = len(calls)
    assert cli.main(["--db", str(db), "eval", "cost", "--collection", cid, "--labels", str(labels), "--n", "2"]) == 0
    assert len(calls) - n_before == 4


def test_the_cached_flag_is_false_for_fresh_calls_and_true_only_for_repeats(real_llm_env, monkeypatch):
    from services.llm import LLMClient
    from store.db import connect as open_db

    cid, db, tmp, labels, calls = real_llm_env
    monkeypatch.setattr(harness, "bertscore_f1", fake_bert)
    llm = LLMClient(api_key="k", model="gemini-3.1-flash-lite", db_path=db, provider="gemini")
    conn = open_db(db)
    rows_before = count(db, "llm_cache")
    first = harness.summary_scores(conn, cid, llm, n=3, bert_fn=fake_bert)
    assert [p["cached"] for p in first["per_paper"]] == [False, False, False]
    assert count(db, "llm_cache") - rows_before == 3 and len(calls) == 3
    second = harness.summary_scores(conn, cid, llm, n=3, bert_fn=fake_bert)
    assert [p["cached"] for p in second["per_paper"]] == [True, True, True]
    assert len(calls) == 3 and count(db, "llm_cache") - rows_before == 3
    conn.close()


def test_cli_passes_n_and_the_bert_model_through(cli_env, monkeypatch):
    cid, tmp, labels, rel = cli_env
    seen = {}

    def spy(candidates, references, **kw):
        seen.update(kw)
        return fake_bert(candidates, references)

    monkeypatch.setattr(harness, "bertscore_f1", spy)
    base = ["--db", str(tmp / "e.sqlite3"), "eval"]
    assert cli.main(base + ["summaries", "--collection", cid, "--n", "2", "--bert-model", "distilbert-base-uncased"]) == 0
    assert seen == {"model_type": "distilbert-base-uncased"}
    sections = lambda: harness.load_report(tmp / "data", cid)["sections"]
    assert sections()["summary_rouge"]["n"] == 2 and sections()["summary_rouge"]["n_requested"] == 2
    assert cli.main(base + ["summaries", "--collection", cid]) == 0
    assert sections()["summary_rouge"]["n_requested"] == 20 and seen["model_type"] == "roberta-large"
    assert cli.main(base + ["cost", "--collection", cid, "--labels", str(labels), "--n", "1"]) == 0
    cost = sections()["cost_per_query"]
    assert cost["questions_run"] == 1 and cost["per_summary"]["calls"] == 1


@pytest.mark.parametrize("section", ["summaries", "cost"])
def test_n_zero_is_rejected(cli_env, section):
    cid, tmp, labels, rel = cli_env
    with pytest.raises(SystemExit) as stop:
        cli.main(["--db", str(tmp / "e.sqlite3"), "eval", section, "--collection", cid, "--n", "0",
                  "--labels", str(labels)])
    assert stop.value.code == 2


def test_cli_applies_refusal_tau_from_the_environment(cli_env, monkeypatch):
    cid, tmp, labels, rel = cli_env
    monkeypatch.setenv("REFUSAL_TAU", "1.5")
    monkeypatch.setenv("REFUSAL_TAU_MODE", "dense")
    assert cli.main(["--db", str(tmp / "e.sqlite3"), "eval", "refusal", "--collection", cid,
                     "--labels", str(labels)]) == 0
    section = harness.load_report(tmp / "data", cid)["sections"]["refusal_end_to_end"]
    assert section["refusal_tau"] == 1.5 and section["refusal_tau_mode"] == "dense"
    assert section["refusal_reasons"] == {"below_threshold": 3}


def test_the_no_labels_hint_no_longer_claims_every_metric_needs_humans(cli_env, capsys):
    cid, tmp, labels, rel = cli_env
    empty = tmp / "none.json"
    empty.write_text(json.dumps({"queries": []}), encoding="utf-8")
    assert cli.main(["--db", str(tmp / "e.sqlite3"), "eval", "bootstrap", "--collection", cid,
                     "--labels", str(empty)]) == 2
    assert "human" not in capsys.readouterr().err
