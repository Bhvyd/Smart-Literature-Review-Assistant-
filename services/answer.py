
from __future__ import annotations

import json
import re
import time
from collections.abc import Sequence
from typing import Any

from rag_core import Passage, build_prompt
from services.llm import UNTRUSTED_TEXT_RULE, LLMClient, LLMUnavailable
from services.retrieval import cached_index, paper_details
from store import repo

SENTINEL = "INSUFFICIENT_CONTEXT"
REFUSAL_MESSAGE = "The collection does not contain enough information to answer this question."
DEFAULT_K = 8
MAX_PASSAGES_PER_PAPER = 2
PROMPT_MAX_CHARS = 9000
QUOTE_CHARS = 420

SYSTEM_PROMPT = (
    "You answer questions about a collection of research papers using ONLY the numbered "
    "context passages you are given.\n"
    "Rules:\n"
    "1. Every sentence that states a fact must end with the number(s) of the passage(s) "
    "that support it, written like [1] or [2][3].\n"
    "2. Use only numbers and names that appear in the context. Never invent any.\n"
    "3. Do not use outside knowledge. Numbers written like (ref. 56) inside a passage are that "
    "passage's own bibliography, NOT your citation numbers; never cite them.\n"
    "3b. A passage may mention other works. When asked which papers do something, name only the "
    "papers you cite, by the title shown after their passage number, not works merely mentioned "
    "inside a passage (say 'one passage mentions ...' if you must refer to such a work).\n"
    f"4. If the context does not contain enough information to answer, reply with exactly: {SENTINEL}\n"
    "5. Be concise: at most 150 words.\n"
    "6. Stay inside the setting the passages describe. If they describe a technique only as part of one "
    "specific paper or application (for example a video-retrieval module, or RLHF as applied to RAG), say "
    "so; do not present it as how the field or a general method works. If the passages do not address "
    "the general question the user asked, reply with exactly: " + SENTINEL
) + UNTRUSTED_TEXT_RULE

_CITATION = re.compile(
    r"\[(\s*\d+(?:\s*[-\u2013]\s*\d+)?(?:\s*[,;]\s*\d+(?:\s*[-\u2013]\s*\d+)?)*\s*)\]"
)
MAX_RANGE = 50
YEAR_FLOOR = 1000


def _slot_numbers(inner: str) -> list[int] | None:
    numbers: list[int] = []
    for part in re.split(r"\s*[,;]\s*", inner.strip()):
        bounds = [int(x) for x in re.findall(r"\d+", part)]
        if any(n >= YEAR_FLOOR for n in bounds):
            return None
        if len(bounds) == 2:
            low, high = bounds
            numbers.extend(range(low, min(high, low + MAX_RANGE) + 1))
        else:
            numbers.extend(bounds)
    return numbers
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def parse_citations(text: str, slots: int) -> tuple[str, list[int], list[int]]:
    valid: list[int] = []
    invalid: list[int] = []

    def keep(match: re.Match[str]) -> str:
        numbers = _slot_numbers(match.group(1))
        if numbers is None:
            return match.group(0)
        good = [n for n in numbers if 1 <= n <= slots]
        invalid.extend(n for n in numbers if not 1 <= n <= slots)
        valid.extend(good)
        return "".join(f"[{n}]" for n in good)

    clean = _CITATION.sub(keep, text)
    clean = re.sub(r"[ \t]{2,}", " ", clean).strip()
    return clean, sorted(set(valid)), sorted(set(invalid))


def citation_coverage(text: str) -> float:
    sentences = [s for s in _SENTENCE_END.split(text.strip()) if s.strip()]
    if not sentences:
        return 0.0
    return sum(1 for s in sentences if _CITATION.search(s)) / len(sentences)


_INLINE_REF = re.compile(r"\[(\s*\d+(?:\s*[-\u2013,;]\s*\d+)*\s*)\]")


def neutralise_inline_references(text: str) -> str:
    return _INLINE_REF.sub(lambda m: f"(ref. {' '.join(m.group(1).split())})", text)


def select_passages(
    index: Any,
    hits: Sequence[Any],
    *,
    k: int,
    per_paper: int = MAX_PASSAGES_PER_PAPER,
) -> list[Passage]:
    texts = {chunk.chunk_id: chunk.text for chunk in index.chunks}
    taken: dict[str, int] = {}
    passages: list[Passage] = []
    for hit in hits:
        paper_id = index.paper_for_document(hit.doc_id)
        if paper_id is None or taken.get(paper_id, 0) >= per_paper:
            continue
        text = texts.get(hit.doc_id) or index.abstract_documents.get(hit.doc_id)
        if not text:
            continue
        taken[paper_id] = taken.get(paper_id, 0) + 1
        passages.append(
            Passage(
                chunk_id=hit.doc_id,
                paper_id=paper_id,
                text=text,
                title=index.paper_by_id[paper_id].title,
            )
        )
        if len(passages) >= k:
            break
    return passages


def _excerpt(text: str, limit: int = QUOTE_CHARS) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip(",;:- ")
    return cut + "…"


def _citation_rows(
    conn: Any, index: Any, passages: Sequence[Passage], bundle: Any
) -> list[dict[str, Any]]:
    details = paper_details(conn, [bundle.papers[slot] for slot in sorted(bundle.papers)])
    rows: list[dict[str, Any]] = []
    for slot in sorted(bundle.citations):
        passage = passages[slot - 1]
        paper = index.paper_by_id[passage.paper_id]
        extra = details.get(passage.paper_id, {})
        rows.append(
            {
                "n": slot,
                "paper_id": passage.paper_id,
                "chunk_id": passage.chunk_id,
                "title": paper.title,
                "year": paper.year,
                "arxiv_id": extra.get("arxiv_id"),
                "url": extra.get("source_url"),
                "quote": _excerpt(passage.text),
            }
        )
    return rows


def ask(
    conn: Any,
    collection_id: str,
    question: str,
    *,
    llm: LLMClient,
    query_vector: Sequence[float] | None,
    k: int = DEFAULT_K,
    mode: str = "hybrid",
    refusal_tau: float | None = None,
    tau_mode: str = "dense",
) -> dict[str, Any]:
    started = time.perf_counter()
    index, retriever = cached_index(conn, collection_id)

    degraded: list[str] = []
    effective = mode
    if effective in {"dense", "hybrid"} and query_vector is None:
        effective = "lexical"
        degraded.append("query_vector_missing")
    if effective in {"dense", "hybrid"} and not index.dense_vectors:
        effective = "lexical"
        degraded.append("no_dense_vectors")

    result = retriever.search(
        question, k=max(k * 6, 40), mode=effective, query_vector=query_vector
    )
    passages = select_passages(index, result.hits, k=k)
    top_score = result.hits[0].score if result.hits else None

    retrieval = {
        "mode": effective,
        "degraded": degraded,
        "top_score": top_score,
        "n_passages": len(passages),
        "tau": refusal_tau,
        "tau_mode": tau_mode,
        "tau_calibrated": refusal_tau is not None,
        "tau_score": None,
        "tau_skipped": None,
    }
    bundle_payload: dict[str, Any] = {
        "question": question,
        "answer": None,
        "refused": False,
        "refusal_reason": None,
        "answer_unavailable": None,
        "citations": [],
        "confidence": None,
        "confidence_basis": "citation_coverage",
        "citation_anomaly": [],
        "retrieval": retrieval,
        "model": llm.model,
        "cached": False,
        "latency_ms": 0,
    }

    def finish(payload: dict[str, Any]) -> dict[str, Any]:
        payload["latency_ms"] = int((time.perf_counter() - started) * 1000)
        payload["answer_id"] = repo.save_answer(
            conn,
            collection_id=collection_id,
            question=question,
            answer=payload["answer"],
            citations=payload["citations"],
            refused=payload["refused"],
            model=llm.model,
            latency_ms=payload["latency_ms"],
            retrieval_snapshot={
                **retrieval,
                "passages": [
                    {"n": i + 1, "chunk_id": p.chunk_id, "paper_id": p.paper_id}
                    for i, p in enumerate(passages)
                ],
            },
        )
        return payload

    if not passages:
        bundle_payload.update(
            refused=True, refusal_reason="no_retrieval_hits", answer=REFUSAL_MESSAGE
        )
        return finish(bundle_payload)

    if refusal_tau is not None:
        needs_vector = tau_mode in {"dense", "hybrid"}
        if needs_vector and (query_vector is None or not index.dense_vectors):
            retrieval["tau_skipped"] = "query_vector_missing"
        else:
            gate = retriever.search(question, k=1, mode=tau_mode, query_vector=query_vector).hits
            gate_score = gate[0].score if gate else None
            retrieval["tau_score"] = gate_score
            if gate_score is None or gate_score < refusal_tau:
                bundle_payload.update(
                    refused=True, refusal_reason="below_threshold", answer=REFUSAL_MESSAGE
                )
                bundle_payload["citations"] = []
                return finish(bundle_payload)

    safe = [
        Passage(chunk_id=p.chunk_id, paper_id=p.paper_id, title=p.title,
                text=neutralise_inline_references(p.text))
        for p in passages
    ]
    prompt = build_prompt(question, safe, max_chars=PROMPT_MAX_CHARS)
    passages = list(passages[: prompt.used])
    retrieval["n_passages"] = prompt.used
    bundle_payload["citations"] = _citation_rows(conn, index, passages, prompt)

    try:
        response = llm.complete(SYSTEM_PROMPT, prompt.text)
    except LLMUnavailable as exc:
        bundle_payload["answer_unavailable"] = {"code": exc.code, "message": exc.message}
        return finish(bundle_payload)

    bundle_payload["cached"] = response.cached
    text = response.text.strip()

    if text.upper().startswith(SENTINEL):
        bundle_payload.update(
            refused=True,
            refusal_reason="model_declined",
            answer=REFUSAL_MESSAGE,
            citations=[],
        )
        return finish(bundle_payload)

    clean, valid, invalid = parse_citations(text, prompt.used)
    bundle_payload["citation_anomaly"] = invalid
    if not valid:
        bundle_payload.update(
            refused=True,
            refusal_reason="answer_had_no_valid_citation",
            answer=None,
            citations=[],
        )
        return finish(bundle_payload)

    bundle_payload["answer"] = clean
    bundle_payload["citations"] = [row for row in bundle_payload["citations"] if row["n"] in valid]
    bundle_payload["confidence"] = round(citation_coverage(clean), 3)
    bundle_payload["uncited_sentences"] = sum(
        1 for s in _SENTENCE_END.split(clean) if s.strip() and not _CITATION.search(s)
    )
    return finish(bundle_payload)
