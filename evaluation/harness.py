
from __future__ import annotations

import hashlib
import json
import random
import re
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.metrics import (
    best_threshold,
    mean,
    ndcg_at_k,
    paired_bootstrap,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from rag_core import cosine_search, document_norms
from services.retrieval import cached_index, load_papers, rank_papers

MODES = ("lexical", "tfidf", "dense", "hybrid")
VectorFn = Callable[[str], "list[float] | None"]


class NoLabels(Exception):
    pass


FATAL_LLM_CODES = frozenset({"insufficient_quota", "invalid_api_key", "no_api_key", "model_not_found"})


def unusable_reason(section: str, result: dict[str, Any]) -> str | None:
    if result.get("stopped_early"):
        return f"stopped early: the provider reported {result['stopped_early']!r}; the run is incomplete"
    if section == "summary_rouge" and not result.get("n"):
        return "no summary was generated, so there is nothing to score"
    if section == "refusal_end_to_end":
        if not result.get("n_questions"):
            return "no labelled question was run"
        if result.get("n_unavailable") == result["n_questions"]:
            return "every question was unavailable (provider failure), so nothing was measured"
    if section == "cost_per_query":
        for part in ("per_answered_question", "per_summary"):
            if not (result.get(part) or {}).get("calls"):
                return (f"{part} has no fresh call with usage metadata (provider unavailable, cached, or no "
                        "usage returned), so the cost is not measured")
    return None


def _delete_answers(conn: Any, answer_ids: Sequence[str]) -> int:
    from store.db import transaction

    with transaction(conn):
        for answer_id in answer_ids:
            conn.execute("DELETE FROM answers WHERE id=?", (answer_id,))
    return len(answer_ids)


def report_path(data_dir: str | Path, collection_id: str) -> Path:
    return Path(data_dir) / "eval" / f"{collection_id}.json"


def load_report(data_dir: str | Path, collection_id: str) -> dict[str, Any] | None:
    path = report_path(data_dir, collection_id)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_section(
    data_dir: str | Path, collection_id: str, name: str, payload: dict[str, Any]
) -> dict[str, Any]:
    report = load_report(data_dir, collection_id) or {"collection_id": collection_id, "sections": {}}
    new = {**payload, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    old = report["sections"].get(name)
    if old:
        skip = {"previous", "preserved_from_earlier_section"}
        carried = {k: v for k, v in old.items() if k not in new and k not in skip}
        replaced = {k: v for k, v in old.items() if k in new and k not in skip}
        new = {**carried, **new}
        if carried:
            new["preserved_from_earlier_section"] = sorted(carried)
        if replaced:
            new["previous"] = replaced
    report["sections"][name] = new
    path = report_path(data_dir, collection_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def title_queries(conn: Any, collection_id: str, n: int = 30, words: int = 8) -> list[str]:
    papers = sorted(load_papers(conn, collection_id), key=lambda p: p.paper_id)
    if not papers:
        return []
    stride = max(1, len(papers) // n)
    chosen = papers[::stride][:n]
    return [" ".join(p.title.split()[:words]) for p in chosen]


def latency(
    conn: Any, collection_id: str, vector_fn: VectorFn, *, queries: Sequence[str] | None = None
) -> dict[str, Any]:
    queries = list(queries or title_queries(conn, collection_id))
    if not queries:
        return {"error": "collection has no papers"}

    started = time.perf_counter()
    cached_index(conn, collection_id)
    index_build_s = time.perf_counter() - started

    vectors: dict[str, list[float] | None] = {}
    embed_ms: list[float] = []
    for query in queries:
        t = time.perf_counter()
        vectors[query] = vector_fn(query)
        embed_ms.append((time.perf_counter() - t) * 1000)

    index, _ = cached_index(conn, collection_id)
    modes: dict[str, Any] = {}
    for mode in MODES:
        times: list[float] = []
        skipped = 0
        for query in queries:
            vec = vectors[query]
            if mode in {"dense", "hybrid"} and vec is None:
                skipped += 1
                continue
            t = time.perf_counter()
            rank_papers(conn, collection_id, query=query, mode=mode, top_k=20, query_vector=vec)
            times.append((time.perf_counter() - t) * 1000)
        modes[mode] = (
            {
                "n": len(times),
                "p50_ms": round(percentile(times, 50), 2),
                "p95_ms": round(percentile(times, 95), 2),
                "max_ms": round(max(times), 2),
                "skipped_no_query_vector": skipped,
            }
            if times
            else {"n": 0, "skipped_no_query_vector": skipped}
        )
    return {
        "n_queries": len(queries),
        "n_chunks": len(index.chunks),
        "index_build_seconds": round(index_build_s, 3),
        "query_embedding_ms": {
            "p50": round(percentile(embed_ms, 50), 1),
            "p95": round(percentile(embed_ms, 95), 1),
        },
        "modes": modes,
        "note": "retrieval and paper ranking only; generation latency is measured separately "
        "because it depends on the provider",
    }


def faiss_oracle(
    conn: Any, collection_id: str, vector_fn: VectorFn, *, queries: Sequence[str] | None = None, k: int = 10
) -> dict[str, Any]:
    try:
        import faiss
        import numpy as np
    except ImportError as exc:
        return {"error": f"faiss/numpy not installed ({exc}); pip install faiss-cpu"}

    index, _ = cached_index(conn, collection_id)
    ids = list(index.dense_vectors)
    if not ids:
        return {"error": "collection has no chunk vectors"}
    matrix = np.asarray([index.dense_vectors[i] for i in ids], dtype="float32")
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    flat = faiss.IndexFlatIP(matrix.shape[1])
    flat.add(matrix)
    norms = document_norms(index.dense_vectors)

    same_order = same_set = 0
    max_diff = 0.0
    hand_ms: list[float] = []
    faiss_ms: list[float] = []
    used = 0
    for query in list(queries or title_queries(conn, collection_id)):
        vec = vector_fn(query)
        if vec is None:
            continue
        used += 1
        t = time.perf_counter()
        hand = cosine_search(vec, index.dense_vectors, k, norms=norms)
        hand_ms.append((time.perf_counter() - t) * 1000)

        q = np.asarray([vec], dtype="float32")
        q /= np.linalg.norm(q)
        t = time.perf_counter()
        scores, positions = flat.search(q, k)
        faiss_ms.append((time.perf_counter() - t) * 1000)
        theirs = [(ids[p], float(s)) for p, s in zip(positions[0], scores[0]) if p >= 0]

        same_order += [d for d, _ in hand] == [d for d, _ in theirs]
        same_set += {d for d, _ in hand} == {d for d, _ in theirs}
        max_diff = max(max_diff, max(abs(a[1] - b[1]) for a, b in zip(hand, theirs)))
    if not used:
        return {"error": "no query vectors available (is SPECTER2 loadable?)"}
    return {
        "n_queries": used,
        "k": k,
        "n_vectors": len(ids),
        "identical_top_k_order": f"{same_order}/{used}",
        "identical_top_k_set": f"{same_set}/{used}",
        "max_abs_score_difference": max_diff,
        "hand_written_ms": {"p50": round(percentile(hand_ms, 50), 2), "p95": round(percentile(hand_ms, 95), 2)},
        "faiss_ms": {"p50": round(percentile(faiss_ms, 50), 3), "p95": round(percentile(faiss_ms, 95), 3)},
        "note": "score differences of ~1e-7 are float32-versus-float64 rounding; order can differ "
        "only between near-tied vectors",
    }


def clustering(conn: Any, collection_id: str) -> dict[str, Any]:
    try:
        import numpy as np
        from sklearn.metrics import silhouette_score
    except ImportError as exc:
        return {"error": f"scikit-learn/numpy not installed ({exc})"}

    from services.map import build_map, select_vectors

    payload = build_map(conn, collection_id)
    ids, vectors, _, source = select_vectors(conn, collection_id)
    label_of = {p.paper_id: p.cluster for p in payload.points}
    labels = np.asarray([label_of[i] for i in ids])
    sizes: dict[int, int] = {}
    for label in labels.tolist():
        sizes[label] = sizes.get(label, 0) + 1
    real = {k: v for k, v in sizes.items() if k >= 0}
    mask = labels >= 0
    silhouette = None
    if len(real) >= 2 and mask.sum() > len(real):
        silhouette = float(
            silhouette_score(np.asarray(vectors)[mask], labels[mask], metric="cosine")
        )
    largest = max(real.values()) if real else 0
    return {
        "vector_source": source,
        "n_papers": len(ids),
        "n_clusters": len(real),
        "cluster_sizes": {str(k): v for k, v in sorted(sizes.items())},
        "noise_fraction": round(sizes.get(-1, 0) / max(1, len(ids)), 3),
        "largest_cluster_share": round(largest / max(1, len(ids)), 3),
        "silhouette_cosine": silhouette,
        "caveat": "Silhouette is measured on the original embedding space but the clusters were "
        "found in a 10-D UMAP projection, which is not optimised for it, so low values are expected. "
        "With few clusters it mostly reflects one split; a largest-cluster share near 1 means the "
        "map found one topic, not sub-topics. Judge cluster quality by reading the labels.",
    }


_REF_HEADING = re.compile(
    r"^[ \t]*(?:\d+\.?[ \t]*|[A-Z]\.[ \t]+)?(?:references|bibliography|literature cited)[ \t]*:?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)


def _squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _reference_text(pdf_bytes: bytes) -> str | None:
    import pymupdf

    document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        pages = [page.get_text() for page in document]
    finally:
        document.close()
    for number in range(int(len(pages) * 0.4), len(pages)):
        match = _REF_HEADING.search(pages[number])
        if match:
            return pages[number][match.end() :] + "\n".join(pages[number + 1 :])
    return None


def graph_corroboration(
    conn: Any, collection_id: str, cache_dir: str | Path, *, sample: int = 300
) -> dict[str, Any]:
    from ingest.cache import Cache
    from ingest.ids import arxiv_pdf_url

    min_squashed = 20
    cache = Cache(cache_dir)
    rows = conn.execute(
        """
        SELECT e.src_paper_id, COALESCE(e.dst_title, d.title) AS dst_title, e.resolved,
               p.arxiv_id, p.title AS src_title
        FROM edges e
        JOIN papers p ON p.id = e.src_paper_id
        LEFT JOIN papers d ON d.id = e.dst_paper_id
        WHERE e.collection_id=? AND p.arxiv_id IS NOT NULL
          AND COALESCE(e.dst_title, d.title) IS NOT NULL
        ORDER BY e.id
        """,
        (collection_id,),
    ).fetchall()
    rng = random.Random(0)
    if len(rows) > sample:
        rows = rng.sample(list(rows), sample)

    references: dict[str, str | None] = {}
    matched = checked = no_pdf = no_reference_section = too_short = 0
    misses: list[dict[str, str]] = []
    internal_matched = internal_checked = 0
    for row in rows:
        src = row["src_paper_id"]
        if src not in references:
            data = cache.get(arxiv_pdf_url(row["arxiv_id"]))
            references[src] = _squash(_reference_text(data) or "") if data else None
            if data is None:
                references[src] = "<no-pdf>"
            elif not references[src]:
                references[src] = "<no-references>"
        squashed = references[src]
        if squashed == "<no-pdf>":
            no_pdf += 1
            continue
        if squashed == "<no-references>":
            no_reference_section += 1
            continue
        wanted = _squash(row["dst_title"])
        if len(wanted) < min_squashed:
            too_short += 1
            continue
        checked += 1
        hit = wanted in squashed
        matched += hit
        if row["resolved"]:
            internal_checked += 1
            internal_matched += hit
        if not hit and len(misses) < 10:
            misses.append({"citing": row["src_title"][:80], "cited": row["dst_title"][:80]})
    return {
        "edges_checked": checked,
        "corroborated": matched,
        "corroboration_rate": round(matched / checked, 3) if checked else None,
        "internal_edges_checked": internal_checked,
        "internal_corroborated": internal_matched,
        "skipped_pdf_not_cached": no_pdf,
        "skipped_no_reference_section": no_reference_section,
        "skipped_title_too_short": too_short,
        "sample_of": len(rows),
        "unmatched_examples": misses,
        "note": "precision of OpenAlex edges against the PDFs' own reference lists; an unmatched edge "
        "may be a title formatted differently in the reference list, not a wrong edge",
    }


def _blind_order(paper_id: str, query: str) -> str:
    return hashlib.sha256(f"{query}|{paper_id}".encode()).hexdigest()


def retrieval_pool(
    conn: Any,
    collection_id: str,
    queries: Sequence[str],
    vector_fn: VectorFn,
    *,
    per_mode: int = 10,
) -> dict[str, Any]:
    from services.retrieval import paper_details

    out: list[dict[str, Any]] = []
    for query in queries:
        vec = vector_fn(query)
        pool: dict[str, str] = {}
        for mode in MODES:
            if mode in {"dense", "hybrid"} and vec is None:
                continue
            result = rank_papers(
                conn, collection_id, query=query, mode=mode, top_k=per_mode, query_vector=vec
            )
            for paper in result.papers:
                pool.setdefault(paper.paper_id, paper.title)
        details = paper_details(conn, list(pool))
        candidates = [
            {
                "paper_id": pid,
                "title": title,
                "abstract": (details.get(pid, {}).get("abstract") or "")[:600],
            }
            for pid, title in sorted(pool.items(), key=lambda kv: _blind_order(kv[0], query))
        ]
        out.append({"query": query, "judged": False, "relevant": [], "candidates": candidates})
    return {
        "collection_id": collection_id,
        "instructions": "For each query, put the paper_id of every candidate you judge relevant "
        "into 'relevant' and set 'judged' to true. Set 'judged_by' to who did the judging.",
        "judged_by": None,
        "queries": out,
    }


def _id_resolver(conn: Any, collection_id: str) -> Callable[[str], "str | None"]:
    by_arxiv = {
        row["arxiv_id"]: row["id"]
        for row in conn.execute(
            "SELECT id, arxiv_id FROM papers WHERE collection_id=? AND arxiv_id IS NOT NULL", (collection_id,)
        )
    }
    paper_ids = {row["id"] for row in conn.execute("SELECT id FROM papers WHERE collection_id=?", (collection_id,))}
    return lambda value: value if value in paper_ids else by_arxiv.get(value)


def retrieval_metrics(
    conn: Any, collection_id: str, labels: dict[str, Any], vector_fn: VectorFn
) -> dict[str, Any]:
    judged = [q for q in labels.get("queries", []) if q.get("judged")]
    resolve = _id_resolver(conn, collection_id)
    judged = [{**q, "relevant": [resolve(r) for r in q.get("relevant", []) if resolve(r)]} for q in judged]
    if not any(q.get("relevant") for q in judged):
        raise NoLabels(
            "no judged queries with relevant papers: run `eval pool`, label the sheet, then rerun"
        )

    per_mode: dict[str, dict[str, list[float]]] = {
        m: {"p@5": [], "p@10": [], "r@10": [], "r@20": [], "ndcg@10": [], "mrr": []} for m in MODES
    }
    skipped_dense = 0
    for entry in judged:
        relevant = set(entry["relevant"])
        vec = vector_fn(entry["query"])
        for mode in MODES:
            if mode in {"dense", "hybrid"} and vec is None:
                skipped_dense += 1
                continue
            ranked = [
                p.paper_id
                for p in rank_papers(
                    conn, collection_id, query=entry["query"], mode=mode, top_k=50, query_vector=vec
                ).papers
            ]
            bucket = per_mode[mode]
            bucket["p@5"].append(precision_at_k(ranked, relevant, 5))
            bucket["p@10"].append(precision_at_k(ranked, relevant, 10))
            bucket["mrr"].append(reciprocal_rank(ranked, relevant))
            if relevant:
                bucket["r@10"].append(recall_at_k(ranked, relevant, 10))
                bucket["r@20"].append(recall_at_k(ranked, relevant, 20))
                bucket["ndcg@10"].append(ndcg_at_k(ranked, relevant, 10))
    return {
        "judged_by": labels.get("judged_by"),
        "n_judged_queries": len(judged),
        "n_judged_with_no_relevant_paper": sum(1 for q in judged if not q.get("relevant")),
        "n_unjudged_queries": len(labels.get("queries", [])) - len(judged),
        "queries_per_mode": {m: len(per_mode[m]["p@5"]) for m in MODES},
        "skipped_dense_runs": skipped_dense,
        "modes": {
            mode: {metric: round(mean(values), 4) for metric, values in buckets.items() if values}
            for mode, buckets in per_mode.items()
        },
        "caveat": "Relevance is judged over the pooled top results of these same four modes, so a "
        "relevant paper none of them retrieved is invisible to every metric (pooling bias). "
        "Recall is therefore recall within the pool, not recall over the whole literature.",
    }


def calibrate_tau(
    conn: Any, collection_id: str, labelled: Sequence[dict[str, Any]], vector_fn: VectorFn,
    *, mode: str = "dense",
) -> dict[str, Any]:
    index, retriever = cached_index(conn, collection_id)
    scored: list[tuple[float, bool]] = []
    for item in labelled:
        if not isinstance(item.get("answerable"), bool):
            continue
        vec = vector_fn(item["question"]) if mode in {"dense", "hybrid"} else None
        if mode in {"dense", "hybrid"} and vec is None:
            continue
        hits = retriever.search(item["question"], k=1, mode=mode, query_vector=vec).hits
        scored.append((hits[0].score if hits else 0.0, bool(item["answerable"])))
    best = best_threshold(scored)
    if best is None:
        raise NoLabels("need both answerable and unanswerable labelled questions")
    return {"mode": mode, "n_questions": len(scored), "suggested": best,
            "set_in_env": f"REFUSAL_TAU={best['tau']!r} REFUSAL_TAU_MODE={mode}",
            "note": "RRF scores are rank-based (a question whose BM25 and dense top results coincide "
            "scores exactly 2/61 however weak the match), so calibrate on dense cosine or BM25"}


def _passage_text(conn: Any, chunk_id: str | None) -> str | None:
    if not chunk_id:
        return None
    row = conn.execute("SELECT text FROM chunks WHERE id=?", (chunk_id,)).fetchone()
    return row["text"] if row else None


def faithfulness_pool(conn: Any, collection_id: str, *, n: int = 30, seed: int = 0) -> dict[str, Any]:
    rows = conn.execute(
        "SELECT id, question, answer, citations FROM answers "
        "WHERE collection_id=? AND refused=0 AND answer IS NOT NULL ORDER BY id",
        (collection_id,),
    ).fetchall()
    claims: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if (row["question"], row["answer"]) in seen:
            continue
        seen.add((row["question"], row["answer"]))
        cites = {c["n"]: c for c in json.loads(row["citations"] or "[]")}
        for sentence in re.split(r"(?<=[.!?])\s+", row["answer"]):
            numbers = [int(x) for x in re.findall(r"\[(\d+)\]", sentence)]
            if numbers:
                claims.append({
                    "answer_id": row["id"], "question": row["question"], "claim": sentence,
                    "cited": [{"n": n_, "title": cites[n_]["title"], "quote": cites[n_]["quote"],
                               "passage": _passage_text(conn, cites[n_].get("chunk_id")) or cites[n_]["quote"]}
                              for n_ in numbers if n_ in cites],
                    "supported": None,
                })
    if not claims:
        raise NoLabels("no stored answers with cited claims yet; ask questions first")
    random.Random(seed).shuffle(claims)
    return {"instructions": "Set 'supported' to true/false for each claim after reading the quote.",
            "claims": claims[:n]}


def faithfulness_score(adjudicated: dict[str, Any]) -> dict[str, Any]:
    judged = [c for c in adjudicated.get("claims", []) if c.get("supported") in (True, False)]
    if not judged:
        raise NoLabels("no adjudicated claims: set 'supported' to true/false on the pool")
    unsupported = sum(1 for c in judged if c["supported"] is False)
    return {"n_judged": len(judged), "unsupported": unsupported,
            "unsupported_rate": round(unsupported / len(judged), 4)}


def retrieval_bootstrap(
    conn: Any, collection_id: str, labels: dict[str, Any], vector_fn: VectorFn,
    *, resamples: int = 5000, seed: int = 0,
) -> dict[str, Any]:
    resolve = _id_resolver(conn, collection_id)
    per_query: dict[str, dict[str, list[float]]] = {
        m: {"ndcg": [], "mrr": []} for m in ("lexical", "dense", "hybrid")
    }
    judged_total = sum(1 for q in labels.get("queries", []) if q.get("judged"))
    used = 0
    for entry in labels.get("queries", []):
        relevant = {resolve(r) for r in entry.get("relevant", []) if entry.get("judged") and resolve(r)}
        if not relevant:
            continue
        vec = vector_fn(entry["query"])
        if vec is None:
            raise NoLabels("query vectors are unavailable (is SPECTER2 loadable?); dense/hybrid cannot be compared")
        used += 1
        for mode, bucket in per_query.items():
            ranked = [
                p.paper_id
                for p in rank_papers(
                    conn, collection_id, query=entry["query"], mode=mode, top_k=50, query_vector=vec
                ).papers
            ]
            bucket["ndcg"].append(ndcg_at_k(ranked, relevant, 10))
            bucket["mrr"].append(reciprocal_rank(ranked, relevant))
    if not used:
        raise NoLabels("no judged queries with relevant papers: nothing to bootstrap")
    comparisons = {}
    for a, b in (("hybrid", "dense"), ("hybrid", "lexical"), ("lexical", "dense")):
        for metric in ("ndcg", "mrr"):
            comparisons[f"{a} - {b} ({metric})"] = {
                **paired_bootstrap(per_query[a][metric], per_query[b][metric], resamples=resamples, seed=seed),
                "wilcoxon_p_two_sided": _wilcoxon_p(per_query[a][metric], per_query[b][metric]),
            }
    return {
        "n_queries_with_a_relevant_paper": used,
        "n_judged_queries": judged_total,
        "mrr_scope": f"MRR here is over the {used} queries with a relevant paper; the 'retrieval' section's MRR "
        f"counts all {judged_total} judged queries (a query with no relevant paper scores 0), so the two differ",
        "method": f"paired bootstrap over queries, {resamples} resamples, 95% percentile interval of the "
        "mean difference; Wilcoxon signed-rank p-value alongside (scipy, two-sided)",
        "resamples": resamples,
        "seed": seed,
        "judged_by": labels.get("judged_by"),
        "comparisons": comparisons,
        "caveat": f"Descriptive, not a hypothesis test: {len(comparisons)} comparisons are made with no "
        f"multiple-comparison correction, on only {used} queries, where a percentile bootstrap can "
        "undercover (intervals may be too narrow), so 'excludes_zero' is a description of the interval, "
        "not significance. The candidate pool the labels were built on comes from these same retrieval "
        "modes, and the labels are provisional (not human). An interval that includes zero means "
        "'not distinguishable here', not 'equal'.",
    }


def _wilcoxon_p(a: Sequence[float], b: Sequence[float]) -> float | None:
    try:
        import warnings

        from scipy.stats import wilcoxon

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = float(wilcoxon(list(a), list(b)).pvalue)
        return None if p != p else round(p, 4)
    except (ImportError, ValueError):
        return None


BERT_MODEL = "roberta-large"


def _versions(*names: str) -> dict[str, str | None]:
    from importlib import metadata

    out: dict[str, str | None] = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def bertscore_f1(
    candidates: Sequence[str], references: Sequence[str], *, model_type: str = BERT_MODEL, batch_size: int = 8
) -> dict[str, Any]:
    from bert_score import BERTScorer

    scorer = BERTScorer(
        model_type=model_type, lang="en", rescale_with_baseline=False, device="cpu", batch_size=batch_size
    )
    _, _, f1 = scorer.score(list(candidates), list(references))
    raw = [float(x) for x in f1]
    try:
        baseline = float(scorer.baseline_vals[2])
        rescaled = [(x - baseline) / (1 - baseline) for x in raw]
    except (AssertionError, FileNotFoundError, KeyError, IndexError, ValueError):
        baseline, rescaled = None, None
    return {
        "raw_f1": raw, "rescaled_f1": rescaled, "baseline_f1": baseline, "model_type": model_type,
        "num_layers": scorer.num_layers, "idf": False, "device": "cpu",
    }


def summary_sample(conn: Any, collection_id: str, n: int) -> list[Any]:
    papers = [p for p in sorted(load_papers(conn, collection_id), key=lambda p: p.paper_id) if p.abstract]
    stride = max(1, len(papers) // max(1, n))
    return papers[::stride][:n]


def summary_scores(
    conn: Any, collection_id: str, llm: Any, *, n: int = 20,
    bert_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    from rouge_score import rouge_scorer

    from services.llm import LLMUnavailable
    from services.summarize import paper_summary

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rows: list[dict[str, Any]] = []
    candidates: list[str] = []
    references: list[str] = []
    stopped: str | None = None
    for paper in summary_sample(conn, collection_id, n):
        try:
            result = paper_summary(conn, paper.paper_id, llm=llm)
        except LLMUnavailable as exc:
            stopped = exc.code
            break
        rouge = scorer.score(paper.abstract, result["summary"])["rougeL"].fmeasure
        candidates.append(result["summary"])
        references.append(paper.abstract)
        rows.append({
            "paper_id": paper.paper_id, "rougeL_f1": round(rouge, 4), "cached": result["cached"],
            "n_unsupported_numbers": len(result["unsupported_numbers"]),
        })

    bert = None
    if rows:
        scored = (bert_fn or bertscore_f1)(candidates, references)
        for row, raw in zip(rows, scored["raw_f1"]):
            row["bertscore_f1_raw"] = round(raw, 4)
        if scored.get("rescaled_f1"):
            for row, rescaled in zip(rows, scored["rescaled_f1"]):
                row["bertscore_f1_rescaled"] = round(rescaled, 4)
        bert = {
            "model_type": scored["model_type"],
            "num_layers": scored.get("num_layers"),
            "f1_raw_mean": round(mean(scored["raw_f1"]), 4),
            "f1_rescaled_mean": round(mean(scored["rescaled_f1"]), 4) if scored.get("rescaled_f1") else None,
            "baseline_f1": scored.get("baseline_f1"),
            "rescaling": "(f1 - baseline) / (1 - baseline), the bert-score package's English baseline for "
            "this model and layer; raw F1 is also reported because raw values sit in a narrow high band",
            "idf": scored.get("idf", False),
            "device": scored.get("device", "cpu"),
        }
    return {
        "n_requested": n,
        "n": len(rows),
        "stopped_early": stopped,
        "sampling": "papers with an abstract, sorted by paper id, every len//n-th, first n",
        "reference": "the paper's stored abstract",
        "rougeL_f1_mean": round(mean([r["rougeL_f1"] for r in rows]), 4) if rows else None,
        "rouge_config": "rouge-score RougeScorer(['rougeL'], use_stemmer=True), F-measure, summary text scored "
        "as returned (the 'Problem:/Method:/Result:' labels are included)",
        "bertscore": bert,
        "summaries_with_unsupported_numbers": sum(1 for r in rows if r["n_unsupported_numbers"]),
        "versions": _versions("rouge-score", "bert-score", "torch", "transformers"),
        "model": getattr(llm, "model", None),
        "per_paper": rows,
        "caveat": "Both scores are floors, not quality. A structured three-line summary is deliberately unlike "
        "the abstract it condenses, so word overlap (ROUGE-L) and embedding similarity (BERTScore) both "
        "understate it. The abstract is part of the model's input (title + abstract + introduction/"
        "conclusion excerpts), so this is similarity to a source the model saw, not agreement with an "
        "independent reference.",
    }


def refusal_end_to_end(
    conn: Any, collection_id: str, labels: dict[str, Any], llm: Any, vector_fn: VectorFn,
    *, refusal_tau: float | None = None, tau_mode: str = "dense",
) -> dict[str, Any]:
    from services.answer import ask

    index, retriever = cached_index(conn, collection_id)
    rows: list[dict[str, Any]] = []
    scored: list[tuple[float, bool]] = []
    created: list[str] = []
    stopped: str | None = None
    labelled = [q for q in labels.get("questions", []) if isinstance(q.get("answerable"), bool)]
    skipped = len(labels.get("questions", [])) - len(labelled)
    try:
        for item in labelled:
            question = item["question"]
            vec = vector_fn(question)
            payload = ask(
                conn, collection_id, question, llm=llm, query_vector=vec,
                refusal_tau=refusal_tau, tau_mode=tau_mode,
            )
            created.append(payload["answer_id"])
            dense = None
            if vec is not None and index.dense_vectors:
                hits = retriever.search(question, k=1, mode="dense", query_vector=vec).hits
                dense = hits[0].score if hits else 0.0
                scored.append((dense, item["answerable"]))
            unavailable = payload["answer_unavailable"] is not None
            rows.append({
                "question": question,
                "answerable": item["answerable"],
                "outcome": "unavailable" if unavailable else "refused" if payload["refused"] else "answered",
                "refused": None if unavailable else bool(payload["refused"]),
                "refusal_reason": payload["refusal_reason"],
                "citation_coverage": payload["confidence"],
                "latency_ms": payload["latency_ms"],
                "cached": payload["cached"],
                "dense_top_score": dense,
            })
            if unavailable and payload["answer_unavailable"]["code"] in FATAL_LLM_CODES:
                stopped = payload["answer_unavailable"]["code"]
                break
    finally:
        deleted = _delete_answers(conn, created)

    done = [r for r in rows if r["outcome"] != "unavailable"]
    unanswerable = [r for r in done if not r["answerable"]]
    answerable = [r for r in done if r["answerable"]]
    refused_un = [r for r in unanswerable if r["refused"]]
    wrong = [r for r in answerable if r["refused"]]
    correct = len(refused_un) + len(answerable) - len(wrong)
    answered = [r for r in done if not r["refused"]]
    fresh = [r["latency_ms"] for r in done if not r["cached"]]

    threshold = None
    best = best_threshold(scored)
    if best is not None:
        tau = best["tau"]
        n_ans = sum(1 for _, a in scored if a)
        n_un = len(scored) - n_ans
        threshold = {
            "best_dense_cosine_tau": round(tau, 4), "f1": round(best["f1"], 3),
            "precision": round(best["precision"], 3), "recall": round(best["recall"], 3),
            "would_falsely_refuse_answerable": f"{sum(1 for s, a in scored if a and s < tau)}/{n_ans}",
            "would_catch_unanswerable": f"{sum(1 for s, a in scored if not a and s < tau)}/{n_un}",
            "caveat": "in-sample: the threshold is fitted and scored on the same labelled questions",
        }
    reasons: dict[str, int] = {}
    for r in done:
        if r["refused"]:
            reasons[r["refusal_reason"]] = reasons.get(r["refusal_reason"], 0) + 1
    coverage = [r["citation_coverage"] for r in answered if r["citation_coverage"] is not None]
    return {
        "judged_by": labels.get("judged_by"),
        "provider": getattr(llm, "provider", None),
        "model": getattr(llm, "model", None),
        "retrieval_mode": "hybrid",
        "refusal_tau": refusal_tau,
        "refusal_tau_mode": tau_mode,
        "n_questions_requested": len(labelled),
        "n_questions": len(rows),
        "n_skipped_unlabelled": skipped,
        "n_unavailable": len(rows) - len(done),
        "stopped_early": stopped,
        "unanswerable_refused": f"{len(refused_un)}/{len(unanswerable)}",
        "answerable_wrongly_refused": f"{len(wrong)}/{len(answerable)}",
        "accuracy_as_labelled": round(correct / len(done), 4) if done else None,
        "unanswerable_that_were_answered": [r["question"] for r in unanswerable if not r["refused"]],
        "refusal_reasons": reasons,
        "mean_citation_coverage_when_answered": round(mean(coverage), 4) if coverage else None,
        "n_cached": sum(1 for r in done if r["cached"]),
        "latency_ms_uncached": (
            {"n": len(fresh), "p50": round(percentile(fresh, 50)), "p95": round(percentile(fresh, 95)),
             "max": max(fresh), "note": "wall clock of the whole answer including client retries and "
             "rate-limit waits, so the tail is the provider's, not the pipeline's"}
            if fresh else None
        ),
        "score_threshold_alone": threshold,
        "answers_rows_created_and_deleted": deleted,
        "per_question": rows,
        "run": "a single stochastic run: the model is not deterministic, so counts move between runs "
        "(14/16 and 0/24 became 15/16 and 1/24 on one re-run); do not read a one-question change as a trend",
        "caveat": "The labels are not human (see judged_by) and 'answerable' is defined over abstracts while "
        "the system answers from full text, so a 'missed' unanswerable question may be answerable in a "
        "narrower setting; read unanswerable_that_were_answered before trusting the rate. The answers "
        "this run stored in the answers table were deleted again (answers_rows_created_and_deleted); "
        "responses remain in llm_cache, as for any question asked through the app.",
    }


PRICES_USD_PER_MILLION: dict[tuple[str, str], dict[str, Any]] = {
    ("gemini", "gemini-3.1-flash-lite"): {
        "input": 0.25,
        "output_including_thinking": 1.5,
        "source": "https://aistudio.google.com/docs/pricing, Gemini 3.1 Flash-Lite, Standard tier, paid; read "
        "2026-11-18 through a summarising fetch - confirm before budgeting; the free tier is $0",
    },
}


def normalise_usage(usage: dict[str, Any]) -> dict[str, Any] | None:
    if "promptTokenCount" in usage:
        return {
            "prompt": usage["promptTokenCount"], "output": usage.get("candidatesTokenCount", 0),
            "thinking": usage.get("thoughtsTokenCount", 0), "latency_ms": usage.get("latency_ms"),
        }
    if "prompt_tokens" in usage:
        reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
        return {
            "prompt": usage["prompt_tokens"], "output": usage.get("completion_tokens", 0) - reasoning,
            "thinking": reasoning, "latency_ms": usage.get("latency_ms"),
        }
    return None


class _Recorder:
    def __init__(self, llm: Any) -> None:
        self._llm = llm
        self.provider = getattr(llm, "provider", None)
        self.model = llm.model
        self.results: list[Any] = []

    def complete(self, *args: Any, **kwargs: Any) -> Any:
        result = self._llm.complete(*args, **kwargs)
        self.results.append(result)
        return result


def _token_stats(usages: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not usages:
        return {"calls": 0}
    latencies = [u["latency_ms"] for u in usages if u["latency_ms"] is not None]
    stats: dict[str, Any] = {"calls": len(usages)}
    for key, name in (("prompt", "prompt_tokens"), ("output", "output_tokens"), ("thinking", "thinking_tokens")):
        stats[f"{name}_mean"] = round(mean([u[key] for u in usages]))
    if latencies:
        stats["latency_ms_p50"] = round(percentile(latencies, 50))
        stats["latency_ms_max"] = max(latencies)
    for key, name in (("prompt", "prompt_tokens"), ("output", "output_tokens"), ("thinking", "thinking_tokens")):
        stats[f"{name}_total"] = sum(u[key] for u in usages)
    return stats


def cost_per_query(
    conn: Any, collection_id: str, labels: dict[str, Any], llm: Any, vector_fn: VectorFn,
    *, n_answers: int = 10, n_summaries: int = 10,
) -> dict[str, Any]:
    from services.answer import ask
    from services.llm import LLMUnavailable
    from services.summarize import paper_summary

    recorder = _Recorder(llm)
    pool = [q["question"] for q in labels.get("questions", []) if q.get("answerable") is True]
    questions = pool[:: max(1, len(pool) // max(1, n_answers))][:n_answers]
    refused = unavailable = 0
    stopped: str | None = None
    created: list[str] = []
    n_answer_calls = 0
    try:
        for question in questions:
            payload = ask(conn, collection_id, question, llm=recorder, query_vector=vector_fn(question))
            created.append(payload["answer_id"])
            refused += bool(payload["refused"])
            if payload["answer_unavailable"] is not None:
                unavailable += 1
                if payload["answer_unavailable"]["code"] in FATAL_LLM_CODES:
                    stopped = payload["answer_unavailable"]["code"]
                    break
        n_answer_calls = len(recorder.results)
        for paper in [] if stopped else summary_sample(conn, collection_id, n_summaries):
            try:
                paper_summary(conn, paper.paper_id, llm=recorder)
            except LLMUnavailable as exc:
                unavailable += 1
                if exc.code in FATAL_LLM_CODES:
                    stopped = exc.code
                    break
    finally:
        deleted = _delete_answers(conn, created)

    def usable(results: Sequence[Any]) -> list[dict[str, Any]]:
        normalised = [normalise_usage(r.usage) for r in results if not r.cached and r.usage]
        return [u for u in normalised if u is not None]

    answers = usable(recorder.results[:n_answer_calls])
    summaries = usable(recorder.results[n_answer_calls:])
    fresh = [r for r in recorder.results if not r.cached]
    out: dict[str, Any] = {
        "provider": getattr(llm, "provider", None),
        "model": llm.model,
        "measured_on": "uncached live calls only (cached repeats cost nothing and are not counted)",
        "per_answered_question": _token_stats(answers),
        "per_summary": _token_stats(summaries),
        "questions_run": len(created),
        "questions_requested": len(questions),
        "refused": refused,
        "llm_unavailable": unavailable,
        "stopped_early": stopped,
        "excluded_cached_calls": sum(1 for r in recorder.results if r.cached),
        "excluded_calls_without_usage": len(fresh) - len(answers) - len(summaries),
        "answers_rows_created_and_deleted": deleted,
        "note": "tokens are as reported by the provider (usageMetadata / usage); nothing is estimated. "
        "Latency is the wall clock of the whole call including client retries and rate-limit waits. "
        "Provider usage is a single stochastic run of a small sample, not a stable mean.",
    }
    price = PRICES_USD_PER_MILLION.get((str(out["provider"]), llm.model))
    out["price_used_usd_per_million_tokens"] = price
    if price is None:
        out["price_note"] = "no confirmed price is recorded for this model, so tokens only"
        return out
    out["usd_note"] = ("USD is an upper bound at list price: provider-side implicit caching "
                       "(cachedContentTokenCount) is not read, and the free tier is $0")
    for key, name in (("usd_per_answered_question_if_billed", "per_answered_question"),
                      ("usd_per_summary_if_billed", "per_summary")):
        stats = out[name]
        if stats["calls"]:
            billed = (
                stats["prompt_tokens_total"] * price["input"]
                + (stats["output_tokens_total"] + stats["thinking_tokens_total"])
                * price["output_including_thinking"]
            )
            out[key] = round(billed / stats["calls"] / 1e6, 6)
    if "usd_per_answered_question_if_billed" in out:
        out["usd_per_1000_answered_questions_if_billed"] = round(
            out["usd_per_answered_question_if_billed"] * 1000, 3
        )
    return out
