# `rag_core` — formula map

The graded artefact is the retrieval engine in this package. Everything below is implemented in plain
Python; the only imports are the standard library, enforced by `tests/test_import_guard.py`.

This file exists so that every claim about the engine can be traced from a formula to a function to a
line of code, without reading the whole package.

## Formula → function → file

| # | Component (`PROJECT.md` §8.2) | Formula / rule | Function | File |
|---|---|---|---|---|
| 1 | Tokenisation | NFKD → strip combining marks → casefold → drop apostrophes → `[a-z0-9]+(?:[.\-][a-z0-9]+)*` → length ≥ 2 → stopword filter | `normalize`, `tokenize` | `tokenize.py` |
| 2 | Inverted index | postings `t → {d: tf(t,d)}`, `df(t) = \|postings(t)\|`, `\|d\| = Σ tf`, `avgdl = Σ\|d\| / N` | `InvertedIndex.add`, `.finalize`, `.postings`, `.df`, `.doc_length`, `.avgdl` | `index.py` |
| 3 | BM25 IDF | `idf(t) = ln(1 + (N − df(t) + 0.5) / (df(t) + 0.5))` | `InvertedIndex.idf_bm25` | `index.py` |
| 4 | TF-IDF IDF | `idf(t) = ln((N + 1) / (df(t) + 1)) + 1` | `InvertedIndex.idf_tfidf` | `index.py` |
| 5 | L2 normalisation (lexical) | `‖d‖ = sqrt(Σ_t (tf(t,d) · idf(t))²)` | `InvertedIndex.finalize` → `doc_norm` | `index.py` |
| 6 | BM25 score | `Σ_{t∈q} idf(t) · tf(t,d)(k₁+1) / (tf(t,d) + k₁(1 − b + b·\|d\|/avgdl))`, `k₁ = 1.5`, `b = 0.75` | `bm25_score`, `bm25_search` | `lexical.py` |
| 7 | TF-IDF cosine | `cos(q,d) = (q̂ · d) / ‖d‖` with `q̂` the L2-normalised idf-weighted query vector, accumulated sparsely over query-term postings | `tfidf_cosine_search` | `lexical.py` |
| 8 | Dense cosine | `cos(a,b) = Σ aᵢbᵢ / (‖a‖‖b‖)`; `‖d‖` precomputed per index | `dot`, `l2_norm`, `cosine`, `document_norms`, `cosine_search` | `dense.py` |
| 9 | Weighted fusion | per-source min-max to `[0,1]`, multiply by weight, sum per document | `min_max_normalise`, `weighted_fusion` | `fuse.py` |
| 10 | Reciprocal Rank Fusion | `score(d) = Σ_lists 1 / (k + rank(d))`, `k = 60`, rank 1-based | `rrf` | `fuse.py` |
| 11 | Top-k | sort by `(−score, doc_id)`; `k ≥ n` sorts fully, `k < n` uses `heapq.nsmallest`, `k ≤ 0` returns nothing | `top_k` | `select.py` |
| 12 | Prompt assembly | numbered slots `[n]`, whole passages in order until `max_chars`, first passage always kept | `build_prompt` | `prompt.py` |
| 13 | Facade | `lexical` / `tfidf` / `dense` / `hybrid` dispatch; hybrid = RRF over BM25 + dense | `Retriever.build`, `.search` | `pipeline.py` |

## Decisions that are deliberate, not oversights

**No stemming.** Stemming changes every term frequency, which changes every number in the BM25 and
TF-IDF tests and every score we would report. The brief is to defend the maths as written, so the
tokenizer keeps surface forms. The cost is that `retrieval` and `retrieve` are different terms; the
benefit is that the hand-computed worked examples in the viva are reproducible.

**NFKD (compatibility decomposition) rather than NFC.** `PROJECT.md` §8.3 says "unicode NFKC"; the
implementation uses NFKD and then drops combining marks, which is what actually makes `café` match
`cafe`. NFKC would leave the accent in place. The requirement was accent-insensitivity, and this is the
composition that delivers it.

**Repeats are preserved by `tokenize`.** Term frequency is the signal BM25 is built on. A de-duplicating
tokenizer would silently turn every `tf` into 1.

**English-centric, honestly.** CJK text without whitespace yields no tokens instead of mojibake. The
corpus is English arXiv; a tokenizer that pretended otherwise would produce confident garbage matches.

**Dense search drops non-positive scores.** Sparse lexical search can only emit documents that share a
query term, so a zero cosine document is not a weak match — it is no match. Returning the whole corpus
ranked by nothing would be a worse failure than returning nothing.

**A TF-IDF query term that occurs in no document is dropped before weighting.** Its idf is the largest
possible (`ln(N+1) + 1`), so keeping it would inflate `‖q‖` and shrink *every* cosine for that query.
Measured on the real corpus: adding one made-up word moved a top score from 0.2604 to 0.1469 while
leaving the ordering unchanged. After the change the scores equal scikit-learn's `TfidfVectorizer`
(smooth idf, L2) to 1e-13 on 19 queries, and scores are comparable across queries.

**Document norms are computed once per index, not once per query.** `‖d‖` depends only on the document,
so `document_norms(docs)` builds `{doc_id: ‖d‖}` once (`Retriever._doc_norms`, lazily) and
`cosine_search(query, docs, k, norms=...)` then spends one dot product per document instead of a dot
product plus two norms. The arithmetic is the same expression as `cosine`, so no score changes
(`tests/test_remediation.py::test_precomputed_norms_give_identical_rankings`). Measured on 3,284 real
chunk vectors: 190 ms per query (p50).

**`weighted_fusion` is not the default.** Min-max normalisation collapses a single-document source to
`0.0`, because there is no spread to scale. RRF has no such dependency, which is why it is what
`docs/FLOW.md` §7 names as the product default.

## Complexity

| Operation | Cost |
|---|---|
| `tokenize` | `O(|text|)` |
| `add` | `O(|tokens|)` |
| `finalize` | `O(total tokens)` |
| `bm25_search` | `O(|q| × postings(q))` — one pass, term-major |
| `tfidf_cosine_search` | `O(|q| × postings(q))` — sparse, no full scan |
| `cosine_search` | `O(N × D)` — brute force by definition |
| `rrf` | `O(Σ\|ranking\|)` |
| `top_k` | `O(n log k)` for `k < n`, `O(n log n)` otherwise |

`cosine_search` over 5,000 chunks × 768 dimensions is ~3.8M multiply-adds in interpreted Python. That
is expected to be slow (`PROJECT.md` §8.6 says to measure and report it — not to hide it).

**Measured on the real 100-paper collection (3,284 chunks × 768-d, warm index, 30 queries drawn from
real titles; `python -m cli eval latency` / `eval faiss`, stored in `data/eval/<collection>.json`):**

| Path | p50 | §8.6 budget |
|---|---|---|
| BM25 (`lexical`), paper-ranked | 9 ms | ≤ 50 ms |
| hand-written dense | 135 ms | ≤ 2 s |
| hybrid (BM25 + dense + RRF), **no FAISS** | 144 ms | ≤ 200 ms *with* FAISS |
| FAISS `IndexFlatIP` for the same top-10 | 0.6 ms | — |

The hand-written scorer and FAISS return the **same top-10 in the same order on 30 of 30 queries**, with
a worst score difference of 2.5e-7 (float32 against float64). So the hand-written code is verified
against an independent implementation, and the price of writing it by hand is ~300× FAISS's latency —
which is still well inside the budget at this scale.

## Tests that pin the maths

| Test file | What it proves |
|---|---|
| `tests/test_tokenize.py` | diacritic folding, hyphen/decimal retention, repeats preserved, stopword-only → `[]` |
| `tests/test_index.py` | `df`, both IDF tables, norms, zero-token documents, no mutation after `finalize` |
| `tests/test_bm25.py` | hand-computed BM25 against `d1 = "beta beta"`, `d2 = "beta gamma"` |
| `tests/test_tfidf.py` | hand-derived cosines, zero-norm documents, sparse behaviour |
| `tests/test_dense.py` | known cosines, zero vectors, dimension mismatch, non-positive filtering |
| `tests/test_fuse.py` | min-max edges, weights, RRF rank positions, tie-breaking |
| `tests/test_select.py` | `(−score, doc_id)` ordering independent of insertion order |
| `tests/test_prompt.py` | slot numbering, citation maps, truncation, empty-input refusal |
| `tests/test_pipeline.py` | mode dispatch, missing-input errors, hybrid fusion |
| `tests/test_import_guard.py` | the scoring path imports stdlib only and never the rest of the project |
