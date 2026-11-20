
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from .tokenize import tokenize as _tokenize


class InvertedIndex:
    def __init__(self) -> None:
        self._tf: dict[str, dict[str, int]] = {}
        self._postings: dict[str, dict[str, int]] = {}
        self._length: dict[str, int] = {}
        self._norm: dict[str, float] = {}
        self._df: dict[str, int] = {}
        self._avgdl: float = 0.0
        self._finalized: bool = False

    def add(self, doc_id: str, text: str | Sequence[str]) -> None:
        if self._finalized:
            raise RuntimeError(
                "index is finalised; adding documents after finalize() would invalidate cached scores"
            )
        if doc_id in self._tf:
            raise ValueError(f"duplicate doc_id: {doc_id!r}")

        tokens = _tokenize(text) if isinstance(text, str) else list(text)

        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1

        self._tf[doc_id] = counts
        self._length[doc_id] = len(tokens)
        for term, freq in counts.items():
            self._postings.setdefault(term, {})[doc_id] = freq

    def finalize(self) -> None:
        self._df = {term: len(postings) for term, postings in self._postings.items()}
        total = sum(self._length.values())
        self._avgdl = total / len(self._tf) if self._tf else 0.0
        n = len(self._tf)

        def idf(term: str) -> float:
            return math.log((n + 1.0) / (self._df.get(term, 0) + 1.0)) + 1.0

        self._norm = {
            doc_id: math.sqrt(
                math.fsum((freq * idf(term)) ** 2 for term, freq in counts.items())
            )
            for doc_id, counts in self._tf.items()
        }
        self._finalized = True

    @property
    def n_docs(self) -> int:
        return len(self._tf)

    @property
    def avgdl(self) -> float:
        self._require_finalized()
        return self._avgdl

    def df(self, term: str) -> int:
        self._require_finalized()
        return self._df.get(term, 0)

    def idf_bm25(self, term: str) -> float:
        self._require_finalized()
        df = self._df.get(term, 0)
        n = len(self._tf)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def idf_tfidf(self, term: str) -> float:
        self._require_finalized()
        df = self._df.get(term, 0)
        return math.log((len(self._tf) + 1.0) / (df + 1.0)) + 1.0

    def doc_length(self, doc_id: str) -> int:
        self._require_finalized()
        return self._length[doc_id]

    def doc_norm(self, doc_id: str) -> float:
        self._require_finalized()
        return self._norm[doc_id]

    def postings(self, term: str) -> Mapping[str, int]:
        self._require_finalized()
        return self._postings.get(term, {})

    def doc_ids(self) -> list[str]:
        return list(self._tf)

    def _require_finalized(self) -> None:
        if not self._finalized:
            raise RuntimeError("call finalize() before querying the index")
