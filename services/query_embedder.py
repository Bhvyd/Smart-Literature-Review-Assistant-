
from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)


class QueryEmbedder:
    def __init__(self, model_dir: str | None = None, *, embedder: object | None = None) -> None:
        self.model_dir = model_dir
        self._embedder: object | None = embedder
        self._lock = threading.Lock()
        self._unavailable = False
        self.last_error: str | None = None

    @property
    def available(self) -> bool:
        return not self._unavailable

    def _get(self) -> object | None:
        with self._lock:
            if self._unavailable:
                return None
            if self._embedder is None:
                try:
                    from embed.specter2 import Specter2Embedder

                    self._embedder = Specter2Embedder(model_dir=self.model_dir)
                except Exception as exc:
                    self._unavailable = True
                    self.last_error = f"{type(exc).__name__}: {exc}"
                    return None
            return self._embedder

    def vector(self, text: str) -> list[float] | None:
        if not text.strip():
            return None

        embedder = self._get()
        if embedder is None:
            return None
        try:
            vectors = embedder.embed_queries([text])
        except Exception as exc:
            from embed.specter2 import EmbedderUnavailable

            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("query embedding failed: %s", self.last_error)
            if isinstance(exc, EmbedderUnavailable):
                self._unavailable = True
            return None
        self.last_error = None
        return vectors[0] if vectors else None

    def warm(self) -> bool:
        return self.vector("warm up") is not None
