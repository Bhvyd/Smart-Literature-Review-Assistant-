
from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

BASE_MODEL = "allenai/specter2_base"
PROXIMITY_ADAPTER = "allenai/specter2"
QUERY_ADAPTER = "allenai/specter2_adhoc_query"
EMBED_DIM = 768
MAX_TOKENS = 512
DEFAULT_BATCH_SIZE = 8

_DOC_NAME = "specter2"
_GLOBAL_LOAD_LOCK = threading.RLock()
_QUERY_NAME = "adhoc_query"


class EmbedderUnavailable(Exception):
    pass


class Specter2Embedder:
    def __init__(
        self,
        model_dir: str | None = None,
        *,
        base: str = BASE_MODEL,
        adapter: str = PROXIMITY_ADAPTER,
        query_adapter: str = QUERY_ADAPTER,
        device: str = "cpu",
    ) -> None:
        self.model_dir = model_dir
        self.base = base
        self.adapter = adapter
        self.query_adapter = query_adapter
        self.device = device
        self.model_name = f"{base}+{adapter}"
        self.dim = EMBED_DIM
        self._tokenizer: Any = None
        self._model: Any = None
        self._query_loaded = False
        self._load_lock = _GLOBAL_LOAD_LOCK
        self._infer_lock = threading.Lock()


    def _load(self) -> None:
        with self._load_lock:
            if self._model is not None:
                return
            try:
                from adapters import AutoAdapterModel
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise EmbedderUnavailable(
                    f"SPECTER2 dependencies are not installed: {exc}. "
                    "Install torch, transformers and adapters."
                ) from exc

            try:
                tokenizer = AutoTokenizer.from_pretrained(self.base, cache_dir=self.model_dir)
                model = AutoAdapterModel.from_pretrained(self.base, cache_dir=self.model_dir)
                model.load_adapter(self.adapter, source="hf", load_as=_DOC_NAME, set_active=True)
            except Exception as exc:
                raise EmbedderUnavailable(f"could not load {self.base}: {exc}") from exc

            if model.active_adapters is None:
                raise EmbedderUnavailable(
                    f"adapter {self.adapter} loaded but is not active; "
                    "embeddings would not be SPECTER2"
                )

            model.to(self.device)
            model.eval()
            self._tokenizer = tokenizer
            self._model = model

    def _load_query_adapter(self) -> None:
        self._load()
        with self._load_lock:
            if self._query_loaded:
                return
            try:
                self._model.load_adapter(
                    self.query_adapter, source="hf", load_as=_QUERY_NAME, set_active=False
                )
            except Exception as exc:
                raise EmbedderUnavailable(
                    f"could not load query adapter {self.query_adapter}: {exc}"
                ) from exc
            self._model.to(self.device)
            self._query_loaded = True


    @property
    def sep_token(self) -> str:
        self._load()
        return str(self._tokenizer.sep_token or "[SEP]")

    def paper_text(self, title: str, abstract: str | None) -> str:
        return f"{title}{self.sep_token}{abstract or ''}"

    def embed_texts(
        self, texts: Sequence[str], batch_size: int = DEFAULT_BATCH_SIZE
    ) -> list[list[float]]:
        return self._embed(texts, batch_size, adapter=_DOC_NAME)

    def embed_queries(
        self, texts: Sequence[str], batch_size: int = DEFAULT_BATCH_SIZE
    ) -> list[list[float]]:
        if texts:
            self._load_query_adapter()
        return self._embed(texts, batch_size, adapter=_QUERY_NAME)

    def count_tokens(self, text: str) -> int:
        self._load()
        encoded = self._tokenizer(text, truncation=False, add_special_tokens=True)
        return len(encoded["input_ids"])

    def close(self) -> None:
        with self._load_lock:
            self._model = None
            self._tokenizer = None
            self._query_loaded = False
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def __enter__(self) -> Specter2Embedder:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


    def _embed(
        self, texts: Sequence[str], batch_size: int, *, adapter: str
    ) -> list[list[float]]:
        if not texts:
            return []

        self._load()
        vectors: list[list[float]] = []
        step = max(1, batch_size)
        for start in range(0, len(texts), step):
            batch = list(texts[start : start + step])
            for vector in self._encode(batch, step, adapter):
                if len(vector) != self.dim:
                    raise ValueError(
                        f"embedding has {len(vector)} dims, expected {self.dim}"
                    )
                vectors.append(vector)
        return vectors


    def _encode(
        self, texts: Sequence[str], batch_size: int, adapter: str = _DOC_NAME
    ) -> list[list[float]]:
        import torch

        with self._infer_lock:
            self._model.set_active_adapters(adapter)
            inputs = self._tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                max_length=MAX_TOKENS,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs)
            pooled = outputs.last_hidden_state[:, 0, :]
            return pooled.detach().cpu().tolist()
