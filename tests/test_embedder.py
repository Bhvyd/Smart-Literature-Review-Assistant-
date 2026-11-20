
import os
import sys
import threading
import time
import types

import pytest

from embed.specter2 import EMBED_DIM, Specter2Embedder

RUN_REAL = pytest.mark.skipif(
    os.environ.get("RUN_MODEL_TESTS") != "1", reason="set RUN_MODEL_TESTS=1"
)


def _stub(dim=4):
    embedder = Specter2Embedder.__new__(Specter2Embedder)
    embedder.dim = dim
    embedder._model = object()
    embedder._tokenizer = object()
    embedder.device = "cpu"
    embedder.adapter = "allenai/specter2"
    embedder.query_adapter = "allenai/specter2_adhoc_query"
    embedder._query_loaded = True
    embedder._load_lock = threading.RLock()
    embedder._infer_lock = threading.Lock()
    return embedder


def test_embed_texts_batches_and_preserves_order(monkeypatch):
    embedder = _stub()
    seen = []

    def fake_encode(texts, batch_size, adapter="specter2"):
        seen.append((tuple(texts), batch_size))
        return [[float(len(t))] * 4 for t in texts]

    monkeypatch.setattr(embedder, "_encode", fake_encode)
    out = embedder.embed_texts(["a", "bb", "ccc"], batch_size=2)
    assert out == [[1.0] * 4, [2.0] * 4, [3.0] * 4]
    assert [len(s[0]) for s in seen] == [2, 1]


def test_embed_texts_empty_returns_empty_without_encoding(monkeypatch):
    embedder = _stub()
    monkeypatch.setattr(
        embedder, "_encode", lambda *a, **k: pytest.fail("must not encode")
    )
    assert embedder.embed_texts([]) == []


def test_embed_texts_rejects_dimension_drift(monkeypatch):
    embedder = _stub()
    monkeypatch.setattr(embedder, "_encode", lambda *a, **k: [[1.0, 2.0]])
    with pytest.raises(ValueError):
        embedder.embed_texts(["a"])


def test_documents_use_the_proximity_adapter_and_queries_the_adhoc_one(monkeypatch):
    embedder = _stub()
    used = []

    def fake_encode(texts, batch_size, adapter="specter2"):
        used.append(adapter)
        return [[0.0] * 4 for _ in texts]

    monkeypatch.setattr(embedder, "_encode", fake_encode)
    embedder.embed_texts(["a paper"])
    embedder.embed_queries(["a query"])
    assert used == ["specter2", "adhoc_query"]


def test_a_missing_query_adapter_is_an_error_not_a_silent_fallback(monkeypatch):
    from embed.specter2 import EmbedderUnavailable

    embedder = _stub()
    embedder._query_loaded = False

    class Model:
        def load_adapter(self, *a, **k):
            raise OSError("offline")

    embedder._model = Model()
    with pytest.raises(EmbedderUnavailable):
        embedder.embed_queries(["q"])


def test_paper_text_is_title_sep_abstract():
    embedder = _stub()
    embedder._tokenizer = types.SimpleNamespace(sep_token="[SEP]")
    assert embedder.paper_text("A Title", "An abstract.") == "A Title[SEP]An abstract."
    assert embedder.paper_text("Only title", None) == "Only title[SEP]"


def test_concurrent_loads_load_the_model_exactly_once(monkeypatch):
    loads = {"count": 0, "in_flight": 0, "max_in_flight": 0}
    guard = threading.Lock()

    class FakeModel:
        active_adapters = ("specter2",)

        @classmethod
        def from_pretrained(cls, *a, **k):
            with guard:
                loads["count"] += 1
                loads["in_flight"] += 1
                loads["max_in_flight"] = max(loads["max_in_flight"], loads["in_flight"])
            time.sleep(0.15)
            with guard:
                loads["in_flight"] -= 1
            return cls()

        def load_adapter(self, *a, **k):
            return "specter2"

        def to(self, *_):
            return self

        def eval(self):
            return self

    monkeypatch.setitem(sys.modules, "adapters", types.SimpleNamespace(AutoAdapterModel=FakeModel))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: object())
        ),
    )
    embedder = Specter2Embedder()
    threads = [threading.Thread(target=embedder._load) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert loads["count"] == 1
    assert loads["max_in_flight"] == 1


def test_construction_does_not_load_the_model():
    embedder = Specter2Embedder()
    assert embedder._model is None
    assert embedder.dim == EMBED_DIM


def test_declared_dimension_is_specter2():
    assert EMBED_DIM == 768


@RUN_REAL
def test_real_model_produces_768_dim_distinct_vectors():
    embedder = Specter2Embedder()
    vectors = embedder.embed_texts(
        [
            "Attention Is All You Need. The Transformer uses self attention mechanisms.",
            "LoRA freezes pretrained weights and injects trainable low-rank matrices.",
        ]
    )
    assert len(vectors) == 2
    assert all(len(v) == 768 for v in vectors)
    assert vectors[0] != vectors[1]


@RUN_REAL
def test_real_model_activates_the_proximity_adapter():
    embedder = Specter2Embedder()
    embedder.embed_texts(["probe"])
    assert embedder._model.active_adapters is not None


@RUN_REAL
def test_count_tokens_uses_the_real_tokenizer():
    embedder = Specter2Embedder()
    assert embedder.count_tokens("transformer") >= 1
    assert embedder.count_tokens("a b c") == 5


@RUN_REAL
def test_real_query_adapter_differs_from_the_proximity_adapter_and_both_are_768d():
    embedder = Specter2Embedder()
    text = "how does retrieval augmentation reduce hallucination"
    as_document = embedder.embed_texts([text])[0]
    as_query = embedder.embed_queries([text])[0]
    assert len(as_document) == len(as_query) == EMBED_DIM
    assert as_document != as_query
    assert embedder.embed_texts([text])[0] == pytest.approx(as_document)


@RUN_REAL
def test_real_paper_text_uses_the_tokenizers_own_separator():
    embedder = Specter2Embedder()
    text = embedder.paper_text("A Title", "An abstract.")
    assert text == f"A Title{embedder._tokenizer.sep_token}An abstract."
    assert embedder.sep_token == "[SEP]"
