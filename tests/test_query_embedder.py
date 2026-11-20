
import pytest

import embed.specter2 as specter2
from services.query_embedder import QueryEmbedder


class FakeEmbedder:
    model_name = "fake"
    dim = 4

    def __init__(self, *, raises=None):
        self.raises = raises
        self.calls = 0

    def embed_queries(self, texts, batch_size=8):
        self.calls += 1
        if self.raises:
            raise self.raises if isinstance(self.raises, Exception) else RuntimeError(self.raises)
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


def install(monkeypatch, embedder):
    monkeypatch.setattr(specter2, "Specter2Embedder", lambda **kw: embedder)
    return embedder


def test_vector_returns_the_embedding(monkeypatch):
    fake = install(monkeypatch, FakeEmbedder())
    service = QueryEmbedder()
    assert service.vector("retrieval augmented generation") == [0.1, 0.2, 0.3, 0.4]
    assert fake.calls == 1


def test_empty_text_returns_none_without_embedding(monkeypatch):
    fake = install(monkeypatch, FakeEmbedder())
    service = QueryEmbedder()
    assert service.vector("   ") is None
    assert fake.calls == 0


def test_the_model_is_not_loaded_at_construction(monkeypatch):
    def explode(**kw):
        raise AssertionError("must not construct the embedder at import or construction")

    monkeypatch.setattr(specter2, "Specter2Embedder", explode)
    service = QueryEmbedder()
    assert service.available


def test_a_transient_inference_failure_is_retried_on_the_next_request(monkeypatch):
    fake = install(monkeypatch, FakeEmbedder(raises="cuda oom"))
    service = QueryEmbedder()
    assert service.vector("attention") is None
    assert service.available is True
    assert "cuda oom" in service.last_error

    fake.raises = None
    assert service.vector("attention") == [0.1, 0.2, 0.3, 0.4]
    assert service.last_error is None


def test_a_model_that_cannot_load_is_not_retried_on_the_next_request(monkeypatch):
    fake = install(
        monkeypatch, FakeEmbedder(raises=specter2.EmbedderUnavailable("weights missing"))
    )
    service = QueryEmbedder()
    assert service.vector("one") is None
    assert service.vector("two") is None
    assert service.available is False
    assert fake.calls == 1


def test_the_embedder_is_built_once_across_requests(monkeypatch):
    fake = install(monkeypatch, FakeEmbedder())
    service = QueryEmbedder()
    service.vector("one")
    service.vector("two")
    assert fake.calls == 2
    assert service.available


def test_a_construction_failure_is_reported_as_unavailable(monkeypatch):
    def explode(**kw):
        raise RuntimeError("weights not downloadable")

    monkeypatch.setattr(specter2, "Specter2Embedder", explode)
    service = QueryEmbedder()
    assert service.vector("attention") is None
    assert service.available is False


def test_warm_reports_whether_the_model_loaded(monkeypatch):
    install(monkeypatch, FakeEmbedder())
    assert QueryEmbedder().warm() is True

    install(monkeypatch, FakeEmbedder(raises=specter2.EmbedderUnavailable("nope")))
    assert QueryEmbedder().warm() is False
