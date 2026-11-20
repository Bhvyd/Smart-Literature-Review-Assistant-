
import json

import httpx
import pytest

from services.labels import (
    central_titles,
    cluster_labels,
    distinctive_terms,
    name_collisions,
    parse_names,
    system_prompt,
    ungrounded_names,
    uninformative_names,
)
from services.llm import UNTRUSTED_TEXT_RULE, LLMClient
from store.db import init_db


class Provider:
    def __init__(self, *replies, status=200, error=None):
        self.replies = list(replies)
        self.status, self.error, self.requests = status, error, []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        if self.status != 200:
            return httpx.Response(self.status, json={"error": self.error or {}})
        text = self.replies[min(len(self.requests), len(self.replies)) - 1]
        if isinstance(text, int):
            return httpx.Response(text, json={"error": {"code": "rate_limit_exceeded"}})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": text}, "finish_reason": "stop"}], "usage": {}},
        )

    def user(self, index=0) -> str:
        return self.requests[index]["messages"][1]["content"]

    def system(self, index=0) -> str:
        return self.requests[index]["messages"][0]["content"]


@pytest.fixture
def llm_for(tmp_path):
    path = tmp_path / "labels.sqlite3"
    init_db(path)
    return lambda provider: LLMClient(
        api_key="sk-test", model="gpt-5-nano", db_path=path, transport=httpx.MockTransport(provider)
    )


def _points(spec):
    return [
        {"paper_id": pid, "x": float(i) + 10 * cluster, "y": 0.0, "cluster": cluster}
        for cluster, ids in spec.items()
        for i, pid in enumerate(ids)
    ]


TITLES = {
    "a1": "Tree-based draft verification for speculative decoding",
    "a2": "Draft tree attention in speculative decoding",
    "a3": "Self-speculative drafting with layer skipping",
    "b1": "Weight quantization for fast decoding kernels",
    "b2": "Low-bit quantization kernels for speculative decoding",
    "b3": "Quantization-aware serving of large models",
    "c": "A lone paper about diffusion sampling",
}
SPEC = {0: ["a1", "a2", "a3"], 1: ["b1", "b2", "b3"], 2: ["c"]}
TOPIC = "speculative decoding"
GOOD = "0: Tree Verification\n1: Weight Quantization\n2: Diffusion Sampling"


def _docs(spec, texts):
    titles = {pid: f"Paper {pid}" for ids in spec.values() for pid in ids}
    return _points(spec), titles, texts


TWO = {0: ["a1", "a2", "a3"], 1: ["b1", "b2", "b3"]}


def test_distinctive_terms_are_frequent_here_rare_elsewhere_and_never_the_topic():
    terms = distinctive_terms(_points(SPEC), TITLES, topic=TOPIC)
    assert "tree" in " ".join(terms[0]) and "draft" in " ".join(terms[0])
    assert "quantization" in " ".join(terms[1])
    assert not any("decoding" in t or "speculative" in t for group in terms.values() for t in group)
    words = [set(" ".join(group).split()) for group in terms.values()]
    assert words[0].isdisjoint(words[1])


def test_a_term_as_common_outside_the_cluster_as_inside_does_not_describe_it():
    texts = {pid: "alpha of zebra" for pid in ("a1", "a2", "a3")}
    texts.update({pid: "alpha of quokka" for pid in ("b1", "b2", "b3")})
    terms = distinctive_terms(*_docs(TWO, texts))
    words = [set(" ".join(group).split()) for group in terms.values()]
    assert "zebra" in words[0] and "quokka" in words[1]
    assert "alpha" not in terms[0] + terms[1]


def test_the_stronger_contrast_ranks_first():
    spec = {0: ["a1", "a2", "a3", "a4"], 1: ["b1", "b2", "b3", "b4"]}
    texts = {"a1": "rare of common", "a2": "rare of common", "a3": "rare of common", "a4": "rare",
             "b1": "common", "b2": "common", "b3": "other", "b4": "other"}
    terms = distinctive_terms(*_docs(spec, texts))
    assert terms[0][0] == "rare"


def test_a_term_in_more_of_the_cluster_can_outrank_one_that_is_merely_exclusive():
    spec = {0: ["a1", "a2", "a3", "a4"], 1: [f"b{i}" for i in range(1, 9)]}
    texts = {"a1": "mango of papaya", "a2": "mango of papaya", "a3": "mango", "a4": "mango"}
    texts.update({"b1": "mango", "b2": "mango"})
    terms = distinctive_terms(*_docs(spec, texts))[0]
    assert terms.index("mango") < terms.index("papaya")


def test_only_words_sharing_the_topics_root_are_banned():
    texts = {pid: "specimen specimens kernels" for pid in ("a1", "a2", "a3")}
    words = " ".join(distinctive_terms(*_docs(TWO, texts), topic="speculative decoding")[0]).split()
    assert "specimen" in words


def test_abstracts_feed_the_terms_and_one_paper_cannot_define_a_clusters_vocabulary():
    texts = {"a1": "ensemble members", "a2": "ensemble verification", "a3": "lonely of lonely of lonely"}
    points, titles, abstracts = _docs(TWO, texts)
    terms = distinctive_terms(points, titles, abstracts)
    assert "ensemble" in terms[0]
    assert "lonely" not in " ".join(terms[0]).split()
    assert "ensemble" not in distinctive_terms(points, titles, None)[0]


def test_links_and_bare_domains_are_not_vocabulary():
    text = "Code at https://github.com/foo/bar-tool and www.example.org/x, see foo.io too. kernels"
    terms = distinctive_terms(*_docs(TWO, {pid: text for pid in ("a1", "a2", "a3")}))
    words = set(" ".join(terms[0]).split())
    assert "kernels" in words
    assert not {"https", "http", "github", "com", "foo", "bar", "tool", "www", "example", "org"} & words


def test_the_topic_is_banned_by_word_root_and_by_acronym():
    text = "speculation speculate decoders rag rag kernels"
    texts = {pid: text for pid in ("a1", "a2", "a3")}
    spec_terms = distinctive_terms(*_docs(TWO, texts), topic="speculative decoding")[0]
    spec_words = " ".join(spec_terms).split()
    assert "kernels" in spec_words
    assert not any("specul" in w or "decoder" in w for w in spec_words)
    rag_terms = distinctive_terms(*_docs(TWO, texts), topic="retrieval augmented generation")[0]
    rag_words = " ".join(rag_terms).split()
    assert "kernels" in rag_words and "rag" not in rag_words
    assert "rag" in " ".join(distinctive_terms(*_docs(TWO, texts), topic="something else")[0]).split()


def test_a_word_is_claimed_by_one_cluster_only_and_spellings_are_not_repeated():
    spec = {0: ["a1", "a2", "a3"], 1: ["b1", "b2", "b3"], 2: ["c1", "c2", "c3"]}
    texts = {pid: "kernel anomaly anomalies" for pid in ("a1", "a2", "a3", "b1", "b2", "b3")}
    texts.update({pid: "plain" for pid in ("c1", "c2", "c3")})
    terms = distinctive_terms(*_docs(spec, texts))
    assert sum("kernel" in group for group in terms.values()) == 1
    chosen = {t for group in terms.values() for t in group}
    assert not {"anomaly", "anomalies"} <= chosen


def test_generic_filler_is_not_a_distinctive_term():
    text = "remains nature prior available components generates propose approach novel kernels"
    terms = distinctive_terms(*_docs(TWO, {pid: text for pid in ("a1", "a2", "a3")}))
    assert terms[0] == ["kernels"]


def test_the_near_synonym_names_measured_live_are_collisions_and_distinct_names_are_not():
    measured = {
        0: "Inference Optimization Techniques",
        1: "Optimized Inference Architectures",
        2: "Efficient LLM Inference Acceleration",
        3: "Diffusion Model Acceleration",
        4: "Inference Efficiency Optimization",
    }
    assert {(0, 1), (0, 4), (1, 4)} <= set(name_collisions(measured))
    assert name_collisions({0: "Tree Verification", 1: "Weight Quantization", 2: "Diffusion Sampling"}) == []


def test_two_shared_content_words_collide_even_in_long_names():
    assert name_collisions({0: "Alpha Beta Gamma Delta", 1: "Alpha Beta Omega Sigma Theta"}) == [(0, 1)]


def test_one_shared_word_collides_only_when_the_names_are_otherwise_short():
    assert name_collisions({0: "Speculative Verification", 1: "Speculative Quantization"}) == [(0, 1)]
    assert name_collisions({0: "Alpha Beta Gamma", 1: "Alpha Omega Sigma"}) == []


def test_a_name_contained_in_another_collides():
    assert name_collisions({0: "Quantization", 1: "Weight Low Bit Quantization Kernels"}) == [(0, 1)]


def test_equal_names_collide_whatever_their_punctuation_or_case():
    assert name_collisions({0: "Tree Verification", 1: "tree-verification."}) == [(0, 1)]
    assert name_collisions({0: "Novel Approaches", 1: "Novel approaches"}) == [(0, 1)]


def test_the_collection_topic_does_not_make_two_names_collide():
    names = {0: "Speculative Verification", 1: "Speculative Quantization"}
    assert name_collisions(names, topic=TOPIC) == []
    assert name_collisions(names) == [(0, 1)]


def test_stems_equate_word_forms():
    assert name_collisions({0: "Quantized Kernels Memory", 1: "Quantization Kernel Sampling"}) == [(0, 1)]


def test_names_with_no_content_words_are_flagged():
    names = {0: "Novel Approaches", 1: "Speculative Decoding", 2: "投机解码", 3: "Tree Verification"}
    assert uninformative_names(names, topic=TOPIC) == [0, 1, 2]
    assert uninformative_names({1: "Speculative Decoding"}) == []


def test_name_words_absent_from_the_evidence_are_reported():
    evidence = {0: ["Tree-based draft verification for speculative decoding", "draft tree attention"]}
    assert ungrounded_names({0: "Tree Verification"}, evidence, topic=TOPIC) == {}
    assert ungrounded_names({0: "Tree Architectures"}, evidence, topic=TOPIC) == {0: ["architectures"]}
    assert ungrounded_names({0: "Tree Verification"}, {}, topic=TOPIC) == {0: ["tree", "verification"]}
    assert ungrounded_names({0: "Acceleration"}, {0: ["Accelerating inference"]}) == {}


def test_parse_names_reads_lines_and_json_and_drops_unknown_ids():
    assert parse_names('1: "A b"\n cluster 2 - C d.\n9: Ghost', {1, 2}) == {1: "A b", 2: "C d"}
    assert parse_names('{"1": "A b", "x": "no"}', {1, 2}) == {1: "A b"}
    assert parse_names("garbage", {1}) == {}


def test_all_clusters_are_named_in_one_call_that_carries_titles_and_terms(llm_for):
    provider = Provider(GOOD)
    result = cluster_labels(_points(SPEC), TITLES, llm=llm_for(provider), topic=TOPIC)
    assert result["labels"] == {0: "Tree Verification", 1: "Weight Quantization", 2: "Diffusion Sampling"}
    assert result["distinct"] is True
    assert result["collisions"] == result["unnamed"] == result["uninformative"] == []
    assert len(provider.requests) == 1
    prompt = provider.user()
    assert "Draft tree attention" in prompt and "Low-bit quantization kernels" in prompt
    assert "Cluster 0 (3 papers)" in prompt and "Cluster 2 (1 papers)" in prompt
    assert "Distinctive terms:" in prompt
    system = provider.system()
    assert "do NOT use those words" in system and TOPIC in system
    assert "supported by most of that cluster's titles" in system
    assert "synonyms or near-paraphrases" in system
    assert "share a content word" not in system
    assert result["terms"][0] and result["terms"][1]


def test_the_system_prompt_ends_with_the_untrusted_text_rule_with_or_without_a_topic():
    assert system_prompt(None).endswith(UNTRUSTED_TEXT_RULE)
    assert system_prompt("  ").endswith(UNTRUSTED_TEXT_RULE)
    assert system_prompt("graph neural networks").endswith(UNTRUSTED_TEXT_RULE)
    assert "graph neural networks" in system_prompt("graph neural networks")


def test_colliding_names_get_one_retry_that_points_out_the_collision(llm_for):
    provider = Provider(
        "0: Tree Verification\n1: Tree Verification Kernels\n2: Diffusion Sampling", GOOD
    )
    result = cluster_labels(_points(SPEC), TITLES, llm=llm_for(provider), topic=TOPIC)
    assert len(provider.requests) == 2
    retry = provider.user(1)
    assert "too similar" in retry and '"Tree Verification Kernels" (cluster 1)' in retry
    assert "share a content word" not in retry
    assert result["labels"][1] == "Weight Quantization" and result["distinct"] is True


def test_names_that_still_collide_are_returned_with_terms_and_flagged(llm_for):
    same = "0: Tree Verification\n1: Tree Verification Kernels\n2: Diffusion Sampling"
    provider = Provider(same, same)
    result = cluster_labels(_points(SPEC), TITLES, llm=llm_for(provider), topic=TOPIC)
    assert len(provider.requests) == 2
    assert result["distinct"] is False and result["collisions"] == [[0, 1]]
    assert result["labels"][0] == "Tree Verification" and result["terms"][0]


def test_an_unavailable_model_still_returns_the_distinctive_terms(llm_for):
    provider = Provider(status=429, error={"code": "insufficient_quota", "message": "empty"})
    result = cluster_labels(_points(SPEC), TITLES, llm=llm_for(provider), topic=TOPIC)
    assert result["labels"] == {} and result["unavailable"]["code"] == "insufficient_quota"
    assert result["terms"][1] and len(provider.requests) == 1
    assert result["unnamed"] == [0, 1, 2] and result["distinct"] is False


def test_a_missing_cluster_gets_one_retry_and_is_reported_when_still_unnamed(llm_for):
    partial = "0: Tree Verification"
    provider = Provider(partial, GOOD)
    healed = cluster_labels(_points(SPEC), TITLES, llm=llm_for(provider), topic=TOPIC)
    assert len(provider.requests) == 2 and "no name was given for cluster 1, 2" in provider.user(1)
    assert healed["distinct"] is True and healed["unnamed"] == []

    stuck = Provider(partial, partial)
    result = cluster_labels(_points(SPEC), TITLES, llm=llm_for(stuck), topic="speculative decoding methods")
    assert len(stuck.requests) == 2
    assert result["labels"] == {0: "Tree Verification"}
    assert result["unnamed"] == [1, 2] and result["distinct"] is False


def test_filler_only_names_trigger_a_retry_and_stay_flagged(llm_for):
    junk = "0: Novel Approaches\n1: Weight Quantization\n2: Speculative Decoding"
    provider = Provider(junk, junk)
    result = cluster_labels(_points(SPEC), TITLES, llm=llm_for(provider), topic=TOPIC)
    assert len(provider.requests) == 2 and "say nothing specific" in provider.user(1)
    assert result["uninformative"] == [0, 2] and result["distinct"] is False


def test_invented_name_words_trigger_a_retry_and_are_reported(llm_for):
    invented = "0: Tree Architectures\n1: Weight Quantization\n2: Diffusion Sampling"
    provider = Provider(invented, invented)
    result = cluster_labels(_points(SPEC), TITLES, llm=llm_for(provider), topic=TOPIC)
    assert len(provider.requests) == 2 and "not supported" in provider.user(1)
    assert result["ungrounded"] == {0: ["architectures"]}


def test_a_failed_retry_keeps_the_first_names_and_the_flags(llm_for):
    same = "0: Tree Verification\n1: Tree Verification Kernels\n2: Diffusion Sampling"
    provider = Provider(same, 429)
    result = cluster_labels(_points(SPEC), TITLES, llm=llm_for(provider), topic=TOPIC)
    assert result["unavailable"] is None and result["labels"][1] == "Tree Verification Kernels"
    assert result["distinct"] is False and result["collisions"] == [[0, 1]]


def test_noise_and_unknown_clusters_are_never_named(llm_for):
    points = _points(SPEC) + [{"paper_id": "n", "x": 0.0, "y": 1.0, "cluster": -1}]
    provider = Provider('```\n0: "Tree Verification."\n7: Ghost Cluster\n```', GOOD)
    result = cluster_labels(points, {**TITLES, "n": "Noise"}, llm=llm_for(provider), topic=TOPIC)
    assert 7 not in result["labels"] and -1 not in result["labels"] and -1 not in result["terms"]
    assert result["labels"][0] == "Tree Verification"


def test_json_replies_are_accepted(llm_for):
    provider = Provider(
        '{"0": "Tree Verification", "1": "Weight Quantization", "2": "Diffusion Sampling"}'
    )
    result = cluster_labels(_points(SPEC), TITLES, llm=llm_for(provider), topic=TOPIC)
    assert result["labels"] == {0: "Tree Verification", 1: "Weight Quantization", 2: "Diffusion Sampling"}
    assert len(provider.requests) == 1


def test_labels_are_made_from_the_titles_nearest_each_centre():
    points = [
        {"paper_id": "a", "x": 0.0, "y": 0.0, "cluster": 0},
        {"paper_id": "b", "x": 0.1, "y": 0.0, "cluster": 0},
        {"paper_id": "far", "x": 9.0, "y": 9.0, "cluster": 0},
        {"paper_id": "noise", "x": 1.0, "y": 1.0, "cluster": -1},
    ]
    titles = central_titles(points, {"a": "Alpha", "b": "Beta", "far": "Faraway", "noise": "Noise"})
    assert titles[0][0] in {"Alpha", "Beta"} and titles[0][-1] == "Faraway" and -1 not in titles


def _client(path, provider=None):
    from fastapi.testclient import TestClient

    from app.config import Settings
    from app.main import create_app
    from test_api_map import fake_clusterer, fake_projector

    settings = Settings(db_path=path, llm_api_key="sk-test" if provider else None, llm_provider="openai")
    app = create_app(
        settings, warm_query_embedder=False,
        llm_transport=httpx.MockTransport(provider) if provider else None,
    )
    app.state.map_projector, app.state.map_clusterer = fake_projector, fake_clusterer
    return TestClient(app)


def test_the_labels_endpoint_returns_terms_even_when_generation_is_unavailable(tmp_path):
    from test_api_map import seed

    path = tmp_path / "api.sqlite3"
    init_db(path)
    cid = seed(path, n=20)
    with _client(path) as api:
        body = api.get(f"/api/collections/{cid}/map/labels").json()
    assert body["labels"] == {} and body["unavailable"]["code"] == "no_api_key"
    assert set(body["terms"]) == {"0", "1", "2"}


def test_the_labels_endpoint_names_every_cluster_and_reports_ungrounded_words(tmp_path):
    from test_api_map import seed

    path = tmp_path / "api.sqlite3"
    init_db(path)
    cid = seed(path, n=20)
    provider = Provider("0: Alpha Topic\n1: Beta Subject\n2: Gamma Area")
    with _client(path, provider) as api:
        body = api.get(f"/api/collections/{cid}/map/labels").json()
    assert body["labels"] == {"0": "Alpha Topic", "1": "Beta Subject", "2": "Gamma Area"}
    assert body["distinct"] is True and len(provider.requests) == 2
    assert set(body["ungrounded"]) == {"0", "1", "2"}
