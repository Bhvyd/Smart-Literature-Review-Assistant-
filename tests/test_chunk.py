
from ingest.chunk import Chunk, chunk_text, word_count


def test_word_count():
    assert word_count("a b   c") == 3
    assert word_count("") == 0


def test_chunks_respect_target_and_cover_the_text():
    words = [f"w{i}" for i in range(1000)]
    chunks = chunk_text(" ".join(words), target=100, overlap=20, min_tokens=10)
    assert all(c.n_tokens <= 100 for c in chunks)
    assert [c.seq for c in chunks] == list(range(len(chunks)))
    assert chunks[0].text.startswith("w0")
    assert "w999" in chunks[-1].text


def test_overlap_repeats_the_tail_of_the_previous_chunk():
    words = [f"w{i}" for i in range(300)]
    chunks = chunk_text(" ".join(words), target=100, overlap=20, min_tokens=10)
    assert chunks[0].text.split()[-20:] == chunks[1].text.split()[:20]


def test_short_text_is_one_chunk():
    chunks = chunk_text("only a few words here to keep it simple", target=350, min_tokens=1)
    assert len(chunks) == 1
    assert chunks[0].seq == 0


def test_trailing_fragment_below_min_tokens_is_merged_not_dropped():
    words = [f"w{i}" for i in range(120)]
    chunks = chunk_text(" ".join(words), target=100, overlap=0, min_tokens=40)
    joined = " ".join(c.text for c in chunks)
    assert "w119" in joined
    assert all(c.n_tokens >= 40 for c in chunks)


def test_empty_text_yields_no_chunks():
    assert chunk_text("", target=100, overlap=10, min_tokens=10) == []


def test_whitespace_only_text_yields_no_chunks():
    assert chunk_text("   \n\t ", target=100, overlap=10, min_tokens=10) == []


def test_custom_token_counter_is_used():
    chunks = chunk_text(
        " ".join(f"w{i}" for i in range(100)),
        target=30,
        overlap=0,
        min_tokens=1,
        count_tokens=lambda t: len(t.split()) * 3,
    )
    assert all(c.n_tokens <= 30 for c in chunks)
    assert len(chunks) > 1


def test_overlap_larger_than_target_cannot_stall():
    chunks = chunk_text(" ".join(f"w{i}" for i in range(50)), target=10, overlap=99, min_tokens=1)
    assert len(chunks) > 1
    assert chunks[-1].text.split()[-1] == "w49"


def test_chunks_are_immutable():
    chunk = Chunk(seq=0, text="a", n_tokens=1)
    try:
        chunk.text = "b"
    except Exception:
        return
    raise AssertionError("Chunk should be frozen")


def test_max_tokens_caps_a_token_dense_region():
    def counter(text):
        return sum(7 if word == "dense" else 1 for word in text.split())

    words = ["filler"] * 95 + ["dense"] * 5
    chunks = chunk_text(
        " ".join(words * 30), target=100, overlap=0, min_tokens=1, count_tokens=counter, max_tokens=120
    )
    assert all(c.n_tokens <= 120 for c in chunks)


def test_max_tokens_is_optional_and_defaults_off():
    words = [f"w{i}" for i in range(50)]
    chunks = chunk_text(" ".join(words), target=100, overlap=0, min_tokens=1)
    assert len(chunks) == 1


def test_splitting_under_a_cap_loses_no_text():
    def counter(text):
        return sum(7 if word == "dense" else 1 for word in text.split())

    words = ["filler"] * 40 + ["dense"] * 10
    text = " ".join(words * 10)
    chunks = chunk_text(text, target=200, overlap=0, min_tokens=1, count_tokens=counter, max_tokens=100)
    joined = " ".join(c.text for c in chunks)
    assert joined.split() == text.split()
    assert [c.seq for c in chunks] == list(range(len(chunks)))


def test_a_bibliography_chunk_is_recognised():
    from ingest.chunk import looks_like_references

    refs = (
        "Huang, L., Yu, W., Ma, W., et al. A survey on hallucination in large language models. "
        "arXiv preprint arXiv:2311.05232, 2023. Lewis, P. et al. Retrieval-augmented generation for "
        "knowledge-intensive NLP tasks. In Proceedings of NeurIPS, pp. 9459-9474, 2020. Gao, Y. "
        "Retrieval-augmented generation for large language models: A survey. arXiv preprint arXiv:2312.10997."
    )
    assert looks_like_references(refs)


def test_related_work_prose_full_of_inline_citations_is_not_mistaken_for_references():
    from ingest.chunk import looks_like_references

    prose = (
        "Prior work builds on HotpotQA (Yang et al., 2018), TriviaQA (Joshi et al., 2017) and Natural "
        "Questions (Kwiatkowski et al., 2019), while RAG (Lewis et al., 2020), as evaluated by Petroni et "
        "al. (2021), and DPR + FiD (Piktus et al., 2021) report KILT-based scores in 2021 and 2022."
    )
    assert not looks_like_references(prose)
    assert not looks_like_references("We evaluate on three datasets and report exact match.")
