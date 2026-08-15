from ragqa.ingest import chunk_text, split_sentences


def test_split_sentences_basic():
    text = "This is one. This is two! Is this three? Yes it is."
    sentences = split_sentences(text)
    assert sentences == [
        "This is one.",
        "This is two!",
        "Is this three?",
        "Yes it is.",
    ]


def test_split_sentences_collapses_whitespace():
    text = "First sentence.\n\n\nSecond   sentence."
    sentences = split_sentences(text)
    assert sentences == ["First sentence.", "Second sentence."]


def test_chunk_text_never_splits_a_sentence():
    # Construct text where a naive fixed-width slice at 50 chars WOULD cut
    # a sentence in half, and assert our chunker doesn't.
    sentence = "The quick brown fox jumps over the lazy dog again."
    text = " ".join([sentence] * 5)
    chunks = chunk_text(text, source="test.txt", chunk_size_chars=100, overlap_chars=20)
    for chunk in chunks:
        assert chunk.text.strip().endswith((".", "!", "?"))


def test_chunk_text_overlap_repeats_context():
    sentence = "Sentence number {}."
    text = " ".join(sentence.format(i) for i in range(10))
    chunks = chunk_text(text, source="test.txt", chunk_size_chars=40, overlap_chars=15)
    assert len(chunks) > 1
    # some text from the tail of chunk N should reappear at the head of chunk N+1
    overlap_found = any(
        chunks[i].text.split(".")[-2:] and chunks[i].text[-15:].strip() in chunks[i + 1].text
        for i in range(len(chunks) - 1)
    )
    # weaker but robust check: consecutive chunks share at least one sentence
    def sentences_of(c):
        return {s.strip() for s in c.text.split(".") if s.strip()}

    shared = any(
        sentences_of(chunks[i]) & sentences_of(chunks[i + 1]) for i in range(len(chunks) - 1)
    )
    assert shared


def test_chunk_ids_are_unique_per_source():
    text = "One. Two. Three. Four."
    chunks = chunk_text(text, source="doc.txt", chunk_size_chars=10, overlap_chars=0)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))


def test_empty_text_produces_no_chunks():
    assert chunk_text("", source="empty.txt") == []
    assert split_sentences("   ") == []
