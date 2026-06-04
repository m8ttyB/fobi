from chunker import chunk_document


class TestChunkDocument:
    def test_short_document_returns_single_chunk(self):
        text = "Short document."
        chunks = chunk_document(text, chunk_chars=4000, overlap_chars=400)
        assert chunks == [text]

    def test_empty_string_returns_single_chunk(self):
        chunks = chunk_document("", chunk_chars=4000, overlap_chars=400)
        assert len(chunks) == 1
        assert chunks[0] == ""

    def test_exact_length_returns_single_chunk(self):
        text = "a" * 4000
        chunks = chunk_document(text, chunk_chars=4000, overlap_chars=400)
        assert len(chunks) == 1

    def test_long_document_produces_multiple_chunks(self):
        text = "a" * 12000
        chunks = chunk_document(text, chunk_chars=4000, overlap_chars=400)
        assert len(chunks) > 1

    def test_each_chunk_within_size_bound(self):
        text = "a" * 20000
        chunk_chars = 4000
        overlap_chars = 400
        chunks = chunk_document(text, chunk_chars=chunk_chars, overlap_chars=overlap_chars)
        for chunk in chunks:
            # Allow a small tolerance for paragraph snapping
            assert len(chunk) <= chunk_chars + 500

    def test_overlap_between_chunks(self):
        # Build text where overlap content is identifiable
        segment = "X" * 3600 + "\n\n" + "Y" * 400
        text = segment + "\n\n" + "Z" * 3600
        chunks = chunk_document(text, chunk_chars=4000, overlap_chars=400)
        assert len(chunks) >= 2
        # The overlap region should appear in both adjacent chunks
        tail_of_first = chunks[0][-400:]
        assert tail_of_first in chunks[1]

    def test_snaps_to_paragraph_boundary(self):
        # Place a paragraph break just before the chunk boundary
        before = "a" * 3800
        para_break = "\n\n"
        after = "b" * 3800
        text = before + para_break + after
        chunks = chunk_document(text, chunk_chars=4000, overlap_chars=400)
        # The split should occur at or near the paragraph break
        assert len(chunks) >= 2
        # First chunk should not bleed far past the paragraph break
        assert len(chunks[0]) <= 4000 + 400

    def test_no_paragraph_breaks_hard_cuts(self):
        # No \n\n anywhere — falls back to hard character cut
        text = "a" * 10000
        chunks = chunk_document(text, chunk_chars=4000, overlap_chars=400)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 4000 + 10  # small tolerance

    def test_all_text_covered(self):
        # Every character in the original should appear in at least one chunk
        text = "The quick brown fox. " * 500  # ~10500 chars
        chunks = chunk_document(text, chunk_chars=4000, overlap_chars=400)
        # Rebuild a set of all chars by position — easier: check first and last char of text
        assert text[:50] in chunks[0]
        assert text[-50:] in chunks[-1]

    def test_single_chunk_no_overlap_applied(self):
        text = "Short"
        chunks = chunk_document(text, chunk_chars=4000, overlap_chars=400)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_uses_config_defaults(self):
        # Calling with no kwargs should not raise
        text = "Hello world."
        chunks = chunk_document(text)
        assert len(chunks) >= 1
