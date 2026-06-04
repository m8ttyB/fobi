import config


def chunk_document(
    text: str,
    chunk_chars: int = config.CHUNK_CHARS,
    overlap_chars: int = config.OVERLAP_CHARS,
) -> list[str]:
    """Split text into overlapping chunks of approximately chunk_chars characters.

    Chunk boundaries snap to the nearest paragraph break (double newline) within
    a tolerance window to avoid cutting mid-sentence. Falls back to a hard
    character cut if no paragraph break is found nearby.

    The first character of each chunk after the first overlaps with the last
    overlap_chars characters of the previous chunk, so entities straddling a
    boundary appear in full context in at least one chunk.
    """
    if len(text) <= chunk_chars:
        return [text]

    chunks = []
    start = 0
    snap_window = chunk_chars // 4  # look this far from the target boundary for a \n\n

    while start < len(text):
        end = start + chunk_chars

        if end >= len(text):
            chunks.append(text[start:])
            break

        # Try to snap to a paragraph boundary near the target end
        snap_start = max(start + 1, end - snap_window)
        snap_end = min(len(text), end + snap_window)
        segment = text[snap_start:snap_end]
        para_pos = segment.find("\n\n")

        if para_pos != -1:
            end = snap_start + para_pos + 2  # include the \n\n in this chunk
        # else: hard cut at end

        chunks.append(text[start:end])
        start = max(start + 1, end - overlap_chars)

    return chunks
