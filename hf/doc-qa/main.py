"""CLI entry point for doc-qa.

Four mutually exclusive commands:
    --ingest FILE      Chunk, embed, and index a single PDF.
    --ingest-dir DIR   Recursively ingest all PDFs in a directory.
    --chat             Start an interactive Q&A REPL against all indexed documents.
    --reset            Delete the entire index store (with confirmation).

Readline is imported at startup when available (macOS, Linux) to enable
up/down-arrow history navigation, left/right cursor movement, and Ctrl+R
reverse search within the chat REPL. On Windows, where readline is absent,
the import is silently skipped and plain input() is used instead.
"""

import argparse
import os
import shutil
import sys

import config

try:
    import readline
    readline.set_history_length(config.MAX_HISTORY_TURNS * 2)
except ImportError:
    pass  # Windows — plain input() used, no history navigation
from contextualizer import contextualize_query
from ingest import chunk_text, embed_chunks, save_index, index_name_for, ingest_directory
from model import load_model, stream_response
from prompts import build_messages
from retriever import load_all_indexes, retrieve_multi
from sentence_transformers import SentenceTransformer


def cmd_ingest(path: str) -> None:
    """Ingest a single PDF into the document store.

    Prompts for confirmation before overwriting an existing index for the same file.
    """
    if not os.path.exists(path):
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(config.STORE_DIR, exist_ok=True)
    stem = index_name_for(path, os.path.dirname(os.path.abspath(path)))
    index_path = os.path.join(config.STORE_DIR, f"{stem}.faiss")
    metadata_path = os.path.join(config.STORE_DIR, f"{stem}.json")

    if os.path.exists(index_path):
        print(
            f"Warning: '{os.path.basename(path)}' is already indexed. "
            "Ingesting will overwrite it."
        )
        answer = input("Continue? [y/N] ").strip().lower()
        if answer != "y":
            print("Cancelled.")
            return

    print(f"Loading {path}...")
    from ingest import PdfReader
    reader = PdfReader(path)
    text = "\n\n".join(page.extract_text() or "" for page in reader.pages)

    if not text.strip():
        print("Error: could not extract text from the document.", file=sys.stderr)
        sys.exit(1)

    print("Chunking text...")
    chunks = chunk_text(text)
    print(f"  {len(chunks)} chunks created")

    print(f"Embedding chunks with '{config.EMBED_MODEL}'...")
    embeddings = embed_chunks(chunks, config.EMBED_MODEL)

    print(f"Saving index to '{config.STORE_DIR}'...")
    save_index(embeddings, chunks, source=path, index_path=index_path, metadata_path=metadata_path)
    print("Done. Run --chat to ask questions.")


def cmd_ingest_dir(dir_path: str) -> None:
    """Recursively ingest all PDFs in dir_path into the document store."""
    if not os.path.isdir(dir_path):
        print(f"Error: directory not found: {dir_path}", file=sys.stderr)
        sys.exit(1)
    os.makedirs(config.STORE_DIR, exist_ok=True)
    ingest_directory(dir_path, config.STORE_DIR, config.EMBED_MODEL)
    print("Run --chat to ask questions.")


def cmd_reset() -> None:
    """Delete the entire document store after user confirmation."""
    answer = input(f"Delete all indexes in '{config.STORE_DIR}'? [y/N] ").strip().lower()
    if answer != "y":
        print("Cancelled.")
        return
    if os.path.isdir(config.STORE_DIR):
        shutil.rmtree(config.STORE_DIR)
        print(f"Removed '{config.STORE_DIR}'.")
    else:
        print("Nothing to remove — no index store found.")


def cmd_chat() -> None:
    """Start a REPL for asking questions against all indexed documents.

    Before each retrieval, the current question is rewritten into a standalone
    query using recent conversation history (query contextualization). This
    ensures follow-up questions like "Tell me more" resolve correctly against
    the document index. The rewritten query is used only for retrieval — the
    original question is preserved for history and the generation prompt.

    Retrieves the top-k most relevant chunks across all indexes, builds a
    grounded prompt, and streams the model's response token by token.
    After each answer, prints the source filenames of the retrieved chunks.
    If no chunks pass the MIN_SCORE threshold, the model is not called and
    the user is prompted to rephrase rather than receiving a hallucinated answer.
    History accumulates across turns; oldest turns are trimmed when the
    context budget fills so retrieval quality is never sacrificed for history.
    """
    try:
        indexes = load_all_indexes(config.STORE_DIR)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    doc_count = len(indexes)
    chunk_count = sum(idx.ntotal for idx, _ in indexes)
    print(f"Loading embedding model '{config.EMBED_MODEL}'...")
    embed_model = SentenceTransformer(config.EMBED_MODEL)

    print(f"Loading language model '{config.MODEL_PATH}'...")
    model, tokenizer = load_model(config.MODEL_PATH)

    history: list[dict] = []
    print(f"\n{doc_count} document(s) indexed ({chunk_count} chunks). "
          "Type /exit to quit, /clear to reset history.\n")

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question == "/exit":
            break
        if question == "/clear":
            history = []
            print("History cleared.")
            continue

        retrieval_query = contextualize_query(question, history, model, tokenizer)
        chunks = retrieve_multi(retrieval_query, embed_model, indexes, top_k=config.TOP_K, min_score=config.MIN_SCORE)

        if not chunks:
            print("No relevant content found for that question. "
                  "Try rephrasing, or check that the relevant document has been ingested.\n")
            continue

        messages = build_messages(history, question, chunks)

        response = ""
        try:
            for text, _ in stream_response(model, tokenizer, messages):
                print(text, end="", flush=True)
                response += text
        except KeyboardInterrupt:
            print("\n[interrupted]")

        print()

        # Show which documents contributed to this answer
        sources = list(dict.fromkeys(
            os.path.basename(c["source"]) for c in chunks
        ))
        print(f"Sources: {', '.join(sources)}\n")

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": response})

        max_entries = config.MAX_HISTORY_TURNS * 2
        if len(history) > max_entries:
            history = history[-max_entries:]


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate command."""
    parser = argparse.ArgumentParser(
        description="doc-qa: answer questions about documents using RAG"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ingest", metavar="FILE", help="ingest a single PDF document")
    group.add_argument("--ingest-dir", metavar="DIR", help="ingest all PDFs in a directory (recursive)")
    group.add_argument("--chat", action="store_true", help="start a Q&A session")
    group.add_argument("--reset", action="store_true", help="delete all indexes")

    args = parser.parse_args()

    if args.ingest:
        cmd_ingest(args.ingest)
    elif args.ingest_dir:
        cmd_ingest_dir(args.ingest_dir)
    elif args.chat:
        cmd_chat()
    elif args.reset:
        cmd_reset()


if __name__ == "__main__":
    main()
