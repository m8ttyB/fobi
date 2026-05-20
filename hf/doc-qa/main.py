import argparse
import os
import sys

import config
from ingest import chunk_text, embed_chunks, save_index
from model import load_model, stream_response
from prompts import build_messages
from retriever import load_index, retrieve
from sentence_transformers import SentenceTransformer


def cmd_ingest(path: str) -> None:
    if not os.path.exists(path):
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    if os.path.exists(config.INDEX_PATH):
        print(
            f"Warning: an existing index was found at '{config.INDEX_PATH}'. "
            "Ingesting will overwrite it — the previous document will no longer be searchable. "
            "Run --reset first if you want to start clean."
        )
        answer = input("Continue? [y/N] ").strip().lower()
        if answer != "y":
            print("Cancelled.")
            return

    print(f"Loading {path}...")
    from pypdf import PdfReader
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

    print(f"Saving index to '{config.INDEX_PATH}'...")
    save_index(embeddings, chunks, source=path)
    print("Done. Run --chat to ask questions.")


def cmd_reset() -> None:
    answer = input("Delete index and metadata? [y/N] ").strip().lower()
    if answer != "y":
        print("Cancelled.")
        return
    removed = []
    for path in (config.INDEX_PATH, config.METADATA_PATH):
        if os.path.exists(path):
            os.remove(path)
            removed.append(path)
    if removed:
        print(f"Removed: {', '.join(removed)}")
    else:
        print("Nothing to remove — no index found.")


def cmd_chat() -> None:
    try:
        index, metadata = load_index()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading embedding model '{config.EMBED_MODEL}'...")
    embed_model = SentenceTransformer(config.EMBED_MODEL)

    print(f"Loading language model '{config.MODEL_PATH}'...")
    model, tokenizer = load_model(config.MODEL_PATH)

    history: list[dict] = []
    print(f"\nIndex loaded ({index.ntotal} chunks). Type /exit to quit, /clear to reset history.\n")

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

        chunks = retrieve(question, embed_model, index, metadata, top_k=config.TOP_K)
        messages = build_messages(history, question, chunks)

        response = ""
        try:
            for text, _ in stream_response(model, tokenizer, messages):
                print(text, end="", flush=True)
                response += text
        except KeyboardInterrupt:
            print("\n[interrupted]")

        print()

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": response})

        # trim history to MAX_HISTORY_TURNS pairs
        max_entries = config.MAX_HISTORY_TURNS * 2
        if len(history) > max_entries:
            history = history[-max_entries:]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="doc-qa: answer questions about a document using RAG"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ingest", metavar="FILE", help="ingest a PDF document")
    group.add_argument("--chat", action="store_true", help="start a Q&A session")
    group.add_argument("--reset", action="store_true", help="delete the index and metadata")

    args = parser.parse_args()

    if args.ingest:
        cmd_ingest(args.ingest)
    elif args.chat:
        cmd_chat()
    elif args.reset:
        cmd_reset()


if __name__ == "__main__":
    main()
