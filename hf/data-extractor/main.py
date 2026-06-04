"""CLI entry point for data-extractor.

Usage:
    python main.py FILE [--model MODEL_PATH]

Reads a .txt or .pdf file, extracts structured entities using the local model,
and displays the result as a formatted table. Truncates input to MAX_CHARS with
a visible warning if the document exceeds that limit.
"""

import argparse
import os
import sys

from pypdf import PdfReader
from rich.console import Console
from rich.table import Table
from rich import box

import config
from extractor import ExtractionError, extract
from model import load_model
from schema import ExtractedDocument

console = Console()
err_console = Console(stderr=True)


def load_text(path: str) -> str:
    """Read a .txt or .pdf file and return its text content."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        with open(path, encoding="utf-8") as f:
            return f.read()
    elif ext == ".pdf":
        reader = PdfReader(path)
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        raise ValueError(f"Unsupported file type: '{ext}'. Use .txt or .pdf")


def truncate(text: str, max_chars: int = config.MAX_CHARS) -> tuple[str, bool]:
    """Return (text, truncated). Truncates to max_chars if needed."""
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def display(result: ExtractedDocument, truncated: bool, original_length: int) -> None:
    """Render the extracted document to the terminal using rich."""
    if truncated:
        console.print(
            f"\n[bold yellow]Warning:[/bold yellow] Document truncated: "
            f"{original_length:,} → {config.MAX_CHARS:,} chars. "
            "Results cover only the first portion.\n"
        )

    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    table.add_column("Field", style="bold cyan", min_width=10)
    table.add_column("Value")

    table.add_row("Title", result.title or "[dim]—[/dim]")
    table.add_row("Topic", result.topic)

    if result.people:
        people_lines = []
        for p in result.people:
            parts = [p.name]
            if p.role:
                parts.append(p.role)
            if p.context:
                parts.append(p.context)
            people_lines.append(" — ".join(parts))
        table.add_row("People", "\n".join(people_lines))
    else:
        table.add_row("People", "[dim]none found[/dim]")

    if result.places:
        place_lines = []
        for pl in result.places:
            line = pl.name
            if pl.context:
                line += f" — {pl.context}"
            place_lines.append(line)
        table.add_row("Places", "\n".join(place_lines))
    else:
        table.add_row("Places", "[dim]none found[/dim]")

    if result.dates:
        date_lines = []
        for d in result.dates:
            line = d.date
            if d.event:
                line += f" — {d.event}"
            date_lines.append(line)
        table.add_row("Dates", "\n".join(date_lines))
    else:
        table.add_row("Dates", "[dim]none found[/dim]")

    table.add_row("Summary", result.summary)

    console.print(table)


def main() -> None:
    """Parse CLI arguments and run extraction."""
    parser = argparse.ArgumentParser(
        description="data-extractor: extract structured entities from a document"
    )
    parser.add_argument("file", metavar="FILE", help="path to a .txt or .pdf file")
    parser.add_argument(
        "--model",
        default=config.MODEL_PATH,
        metavar="MODEL_PATH",
        help=f"HuggingFace repo ID or local MLX model path (default: {config.MODEL_PATH})",
    )
    args = parser.parse_args()

    if not os.path.exists(args.file):
        err_console.print(f"[bold red]Error:[/bold red] file not found: {args.file}")
        sys.exit(1)

    try:
        text = load_text(args.file)
    except ValueError as e:
        err_console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)
    except Exception as e:
        err_console.print(f"[bold red]Error reading file:[/bold red] {e}")
        sys.exit(1)

    original_length = len(text)
    text, truncated = truncate(text)

    if not text.strip():
        err_console.print("[bold red]Error:[/bold red] no text could be extracted from the file.")
        sys.exit(1)

    console.print(f"Loading model [cyan]{args.model}[/cyan]...")
    model, tokenizer = load_model(args.model)

    console.print("Extracting structured entities...\n")
    try:
        result = extract(text, model, tokenizer)
    except ExtractionError as e:
        err_console.print(f"[bold red]Extraction failed:[/bold red] {e}")
        sys.exit(1)

    display(result, truncated=truncated, original_length=original_length)


if __name__ == "__main__":
    main()
