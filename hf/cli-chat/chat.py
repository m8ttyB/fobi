import sys
from rich.console import Console

import config
import history as hist
import commands
from model import load_model, stream_response

console = Console()


def main() -> None:
    # Load history
    h = hist.load(config.HISTORY_PATH)
    if h is None:
        console.print("[yellow]Warning: history file was corrupted — starting fresh.[/]")
        h = {"system": config.DEFAULT_SYSTEM_PROMPT, "messages": []}
    if not h.get("system"):
        h["system"] = config.DEFAULT_SYSTEM_PROMPT

    # Load model
    with console.status("[bold green]Loading Gemma 4...[/]", spinner="dots"):
        try:
            model, tokenizer = load_model(config.MODEL_PATH)
        except Exception as e:
            console.print(f"[red]Failed to load model at {config.MODEL_PATH!r}:[/] {e}")
            sys.exit(1)

    console.print("[dim]Model loaded. Type /help for commands, Ctrl+C to exit.[/]\n")

    while True:
        try:
            user_input = console.input("[bold green]you > [/]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/]")
            break

        if not user_input:
            continue

        if commands.handle(user_input, h, config.HISTORY_PATH):
            continue

        hist.append(h, "user", user_input)
        console.print("[dim]gemma >[/] ", end="")

        full_response = ""
        try:
            for token in stream_response(model, tokenizer, h):
                full_response += token
                print(token, end="", flush=True)
        except KeyboardInterrupt:
            # Interrupted mid-generation: discard the incomplete user turn
            print()
            h["messages"].pop()
            console.print("[dim](generation cancelled)[/]")
            continue

        print()
        hist.append(h, "assistant", full_response)
        hist.save(config.HISTORY_PATH, h)


if __name__ == "__main__":
    main()
