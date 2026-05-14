import sys
from rich.console import Console

import config
import history as hist
import commands
from model import load_model, stream_response
from metrics import MetricsCollector, print_stats

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
    with console.status(f"[bold green]Loading model from {config.MODEL_PATH!r}...[/]", spinner="dots"):
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

        collector = MetricsCollector(backend=config.METRICS_BACKEND)
        collector.start()
        full_response = ""
        try:
            for text, chunk in stream_response(model, tokenizer, h):
                full_response += text
                print(text, end="", flush=True)
                collector.record(chunk)
        except KeyboardInterrupt:
            # Interrupted mid-generation: discard the incomplete user turn
            print()
            h["messages"].pop()
            print_stats(collector.finish(cancelled=True), config.METRICS_BACKEND, console)
            continue
        except Exception as e:
            # Any other generation failure: clean up and let user retry
            print()
            h["messages"].pop()
            console.print(f"[red]Generation error: {e}[/]")
            continue

        print()
        print_stats(collector.finish(), config.METRICS_BACKEND, console)
        hist.append(h, "assistant", full_response)
        try:
            hist.save(config.HISTORY_PATH, h)
        except OSError as e:
            console.print(f"[yellow]Warning: could not save history: {e}[/]")


if __name__ == "__main__":
    main()
