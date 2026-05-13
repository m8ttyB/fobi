# cli-chat/commands.py
import sys
import history as hist
from rich.console import Console
from rich.panel import Panel

console = Console()


def handle(line: str, history: dict, path: str) -> bool:
    if not line.startswith("/"):
        return False

    parts = line.strip().split(None, 1)
    cmd = parts[0].lower()
    args = parts[1].strip() if len(parts) > 1 else ""

    if cmd == "/quit":
        console.print("[dim]Goodbye![/]")
        sys.exit(0)

    elif cmd == "/clear":
        history["messages"].clear()
        hist.save(path, history)
        console.print("[dim]History cleared.[/]")

    elif cmd == "/history":
        _show_history(history)

    elif cmd == "/system":
        prompt = args.strip().strip('"').strip("'")
        if not prompt:
            console.print("[red]Usage: /system \"your prompt here\"[/]")
        else:
            history["system"] = prompt
            history["messages"].clear()
            hist.save(path, history)
            console.print("[dim]System prompt updated. History cleared.[/]")

    elif cmd == "/help":
        console.print(
            "[dim]Commands:[/]\n"
            "  /clear           — clear conversation history\n"
            "  /history         — show conversation history\n"
            "  /system \"...\"    — set system prompt and clear history\n"
            "  /quit            — exit\n"
            "  /help            — show this message"
        )

    else:
        console.print(f"[red]Unknown command: {cmd}. Type /help for commands.[/]")

    return True


def _show_history(history: dict) -> None:
    if history.get("system"):
        console.print(Panel(history["system"], title="system", border_style="dim"))
    if not history["messages"]:
        console.print("[dim]No messages yet.[/]")
        return
    for msg in history["messages"]:
        style = "green" if msg["role"] == "user" else "blue"
        console.print(Panel(msg["content"], title=msg["role"], border_style=style))
