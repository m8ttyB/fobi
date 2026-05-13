# cli-chat/history.py
import json
from pathlib import Path


def load(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return {"system": "", "messages": []}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def save(path: str, history: dict) -> None:
    Path(path).write_text(json.dumps(history, indent=2))


def append(history: dict, role: str, content: str) -> None:
    history["messages"].append({"role": role, "content": content})
