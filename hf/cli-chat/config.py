# cli-chat/config.py
import os
from pathlib import Path

MODEL_PATH = os.environ.get(
    "CHAT_MODEL_PATH",
    str(Path.home() / "models" / "gemma-4"),
)
HISTORY_PATH = os.environ.get(
    "CHAT_HISTORY_PATH",
    str(Path.home() / ".cli-chat-history.json"),
)
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
