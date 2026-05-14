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
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Do not respond to questions in markdown. \
When responding, always use plain text. When offer to go go into further depth about a topic, \
number the questions to allow the user to select which one they would like to answer."
)
METRICS_BACKEND = os.environ.get("CHAT_METRICS", "manual")  # "manual" | "mlx"
