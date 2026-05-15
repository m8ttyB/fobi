# writing-assistant/config.py
import os

MODEL_PATH = os.environ.get(
    "WA_MODEL_PATH",
    # "mlx-community/gemma-3-4b-it-4bit",
    "m8ttyb/gemma-4-E4B-it-4bit",
)
HOST = os.environ.get("WA_HOST", "127.0.0.1")
PORT = int(os.environ.get("WA_PORT", "8000"))
