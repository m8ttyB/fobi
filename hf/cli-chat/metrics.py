# cli-chat/metrics.py
import time


class MetricsCollector:
    def __init__(self, backend: str):
        self._backend = backend
        self._start_time: float = 0.0
        self._first_token_time: float | None = None
        self._token_count: int = 0
        self._last_chunk = None

    def start(self) -> None:
        self._start_time = time.perf_counter()
        self._first_token_time = None
        self._token_count = 0
        self._last_chunk = None

    def record(self, chunk) -> None:
        if self._first_token_time is None:
            self._first_token_time = time.perf_counter()
        self._token_count += 1
        self._last_chunk = chunk

    def finish(self, cancelled: bool = False) -> dict:
        elapsed = time.perf_counter() - self._start_time

        if self._backend == "mlx":
            if cancelled or self._last_chunk is None:
                return {"tokens": None, "tps": None, "ttft": None, "cancelled": cancelled}
            return {
                "tokens": self._last_chunk.generation_tokens,
                "tps": round(self._last_chunk.generation_tps, 1),
                "ttft": None,
                "cancelled": False,
            }

        # manual backend
        ttft = round(self._first_token_time - self._start_time, 2) if self._first_token_time else None
        tps = round(self._token_count / elapsed, 1) if elapsed > 0 and self._token_count > 0 else 0.0
        return {
            "tokens": self._token_count,
            "tps": tps,
            "ttft": ttft,
            "cancelled": cancelled,
        }


def format_stats(stats: dict, backend: str) -> str:
    if stats.get("cancelled") and stats.get("tokens") is None:
        return "(generation cancelled — no metrics)"

    parts = []

    tokens = stats.get("tokens")
    if tokens is not None:
        parts.append(f"{tokens} tokens")

    tps = stats.get("tps")
    if tps is not None:
        parts.append(f"{tps} tok/s")

    ttft = stats.get("ttft")
    if ttft is not None:
        parts.append(f"TTFT {ttft}s")

    base = " · ".join(parts)

    if stats.get("cancelled"):
        return f"{base} (cancelled)"
    return base


def print_stats(stats: dict, backend: str, console) -> None:
    line = format_stats(stats, backend)
    console.print(f"[dim]{line}[/]")
