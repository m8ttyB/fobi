# cli-chat/tests/test_metrics.py
import time
from unittest.mock import MagicMock
import metrics


def make_chunk(text, generation_tps=45.0, generation_tokens=10, prompt_tokens=20):
    chunk = MagicMock()
    chunk.text = text
    chunk.generation_tps = generation_tps
    chunk.generation_tokens = generation_tokens
    chunk.prompt_tokens = prompt_tokens
    return chunk


# --- manual backend ---

def test_manual_records_token_count():
    c = metrics.MetricsCollector(backend="manual")
    c.start()
    c.record(make_chunk("Hello"))
    c.record(make_chunk(" world"))
    stats = c.finish()
    assert stats["tokens"] == 2


def test_manual_reports_ttft():
    c = metrics.MetricsCollector(backend="manual")
    c.start()
    time.sleep(0.05)
    c.record(make_chunk("Hi"))
    stats = c.finish()
    assert stats["ttft"] >= 0.05


def test_manual_partial_stats_on_cancel():
    c = metrics.MetricsCollector(backend="manual")
    c.start()
    c.record(make_chunk("Hello"))
    c.record(make_chunk(" world"))
    stats = c.finish(cancelled=True)
    assert stats["tokens"] == 2
    assert stats["cancelled"] is True


def test_manual_tps_is_nonzero_after_tokens():
    c = metrics.MetricsCollector(backend="manual")
    c.start()
    c.record(make_chunk("a"))
    c.record(make_chunk("b"))
    c.record(make_chunk("c"))
    stats = c.finish()
    assert stats["tps"] > 0


# --- mlx backend ---

def test_mlx_uses_chunk_tps():
    c = metrics.MetricsCollector(backend="mlx")
    c.start()
    c.record(make_chunk("Hello", generation_tps=47.5))
    c.record(make_chunk(" world", generation_tps=47.5))
    stats = c.finish()
    assert stats["tps"] == 47.5


def test_mlx_uses_chunk_token_count():
    c = metrics.MetricsCollector(backend="mlx")
    c.start()
    c.record(make_chunk("Hi", generation_tokens=5))
    c.record(make_chunk("!", generation_tokens=6))
    stats = c.finish()
    assert stats["tokens"] == 6


def test_mlx_has_no_ttft():
    c = metrics.MetricsCollector(backend="mlx")
    c.start()
    c.record(make_chunk("Hi"))
    stats = c.finish()
    assert stats.get("ttft") is None


def test_mlx_cancelled_returns_none_stats():
    c = metrics.MetricsCollector(backend="mlx")
    c.start()
    c.record(make_chunk("Hi"))
    stats = c.finish(cancelled=True)
    assert stats["tps"] is None
    assert stats["tokens"] is None


# --- format_stats ---

def test_format_stats_manual_complete():
    stats = {"tokens": 42, "tps": 38.1, "ttft": 1.23, "cancelled": False}
    line = metrics.format_stats(stats, backend="manual")
    assert "42" in line
    assert "38.1" in line
    assert "1.23" in line


def test_format_stats_manual_cancelled():
    stats = {"tokens": 10, "tps": 20.0, "ttft": 0.5, "cancelled": True}
    line = metrics.format_stats(stats, backend="manual")
    assert "cancel" in line.lower()


def test_format_stats_mlx_no_ttft():
    stats = {"tokens": 30, "tps": 45.0, "ttft": None, "cancelled": False}
    line = metrics.format_stats(stats, backend="mlx")
    assert "TTFT" not in line


def test_format_stats_mlx_cancelled():
    stats = {"tokens": None, "tps": None, "ttft": None, "cancelled": True}
    line = metrics.format_stats(stats, backend="mlx")
    assert "cancel" in line.lower()
