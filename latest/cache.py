"""
Disk-based JSON cache for LLM calls.
Keyed by (task, date, content_hash) to avoid repeated Gemini calls.
"""

import hashlib
import json
import os
from datetime import date, datetime

from config import CACHE_DIR


def _make_key(task: str, dt: str, content_hash: str) -> str:
    """Create a filesystem-safe cache key."""
    return f"{task}_{dt}_{content_hash}"


def _content_hash(content: str) -> str:
    """SHA-256 hash of content string, truncated to 12 hex chars."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]


def _cache_path(key: str) -> str:
    return os.path.join(CACHE_DIR, f"{key}.json")


class LLMCache:
    """Simple disk-backed JSON cache for LLM responses."""

    def __init__(self, cache_dir: str = CACHE_DIR):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def get(self, task: str, dt: str, context: str):
        """Retrieve cached response. Returns None on miss."""
        ch = _content_hash(context)
        key = _make_key(task, dt, ch)
        path = _cache_path(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("response")
        except (json.JSONDecodeError, IOError):
            return None

    def put(self, task: str, dt: str, context: str, response: dict):
        """Store a response in cache."""
        ch = _content_hash(context)
        key = _make_key(task, dt, ch)
        path = _cache_path(key)
        payload = {
            "task": task,
            "date": dt,
            "content_hash": ch,
            "cached_at": datetime.now().isoformat(),
            "response": response,
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
        except IOError as e:
            print(f"[Cache] Warning: could not write cache file {path}: {e}")

    def has(self, task: str, dt: str, context: str) -> bool:
        ch = _content_hash(context)
        key = _make_key(task, dt, ch)
        return os.path.exists(_cache_path(key))

    def stats(self) -> dict:
        """Return cache statistics."""
        files = [f for f in os.listdir(self.cache_dir) if f.endswith(".json")]
        total_size = sum(
            os.path.getsize(os.path.join(self.cache_dir, f)) for f in files
        )
        return {"entries": len(files), "total_size_mb": round(total_size / 1e6, 2)}
