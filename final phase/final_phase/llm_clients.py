"""
LLM backends for persona ablation.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Optional
import json
import os
import re

import requests

from .config import MODULE_ROOT


def _content_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts = [_content_to_text(item) for item in value]
        return "\n".join(part for part in parts if part).strip()
    if isinstance(value, dict):
        parts = []
        for key in ("content", "text", "output_text", "input_text", "value", "arguments"):
            text = _content_to_text(value.get(key))
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
    return str(value)


def _extract_json(text: Any) -> Optional[dict]:
    text = _content_to_text(text).strip()
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    inline = re.search(r"\{.*\}", text, re.DOTALL)
    if inline:
        try:
            return json.loads(inline.group(0))
        except json.JSONDecodeError:
            return None
    return None


class DiskResponseCache:
    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or (MODULE_ROOT / "cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _key(self, provider: str, model: str, prompt: str) -> str:
        digest = sha256(prompt.encode("utf-8")).hexdigest()[:16]
        clean_provider = re.sub(r"[^a-zA-Z0-9]+", "_", provider)
        clean_model = re.sub(r"[^a-zA-Z0-9]+", "_", model)
        return f"{clean_provider}_{clean_model}_{digest}"

    def get(self, provider: str, model: str, prompt: str) -> Optional[dict]:
        path = self._path(self._key(provider, model, prompt))
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            return payload.get("response")
        except (OSError, json.JSONDecodeError):
            return None

    def put(self, provider: str, model: str, prompt: str, response: dict) -> None:
        path = self._path(self._key(provider, model, prompt))
        payload = {
            "provider": provider,
            "model": model,
            "response": response,
        }
        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except OSError:
            return


@dataclass
class ChatCompletion:
    text: str
    json_payload: Optional[dict]


class BaseChatClient:
    provider = "base"

    def __init__(self, model: str):
        self.model = model
        self.available = False
        self.cache = DiskResponseCache()

    def complete_json(self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> Optional[ChatCompletion]:
        raise NotImplementedError


class OpenRouterChatClient(BaseChatClient):
    provider = "openrouter"

    def __init__(self, model: str):
        super().__init__(model)
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
        self.base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")
        self.available = bool(self.api_key)

    def complete_json(self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> Optional[ChatCompletion]:
        if not self.available:
            return None
        prompt = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_prompt}"
        cached = self.cache.get(self.provider, self.model, prompt)
        if cached is not None:
            return ChatCompletion(text=json.dumps(cached), json_payload=cached)

        try:
            resp = requests.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=90,
            )
            resp.raise_for_status()
            data = resp.json()
            message = data["choices"][0]["message"]
            text = _content_to_text(message.get("content"))
            payload = _extract_json(text)
            if payload is not None:
                self.cache.put(self.provider, self.model, prompt, payload)
            return ChatCompletion(text=text, json_payload=payload)
        except (requests.RequestException, KeyError, ValueError, TypeError):
            return None


class OllamaChatClient(BaseChatClient):
    provider = "ollama"

    def __init__(self, model: str):
        super().__init__(model)
        self.base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.available = self._check_available()

    def _check_available(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code != 200:
                return False
            models = [m["name"] for m in resp.json().get("models", [])]
            return any(self.model in m or m.startswith(self.model) for m in models)
        except requests.RequestException:
            return False

    def complete_json(self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> Optional[ChatCompletion]:
        if not self.available:
            return None
        prompt = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_prompt}"
        cached = self.cache.get(self.provider, self.model, prompt)
        if cached is not None:
            return ChatCompletion(text=json.dumps(cached), json_payload=cached)

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            text = _content_to_text(data.get("message", {}).get("content", ""))
            payload = _extract_json(text)
            if payload is not None:
                self.cache.put(self.provider, self.model, prompt, payload)
            return ChatCompletion(text=text, json_payload=payload)
        except (requests.RequestException, ValueError):
            return None


def make_chat_client(spec: str) -> Optional[BaseChatClient]:
    if spec == "rule":
        return None
    if ":" not in spec:
        raise ValueError(f"Unsupported LLM spec: {spec}")
    provider, model = spec.split(":", 1)
    provider = provider.strip().lower()
    model = model.strip()
    if provider == "openrouter":
        return OpenRouterChatClient(model)
    if provider == "ollama":
        return OllamaChatClient(model)
    raise ValueError(f"Unsupported LLM provider: {provider}")
