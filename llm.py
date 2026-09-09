# -*- coding: utf-8 -*-
"""Minimal multi-provider LLM client.

Deliberately dependency-free (urllib, not the vendor SDKs) so the repo installs
with just scikit-learn and the pipeline stays reproducible in under 15 minutes.

Providers are tried in order: Anthropic, OpenAI, then a local Ollama. Every call
returns (text, input_tokens, output_tokens) so cost per unit of work is measured
rather than assumed.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

OLLAMA_URL = "http://127.0.0.1:11434"


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, i: int, o: int) -> None:
        self.calls += 1
        self.input_tokens += i
        self.output_tokens += o

    def __str__(self) -> str:
        return (f"{self.calls} calls, {self.input_tokens:,} in / "
                f"{self.output_tokens:,} out tokens")


def _post(url: str, payload: dict, headers: dict, timeout: int = 180) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _anthropic(system: str, prompt: str, model: str, max_tokens: int):
    r = _post(
        "https://api.anthropic.com/v1/messages",
        {"model": model, "max_tokens": max_tokens, "temperature": 0,
         "system": system, "messages": [{"role": "user", "content": prompt}]},
        {"x-api-key": os.environ["ANTHROPIC_API_KEY"],
         "anthropic-version": "2023-06-01"},
    )
    u = r.get("usage", {})
    return r["content"][0]["text"], u.get("input_tokens", 0), u.get("output_tokens", 0)


def _openai(system: str, prompt: str, model: str, max_tokens: int):
    r = _post(
        "https://api.openai.com/v1/chat/completions",
        {"model": model, "temperature": 0, "max_tokens": max_tokens,
         "messages": [{"role": "system", "content": system},
                      {"role": "user", "content": prompt}]},
        {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
    )
    u = r.get("usage", {})
    return (r["choices"][0]["message"]["content"],
            u.get("prompt_tokens", 0), u.get("completion_tokens", 0))


def _ollama(system: str, prompt: str, model: str, max_tokens: int):
    r = _post(
        f"{OLLAMA_URL}/api/chat",
        {"model": model, "stream": False,
         "options": {"temperature": 0, "num_predict": max_tokens},
         "messages": [{"role": "system", "content": system},
                      {"role": "user", "content": prompt}]},
        {}, timeout=600,
    )
    return (r["message"]["content"],
            r.get("prompt_eval_count", 0), r.get("eval_count", 0))


BACKENDS = {
    "anthropic": (_anthropic, "claude-haiku-4-5-20251001"),
    "openai": (_openai, "gpt-4o-mini"),
    "ollama": (_ollama, "qwen2.5:7b"),
}


def detect_backend() -> str:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    try:
        urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3)
        return "ollama"
    except Exception:
        raise SystemExit(
            "No LLM backend available. Set ANTHROPIC_API_KEY or OPENAI_API_KEY, "
            "or run Ollama with a model pulled (ollama pull qwen2.5:7b)."
        )


def preflight(backend: str, model: str) -> None:
    """Ollama answers /api/tags with nothing pulled, so reachable != usable."""
    if backend != "ollama":
        return
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as r:
            have = [m["name"] for m in json.loads(r.read()).get("models", [])]
    except Exception as e:
        raise SystemExit(f"Ollama unreachable: {e}")
    if not have:
        raise SystemExit(
            "Ollama is running but has no models pulled.\n"
            f"  Run:  ollama pull {model}\n"
            "  Or set ANTHROPIC_API_KEY / OPENAI_API_KEY."
        )
    if not any(m == model or m.startswith(model + ":") for m in have):
        raise SystemExit(
            f"Model '{model}' not pulled. Available: {', '.join(have)}\n"
            f"  Run:  ollama pull {model}   (or pass --model)"
        )


class Client:
    def __init__(self, backend: str | None = None, model: str | None = None):
        self.backend = backend or detect_backend()
        fn, default_model = BACKENDS[self.backend]
        self._fn = fn
        self.model = model or default_model
        preflight(self.backend, self.model)
        self.usage = Usage()

    def complete(self, system: str, prompt: str, max_tokens: int = 2048,
                 retries: int = 3) -> str:
        last: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                text, i_tok, o_tok = self._fn(system, prompt, self.model, max_tokens)
                self.usage.add(i_tok, o_tok)
                return text
            except (urllib.error.HTTPError, urllib.error.URLError,
                    KeyError, json.JSONDecodeError) as e:
                last = e
                if attempt < retries:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM call failed after {retries} attempts: {last}")

    def __str__(self) -> str:
        return f"{self.backend}/{self.model} ({self.usage})"
