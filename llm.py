# -*- coding: utf-8 -*-
"""Minimal multi-provider LLM client.

Deliberately dependency-free (urllib, not the vendor SDKs) so the repo installs
with just scikit-learn and the pipeline stays reproducible in under 15 minutes.

Providers are tried in order: Anthropic, OpenAI, Gemini, then a local Ollama.
Keys are read from the environment or a local .env (gitignored). Every call
returns (text, input_tokens, output_tokens) so cost per unit of work is measured
rather than assumed.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

OLLAMA_URL = "http://127.0.0.1:11434"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models"


def _load_dotenv() -> None:
    """KEY=VALUE lines from ./.env into the environment, never overriding."""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(p):
        return
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


def utf8_console() -> None:
    """Windows consoles default to cp1252; tweets contain emoji. Never crash on print."""
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


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


def _gemini(system: str, prompt: str, model: str, max_tokens: int):
    key = os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"]
    r = _post(
        f"{GEMINI_URL}/{model}:generateContent",
        {"systemInstruction": {"parts": [{"text": system}]},
         "contents": [{"role": "user", "parts": [{"text": prompt}]}],
         "generationConfig": {"temperature": 0, "maxOutputTokens": max_tokens}},
        {"x-goog-api-key": key},
    )
    parts = r["candidates"][0]["content"]["parts"]
    u = r.get("usageMetadata", {})
    return ("".join(p.get("text", "") for p in parts),
            u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0))


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
    # flash-lite, not flash: the free tier allows far more requests/day on
    # lite, and a full evaluate.py run is ~450 calls. (2.5-lite is closed to
    # new users as of 2026-09; the API's own 404 points at 3.5-lite.)
    "gemini": (_gemini, "gemini-3.5-flash-lite"),
    "ollama": (_ollama, "qwen2.5:3b"),
}


def detect_backend() -> str:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "gemini"
    try:
        urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3)
        return "ollama"
    except Exception:
        raise SystemExit(
            "No LLM backend available. Set ANTHROPIC_API_KEY, OPENAI_API_KEY or GEMINI_API_KEY, "
            "or run Ollama with a model pulled (ollama pull qwen2.5:3b)."
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
            "  Or set ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY."
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
                 retries: int = 5) -> str:
        last: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                text, i_tok, o_tok = self._fn(system, prompt, self.model, max_tokens)
                self.usage.add(i_tok, o_tok)
                return text
            except (urllib.error.HTTPError, urllib.error.URLError,
                    KeyError, IndexError, json.JSONDecodeError) as e:
                last = e
                if attempt < retries:
                    # 429 on a free tier is a per-minute window, not a fault:
                    # wait it out rather than burn the remaining retries.
                    rate_limited = isinstance(e, urllib.error.HTTPError) and e.code == 429
                    time.sleep(max(20, 2 ** attempt) if rate_limited else 2 ** attempt)
        raise RuntimeError(f"LLM call failed after {retries} attempts: {last}")

    def __str__(self) -> str:
        return f"{self.backend}/{self.model} ({self.usage})"
