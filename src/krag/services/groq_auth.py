"""Groq API access via the Secure Vault surrogate pattern.

Critical gotchas (verified live 2026-09-30):
1. Cloudflare blocks non-browser User-Agents (HTTP 403, error 1010) — every
   request MUST carry a browser-like User-Agent header.
2. Credentials are applied with add_surrogate_to_request(req, "custom.groq",
   allowed_hosts=("api.groq.com",)) — never handled as raw strings here.
3. gpt-oss models return reasoning tokens; use reasoning_effort="low" and a
   generous max_tokens budget or the model spends the whole budget reasoning
   and returns empty content with finish_reason="length".

The real API key is NEVER visible to this process and is never written to
disk, logs, or chat output.
"""

from __future__ import annotations

import json
import logging
import sys
import urllib.error
import urllib.request
from typing import Any, cast

logger = logging.getLogger(__name__)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_DEFAULT_MODEL = "openai/gpt-oss-20b"  # default benchmark model
GROQ_QUALITY_MODEL = "openai/gpt-oss-120b"  # quality comparison runs

# Browser-like UA is REQUIRED — Cloudflare 403s urllib's default UA.
BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


def _credential_module() -> tuple[type[BaseException], Any, Any]:
    sys.path.insert(0, "/opt/hatch/skills/skill-creator/bin")
    from dynamic_credentials import (
        DynamicCredentialError,
        add_surrogate_to_request,
        dynamic_credential_entry,
    )

    return DynamicCredentialError, add_surrogate_to_request, dynamic_credential_entry


def groq_available() -> bool:
    """True if a Groq credential can be resolved right now."""
    try:
        _, _, dynamic_credential_entry = _credential_module()
        entry = dynamic_credential_entry("custom.groq")
        return bool(entry and entry.get("surrogate"))
    except Exception:  # noqa: BLE001 - unavailable is a normal state
        return False


def groq_chat(
    messages: list[dict[str, str]],
    model: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    timeout: int = 120,
) -> dict[str, Any]:
    """POST to Groq /chat/completions. Returns the parsed JSON body.

    Raises RuntimeError on transport/API errors (callers decide retry policy).
    """
    DynamicCredentialError, add_surrogate_to_request, _ = _credential_module()
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "reasoning_effort": "low",  # gpt-oss: keep reasoning from eating the budget
    }
    req = urllib.request.Request(
        f"{GROQ_BASE_URL}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "User-Agent": BROWSER_UA},
        method="POST",
    )
    try:
        add_surrogate_to_request(req, "custom.groq", allowed_hosts=("api.groq.com",))
    except DynamicCredentialError as exc:
        raise RuntimeError(f"Groq credential unavailable: {exc}") from exc
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return cast("dict[str, Any]", json.loads(resp.read().decode()))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")[:500]
        raise RuntimeError(f"Groq API HTTP {exc.code}: {body}") from exc
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Groq request failed: {exc}") from exc


def extract_content(response: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
    """Defensively pull (content, usage, finish_reason) from a response.

    gpt-oss models can return empty ``content`` with all tokens spent on
    reasoning (``finish_reason="length"``). We do NOT substitute the
    internal reasoning trace as the answer — callers decide (retry with a
    larger budget, or surface an explicit failure).
    """
    choices = response.get("choices") or []
    first = choices[0] if choices else {}
    message = first.get("message", {}) or {}
    content = (message.get("content") or "").strip()
    usage = response.get("usage", {}) or {}
    finish_reason = str(first.get("finish_reason") or "")
    return content, usage, finish_reason
