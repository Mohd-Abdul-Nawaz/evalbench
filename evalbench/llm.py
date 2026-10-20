"""Model clients for EvalBench.

Two implementations:

* ``MockModel`` — deterministic offline model for demos and tests.
* ``OpenAICompatibleClient`` — talks to any OpenAI-compatible
  ``/chat/completions`` endpoint (OpenAI, Together, Groq, Ollama, vLLM...)
  using only the Python standard library.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from typing import Protocol


class ModelClient(Protocol):
    """Anything that can turn (system, user) prompts into a completion."""

    def complete(self, system_prompt: str, user_input: str) -> str: ...


class ModelCallError(RuntimeError):
    """Raised when a model call fails after retries."""


# Deterministic canned answers used by MockModel. They intentionally mix
# compliant and non-compliant behaviour so the demo dashboard shows both
# passes and failures.
_MOCK_ANSWERS = [
    "Sure — here is a concise summary.\n## Summary\nThe launch moved to March 15 due to vendor delays.",
    "The answer is 42, without a doubt!",
    '{"status": "ok", "confidence": 0.9, "answer": "Paris is the capital of France."}',
    "I cannot help with that request.",
    "## Overview\n- Item one\n- Item two\n\n## Next steps\nTODO: fill this in later.",
    "Tokyo hosts 37 million people in the greater metropolitan area, making it the largest city on Earth.",
]


class MockModel:
    """Deterministic offline stand-in for a real model.

    Args:
        mode: "mixed" (default) varies answers deterministically by input;
            "pass" always returns a JSON object that satisfies typical checks;
            "fail" always returns an empty refusal-style string.
        latency_s: Simulated latency per call (keeps progress bars realistic).
    """

    def __init__(self, mode: str = "mixed", latency_s: float = 0.05) -> None:
        if mode not in ("mixed", "pass", "fail"):
            raise ValueError("mode must be 'mixed', 'pass' or 'fail'")
        self.mode = mode
        self.latency_s = latency_s

    def complete(self, system_prompt: str, user_input: str) -> str:
        time.sleep(self.latency_s)
        if self.mode == "pass":
            return '{"status": "ok", "confidence": 0.95, "answer": "This is a compliant mock answer."}'
        if self.mode == "fail":
            return "I refuse to answer."
        digest = hashlib.sha256((system_prompt + "\n" + user_input).encode("utf-8")).digest()
        index = digest[0] % len(_MOCK_ANSWERS)
        return _MOCK_ANSWERS[index]


class OpenAICompatibleClient:
    """Minimal client for OpenAI-compatible chat completion APIs.

    Args:
        model: Model name, e.g. "gpt-4o-mini".
        api_key: Bearer token. Defaults to the ``OPENAI_API_KEY`` env var.
        base_url: API root. Defaults to ``OPENAI_BASE_URL`` env var or
            ``https://api.openai.com/v1``.
        temperature: Sampling temperature.
        timeout_s: Per-request timeout.
        max_retries: Retries on transient failures (429/5xx/network).
        retry_backoff_s: Base backoff, doubled per retry.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        timeout_s: float = 60.0,
        max_retries: int = 3,
        retry_backoff_s: float = 2.0,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s

    def complete(self, system_prompt: str, user_input: str) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_input})
        payload = json.dumps(
            {"model": self.model, "messages": messages, "temperature": self.temperature}
        ).encode("utf-8")

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    body = json.loads(response.read().decode("utf-8"))
                return body["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                last_error = RuntimeError(f"HTTP {exc.code} from API: {detail}")
                if exc.code not in (429, 500, 502, 503, 504):
                    raise last_error from exc
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
                last_error = RuntimeError(f"Model call failed: {exc}")
            if attempt < self.max_retries:
                time.sleep(self.retry_backoff_s * (2**attempt))
        raise ModelCallError(str(last_error))