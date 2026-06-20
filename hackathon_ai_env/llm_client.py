"""LLM client that uses the platform-injected API_BASE_URL and API_KEY."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _get_config() -> tuple[str, str]:
    """Return (base_url, api_key) from environment variables."""
    base_url = os.environ.get("API_BASE_URL", "").strip().rstrip("/")
    api_key = os.environ.get("API_KEY", "").strip()
    return base_url, api_key


def llm_enhance(query: str, base_answer: str, agent_name: str) -> str:
    """Enhance a rule-based answer using the LLM proxy.

    Falls back to the original *base_answer* when the proxy is
    unavailable or no credentials are configured.
    """
    base_url, api_key = _get_config()
    if not base_url or not api_key:
        return base_answer

    system_prompt = (
        "You are a helpful AI assistant that is part of a multi-agent system. "
        f"The '{agent_name}' agent has drafted a response. "
        "Your job is to refine and polish this draft into a clear, well-structured, "
        "and helpful final answer. Keep the core information intact but improve "
        "clarity, tone, and completeness. Be concise."
    )

    user_prompt = (
        f"User query: {query}\n\n"
        f"Draft response from {agent_name} agent:\n{base_answer}\n\n"
        "Please refine this into a polished final answer."
    )

    payload: dict[str, Any] = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 1024,
        "temperature": 0.7,
    }

    try:
        url = f"{base_url}/chat/completions"
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        with urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
            enhanced = result["choices"][0]["message"]["content"].strip()
            return enhanced if enhanced else base_answer
    except (HTTPError, URLError, TimeoutError, KeyError, IndexError, Exception) as exc:
        print(f"[LLM] Enhancement failed ({exc}), using base answer")
        return base_answer
