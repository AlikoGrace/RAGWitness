from __future__ import annotations

import json
import time
import urllib.request
from typing import Any, Dict

_MAX_RETRIES = 3
_RETRY_DELAY = 10  # seconds between retries


def ollama_chat(
    model: str,
    system: str,
    user: str,
    temperature: float = 0.2,
    top_p: float = 0.9,
    seed: int | None = None,
) -> Dict[str, Any]:
    """
    Minimal Ollama chat client (local).
    Uses http://localhost:11434/api/chat
    Retries up to _MAX_RETRIES times on timeout or connection error.
    """
    url = "http://localhost:11434/api/chat"
    options: Dict[str, Any] = {
        "temperature": temperature,
        "top_p": top_p,
    }
    if seed is not None:
        options["seed"] = seed

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": options,
        "stream": False,
    }

    data = json.dumps(payload).encode("utf-8")
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(
                url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (TimeoutError, OSError) as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES:
                print(f"  [ollama] attempt {attempt} failed ({exc}), retrying in {_RETRY_DELAY}s…")
                time.sleep(_RETRY_DELAY)
    raise RuntimeError(f"Ollama failed after {_MAX_RETRIES} attempts") from last_exc
