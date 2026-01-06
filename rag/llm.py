from __future__ import annotations

import json
import os
import urllib.request


def llm_available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY") or os.getenv("LOCAL_LLM_ENDPOINT"))


def generate_with_openai(prompt: str) -> str | None:
    try:
        from openai import OpenAI
    except Exception:
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception:
        return None


def generate_with_local(prompt: str) -> str | None:
    endpoint = os.getenv("LOCAL_LLM_ENDPOINT")
    if not endpoint:
        return None
    model = os.getenv("LOCAL_LLM_MODEL", "llama3")
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
    try:
        req = urllib.request.Request(endpoint, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
        data = json.loads(body)
        return data.get("response") or data.get("text")
    except Exception:
        return None


def generate_text(prompt: str) -> str | None:
    text = generate_with_openai(prompt)
    if text:
        return text
    return generate_with_local(prompt)
