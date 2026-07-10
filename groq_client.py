"""
groq_client.py — Thin wrapper around the Groq Chat Completions API.

Why Groq: it runs open models (Llama-3.3-70B etc.) on custom LPU hardware,
so responses come back in a few hundred ms instead of seconds — important
for a chat feature embedded in a live trading UI. Free tier is generous
enough for a resume/demo project.

Install:  pip install groq
Env var:  GROQ_API_KEY=gsk_...   (get one free at https://console.groq.com)
"""
import os
from groq import Groq

_client = None


def get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Add it to your .env / Fly secrets."
            )
        _client = Groq(api_key=api_key)
    return _client


# llama-3.3-70b-versatile is a strong general model on Groq's free tier.
# Swap to "llama-3.1-8b-instant" if you want lower latency for the chat route.
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def chat(messages: list, model: str = None, temperature: float = 0.4,
         max_tokens: int = 700, json_mode: bool = False) -> str:
    """
    messages: list of {"role": "system"|"user"|"assistant", "content": str}
    json_mode: if True, forces the model to return valid JSON (Groq supports
               OpenAI-style response_format).

    Raises RuntimeError with a clear message on API failure instead of
    letting a raw exception bubble up to the chat route — callers can
    catch this and show a friendly "AI temporarily unavailable" message.
    """
    client = get_client()
    kwargs = dict(
        model=model or DEFAULT_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as e:
        raise RuntimeError(f"Groq API call failed: {e}") from e

    if not resp.choices:
        raise RuntimeError("Groq API returned no choices in response.")

    return resp.choices[0].message.content