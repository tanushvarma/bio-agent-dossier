"""Shared Gemini client using google-genai (supports AQ. auth keys)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(Path(__file__).resolve().parent / ".env")

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")

_client: genai.Client | None = None


def get_gemini_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set. Add it to your .env file.")
    return api_key


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=get_gemini_api_key())
    return _client


def generate_text(
    prompt: str,
    *,
    model: str | None = None,
    temperature: float = 0.2,
    json_mode: bool = False,
) -> str:
    config_kwargs: dict[str, Any] = {"temperature": temperature}
    if json_mode:
        config_kwargs["response_mime_type"] = "application/json"

    response = get_client().models.generate_content(
        model=model or DEFAULT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(**config_kwargs),
    )
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("Gemini returned an empty response.")
    return text
