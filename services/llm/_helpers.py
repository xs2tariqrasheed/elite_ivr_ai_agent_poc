"""Helper functions for LLM service."""
import logging
import re

import config

from .llm_client import get_ollama_client

logger = logging.getLogger(__name__)


_DIGIT_WORDS = {
    "zero": "0", "oh": "0", "o": "0",
    "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}


def _llm_generate(prompt: str) -> str:
    """Send a prompt to the model and return the (stripped) response."""
    client = get_ollama_client()
    resp = client.generate(
        model=config.OLLAMA_MODEL,
        prompt=prompt,
        options={
            "temperature": 0,
            "num_predict": 16,
        },
    )
    return (resp.get("response") or "").strip()


def _digits_from_text(text: str) -> str:
    """Best-effort fallback: pull digits and digit-words out of ``text``."""
    out = []
    for token in re.findall(r"[A-Za-z]+|\d", text.lower()):
        if token.isdigit():
            out.append(token)
        elif token in _DIGIT_WORDS:
            out.append(_DIGIT_WORDS[token])
    return "".join(out)
