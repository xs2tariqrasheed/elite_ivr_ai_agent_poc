"""LLM service backed by a locally-running Ollama model (qwen2.5:1.5b).

Two helpers are exposed: ``extract_account_number`` and
``extract_phone_number``.  Both ask the LLM for a deterministic answer
and fall back to ``None`` whenever the model declines or returns junk.
"""
import logging
import re
from typing import Optional

import ollama

import config

logger = logging.getLogger(__name__)


_client: Optional[ollama.Client] = None


def get_ollama_client() -> ollama.Client:
    """Return the lazily-initialised Ollama client."""
    global _client
    if _client is None:
        logger.info("Creating Ollama client at %s", config.OLLAMA_HOST)
        _client = ollama.Client(host=config.OLLAMA_HOST)
    return _client


def warm_up_model() -> None:
    """Issue a tiny request so the model is loaded into Ollama's RAM.

    Called once at app startup so the first real request isn't slow.
    """
    try:
        client = get_ollama_client()
        logger.info("Warming up Ollama model %s", config.OLLAMA_MODEL)
        client.generate(
            model=config.OLLAMA_MODEL,
            prompt="ok",
            options={"num_predict": 1},
        )
        logger.info("Ollama model %s warmed up", config.OLLAMA_MODEL)
    except Exception:
        logger.exception("Ollama warm-up failed (continuing anyway)")


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


_DIGIT_WORDS = {
    "zero": "0", "oh": "0", "o": "0",
    "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}


def _digits_from_text(text: str) -> str:
    """Best-effort fallback: pull digits and digit-words out of ``text``."""
    out = []
    for token in re.findall(r"[A-Za-z]+|\d", text.lower()):
        if token.isdigit():
            out.append(token)
        elif token in _DIGIT_WORDS:
            out.append(_DIGIT_WORDS[token])
    return "".join(out)


def extract_account_number(text: str) -> Optional[str]:
    """Use the LLM to extract a 4-digit account number from ``text``.

    Returns the 4-digit string on success, or ``None`` if no number
    could be extracted.
    """
    text = (text or "").strip()
    if not text:
        return None

    prompt = (
        "You are an information extraction assistant. The user said the "
        "following sentence which may contain a 4-digit account number "
        "(spoken as digits or digit words). Extract the 4-digit account "
        "number and respond with ONLY those 4 digits and nothing else. "
        "If there is no 4-digit account number, respond with NOT_FOUND.\n\n"
        f"Sentence: {text}\n"
        "Answer:"
    )
    try:
        raw = _llm_generate(prompt)
    except Exception:
        logger.exception("Ollama call failed for account-number extraction")
        raw = ""

    logger.debug("LLM account-number raw response: %r", raw)

    if "NOT_FOUND" in raw.upper():
        digits = _digits_from_text(text)
    else:
        digits = re.sub(r"\D", "", raw)
        if len(digits) != 4:
            digits = _digits_from_text(text)

    if len(digits) >= 4:
        digits = digits[:4]
    if len(digits) == 4:
        logger.info("Extracted account number: %s", digits)
        return digits

    logger.info("Could not extract a 4-digit account number from %r", text)
    return None


def extract_phone_number(text: str) -> Optional[str]:
    """Use the LLM to extract a 10-digit phone number from ``text``.

    Returns the 10-digit string on success, or ``None`` if no number
    could be extracted.
    """
    text = (text or "").strip()
    if not text:
        return None

    prompt = (
        "You are an information extraction assistant. The user said the "
        "following sentence which may contain a 10-digit US phone number "
        "(spoken as digits or digit words). Extract the 10-digit phone "
        "number and respond with ONLY those 10 digits and nothing else. "
        "If there is no 10-digit phone number, respond with NOT_FOUND.\n\n"
        f"Sentence: {text}\n"
        "Answer:"
    )
    try:
        raw = _llm_generate(prompt)
    except Exception:
        logger.exception("Ollama call failed for phone-number extraction")
        raw = ""

    logger.debug("LLM phone-number raw response: %r", raw)

    if "NOT_FOUND" in raw.upper():
        digits = _digits_from_text(text)
    else:
        digits = re.sub(r"\D", "", raw)
        if len(digits) != 10:
            digits = _digits_from_text(text)

    # Strip leading "1" country code if it makes the number 11 digits.
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]

    if len(digits) == 10:
        logger.info("Extracted phone number: %s", digits)
        return digits

    logger.info("Could not extract a 10-digit phone number from %r", text)
    return None
