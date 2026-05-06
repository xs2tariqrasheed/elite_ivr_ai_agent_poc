"""Phone number extraction using LLM."""
import logging
import re
from typing import Optional

from ._helpers import _llm_generate, _digits_from_text

logger = logging.getLogger(__name__)


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
