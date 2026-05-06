"""Account number extraction using LLM."""
import logging
import re
from typing import Optional

from ._helpers import _llm_generate, _digits_from_text

logger = logging.getLogger(__name__)


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
