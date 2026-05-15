"""Pickup date and time extraction using LLM."""
import json
import logging
import re
from datetime import datetime
from typing import Optional, Tuple

from ._helpers import _llm_generate

logger = logging.getLogger(__name__)


_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_TIME_RE = re.compile(r"\b(\d{1,2}:\d{2}(?::\d{2})?)\b")


def _normalise_time(value: str) -> Optional[str]:
    """Coerce ``HH:MM`` or ``HH:MM:SS`` to a zero-padded ``HH:MM:SS``."""
    match = _TIME_RE.search(value)
    if not match:
        return None
    parts = match.group(1).split(":")
    try:
        hour = int(parts[0])
        minute = int(parts[1])
        second = int(parts[2]) if len(parts) > 2 else 0
    except ValueError:
        return None
    if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
        return None
    return f"{hour:02d}:{minute:02d}:{second:02d}"


def _normalise_date(value: str) -> Optional[str]:
    """Validate a ``YYYY-MM-DD`` string."""
    match = _DATE_RE.search(value)
    if not match:
        return None
    try:
        datetime.strptime(match.group(1), "%Y-%m-%d")
    except ValueError:
        return None
    return match.group(1)


def _parse_llm_response(raw: str) -> Tuple[Optional[str], Optional[str]]:
    """Pull date/time out of the LLM response.

    Tries strict JSON first, then falls back to regex scraping.
    """
    if not raw:
        return None, None

    # Strict JSON path. The model is told to emit a JSON object.
    json_match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if json_match:
        try:
            payload = json.loads(json_match.group(0))
        except (json.JSONDecodeError, TypeError):
            payload = None
        if isinstance(payload, dict):
            date = _normalise_date(str(payload.get("date") or ""))
            time = _normalise_time(str(payload.get("time") or ""))
            if date or time:
                return date, time

    # Regex fallback over the entire response.
    return _normalise_date(raw), _normalise_time(raw)


def extract_pickup_date_time(
    text: str, today: Optional[datetime] = None
) -> Tuple[Optional[str], Optional[str]]:
    """Extract a pickup date and time from a spoken sentence.

    ``text`` is a speech-to-text transcript that may contain the date
    and time in any order, in any format, possibly incomplete (only a
    date, only a time), and with STT errors. The LLM is asked to emit
    ``{"date": "YYYY-MM-DD", "time": "HH:MM:SS"}``; missing components
    come back as ``null``.

    Relative references like "tomorrow" or "next Monday" are resolved
    against ``today`` (defaults to ``datetime.now()``).

    Returns ``(date, time)`` where each element is the normalised
    string or ``None`` if it couldn't be extracted.
    """
    text = (text or "").strip()
    if not text:
        return None, None

    today = today or datetime.now()
    today_str = today.strftime("%Y-%m-%d")
    weekday = today.strftime("%A")
        
    prompt = (
        "You are an information extraction assistant for a phone-call IVR "
        "system. The caller was asked for the pickup date and time of "
        "their reservation. The sentence below is the speech-to-text "
        "transcript of their reply. The date and time may appear in any "
        "order and any format (e.g. \"tomorrow at 5pm\", \"June 3rd 14:30\", "
        "\"next Monday morning at nine thirty\", \"5 PM on the 12th\"). "
        "The transcript may contain only a date, only a time, or both. "
        "It may contain speech-to-text errors.\n\n"
        f"Today is {today_str} ({weekday}). Resolve relative references "
        "(today, tomorrow, tonight, next Monday, etc.) against this date. "
        "If the caller says just a time with no date, assume today if the "
        "time hasn't passed yet, otherwise tomorrow. If the caller gives "
        "an AM/PM hint, convert to 24-hour time. If a part is missing or "
        "unclear, emit null for that field.\n\n"
        "Respond with ONLY a single JSON object on one line, no prose, no "
        "code fences, with exactly these two keys: \"date\" (string in "
        "YYYY-MM-DD format or null) and \"time\" (string in HH:MM:SS "
        "24-hour format or null).\n\n"
        f"Sentence: {text}\n"
        "Answer:"
    )

    try:
        raw = _llm_generate(prompt, num_predict=64)
    except Exception:
        logger.exception("Ollama call failed for pickup date/time extraction")
        raw = ""

    logger.debug("LLM pickup date/time raw response: %r", raw)

    date, time = _parse_llm_response(raw)
    logger.info(
        "Extracted pickup date/time: date=%s time=%s (from %r)",
        date,
        time,
        text,
    )
    return date, time
