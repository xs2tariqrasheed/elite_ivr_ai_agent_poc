"""Duckling client for parsing dates/times out of natural language.

The Duckling HTTP server is expected to be running at ``DUCKLING_URL``
(``http://localhost:8000`` by default) and to expose the standard
``/parse`` endpoint.
"""

import logging
from datetime import datetime
from typing import Optional

import requests

import config

logger = logging.getLogger(__name__)


def parse_date_with_duckling(
    text: str,
    reference_time: Optional[datetime] = None,
    locale: Optional[str] = None,
    url: Optional[str] = None,
    timeout: Optional[float] = None,
) -> Optional[str]:
    """Send ``text`` to Duckling and return a ``YYYY-MM-DD`` date or ``None``.

    Only ``time``-dimension entities are requested; the highest-scoring
    one whose grain is day-level or finer is returned as a date.
    """
    text = (text or "").strip()
    if not text:
        return None

    url = (url or config.DUCKLING_URL).rstrip("/") + "/parse"
    locale = locale or config.DUCKLING_LOCALE
    timeout = timeout if timeout is not None else config.DUCKLING_TIMEOUT
    ref = reference_time or datetime.now()
    # Duckling wants milliseconds since the epoch for reftime.
    reftime_ms = int(ref.timestamp() * 1000)

    payload = {
        "text": text,
        "locale": locale,
        "dims": '["time"]',
        "reftime": reftime_ms,
    }

    try:
        resp = requests.post(url, data=payload, timeout=timeout)
        resp.raise_for_status()
        results = resp.json()
    except Exception:
        logger.exception("Duckling request failed for text=%r", text)
        return None

    return _extract_date_from_results(results)


def _extract_date_from_results(results) -> Optional[str]:
    """Pull the first usable ``YYYY-MM-DD`` out of Duckling's response."""
    if not isinstance(results, list):
        return None

    for entity in results:
        if not isinstance(entity, dict):
            continue
        if entity.get("dim") != "time":
            continue
        value = entity.get("value") or {}

        # Duckling returns either a single value (``type: "value"``) or
        # an interval (``type: "interval"``). For an interval, prefer the
        # ``from`` side.
        iso = None
        if value.get("type") == "value":
            iso = value.get("value")
        elif value.get("type") == "interval":
            iso = (value.get("from") or {}).get("value")
        if not iso:
            # Some Duckling builds put the canonical ISO under "values".
            values = value.get("values") or []
            if values and isinstance(values[0], dict):
                iso = values[0].get("value")
        if not iso:
            continue

        date = _iso_to_date(iso)
        if date:
            return date

    return None


def _iso_to_date(iso: str) -> Optional[str]:
    """Parse Duckling's ISO-8601 timestamp into ``YYYY-MM-DD``."""
    try:
        # Duckling emits e.g. ``2026-05-28T00:00:00.000-07:00``.
        # ``fromisoformat`` handles the timezone in Python 3.11+; for
        # older Pythons we strip the millisecond fragment.
        cleaned = iso.replace("Z", "+00:00")
        if "." in cleaned:
            # Strip ``.fff`` between seconds and the timezone offset.
            head, _, tail = cleaned.partition(".")
            # ``tail`` looks like ``000-07:00`` or ``000+00:00``.
            tz = ""
            for i, ch in enumerate(tail):
                if ch in "+-":
                    tz = tail[i:]
                    break
            cleaned = head + tz
        dt = datetime.fromisoformat(cleaned)
    except ValueError:
        logger.debug("Could not parse Duckling ISO timestamp: %r", iso)
        return None
    return dt.strftime("%Y-%m-%d")
