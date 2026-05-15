"""Caller intent classification using LLM."""
import logging

from ._helpers import _llm_generate

logger = logging.getLogger(__name__)

NEW_RESERVATION = "new_reservation"
OTHER = "other"

# Words a speech-to-text engine commonly produces in place of "reservation"
# when the caller actually said "reservation". Used as a fast-path fallback
# in case the LLM is unavailable or returns junk.
_RESERVATION_LIKE_WORDS = (
    "reservation",
    "reservations",
    "observation",
    "observations",
    "preservation",
    "conservation",
    "reservetion",
    "reservaton",
    "reserveation",
    "reservasion",
    "reservacion",
)

_NEW_LIKE_WORDS = (
    "new",
    "knew",
    "nu",
    "noo",
    "a new",
    "another",
    "make",
    "book",
    "booking",
    "schedule",
    "create",
)


def _keyword_intent(text: str) -> str:
    """Best-effort fallback that scans for reservation-like words."""
    lowered = text.lower()
    if any(word in lowered for word in _RESERVATION_LIKE_WORDS):
        return NEW_RESERVATION
    return OTHER


def classify_intent(text: str) -> str:
    """Classify the caller's spoken response as a reservation intent.

    The caller has just been asked: "Do you want a new reservation or
    something else?" ``text`` is the speech-to-text transcript of their
    reply. STT mistakes (e.g. "observation" for "reservation",
    "knew" for "new") are common, so the classifier is intentionally
    lenient about phonetic look-alikes.

    Returns ``"new_reservation"`` when the caller wants to book a new
    reservation, otherwise ``"other"``.
    """
    text = (text or "").strip()
    if not text:
        return OTHER

    prompt = (
        "You are an intent classification assistant for a phone-call IVR "
        "system. The caller was asked: \"Do you want a new reservation or "
        "something else?\" The sentence below is the speech-to-text "
        "transcript of their reply. The transcript can contain "
        "speech-to-text errors where words sound similar to what was "
        "actually said. In particular, treat the following as the caller "
        "asking for a NEW RESERVATION: 'reservation', 'observation', "
        "'preservation', 'conservation', 'reservetion', or any similar "
        "sounding word, especially when combined with words like 'new', "
        "'knew', 'a new', 'another', 'make', 'book', 'schedule', or "
        "'create'. Phrases like 'I want a new observation', 'make a "
        "reservation', 'I'd like to book', or 'new booking' all mean "
        "NEW RESERVATION. Anything else (existing reservation, cancel, "
        "change, billing question, agent, etc.) is OTHER.\n\n"
        "Respond with ONLY one of these two tokens and nothing else: "
        "NEW_RESERVATION or OTHER.\n\n"
        f"Sentence: {text}\n"
        "Answer:"
    )
    try:
        raw = _llm_generate(prompt)
    except Exception:
        logger.exception("Ollama call failed for intent classification")
        raw = ""

    logger.debug("LLM intent raw response: %r", raw)

    upper = raw.upper()
    if "NEW_RESERVATION" in upper or "NEW RESERVATION" in upper:
        intent = NEW_RESERVATION
    elif "OTHER" in upper:
        intent = OTHER
    else:
        intent = _keyword_intent(text)

    logger.info("Classified intent: %s (from %r)", intent, text)
    return intent
