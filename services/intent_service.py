"""fastText-based intent recognition.

Two intents are supported: ``new_reservation`` and ``other``.  The
trained binary lives at ``INTENT_MODEL_PATH`` and is loaded once at
startup.
"""
import logging
import os
from typing import Optional, Tuple

import fasttext

import config

logger = logging.getLogger(__name__)


INTENT_NEW_RESERVATION = "new_reservation"
INTENT_OTHER = "other"


_model = None


def load_model() -> None:
    """Load the trained fastText model.  Call once at startup."""
    global _model
    if _model is not None:
        return

    path = config.INTENT_MODEL_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Intent model not found at {path}")

    logger.info("Loading fastText intent model from %s", path)
    # fastText prints a deprecation warning to stderr on import; ignore.
    _model = fasttext.load_model(path)
    logger.info("Intent model loaded")


def predict_intent(text: str) -> Tuple[str, float]:
    """Return (intent_label, confidence) for ``text``.

    Empty / ``None`` input returns ``("other", 0.0)``.
    """
    if _model is None:
        raise RuntimeError("Intent model not loaded. Call load_model() first.")

    if not text or not text.strip():
        return INTENT_OTHER, 0.0

    # fastText needs a single line of normalised text.
    cleaned = text.strip().replace("\n", " ").lower()
    try:
        labels, probs = _model.predict(cleaned)
        intent = labels[0].replace("__label__", "")
        confidence = float(probs[0])
    except Exception:
        logger.exception("Intent prediction failed for text=%r", cleaned)
        return INTENT_OTHER, 0.0

    if intent not in (INTENT_NEW_RESERVATION, INTENT_OTHER):
        # Unknown label — treat as "other" to be safe.
        logger.warning("Unexpected intent label %r — coercing to 'other'", intent)
        intent = INTENT_OTHER

    logger.info("Intent predicted: %s (confidence=%.4f) text=%r", intent, confidence, cleaned)
    return intent, confidence


def classify_with_threshold(
    text: str,
    threshold: Optional[float] = None,
) -> str:
    """Helper that applies the confidence threshold from config.

    If confidence is below ``threshold`` we fall back to
    ``"other"`` (safer default for the Elite IVR flow).
    """
    if threshold is None:
        threshold = config.INTENT_CONFIDENCE_THRESHOLD

    intent, confidence = predict_intent(text)
    if intent == INTENT_NEW_RESERVATION and confidence < threshold:
        logger.info(
            "Intent %s confidence %.4f below threshold %.2f → falling back to 'other'",
            intent, confidence, threshold,
        )
        return INTENT_OTHER
    return intent
