"""Phase manager: knows which phase comes after which.

This is intentionally tiny. The orchestrator (in ``main.py``) does
the heavy lifting; the phase manager only encapsulates the rules
about phase ordering and "is this the terminal phase?".
"""
import logging
from typing import Optional

from constants.call_phases import (
    PHASE_HANGUP,
    PHASE_ORDER,
)

logger = logging.getLogger(__name__)


def next_phase(current: str) -> Optional[str]:
    """Return the next phase name after ``current``.

    Returns ``None`` if ``current`` is unknown or is already the last
    phase in the linear flow.
    """
    if current not in PHASE_ORDER:
        logger.debug("next_phase: unknown current phase %r", current)
        return None

    idx = PHASE_ORDER.index(current)
    if idx + 1 >= len(PHASE_ORDER):
        return None
    return PHASE_ORDER[idx + 1]


def is_terminal(phase: str) -> bool:
    """True once the flow returns ``PHASE_HANGUP`` (after goodbye in ``PHASE_END``)."""
    return phase == PHASE_HANGUP
