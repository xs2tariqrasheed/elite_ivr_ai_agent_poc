"""Reservation state for a single express (one-shot) reservation call.

Identical shape to :class:`agents.reservation.session.ReservationSession` so it
can be persisted by the shared :func:`agents.reservation.store.save_reservation`.
Kept as its own subclass to leave room for the express flow to diverge later
without touching the original reservation agent.
"""
from typing import Optional

from agents.reservation.session import ReservationSession


class QuickReservationSession(ReservationSession):
    """Mutable reservation state for one express-booking call."""

    def __init__(self, account=None) -> None:
        super().__init__(account)
        # Turn-boundary gate for the mandatory "CONFIRM EVERYTHING" step.
        #
        # `turn_index` is bumped once per caller utterance by the runtime.
        # `readback_spoken_turn` records the turn in which the agent read the
        # full reservation back (via read_back_reservation). finalize_reservation
        # refuses to run unless the read-back happened in a STRICTLY EARLIER turn
        # than the current one — i.e. the caller actually got a turn to respond
        # to the read-back. Because the model cannot advance the turn counter
        # itself (only a real caller utterance does), it cannot read back and
        # finalize in the same burst of tool calls; the confirmation step can
        # never be skipped.
        self.turn_index: int = 0
        self.readback_spoken_turn: Optional[int] = None
