"""Reservation state for a single express (one-shot) reservation call.

Identical shape to :class:`agents.reservation.session.ReservationSession` so it
can be persisted by the shared :func:`agents.reservation.store.save_reservation`.
Kept as its own subclass to leave room for the express flow to diverge later
without touching the original reservation agent.
"""
from agents.reservation.session import ReservationSession


class QuickReservationSession(ReservationSession):
    """Mutable reservation state for one express-booking call."""
