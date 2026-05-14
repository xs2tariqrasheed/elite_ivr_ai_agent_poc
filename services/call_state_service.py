"""Per-call in-memory state.

A ``CallState`` instance holds everything we know about a single
in-flight call: the Twilio identifiers, the current phase, the
reservation fields collected so far, and a couple of flags that the
WebSocket handler uses to decide whether inbound audio should be
buffered.
"""
import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional

from constants.call_phases import PHASE_START

logger = logging.getLogger(__name__)


@dataclass
class Reservation:
    """The bag of fields we collect from the caller during the flow."""
    account_number: Optional[str] = None
    account_record: Optional[dict] = None  # row from dummy_data.json
    account_name: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    pickup_date_time: Optional[str] = None
    pickup_address: Optional[str] = None
    dropoff_address: Optional[str] = None
    callback_number: Optional[str] = None
    email: Optional[str] = None

    def as_summary_dict(self) -> dict:
        return {
            "account_number": self.account_number,
            "account_name": self.account_name,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "pickup_date_time": self.pickup_date_time,
            "pickup_address": self.pickup_address,
            "dropoff_address": self.dropoff_address,
            "callback_number": self.callback_number,
            "email": self.email,
        }


@dataclass
class CallState:
    call_sid: str
    stream_sid: str
    caller_phone: Optional[str] = None
    phase: str = PHASE_START
    reservation: Reservation = field(default_factory=Reservation)

    # Flow-control flags. These are toggled by the orchestrator and
    # read by the WebSocket receive loop.
    agent_speaking: bool = True   # we always start with agent talking
    capturing_audio: bool = False  # set True while we want to keep mic data

    # Inbound mu-law audio captured from Twilio while ``capturing_audio``
    # is True. Bytes, raw G.711 mu-law @ 8 kHz, 8 bit.
    inbound_mulaw: bytearray = field(default_factory=bytearray)

    # Per-phase retry counters
    attempts: Dict[str, int] = field(default_factory=dict)

    # Set by the WebSocket loop when Twilio echoes a ``mark`` event.
    _mark_event: threading.Event = field(default_factory=threading.Event)
    _last_mark_name: Optional[str] = None

    # ----- attempts -----
    def increment_attempts(self, key: str) -> int:
        self.attempts[key] = self.attempts.get(key, 0) + 1
        return self.attempts[key]

    def get_attempts(self, key: str) -> int:
        return self.attempts.get(key, 0)

    # ----- audio buffer -----
    def reset_inbound_buffer(self) -> None:
        self.inbound_mulaw = bytearray()

    # ----- marks -----
    def signal_mark(self, name: str) -> None:
        self._last_mark_name = name
        self._mark_event.set()

    def clear_mark(self) -> None:
        self._mark_event.clear()
        self._last_mark_name = None

    def wait_for_mark(self, timeout: float) -> Optional[str]:
        if self._mark_event.wait(timeout=timeout):
            return self._last_mark_name
        return None


# Module-level registry of active calls keyed by stream_sid. Twilio
# delivers the stream_sid in every WebSocket frame, so it's the most
# convenient handle. Access is single-threaded inside the asyncio
# loop, so we don't need a lock.
_calls: Dict[str, CallState] = {}


def create_call_state(
    call_sid: str, stream_sid: str, caller_phone: Optional[str] = None
) -> CallState:
    state = CallState(
        call_sid=call_sid,
        stream_sid=stream_sid,
        caller_phone=caller_phone.strip() if caller_phone else None,
    )
    _calls[stream_sid] = state
    logger.info(
        "Created call state call_sid=%s stream_sid=%s caller_phone=%s",
        call_sid,
        stream_sid,
        state.caller_phone,
    )
    return state


def get_call_state(stream_sid: str) -> Optional[CallState]:
    return _calls.get(stream_sid)


def remove_call_state(stream_sid: str) -> None:
    if stream_sid in _calls:
        logger.info("Removing call state stream_sid=%s", stream_sid)
        _calls.pop(stream_sid, None)
