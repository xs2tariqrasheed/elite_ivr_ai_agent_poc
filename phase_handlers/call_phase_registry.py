"""Call flow orchestration.

This module owns the cross-phase concerns of a Twilio call:

* ``_hangup_call`` — Twilio REST hang-up for cleanly ending a call.
* ``_PHASE_HANDLERS`` — phase identifier → handler coroutine dispatch.
* ``_run_call_flow`` — drives the call from PHASE_INTENT through hang-up.

The shared ``_speak`` / ``_listen`` helpers live in their own sibling
modules (``phase_handlers.speak`` and ``phase_handlers.listen``) and are
imported directly by the phase handler modules.

To avoid a circular import (``call_phase_registry`` → phase modules →
``call_phase_registry``), the dispatch table is built lazily inside
``_run_call_flow`` rather than at module import time.
"""

import asyncio
import logging
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect
from twilio.rest import Client as TwilioClient

import config
from constants import call_phases as phases
from services import agent_voice_service as voice
from services import call_state_service
from services import phase_manager


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Twilio call hang-up via REST
# ---------------------------------------------------------------------------


def _hangup_call(call_sid: str) -> None:
    if not (config.TWILIO_ACCOUNT_SID and config.TWILIO_AUTH_TOKEN):
        logger.warning("Twilio creds missing — cannot hang up call %s", call_sid)
        return
    try:
        client = TwilioClient(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)
        client.calls(call_sid).update(status="completed")
        call_state_service.remove_call_state(call_sid)
        logger.info("Hung up call %s via Twilio REST", call_sid)
    except Exception:
        logger.exception("Failed to hang up call %s", call_sid)


# ---------------------------------------------------------------------------
# Phase dispatch table (built lazily to avoid circular imports)
# ---------------------------------------------------------------------------


_PHASE_HANDLERS: Optional[dict] = None


def _build_phase_handlers() -> dict:
    """Import every phase handler and assemble the dispatch table.

    Imported lazily so that the individual phase modules can themselves
    import ``_speak`` / ``_listen`` without creating a circular import at
    package load time.
    """
    from phase_handlers.phase_intent import _run_phase_intent
    from phase_handlers.phase_passenger_info_verification import (
        _run_phase_passenger_info_verification,
    )
    from phase_handlers.phase_account_number import _run_phase_account_number
    from phase_handlers.phase_account_name import _run_phase_account_name
    from phase_handlers.phase_first_name import _run_phase_first_name
    from phase_handlers.phase_last_name import _run_phase_last_name
    from phase_handlers.phase_pickup_date_time import _run_phase_pickup_date_time
    from phase_handlers.phase_pickup_address import _run_phase_pickup_address
    from phase_handlers.phase_dropoff_address import _run_phase_dropoff_address
    from phase_handlers.phase_callback_number import _run_phase_callback_number
    from phase_handlers.phase_email import _run_phase_email
    from phase_handlers.phase_end import _run_phase_end

    return {
        phases.PHASE_INTENT: _run_phase_intent,
        phases.PHASE_PASSENGER_INFO_VERIFICATION: _run_phase_passenger_info_verification,
        phases.PHASE_ACCOUNT_NUMBER: _run_phase_account_number,
        phases.PHASE_ACCOUNT_NAME: _run_phase_account_name,
        phases.PHASE_FIRST_NAME: _run_phase_first_name,
        phases.PHASE_LAST_NAME: _run_phase_last_name,
        phases.PHASE_PICKUP_DATE_TIME: _run_phase_pickup_date_time,
        phases.PHASE_PICKUP_ADDRESS: _run_phase_pickup_address,
        phases.PHASE_DROPOFF_ADDRESS: _run_phase_dropoff_address,
        phases.PHASE_CALLBACK_NUMBER: _run_phase_callback_number,
        phases.PHASE_EMAIL: _run_phase_email,
        phases.PHASE_END: _run_phase_end,
    }


# ---------------------------------------------------------------------------
# The orchestrator coroutine
# ---------------------------------------------------------------------------


async def _run_call_flow(websocket: WebSocket, state) -> None:
    """Drive the call from PHASE_INTENT through to hang-up."""
    global _PHASE_HANDLERS
    if _PHASE_HANDLERS is None:
        _PHASE_HANDLERS = _build_phase_handlers()

    current = phases.PHASE_INTENT
    try:
        # If the caller's phone didn't match any known account, play the
        # "account not found" clip and short-circuit straight to hang-up.
        if state.account_info is None:
            from phase_handlers.speak import _speak

            logger.info(
                "Caller account not found for caller_phone=%s — playing "
                "account_not_found and hanging up",
                state.caller_phone or "n/a",
            )
            await _speak(
                websocket, state, [["account_not_found"]]
            )
            return

        while not phase_manager.is_terminal(current):
            handler = _PHASE_HANDLERS.get(current)
            if handler is None:
                logger.error("No handler for phase %r — aborting", current)
                break
            logger.info("Entering %s", current)
            current = await handler(websocket, state)
            logger.info("Phase returned %s", current)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected mid-flow")
        return
    except Exception:
        logger.exception("Call flow crashed")
    finally:
        # Always try to hang up cleanly when we exit the loop.
        if state.call_sid:
            await asyncio.to_thread(_hangup_call, state.call_sid)
        try:
            await voice.send_clear(websocket, state.stream_sid)
        except Exception:
            pass
