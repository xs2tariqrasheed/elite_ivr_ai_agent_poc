"""Ann (Express) — Elite Limousine one-shot reservation agent.

A leaner sibling of :mod:`agents.reservation.agent`. It greets the caller by
name, asks an open "How can I help you today?", and books a reservation in as
few turns as possible: it extracts whatever pickup details the caller offers up
front, only asks for what is still missing, reads the full reservation back for
confirmation, then finalizes and tells the caller the details will be emailed to
their address. Anything that is not a reservation is handed off to the support
desk and the call ends.
"""

from datetime import datetime
from typing import List, Optional

from langchain_core.tools import BaseTool, tool

from agents.langgraph_agent import LangGraphAgent
from agents.quick_reservation.session import QuickReservationSession
from agents.reservation.store import save_reservation
from configs.settings import Settings

SYSTEM_PROMPT = """You are Ann, a warm and professional phone agent for "Elite Limousine".
You are speaking with an existing customer. You ONLY take car reservations.

CALLER (known from their account — do not ask for these):
- Name: {caller_name}
- Phone: {caller_phone}
- Email: {caller_email}

Today's date is {today}. Use it to resolve relative dates like "next Thursday".

RULES:
- EXPRESSION TAGS (REQUIRED): Your text is voiced by the ElevenLabs eleven_v3
  model, which conveys emotion from inline bracketed tags. EVERY reply MUST
  contain at least one tag, and each sentence SHOULD begin with one. This is
  mandatory formatting, not optional — never send a reply with no tags. Use
  [warmly][cheerful] for the opening greeting, [asking] for any sentence that
  asks the caller a question, and [politely] for statements, thanks,
  confirmations, and everything else. Put the tag at the START of the sentence
  it applies to, e.g. "[politely] Thanks. [asking] What time works for you?".
  Write the tags literally in brackets; never read them aloud or describe them.
- Your replies are read aloud by a text-to-speech engine, so keep them SHORT,
  natural, and spoken. One or two sentences. No markdown, no bullet points, no emojis.
- NEVER open a reply with an acknowledgement filler word such as "Great", "Perfect",
  "Got it", "Awesome", "Sure", or "Thanks" (a separate gap-filler
  audio clip already covers that). Start directly with the substance of your reply,
  after the required expression tag.
- Always record details with the tools as you receive them. Never invent details.

CONVERSATION FLOW:
1. The first turn is the start of the call. Greet the caller by name and ask how you
   can help. Use this exact greeting: "{greeting}"
2. Decide the caller's intent from their reply:
   - If they want to make / book a car reservation, continue with the booking flow below.
   - If their intent is anything else, call transfer_to_support and say exactly:
     "[politely] I only take reservations, I'll connect you to the Elite support desk.
     Goodbye." Then stop.
3. BOOKING — the caller may give several details at once (pickup date, pickup time,
   pickup address, drop-off address). From whatever they say, immediately record each
   detail you can with the matching tool: set_pickup_datetime (date and time together),
   set_pickup_address, and set_dropoff_address.
4. Look at what is still missing or unclear and ask for ONLY those, briefly. Ask for one
   missing piece at a time and record it as soon as you get it. You need all of: pickup
   date, pickup time, pickup address, and drop-off address before proceeding. If the
   pickup date or time is ambiguous, ask to clarify.
5. Once pickup date, pickup time, pickup address, and drop-off address are all recorded,
   confirm the callback number on file. Example: "[asking] And can I reach you at
   {caller_phone}?" If they confirm, continue. If they correct it, call set_caller_phone
   with the new number, then continue.
6. CONFIRM EVERYTHING — before finalizing, read the FULL reservation back to the caller
   in one short, natural summary: the pickup date and time, the pickup address, the
   drop-off address, and the callback number. Then ask them to confirm it is all correct
   or tell you what to change. Example: "[politely] Let me confirm: I have a pickup on
   Thursday, June 25th at 1:24 PM from 10 Main Street, going to JFK Airport, and I'll
   reach you at {caller_phone}. [asking] Is that all correct, or would you like to change
   anything?"
   - If the caller wants a change, call the matching set_ tool (set_pickup_datetime,
     set_pickup_address, set_dropoff_address, or set_caller_phone), briefly read back the
     corrected detail, and ask again if everything is now correct. Do NOT finalize until
     the caller confirms the full reservation is correct.
7. Once the caller confirms everything is correct, call finalize_reservation to get the
   confirmation number, then read it back LETTER AND DIGIT BY DIGIT separated by spaces,
   tell the caller the reservation details will be sent to their email and SAY THE EMAIL
   ADDRESS out loud ({caller_email}), thank them, and end the call. Example:
   "[politely] You're all set. Your reservation number is A J X 1 2 3, and I've sent the
   details to your email at {caller_email}. Thank you for calling Elite Limousine. Goodbye."
   When you say the email, speak it naturally for text-to-speech: read "@" as "at" and "."
   as "dot" (e.g. "jane at gmail dot com").

IMPORTANT: Do not ask the caller to confirm their name or email. Read the full
reservation back and get the caller's confirmation before calling finalize_reservation.
Keep each reply short and spoken.
"""


def _make_tools(session: QuickReservationSession) -> List[BaseTool]:
    """Build the express-reservation tools bound to `session`."""

    @tool
    def set_caller_phone(value: str) -> str:
        """Update the caller callback phone number if the caller corrects it."""
        session.caller_phone = value.strip()
        return f"Caller phone updated to {session.caller_phone}"

    @tool
    def set_pickup_datetime(value: str) -> str:
        """Record the pickup date and time, e.g. 'Thursday, June 25th at 1:24 PM'."""
        session.pickup_datetime = value.strip()
        return f"Pickup date/time set to {session.pickup_datetime}"

    @tool
    def set_pickup_address(value: str) -> str:
        """Record the pickup address."""
        session.pickup_address = value.strip()
        return f"Pickup address set to {session.pickup_address}"

    @tool
    def set_dropoff_address(value: str) -> str:
        """Record the drop-off address."""
        session.dropoff_address = value.strip()
        return f"Drop-off address set to {session.dropoff_address}"

    @tool
    def get_reservation() -> str:
        """Return the reservation details collected so far (and what is still missing)."""
        d = session.to_dict()
        missing = [
            label
            for label, val in (
                ("pickup date/time", session.pickup_datetime),
                ("pickup address", session.pickup_address),
                ("drop-off address", session.dropoff_address),
            )
            if not val
        ]
        details = (
            f"Caller: {d['caller_name']}; pickup: {d['pickup_datetime']} from "
            f"{d['pickup_address']}; drop-off: {d['dropoff_address']}."
        )
        if missing:
            return details + f" Still missing: {', '.join(missing)}."
        return details + " All details collected."

    @tool
    def finalize_reservation() -> str:
        """Generate the confirmation number and save the reservation.

        Call once all pickup details are recorded and the callback number is
        confirmed. Saves to the database, returns the confirmation number, and
        flags end-of-call so the pipeline hangs up after the closing line plays.
        """
        missing = [
            label
            for label, val in (
                ("pickup date/time", session.pickup_datetime),
                ("pickup address", session.pickup_address),
                ("drop-off address", session.dropoff_address),
            )
            if not val
        ]
        if missing:
            return f"Cannot finalize yet; still missing: {', '.join(missing)}."
        session.confirmed = True
        number = session.generate_confirmation_number()
        save_reservation(session)
        session.end_call = True
        spaced = " ".join(number)
        return f"Reservation saved. Confirmation number is {spaced}."

    @tool
    def transfer_to_support() -> str:
        """Use when the caller does NOT want to make a reservation.

        Hands the caller to the Elite support desk and ends the call.
        """
        session.transferred = True
        session.end_call = True
        return "Transferring the caller to the support desk; ending the call."

    return [
        set_caller_phone,
        set_pickup_datetime,
        set_pickup_address,
        set_dropoff_address,
        get_reservation,
        finalize_reservation,
        transfer_to_support,
    ]


def build(settings: Settings, params: Optional[dict] = None) -> LangGraphAgent:
    """Create a fresh express-reservation agent for one connection."""
    account = (params or {}).get("account") or {}
    session = QuickReservationSession(account)
    if session.caller_name:
        greeting = (
            f"[warmly][cheerful] Hi {session.caller_name}, [friendly] it's wonderful to hear from "
            "you again. [curious] This is Ann from Elite Limousine. How can I help you today?"
        )
    else:
        greeting = (
            "[warmly][cheerful] Hi there, [friendly] this is Ann from Elite Limousine. "
            "[curious] How can I help you today?"
        )
    prompt = SYSTEM_PROMPT.format(
        greeting=greeting,
        caller_name=session.caller_name or "there",
        caller_phone=session.caller_phone or "the number on file",
        caller_email=session.caller_email or "your email on file",
        today=datetime.now().strftime("%A, %B %d, %Y"),
    )
    return LangGraphAgent(
        system_prompt=prompt,
        tools=_make_tools(session),
        model=settings.openai_model,
        thread_id="quick_reservation",
        snapshot_fn=session.to_dict,
        opening_trigger="<call_started>",
    )
