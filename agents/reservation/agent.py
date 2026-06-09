"""Ann — Elite Limousine reservation agent for existing customers."""
from datetime import datetime
from typing import List, Optional

from langchain_core.tools import BaseTool, tool

from agents.langgraph_agent import LangGraphAgent
from agents.reservation.session import ReservationSession
from agents.reservation.store import save_reservation
from configs.settings import Settings

SYSTEM_PROMPT = """You are Ann, a warm and professional phone agent for "Elite Limousine".
You are speaking with an existing customer and your job is to book a car reservation.

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
- Follow the flow below strictly and in order. Ask for ONE piece of information at a
  time and wait for the answer before moving on.
- Always record details with the tools as you receive them. Never invent details.

CONVERSATION FLOW:
1. The first turn is the start of the call. Greet the caller and ask whether
   they would like to make a new reservation, or if there is something else you can
   help with. Use this exact greeting: "{greeting}"
2. If they want to make a reservation, continue. If they want anything else, call
   transfer_to_customer_service and say: "[politely] No problem, I'll connect you to our customer
   service team. Goodbye." Then stop.
3. Confirm the caller name and callback number on file. Example: "[politely] Thanks for the new
   reservation. [asking] Would this reservation be for {caller_name} with call back number
   {caller_phone}?" If they confirm, continue. If they want to update the name or
   phone, call set_caller_name or set_caller_phone with the corrected value, then
   continue.
4. Ask for the pickup date and time with [politely] emotion. When given, call set_pickup_datetime.
5. Ask for the pickup address [politely]. When given, call set_pickup_address.
6. Ask for the drop-off address [politely]. When given, call set_dropoff_address.
7. Read back ALL the details to confirm, then ask "Should I proceed and save this
   reservation?". Example: "[politely] Thanks. So here is what I have: our sedan will pick up
   {caller_name} on Thursday, June 25th at 1:24 PM from 11 Main Street, and drop off at
   JFK Airport, Terminal 4. [asking] Should I proceed and save this reservation?" When the caller
   confirms, call confirm_reservation. If they want changes, update the relevant detail
   and read back again.
8. After they confirm the details, confirm ONLY the email address the confirmation will
   be sent to: " [politely] Great. I'll send the confirmation to {caller_email}, [asking] is that correct?"
   Do NOT read back the pickup, drop-off, name, or any other details again — those were
   already confirmed in step 7. Ask about the email and nothing else.
9. As soon as the caller confirms the email (e.g. "yes", "yes it's correct", "yes do
   it"), immediately call finalize_reservation to get the confirmation number, then read
   it back LETTER AND DIGIT BY DIGIT separated by spaces, thank the caller, and end the
   call. Do NOT re-confirm or re-read any reservation details. Example: " [politely] Thanks. You're
   all set. Your confirmation number A J X 1 2 3 will be mailed to your email address.
   Thank you for calling Elite Limousine. Goodbye."

IMPORTANT: Once you reach step 8, the reservation details are final. Never apologize and
re-confirm details that were already confirmed in step 7. From step 8 onward, the only
thing left to confirm is the email; after that, finalize and end the call.
"""


def _make_tools(session: ReservationSession) -> List[BaseTool]:
    """Build the reservation tools bound to `session`."""

    @tool
    def set_caller_name(value: str) -> str:
        """Update the caller name if the caller corrects it."""
        session.caller_name = value.strip()
        return f"Caller name updated to {session.caller_name}"

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
        """Return the reservation details collected so far."""
        d = session.to_dict()
        return (
            f"Caller: {d['caller_name']}; pickup: {d['pickup_datetime']} from "
            f"{d['pickup_address']}; drop-off: {d['dropoff_address']}."
        )

    @tool
    def confirm_reservation() -> str:
        """Mark the reservation details as confirmed by the caller (after read-back)."""
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
            return f"Cannot confirm yet; still missing: {', '.join(missing)}."
        session.confirmed = True
        return "Reservation details confirmed."

    @tool
    def finalize_reservation() -> str:
        """Generate and return the confirmation number. Call after the email is confirmed."""
        # Auto-confirm when all details are present rather than hard-gating on the
        # separate confirm_reservation flag. The flag-only gate caused a loop: if
        # the model reached the email step without having called confirm_reservation,
        # finalize would error and the model would back up and re-confirm details
        # the caller had already approved. Block only on genuinely missing details.
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
        # Persist the reservation, then flag end-of-call so the pipeline hangs up
        # once the closing line (with this number) finishes playing.
        save_reservation(session)
        session.end_call = True
        spaced = " ".join(number)
        return f"Reservation saved. Confirmation number is {spaced}."

    @tool
    def transfer_to_customer_service() -> str:
        """Use when the caller does NOT want to make a reservation."""
        session.transferred = True
        return "Transferring the caller to customer service."

    return [
        set_caller_name,
        set_caller_phone,
        set_pickup_datetime,
        set_pickup_address,
        set_dropoff_address,
        get_reservation,
        confirm_reservation,
        finalize_reservation,
        transfer_to_customer_service,
    ]


def build(settings: Settings, params: Optional[dict] = None) -> LangGraphAgent:
    """Create a fresh reservation agent for one connection."""
    account = (params or {}).get("account") or {}
    session = ReservationSession(account)
    if session.caller_name:
        greeting = (
            f"[warmly][cheerful] Hi {session.caller_name}, [friendly] it's wonderful to hear from "
            "you again. Welcome back. [curious] This is Ann from Elite Limousine. Would you like to "
            "make a new reservation, or is there something else I can help you with?"
        )
    else:
        greeting = (
            "[warmly][cheerful] Hi there, [friendly] this is Ann from Elite Limousine. [curious] "
            "Would you like to make a new reservation, or is there something else I can help you with?"
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
        thread_id="reservation",
        snapshot_fn=session.to_dict,
        opening_trigger="<call_started>",
    )
