"""Phase 0 — the eval harness.

The whole point of this file is one function, `run_conversation`, that drives
the real `quick_reservation` agent WITHOUT any of the voice pipeline (no audio,
no STT, no TTS, no WebSocket). It talks to the agent at the same text layer the
pipeline does — `agent.respond(text)` — feeds it a scripted list of caller
utterances, and hands back everything a later phase might want to grade:

    * the full transcript (every caller line and the agent's reply)
    * the final session snapshot (the slots it recorded + terminal flags)

There is NO scoring here yet. Phase 0 is just "run the agent, collect outputs".
Seeing that object print out is the goal.

Run it directly:

    python evals/harness.py
"""
import asyncio
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

# Standalone-script convention (same as tests/test_barge_in.py): make the repo
# root importable so `configs`, `agents`, etc. resolve when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.settings import settings  # noqa: E402

# ---------------------------------------------------------------------------
# Hermetic DB: an eval must not depend on Postgres running.
#
# When the agent finalizes a reservation it calls save_reservation(), which
# writes a row to Postgres. We don't care about that row — the eval reads the
# agent's in-memory *snapshot*, not the database. So we replace the DB write
# with a no-op BEFORE building any agent. This keeps every eval run repeatable
# on any machine with just an OPENAI_API_KEY and no other services.
#
# We patch it on the module that imported the name (agents.quick_reservation.
# agent did `from agents.reservation.store import save_reservation`), so the
# agent's own reference points at our stub.
# ---------------------------------------------------------------------------
import agents.quick_reservation.agent as agent_module  # noqa: E402


def _stub_db() -> None:
    """Replace the reservation DB write with a no-op (returns True)."""
    agent_module.save_reservation = lambda session: True


_stub_db()

# Import build AFTER the stub so we don't accidentally re-bind the real one.
from agents.quick_reservation.agent import build  # noqa: E402

# Quiet the app's noisy "voice" logger during evals; we want to read the
# transcript, not pipeline chatter.
logging.getLogger("voice").setLevel(logging.ERROR)


# A sample existing customer. The agent greets by name and reads this back;
# `id` is what save_reservation would use — harmless now that the DB is stubbed.
SAMPLE_ACCOUNT = {
    "id": 1,
    "name": "Jane Cooper",
    "phone": "(212) 555-0147",
    "email": "jane.cooper@example.com",
}

# The synthetic first turn. build() sets opening_trigger="<call_started>"; the
# pipeline sends this once so the agent speaks its greeting first. We reproduce
# that here so the conversation starts exactly like a real call.
OPENING_TRIGGER = "<call_started>"


@dataclass
class ConversationResult:
    """Everything a scorer might need from one simulated call.

    transcript : list of (speaker_text, agent_reply) pairs, in order. The first
                 pair's speaker_text is the opening trigger and its reply is the
                 greeting. Each later pair is one scripted caller line and the
                 agent's spoken reply to it.
    snapshot   : the final session state dict (slots + confirmed / transferred /
                 end_call / confirmation_number). This is the ground-truth record
                 of what the agent actually DID, independent of what it SAID.
    """

    transcript: List[Tuple[str, str]] = field(default_factory=list)
    snapshot: Optional[dict] = None

    @property
    def replies(self) -> List[str]:
        """Just the agent's spoken replies, in order (handy for scorers)."""
        return [reply for _caller, reply in self.transcript]


async def run_conversation(
    caller_turns: List[str],
    account: Optional[dict] = None,
) -> ConversationResult:
    """Drive one full call and return the transcript + final snapshot.

    caller_turns : the caller's utterances, in order. These are SCRIPTED — we
                   decide them up front. (A real caller would react to what the
                   agent just said; a fixed script can drift out of sync with an
                   agent that asks things in a different order. That limitation
                   is exactly what motivates the LLM-simulated caller in Phase 4.
                   For now, scripted is simple and good enough to see the loop.)
    account      : the caller's account dict; defaults to SAMPLE_ACCOUNT.

    A fresh agent is built for every call so no state leaks between cases —
    each ConversationResult is fully independent.
    """
    account = account or SAMPLE_ACCOUNT
    agent = build(settings, {"account": account})

    result = ConversationResult()

    # Turn 1: the opening trigger -> the agent's greeting.
    greeting = await agent.respond(OPENING_TRIGGER)
    result.transcript.append((OPENING_TRIGGER, greeting))

    # Each scripted caller line -> the agent's reply.
    for line in caller_turns:
        reply = await agent.respond(line)
        result.transcript.append((line, reply))

        # Fidelity to production: once the agent sets end_call (it finalized or
        # transferred), the real voice pipeline HANGS UP — the caller never gets
        # another turn. So we stop feeding the script here too. Without this, a
        # script longer than the agent needs would keep talking to a "hung-up"
        # agent and, e.g., trigger a second finalize. Stopping makes the harness
        # match what a real call would do.
        if agent.snapshot().get("end_call"):
            break

    # The snapshot is captured at the END, after the last turn — it reflects the
    # final recorded state of the reservation.
    result.snapshot = agent.snapshot()
    return result


def print_result(result: ConversationResult) -> None:
    """Human-friendly dump of a conversation, for eyeballing in Phase 0."""
    print("=" * 70)
    print("TRANSCRIPT")
    print("=" * 70)
    for caller, reply in result.transcript:
        speaker = "CALL START" if caller == OPENING_TRIGGER else "CALLER"
        print(f"\n[{speaker}] {caller}")
        print(f"[ANN]    {reply}")
    print("\n" + "=" * 70)
    print("FINAL SNAPSHOT (what the agent actually recorded)")
    print("=" * 70)
    for key, val in (result.snapshot or {}).items():
        print(f"  {key:20} = {val!r}")


# A single happy-path script: the caller offers every detail up front, confirms
# the callback number, then confirms the read-back. Because replies are
# non-deterministic, the agent may not ask things in exactly this order — that's
# fine and worth noticing. Phase 0 just shows us a real run.
HAPPY_PATH = [
    "Hi, I'd like to book a car for next Thursday at 1:30 in the afternoon. "
    "Pick me up at 10 Main Street in Brooklyn, and I'm heading to JFK Airport.",
    "Yes, that number's fine.",
    "Yes, that's all correct.",
]


async def _main() -> None:
    print("Running one scripted happy-path conversation "
          "(this makes real LLM calls)...\n")
    result = await run_conversation(HAPPY_PATH)
    print_result(result)


if __name__ == "__main__":
    asyncio.run(_main())
