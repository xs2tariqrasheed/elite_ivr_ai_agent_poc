"""Phase 4 — the LLM-simulated caller (realistic multi-turn).

A scripted list of caller lines (Phases 0-1) breaks the moment the agent asks
for things in a different order than the script assumed — so a case can fail for
a reason that has nothing to do with the agent being wrong. Here we replace the
script with an LLM that ROLE-PLAYS the caller: it reads the agent's actual reply
each turn and responds to it, using a persona (`caller_goal` + `caller_facts`
from the dataset). It gives one detail at a time when asked, corrects the agent
if a read-back is wrong, and stops when the agent says goodbye.

The important design win: `run_simulated_conversation` returns the SAME
`ConversationResult` shape as the scripted harness, so every Phase 1 scorer and
the Phase 3 judge grade a simulated call with ZERO changes. That uniform output
type is exactly why the abstraction was built in Phase 0.

A tradeoff to understand: the caller is itself an LLM, so simulated runs add a
second source of non-determinism. We keep its temperature low for stability, but
this is the other half of the "one run isn't a measurement" lesson — it's why
Phase 5 will run each case several times and report a pass RATE.

Run it:

    python evals/caller.py     # drive a few cases with the simulated caller
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import (  # noqa: E402
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI  # noqa: E402

from configs.settings import settings  # noqa: E402  (loads .env)
from evals.dataset import EvalCase  # noqa: E402
from evals.harness import (  # noqa: E402
    OPENING_TRIGGER,
    SAMPLE_ACCOUNT,
    ConversationResult,
    build,
)

# The caller model. gpt-4o role-plays a human more convincingly than the agent's
# gpt-4o-mini. Low temperature keeps behavior fairly stable run to run.
CALLER_MODEL = "gpt-4o"
CALLER_TEMPERATURE = 0.3

# The caller emits this when the agent has clearly ended the call, so we can stop
# even in the buggy case where the agent says "goodbye" WITHOUT calling a tool
# (so end_call never flips). Without it, such a call would loop until the cap.
END_SENTINEL = "[[END]]"

# Hard cap on caller turns, so a confused conversation can't run forever.
MAX_TURNS = 14


def _caller_system_prompt(case: EvalCase) -> str:
    """Turn a case's persona (goal + facts) into the caller LLM's instructions."""
    return f"""You are a customer phoning Elite Limousine. Speak like a real \
person on the phone: short, natural, one or two sentences. Say ONLY the words \
you speak out loud — no stage directions, no narration, no bracketed tags.

YOUR GOAL: {case.caller_goal}

FACTS YOU CAN PROVIDE (supply these when the agent asks; do not dump them all at \
once unless your goal explicitly says to): {case.caller_facts}

How to behave:
- Answer what the agent asks. Give one detail at a time unless your goal says to
  provide everything up front.
- If the agent reads your reservation back, confirm it if it's correct, or
  correct it if it's wrong.
- Do NOT invent details beyond the facts above. If asked for something you were
  not given, make a brief reasonable choice consistent with your goal.
- When the agent has clearly ended the call (said goodbye, or is transferring
  you to support), reply with exactly {END_SENTINEL} and nothing else."""


async def run_simulated_conversation(
    case: EvalCase,
    account: dict | None = None,
    max_turns: int = MAX_TURNS,
) -> ConversationResult:
    """Drive one call with an LLM playing the caller; return transcript+snapshot.

    Mirrors harness.run_conversation but generates each caller utterance live
    from the agent's latest reply instead of reading a fixed script.
    """
    account = account or SAMPLE_ACCOUNT
    agent = build(settings, {"account": account})
    caller = ChatOpenAI(
        model=CALLER_MODEL, temperature=CALLER_TEMPERATURE, timeout=30
    )

    # The caller LLM's own message history. From ITS point of view it is the
    # assistant, and the AGENT's replies arrive as "user" (human) messages.
    caller_msgs = [SystemMessage(content=_caller_system_prompt(case))]

    result = ConversationResult()

    # Opening trigger -> greeting, exactly as the real pipeline starts a call.
    greeting = await agent.respond(OPENING_TRIGGER)
    result.transcript.append((OPENING_TRIGGER, greeting))
    caller_msgs.append(HumanMessage(content=greeting))

    for _ in range(max_turns):
        # The caller decides what to say next, given the whole conversation.
        utterance = (await caller.ainvoke(caller_msgs)).content.strip()
        if END_SENTINEL in utterance:
            break
        caller_msgs.append(AIMessage(content=utterance))

        reply = await agent.respond(utterance)
        result.transcript.append((utterance, reply))
        caller_msgs.append(HumanMessage(content=reply))

        # Production fidelity: the agent hung up (finalized or transferred).
        if agent.snapshot().get("end_call"):
            break

    result.snapshot = agent.snapshot()
    return result


async def _main() -> None:
    # Import here to avoid any import-time coupling; scorers already work on the
    # ConversationResult this file produces — no changes needed there.
    from evals.dataset import get_case
    from evals.scorers import print_case_scorecard, score_case

    # The three cases that were fragile / buggy under scripting. Watch the
    # multi-turn ones (drip_fed, ambiguous_date) actually complete now that the
    # caller reacts to what Ann asks.
    for cid in ["drip_fed_details", "ambiguous_date_clarified",
                "non_reservation_transfers"]:
        case = get_case(cid)
        print("\n" + "#" * 70)
        print(f"# {cid}  (simulated caller)")
        print("#" * 70)
        result = await run_simulated_conversation(case)
        for caller_line, reply in result.transcript:
            who = "START " if caller_line == OPENING_TRIGGER else "CALLER"
            print(f"  [{who}] {caller_line}")
            print(f"  [ANN ] {reply}")
        print_case_scorecard(case, score_case(case, result))


if __name__ == "__main__":
    asyncio.run(_main())
