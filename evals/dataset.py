"""Phase 2 — the dataset (the test cases).

A dataset is nothing more than a list of scenarios. Each scenario (`EvalCase`)
pairs an INPUT — the caller's scripted utterances — with an EXPECTED outcome:
what should be true about the reservation once the call ends.

The hard part of a dataset is NOT code; it's *coverage*. A suite that only
tests the happy path tells you almost nothing. The cases below deliberately
stress different behaviors: booking everything up front, dripping details out
one at a time, correcting a detail, a caller who doesn't want a reservation at
all (should transfer), a phrasing that tempts the agent to swap pickup/drop-off,
a corrected callback number, and an ambiguous date the agent must pin down.

Two design decisions worth internalizing — both are about beating the
non-determinism we saw in Phase 0:

  1. Assert on STABLE facts, not brittle absolutes. "Next Thursday" resolves to
     a different calendar date every week, so we never hard-code "July 23rd".
     We assert the day-of-week and time ("thursday", "1:30") — which are fixed
     by the caller's words regardless of when the eval runs.

  2. Match slots by REQUIRED SUBSTRINGS (case-insensitive), not exact strings.
     The agent might store "10 Main Street, Brooklyn" or "10 Main St, Brooklyn";
     requiring the substrings "10 main" and "brooklyn" accepts both while still
     rejecting a wrong address. (Phase 1's scorers will implement this check;
     here we just declare the contract.)

This file does no grading yet. Run it to SEE the suite:

    python evals/dataset.py          # print the catalog of cases
    python evals/dataset.py --run    # actually drive each case through the agent
"""
import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.harness import run_conversation  # noqa: E402


# The three ways a call can end, from the caller's point of view. Every case
# declares which one it expects. (These map onto the session snapshot: a
# finalize sets confirmed + confirmation_number; a transfer sets transferred;
# "neither" means the agent never reached a terminal state — usually a failure.)
FINALIZE = "finalize"
TRANSFER = "transfer"
NEITHER = "neither"


@dataclass
class Expected:
    """The contract a case must satisfy once the call ends.

    terminal_action : FINALIZE | TRANSFER | NEITHER — how the call should end.
    slots           : maps a snapshot field (e.g. "pickup_address") to a list of
                      substrings that must ALL appear in that field, matched
                      case-insensitively. Empty for transfer cases (nothing is
                      booked). This is a pragmatic, dependency-free stand-in for
                      "semantic equality": lenient about formatting, strict about
                      the facts that matter.
    notes           : free text — why this case exists, and any fragility.
    """

    terminal_action: str
    slots: Dict[str, List[str]] = field(default_factory=dict)
    notes: str = ""


@dataclass
class EvalCase:
    """One test scenario: an id, a caller script, and the expected outcome.

    A case can be driven two ways against the SAME `expected` contract:
      * SCRIPTED — replay `caller_turns` verbatim (Phases 0-1). Simple, but a
        fixed script desyncs if the agent asks things in a different order.
      * SIMULATED — let an LLM play the caller from `caller_goal` + `caller_facts`
        (Phase 4). The caller reacts to what the agent actually says, so multi-
        turn cases stop failing for the wrong reason.
    """

    id: str
    description: str
    caller_turns: List[str]
    expected: Expected
    account: Optional[dict] = None  # None -> harness SAMPLE_ACCOUNT
    # Persona for the Phase 4 simulated caller. `caller_goal` is the caller's
    # objective; `caller_facts` are the concrete details it may supply when
    # asked. Kept as plain strings so dataset.py needs no extra imports.
    caller_goal: str = ""
    caller_facts: str = ""


# ---------------------------------------------------------------------------
# THE SUITE
#
# Each case targets a distinct behavior. The `caller_turns` are scripted, so a
# case with several clarification turns can drift if the agent asks things in a
# different order than the script assumes — cases prone to that are flagged in
# `notes`. Phase 4 (an LLM that role-plays the caller) removes that fragility;
# for now, front-loading details keeps most cases on the rails.
# ---------------------------------------------------------------------------
DATASET: List[EvalCase] = [
    EvalCase(
        id="happy_path_upfront",
        description="Caller offers every detail in the first turn.",
        caller_turns=[
            "Hi, I'd like to book a car for next Thursday at 1:30 in the "
            "afternoon. Pick me up at 10 Main Street in Brooklyn, and I'm "
            "heading to JFK Airport.",
            "Yes, that number's fine.",
            "Yes, that's all correct.",
        ],
        expected=Expected(
            terminal_action=FINALIZE,
            slots={
                "pickup_datetime": ["thursday", "1:30"],
                "pickup_address": ["10 main", "brooklyn"],
                "dropoff_address": ["jfk"],
            },
            notes="The baseline. If this ever fails, something is badly wrong.",
        ),
        caller_goal="Book a car reservation in one go, then confirm and finish.",
        caller_facts="Give all of this in your very first sentence: pickup next "
        "Thursday at 1:30 in the afternoon, pickup at 10 Main Street in Brooklyn, "
        "drop-off at JFK Airport. The callback number on file is fine. Confirm "
        "when the agent reads it back.",
    ),
    EvalCase(
        id="drip_fed_details",
        description="Caller reveals one detail per turn; agent must ask for the "
        "rest without re-asking for what it already has.",
        caller_turns=[
            "I need to book a ride.",
            "Next Monday at 9 in the morning.",
            "From 200 Park Avenue in Manhattan.",
            "To LaGuardia Airport.",
            "Yes, that's my number.",
            "Yes, that's correct.",
        ],
        expected=Expected(
            terminal_action=FINALIZE,
            slots={
                "pickup_datetime": ["monday", "9"],
                "pickup_address": ["200 park"],
                "dropoff_address": ["laguardia"],
            },
            notes="Fragile under scripting: the agent chooses what to ask next, "
            "so turn order may drift. A good candidate for the Phase 4 caller.",
        ),
        caller_goal="Book a ride, but reveal only ONE detail at a time so the "
        "agent has to ask for each piece.",
        caller_facts="Pickup is next Monday at 9 in the morning; pickup address "
        "is 200 Park Avenue in Manhattan; drop-off is LaGuardia Airport; the "
        "callback number on file is fine. Open by just saying you'd like to book "
        "a ride, then give each detail only when asked. Confirm when read back.",
    ),
    EvalCase(
        id="correction_at_readback",
        description="Caller changes the drop-off during the read-back; the "
        "corrected value must win and the agent must re-confirm.",
        caller_turns=[
            "Book me a car for this Saturday at 6 PM, pickup at 5 Elm Street, "
            "drop-off at the Plaza Hotel.",
            "Actually, change the drop-off to Grand Central Station instead.",
            "Yes, that number is fine.",
            "Yes, perfect.",
        ],
        expected=Expected(
            terminal_action=FINALIZE,
            slots={
                "pickup_address": ["5 elm"],
                "dropoff_address": ["grand central"],
            },
            notes="Tests that a correction overwrites the slot AND re-locks "
            "finalize until the caller confirms the corrected read-back.",
        ),
        caller_goal="Book a car, but change the drop-off during the read-back.",
        caller_facts="Pickup this Saturday at 6 PM, pickup at 5 Elm Street. At "
        "first say the drop-off is the Plaza Hotel. When the agent reads the "
        "reservation back, change the drop-off to Grand Central Station instead. "
        "The callback number on file is fine. Confirm once the corrected "
        "read-back is right.",
    ),
    EvalCase(
        id="non_reservation_transfers",
        description="Caller wants something that isn't a booking; agent must "
        "transfer to support and end the call — never start a reservation.",
        caller_turns=[
            "Hi, I have a question about a charge on my last invoice.",
        ],
        expected=Expected(
            terminal_action=TRANSFER,
            slots={},
            notes="Intent routing. No slots should be filled.",
        ),
        caller_goal="You are NOT booking a car — you have a billing question "
        "about a charge on your last invoice.",
        caller_facts="You do not want a reservation. State your billing "
        "question. If the agent offers to transfer you to the support desk, "
        "that's fine — let the call end.",
    ),
    EvalCase(
        id="address_swap_trap",
        description="Caller states the destination BEFORE the pickup, tempting "
        "the agent to swap the two addresses.",
        caller_turns=[
            "I'm flying out of JFK on Friday at noon, so I need a pickup from "
            "88 Willow Road in Queens.",
            "Yes, that's my number.",
            "Yes, that's right.",
        ],
        expected=Expected(
            terminal_action=FINALIZE,
            slots={
                "pickup_address": ["88 willow"],
                "dropoff_address": ["jfk"],
                "pickup_datetime": ["friday"],
            },
            notes="The trap: JFK is mentioned first but is the DROP-OFF; 88 "
            "Willow is the PICKUP. A swap makes pickup_address contain 'jfk'.",
        ),
        caller_goal="Book a car, mentioning your destination before your pickup.",
        caller_facts="Say it like this: you're flying out of JFK on Friday at "
        "noon, so you need a pickup from 88 Willow Road in Queens. (Destination "
        "is JFK; pickup is 88 Willow Road.) Callback number on file is fine. "
        "Confirm when read back.",
    ),
    EvalCase(
        id="callback_number_correction",
        description="Caller corrects the callback number; the new number must "
        "replace the one on file.",
        caller_turns=[
            "I'd like a car Tuesday at 3 PM from 12 Oak Lane to Newark Airport.",
            "Actually, call me at 917-555-0199 instead.",
            "Yes, that's all correct.",
        ],
        expected=Expected(
            terminal_action=FINALIZE,
            slots={
                "caller_phone": ["917", "0199"],
                "dropoff_address": ["newark"],
            },
            notes="Tests set_caller_phone. Substrings '917' + '0199' tolerate "
            "any formatting the agent stores the number in.",
        ),
        caller_goal="Book a car and correct your callback number.",
        caller_facts="Pickup Tuesday at 3 PM from 12 Oak Lane to Newark Airport. "
        "When the agent mentions or confirms the callback number, tell them to "
        "call you at 917-555-0199 instead. Confirm when the corrected read-back "
        "is right.",
    ),
    EvalCase(
        id="ambiguous_date_clarified",
        description="Caller is vague about timing; agent must ask to pin down "
        "the day and time before it can finalize.",
        caller_turns=[
            "I need a pickup sometime next week from 3 River Road, going to the "
            "airport.",
            "Let's do Wednesday at 10 in the morning.",
            "JFK, please.",
            "Yes, that number works.",
            "Yes, correct.",
        ],
        expected=Expected(
            terminal_action=FINALIZE,
            slots={
                "pickup_datetime": ["wednesday", "10"],
                "pickup_address": ["3 river"],
                "dropoff_address": ["jfk"],
            },
            notes="Fragile under scripting (multi-clarification). Its real value "
            "arrives with the Phase 4 caller.",
        ),
        caller_goal="Book a car, but be vague about the date at first.",
        caller_facts="You want a pickup from 3 River Road going to JFK Airport. "
        "At first be vague about timing ('sometime next week'). Only when the "
        "agent asks you to be specific, say Wednesday at 10 in the morning. "
        "Callback number on file is fine. Confirm when read back.",
    ),
]


def get_case(case_id: str) -> EvalCase:
    """Look up a single case by id (raises if unknown)."""
    for case in DATASET:
        if case.id == case_id:
            return case
    known = ", ".join(c.id for c in DATASET)
    raise KeyError(f"No case '{case_id}'. Known: {known}")


def print_catalog() -> None:
    """Show the suite as data — no agent is run."""
    print(f"DATASET — {len(DATASET)} cases\n" + "=" * 70)
    for case in DATASET:
        exp = case.expected
        print(f"\n• {case.id}   (expect: {exp.terminal_action})")
        print(f"    {case.description}")
        print(f"    caller turns: {len(case.caller_turns)}")
        if exp.slots:
            for slot, subs in exp.slots.items():
                print(f"    expect {slot}: contains all of {subs}")
        if exp.notes:
            print(f"    note: {exp.notes}")


def actual_terminal_action(snapshot: dict) -> str:
    """Read the terminal action back off a finished snapshot.

    The single source of truth for "how did the call end", derived from the
    session flags. Phase 1's scorers import this so the dataset and the scorers
    can never disagree about what a snapshot means.
    """
    if snapshot.get("transferred"):
        return TRANSFER
    if snapshot.get("confirmed") and snapshot.get("confirmation_number"):
        return FINALIZE
    return NEITHER


async def _run_all() -> None:
    """Drive every case through the real agent and show a provisional result.

    Note the cost: this makes several LLM calls per case. Multi-turn cases may
    read messily if the script drifts from what the agent asks — that's the
    scripted-caller limitation on display, not a bug in the dataset.
    """
    for case in DATASET:
        print("\n" + "#" * 70)
        print(f"# {case.id}  (expect {case.expected.terminal_action})")
        print("#" * 70)
        result = await run_conversation(case.caller_turns, case.account)
        for caller, reply in result.transcript:
            who = "START " if caller == "<call_started>" else "CALLER"
            print(f"  [{who}] {caller}")
            print(f"  [ANN ] {reply}")
        got = actual_terminal_action(result.snapshot or {})
        ok = "PASS" if got == case.expected.terminal_action else "FAIL"
        print(f"  -> terminal action: expected "
              f"{case.expected.terminal_action}, got {got}   [{ok}]")


if __name__ == "__main__":
    if "--run" in sys.argv:
        asyncio.run(_run_all())
    else:
        print_catalog()
        print("\n(Use `python evals/dataset.py --run` to drive these through "
              "the agent.)")
