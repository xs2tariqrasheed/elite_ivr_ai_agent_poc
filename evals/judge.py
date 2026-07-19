"""Phase 3 — LLM-as-judge (grading subjective quality).

Code scorers (Phase 1) answer mechanical questions: "is dropoff_address 'jfk'?".
They CANNOT answer judgment questions: "did that reply sound like a warm human?"
or "was the read-back accurate and complete?" or "did the agent re-ask for
something the caller already gave?". For those we hand the transcript to a
SECOND LLM acting as a QA reviewer, with a RUBRIC, and it returns a structured
verdict with a reason per criterion.

Three practices that separate a useful judge from a misleading one:

  1. Judge with a STRONGER, DIFFERENT model than the one under test. Ann runs on
     gpt-4o-mini; the judge runs on gpt-4o. A model grading its own family tends
     to be too lenient ("self-preference bias").

  2. Give it a RUBRIC, not vibes. Concrete yes/no criteria, each requiring a
     written reason. Vague "rate 1-10" judges are noisy and unactionable.

  3. VALIDATE THE JUDGE. It is just another model and can be wrong. Before you
     trust it on unknown transcripts, confirm it agrees with YOU on transcripts
     whose answer you already know — at minimum, that it fails an obviously bad
     one. The __main__ below demonstrates this with a canned bad transcript.

Run it:

    python evals/judge.py         # judge a live happy-path call, then validate
"""
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

from configs.settings import settings  # noqa: E402  (also loads .env)
from evals.harness import ConversationResult, run_conversation  # noqa: E402


# A capable judge model, deliberately NOT the agent's gpt-4o-mini.
JUDGE_MODEL = "gpt-4o"


# ---------------------------------------------------------------------------
# THE RUBRIC
#
# Each criterion is a single, answerable yes/no question about the WHOLE
# conversation. Keep them about things a human reviewer would judge — not things
# a code scorer already checks (those belong in Phase 1). The judge must return
# {"pass": bool, "reason": str} for each key below, plus an overall verdict.
# ---------------------------------------------------------------------------
RUBRIC: List[Dict[str, str]] = [
    {
        "key": "natural_spoken",
        "criterion": "Every agent reply sounds like a warm, natural human phone "
        "agent speaking out loud: short (one or two sentences), conversational, "
        "no markdown, bullet points, or robotic/boilerplate phrasing.",
    },
    {
        "key": "readback_accurate",
        "criterion": "When the agent read the reservation back to confirm, the "
        "read-back was CONSISTENT with what the caller actually said — same "
        "day-of-week and time, same pickup address, same drop-off address, same "
        "callback number — with nothing invented, contradicted, or omitted. "
        "IMPORTANT: turning a relative date the caller gave (e.g. 'next "
        "Thursday') into a specific calendar date is CORRECT and expected; do "
        "NOT flag it as inaccurate as long as the weekday and time still match.",
    },
    {
        "key": "asked_only_missing",
        "criterion": "The agent only asked for details it did not already have, "
        "and never re-asked for information the caller had already given.",
    },
    {
        "key": "no_redundant_stalls",
        "criterion": "The conversation moved forward without stalling — the "
        "agent did not repeat the same read-back or the same question two turns "
        "in a row, and did not emit a generic error/fallback line.",
    },
    {
        "key": "correct_intent_handling",
        "criterion": "The agent did the right high-level thing for the caller's "
        "intent: it booked the reservation for a booking request, or clearly "
        "handed off to support for anything that is not a reservation.",
    },
]


def _build_judge_prompt() -> str:
    """Assemble the judge's system prompt from the rubric.

    We tell the judge who the agent is and what 'good' means, list the criteria
    by key, and pin the OUTPUT FORMAT to strict JSON so we can parse it. Asking
    for a reason per criterion isn't decoration — it forces the model to
    actually inspect the transcript instead of pattern-matching a score, and it
    gives US something to audit when we validate the judge.
    """
    criteria_block = "\n".join(
        f'  - "{c["key"]}": {c["criterion"]}' for c in RUBRIC
    )
    keys_json = ", ".join(f'"{c["key"]}": {{"pass": bool, "reason": str}}'
                          for c in RUBRIC)
    # The judge needs the SAME context the agent had, or it will penalize
    # correct answers it can't verify (e.g. that "next Thursday" is July 23rd).
    today = datetime.now().strftime("%A, %B %d, %Y")
    return f"""You are a strict QA reviewer for "Ann", an automated phone agent \
that takes car reservations for Elite Limousine. For reference, today's date is \
{today}; use it to check any relative dates the caller gave. You are given the \
full transcript of one call (the agent's replies contain bracketed expression \
tags like [politely] that a text-to-speech engine reads; judge the words, not \
the tags). Evaluate ONLY the criteria below. Be exacting: if a criterion is even \
partly violated, it does not pass. Base every judgment strictly on the \
transcript — never assume facts that are not shown.

CRITERIA (answer each with pass=true/false and a one-sentence reason):
{criteria_block}

Respond with ONLY a JSON object, no prose, no code fences, of exactly this shape:
{{"criteria": {{{keys_json}}}, "overall_pass": bool, "summary": "one sentence"}}
overall_pass is true only if every criterion passes."""


def format_transcript(result: ConversationResult) -> str:
    """Render a ConversationResult as plain text for the judge to read."""
    lines = []
    for caller, reply in result.transcript:
        if caller == "<call_started>":
            lines.append("(call begins)")
        else:
            lines.append(f"CALLER: {caller}")
        lines.append(f"ANN: {reply}")
    return "\n".join(lines)


def _parse_json(text: str) -> dict:
    """Parse the judge's reply, tolerating accidental ```json code fences."""
    text = text.strip()
    if text.startswith("```"):
        # Strip a leading ```json / ``` fence and the trailing ```.
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[len("json"):]
        text = text.strip().rstrip("`").strip()
    return json.loads(text)


async def judge_conversation(result: ConversationResult) -> dict:
    """Send one conversation to the judge model and return its parsed verdict.

    temperature=0 so the same transcript gets the same verdict run to run — a
    judge you can't reproduce is a judge you can't trust or debug.
    """
    llm = ChatOpenAI(model=JUDGE_MODEL, temperature=0, timeout=30)
    messages = [
        SystemMessage(content=_build_judge_prompt()),
        HumanMessage(content="TRANSCRIPT:\n" + format_transcript(result)),
    ]
    resp = await llm.ainvoke(messages)
    return _parse_json(resp.content)


def print_verdict(title: str, verdict: dict) -> None:
    """Human-friendly dump of a judge verdict."""
    overall = "PASS" if verdict.get("overall_pass") else "FAIL"
    print(f"\n{'=' * 70}\nJUDGE VERDICT — {title}  [{overall}]\n{'=' * 70}")
    for key, item in verdict.get("criteria", {}).items():
        mark = "✅" if item.get("pass") else "❌"
        print(f"  {mark} {key}")
        print(f"       {item.get('reason', '')}")
    print(f"  summary: {verdict.get('summary', '')}")


# A deliberately AWFUL transcript used to VALIDATE the judge. It is robotic, uses
# markdown bullets, invents/omits details in the read-back, re-asks for known
# info, and stalls. A trustworthy judge must FAIL this. If it passes, the judge
# (or the rubric) is broken and nothing it says can be believed.
BAD_TRANSCRIPT = ConversationResult(
    transcript=[
        ("<call_started>", "Hello. State your request."),
        (
            "I'd like a car next Thursday at 1:30 PM from 10 Main Street, "
            "Brooklyn to JFK Airport.",
            "Understood. Please provide:\n- Pickup date\n- Pickup address\n"
            "- Destination",  # re-asks for everything just given; markdown list
        ),
        (
            "I just told you all of that.",
            "Confirming your reservation: pickup Friday at 9:00 AM from 500 "
            "Broadway to LaGuardia. Correct?",  # totally wrong read-back
        ),
        ("No, that's all wrong.", "Sorry, need more steps to process this "
                                  "request."),  # generic stall/fallback
    ],
    snapshot={"confirmed": False, "transferred": False, "end_call": False},
)


async def _main() -> None:
    from evals.dataset import get_case

    # (1) Judge a REAL, good call: run the happy-path case, then grade it.
    print("Running a live happy-path call, then judging it (makes LLM calls)...")
    good = await run_conversation(get_case("happy_path_upfront").caller_turns)
    good_verdict = await judge_conversation(good)
    print_verdict("live happy_path_upfront", good_verdict)

    # (2) VALIDATE the judge: it must FAIL the deliberately awful transcript.
    print("\nValidating the judge against a known-bad transcript "
          "(it should FAIL)...")
    bad_verdict = await judge_conversation(BAD_TRANSCRIPT)
    print_verdict("known-bad transcript", bad_verdict)

    ok = good_verdict.get("overall_pass") and not bad_verdict.get("overall_pass")
    print("\n" + ("JUDGE VALIDATION PASSED: it passed the good call and failed "
                  "the bad one." if ok else
                  "JUDGE VALIDATION FAILED: the judge did not cleanly separate "
                  "good from bad — tighten the rubric before trusting it."))


if __name__ == "__main__":
    asyncio.run(_main())
