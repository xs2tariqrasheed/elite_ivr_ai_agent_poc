"""Phase 1 — code-based scorers (deterministic grading).

A scorer is a tiny pure function: given a finished conversation (and, for some,
the case's expected outcome), return pass/fail plus a human-readable reason. No
LLM, no judgment — same input always yields the same score. That reproducibility
is the whole appeal: these catch regressions cheaply and never flake.

Two families:

  * CASE scorers need the dataset's `Expected` contract:
      - terminal action (finalize vs transfer vs neither)
      - each declared slot contains its required substrings
  * UNIVERSAL scorers apply to EVERY conversation, no expectation needed:
      - every agent reply carries at least one expression tag
      - no reply contains markdown
      - replies are short enough to speak
      - the agent finalized at most once (the Phase 0 double-finalize bug)
      - no reply is the generic error/fallback line (the ambiguous-date bug)

Run it to score the whole dataset and print a scorecard:

    python evals/scorers.py
"""
import asyncio
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.dataset import (  # noqa: E402
    DATASET,
    EvalCase,
    actual_terminal_action,
)
from evals.harness import ConversationResult, run_conversation  # noqa: E402


@dataclass
class ScoreResult:
    """The verdict of one scorer on one conversation.

    name   : short label shown in the scorecard (e.g. "terminal_action",
             "slot:pickup_address", "tags_present").
    passed : did this check pass?
    detail : one line explaining the result — especially WHY it failed.
    """

    name: str
    passed: bool
    detail: str


# ---------------------------------------------------------------------------
# CASE SCORERS — graded against the dataset's Expected contract.
# ---------------------------------------------------------------------------
def score_terminal_action(result: ConversationResult, case: EvalCase) -> ScoreResult:
    """Did the call end the way the case expects (finalize / transfer / neither)?"""
    got = actual_terminal_action(result.snapshot or {})
    want = case.expected.terminal_action
    return ScoreResult(
        name="terminal_action",
        passed=(got == want),
        detail=f"expected {want}, got {got}",
    )


def score_slots(result: ConversationResult, case: EvalCase) -> List[ScoreResult]:
    """For each expected slot, does the recorded value contain all its substrings?

    This is the Phase 2 design rule in code: match by required, case-insensitive
    substrings so formatting differences ("10 Main Street" vs "10 Main St") pass
    while genuinely wrong values ("jfk" in the pickup) fail. One ScoreResult per
    slot so a scorecard pinpoints exactly which field went wrong.
    """
    snapshot = result.snapshot or {}
    out: List[ScoreResult] = []
    for slot, required in case.expected.slots.items():
        value = snapshot.get(slot)
        haystack = (value or "").lower()
        missing = [sub for sub in required if sub.lower() not in haystack]
        out.append(
            ScoreResult(
                name=f"slot:{slot}",
                passed=(not missing and bool(value)),
                detail=(
                    f"value={value!r}"
                    if not missing and value
                    else f"value={value!r} missing {missing}"
                ),
            )
        )
    return out


# ---------------------------------------------------------------------------
# UNIVERSAL SCORERS — apply to every conversation, no expectation needed.
# ---------------------------------------------------------------------------
_TAG_RE = re.compile(r"\[[a-zA-Z]+\]")  # e.g. [politely], [asking], [warmly]
_FALLBACK_MARK = "need more steps to process"  # the agent's generic error line
_MAX_REPLY_CHARS = 400  # a spoken one-or-two-sentence reply proxy


def _has_markdown(text: str) -> bool:
    """True if a reply looks like markdown (Ann must speak, not format)."""
    if "**" in text or "•" in text:
        return True
    for line in text.splitlines():
        if line.strip().startswith(("- ", "* ", "# ")):
            return True
    return False


def score_tags_present(result: ConversationResult) -> ScoreResult:
    """Every agent reply must contain at least one expression tag (prompt rule)."""
    bad = [r for r in result.replies if not _TAG_RE.search(r)]
    return ScoreResult(
        name="tags_present",
        passed=(not bad),
        detail="all replies tagged" if not bad
        else f"{len(bad)} reply(ies) missing a tag, e.g. {bad[0][:60]!r}",
    )


def score_no_markdown(result: ConversationResult) -> ScoreResult:
    """No reply may contain markdown formatting."""
    bad = [r for r in result.replies if _has_markdown(r)]
    return ScoreResult(
        name="no_markdown",
        passed=(not bad),
        detail="clean" if not bad else f"{len(bad)} reply(ies) contain markdown",
    )


def score_reply_length(result: ConversationResult) -> ScoreResult:
    """Replies should be short enough to speak (heuristic char cap).

    A proxy, not truth: length is a cheap stand-in for "spoken and brief". The
    judge (Phase 3) grades naturalness properly; this just flags obvious runaway
    replies without an LLM call.
    """
    longest = max((len(r) for r in result.replies), default=0)
    return ScoreResult(
        name="reply_length",
        passed=(longest <= _MAX_REPLY_CHARS),
        detail=f"longest reply {longest} chars (cap {_MAX_REPLY_CHARS})",
    )


def score_single_finalize(result: ConversationResult) -> ScoreResult:
    """The agent must announce a confirmation number at most once.

    The Phase 0 double-finalize bug produced two 'reservation number is …'
    replies (two numbers, two emails) for one booking. Counting those replies
    catches it deterministically. Zero is fine (transfer / unfinished calls).
    """
    n = sum(1 for r in result.replies if "reservation number is" in r.lower())
    return ScoreResult(
        name="single_finalize",
        passed=(n <= 1),
        detail=f"{n} finalize announcement(s)",
    )


def score_no_error_fallback(result: ConversationResult) -> ScoreResult:
    """No reply may be the generic 'need more steps to process' fallback."""
    bad = [r for r in result.replies if _FALLBACK_MARK in r.lower()]
    return ScoreResult(
        name="no_error_fallback",
        passed=(not bad),
        detail="no fallback" if not bad else "agent emitted the error fallback",
    )


UNIVERSAL_SCORERS = [
    score_tags_present,
    score_no_markdown,
    score_reply_length,
    score_single_finalize,
    score_no_error_fallback,
]


def score_case(case: EvalCase, result: ConversationResult) -> List[ScoreResult]:
    """Run every scorer (case + universal) against one finished conversation."""
    scores = [score_terminal_action(result, case)]
    scores += score_slots(result, case)
    scores += [scorer(result) for scorer in UNIVERSAL_SCORERS]
    return scores


# ---------------------------------------------------------------------------
# Runner + scorecard.
# ---------------------------------------------------------------------------
def print_case_scorecard(case: EvalCase, scores: List[ScoreResult]) -> bool:
    """Print one case's scores; return True if EVERY scorer passed."""
    passed = sum(s.passed for s in scores)
    all_ok = passed == len(scores)
    head = "PASS" if all_ok else "FAIL"
    print(f"\n{case.id}  [{head}]  ({passed}/{len(scores)} checks)")
    for s in scores:
        mark = "✅" if s.passed else "❌"
        print(f"    {mark} {s.name:22} {s.detail}")
    return all_ok


async def score_dataset() -> None:
    """Drive every case through the agent, score it, and print an aggregate."""
    cases_passed = 0
    total_checks = 0
    checks_passed = 0
    for case in DATASET:
        result = await run_conversation(case.caller_turns, case.account)
        scores = score_case(case, result)
        if print_case_scorecard(case, scores):
            cases_passed += 1
        total_checks += len(scores)
        checks_passed += sum(s.passed for s in scores)

    print("\n" + "=" * 70)
    print("AGGREGATE")
    print("=" * 70)
    print(f"  cases fully passing : {cases_passed}/{len(DATASET)}")
    print(f"  individual checks   : {checks_passed}/{total_checks} "
          f"({100 * checks_passed // max(total_checks, 1)}%)")


if __name__ == "__main__":
    asyncio.run(score_dataset())
