"""Phase 5 — the scorecard & regression gate (the whole suite in one command).

This ties every earlier phase together:

    for each case:
        run it N times through the SIMULATED CALLER (Phase 4)
        grade each run with the CODE SCORERS (Phase 1) AND the JUDGE (Phase 3)
    report a PASS RATE per case, save the results, and (optionally) diff a
    saved BASELINE so you can see the number move when you change the agent.

Three ideas make this more than "run the tests once":

  1. PASS RATE, not pass/fail. LLM agents are non-deterministic — we saw the
     same case pass and fail on different single runs. Running each case N times
     and reporting passes/N turns that flakiness into a measurement.

  2. BOTH GRADERS in one verdict. Each run's checks are the code scorers plus
     the judge's criteria (prefixed "judge:"). A run passes only if ALL of them
     pass — objective facts AND subjective quality.

  3. SAVE + COMPARE. Results are written to evals/results/. Point --baseline at
     an earlier file and the scorecard prints the delta. Fix a bug, re-run,
     watch the pass rate climb: that before/after loop is why evals exist.

Examples:
    python evals/run.py                              # all cases, 3 runs each
    python evals/run.py --runs 5                     # more runs = tighter rate
    python evals/run.py --cases drip_fed_details     # just one case
    python evals/run.py --no-judge                   # code scorers only (cheaper)
    python evals/run.py --label baseline             # name this run's saved file
    python evals/run.py --baseline evals/results/baseline-<ts>.json
"""
import argparse
import asyncio
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.caller import run_simulated_conversation  # noqa: E402
from evals.dataset import DATASET, get_case  # noqa: E402
from evals.judge import judge_conversation  # noqa: E402
from evals.scorers import score_case  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass
class RunOutcome:
    """One (case, run) result: did every check pass, and each check's verdict."""

    case_id: str
    passed: bool
    checks: Dict[str, bool] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class CaseAggregate:
    """All runs of one case, rolled up into pass counts."""

    case_id: str
    runs: int = 0
    passes: int = 0
    # per-check: how many runs it passed, and how many runs it appeared in
    # (slots differ per case; errored runs contribute no checks).
    check_pass: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    check_total: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    errors: List[str] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passes / self.runs if self.runs else 0.0


async def _one_run(case, run_idx: int, use_judge: bool, sem) -> RunOutcome:
    """Drive one simulated call and grade it with code scorers (+ judge).

    A run passes only if every check passes. Any exception (agent error, judge
    parse failure, timeout) is caught and recorded as a failed, errored run —
    the suite must never crash just because one conversation went sideways.
    """
    async with sem:
        try:
            result = await run_simulated_conversation(case)
            checks: Dict[str, bool] = {
                s.name: s.passed for s in score_case(case, result)
            }
            if use_judge:
                verdict = await judge_conversation(result)
                for key, item in verdict.get("criteria", {}).items():
                    checks[f"judge:{key}"] = bool(item.get("pass"))
            outcome = RunOutcome(case.id, all(checks.values()), checks, None)
        except Exception as exc:  # noqa: BLE001
            outcome = RunOutcome(case.id, False, {}, f"{type(exc).__name__}: {exc}")
    mark = "PASS" if outcome.passed else ("ERROR" if outcome.error else "FAIL")
    print(f"  {case.id:28} run {run_idx + 1}: {mark}"
          + (f"  ({outcome.error})" if outcome.error else ""))
    return outcome


def _aggregate(outcomes: List[RunOutcome]) -> Dict[str, CaseAggregate]:
    """Roll per-run outcomes up into one CaseAggregate per case."""
    aggs: Dict[str, CaseAggregate] = {}
    for o in outcomes:
        agg = aggs.setdefault(o.case_id, CaseAggregate(o.case_id))
        agg.runs += 1
        agg.passes += int(o.passed)
        if o.error:
            agg.errors.append(o.error)
        for name, passed in o.checks.items():
            agg.check_total[name] += 1
            agg.check_pass[name] += int(passed)
    return aggs


def _pct(n: int, d: int) -> int:
    return 100 * n // d if d else 0


def print_scorecard(
    aggs: Dict[str, CaseAggregate], baseline: Optional[dict]
) -> None:
    """Print each case's pass rate, its imperfect checks, and any baseline delta."""
    print("\n" + "=" * 70)
    print("SCORECARD")
    print("=" * 70)
    for case in DATASET:
        agg = aggs.get(case.id)
        if not agg:
            continue
        rate = f"{agg.passes}/{agg.runs} ({_pct(agg.passes, agg.runs)}%)"
        delta = ""
        if baseline:
            prev = baseline.get("cases", {}).get(case.id)
            if prev:
                d = _pct(agg.passes, agg.runs) - prev["pass_rate_pct"]
                delta = f"   Δ {d:+d}% vs baseline"
        print(f"\n{case.id:30} {rate}{delta}")
        # Only show checks that were not a clean sweep — those carry the signal.
        for name in sorted(agg.check_total):
            p, t = agg.check_pass[name], agg.check_total[name]
            if p < t:
                print(f"    ✗ {name:26} {p}/{t} passed")
        for err in agg.errors[:2]:
            print(f"    ! errored: {err}")

    total_runs = sum(a.runs for a in aggs.values())
    total_pass = sum(a.passes for a in aggs.values())
    mean_rate = (sum(a.pass_rate for a in aggs.values()) / len(aggs)) if aggs else 0
    print("\n" + "=" * 70)
    print("AGGREGATE")
    print("=" * 70)
    print(f"  run-level pass rate : {total_pass}/{total_runs} "
          f"({_pct(total_pass, total_runs)}%)")
    print(f"  mean case pass rate : {mean_rate * 100:.0f}%")


def _serialize(aggs: Dict[str, CaseAggregate], runs: int, label: str) -> dict:
    """Shape the results for JSON so a later run can diff against them."""
    return {
        "label": label,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "runs_per_case": runs,
        "cases": {
            cid: {
                "passes": a.passes,
                "runs": a.runs,
                "pass_rate_pct": _pct(a.passes, a.runs),
                "checks": {
                    name: {"pass": a.check_pass[name], "total": a.check_total[name]}
                    for name in a.check_total
                },
                "errors": a.errors,
            }
            for cid, a in aggs.items()
        },
    }


def _save(payload: dict, label: str) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = payload["timestamp"].replace(":", "").replace("-", "")
    path = RESULTS_DIR / f"{label}-{stamp}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run the quick_reservation eval.")
    parser.add_argument("--runs", type=int, default=3,
                        help="how many times to run each case (default 3)")
    parser.add_argument("--cases", default="",
                        help="comma-separated case ids (default: all)")
    parser.add_argument("--label", default="run",
                        help="name for the saved results file")
    parser.add_argument("--baseline", default="",
                        help="path to an earlier results JSON to diff against")
    parser.add_argument("--no-judge", action="store_true",
                        help="skip the LLM judge (code scorers only, cheaper)")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="max conversations in flight at once")
    args = parser.parse_args()

    cases = ([get_case(c.strip()) for c in args.cases.split(",") if c.strip()]
             if args.cases else list(DATASET))
    baseline = (json.loads(Path(args.baseline).read_text())
                if args.baseline else None)

    print(f"Running {len(cases)} case(s) x {args.runs} run(s) "
          f"{'with' if not args.no_judge else 'without'} the judge "
          f"(concurrency {args.concurrency})...\n")

    sem = asyncio.Semaphore(args.concurrency)
    tasks = [
        _one_run(case, i, not args.no_judge, sem)
        for case in cases
        for i in range(args.runs)
    ]
    outcomes = await asyncio.gather(*tasks)

    aggs = _aggregate(outcomes)
    print_scorecard(aggs, baseline)

    saved = _save(_serialize(aggs, args.runs, args.label), args.label)
    print(f"\nresults saved to {saved}")
    if baseline:
        print(f"compared against baseline: {args.baseline}")


if __name__ == "__main__":
    asyncio.run(main())
