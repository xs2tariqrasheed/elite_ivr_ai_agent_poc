"""Standalone test for the OpenAI → Duckling pickup-time pipeline.

Run from the project root:

    python -m scripts.test_duckling_pickup_time
    python -m scripts.test_duckling_pickup_time "pick me up at 5pm please"

Prereqs:
    * ``OPENAI_API_KEY`` set (in env or .env).
    * Duckling running locally — by default ``http://localhost:8000``.
      Override with ``DUCKLING_URL`` if it lives elsewhere. A quick way
      to start one::

          docker run --rm -p 8000:8000 rasa/duckling

The script:
    1. Sends each transcript to OpenAI to produce a clean time phrase.
    2. Sends that phrase to Duckling's ``/parse`` endpoint.
    3. Prints the normalised phrase, the parsed HH:MM:SS, and timings.
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime

# Allow running as a script *and* as ``python -m scripts.<name>``.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config  # noqa: E402
from services.duckling_service import parse_time_with_duckling  # noqa: E402
from services.llm.pickup_date_time import (  # noqa: E402
    normalize_time_for_duckling_openai,
)


DEFAULT_TRANSCRIPTS = [
    "5pm",
    "five pm",
    "at 5 PM please",
    "pick me up at five thirty in the evening",
    "nine thirty in the morning",
    "quarter past three in the afternoon",
    "half past two",
    "noon",
    "midnight",
    "around 14:30",
    "uh yeah like seven o'clock at night",
    "ten to four PM",
    "eight am sharp",
    "umm I think 6:45 in the evening",
]


def _run_one(transcript: str, reference_time: datetime) -> dict:
    """Run the OpenAI normalise → Duckling parse pipeline for one transcript."""
    t0 = time.perf_counter()
    phrase = normalize_time_for_duckling_openai(transcript, reference_time)
    t1 = time.perf_counter()
    parsed_time = (
        parse_time_with_duckling(phrase, reference_time)
        if phrase
        else None
    )
    t2 = time.perf_counter()
    return {
        "transcript": transcript,
        "phrase": phrase,
        "time": parsed_time,
        "openai_ms": round((t1 - t0) * 1000, 1),
        "duckling_ms": round((t2 - t1) * 1000, 1),
        "total_ms": round((t2 - t0) * 1000, 1),
    }


def _print_row(result: dict) -> None:
    print("-" * 72)
    print(f"transcript : {result['transcript']!r}")
    print(f"phrase     : {result['phrase']!r}")
    print(f"time       : {result['time']}")
    print(
        f"timing     : openai={result['openai_ms']}ms  "
        f"duckling={result['duckling_ms']}ms  "
        f"total={result['total_ms']}ms"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "transcripts",
        nargs="*",
        help="Transcripts to test. If omitted, a built-in sample list is used.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    parser.add_argument(
        "--reference-time",
        default=None,
        help="Override 'now' for relative times, ISO-8601 (e.g. 2026-05-27T10:00).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not config.OPENAI_API_KEY:
        print(
            "ERROR: OPENAI_API_KEY is not set. Add it to your environment or .env.",
            file=sys.stderr,
        )
        return 2

    if args.reference_time:
        try:
            reference_time = datetime.fromisoformat(args.reference_time)
        except ValueError:
            print(
                f"ERROR: --reference-time is not a valid ISO-8601 timestamp: "
                f"{args.reference_time!r}",
                file=sys.stderr,
            )
            return 2
    else:
        reference_time = datetime.now()

    transcripts = args.transcripts or DEFAULT_TRANSCRIPTS

    print(f"Duckling URL : {config.DUCKLING_URL}")
    print(f"OpenAI model : {config.OPENAI_MODEL}")
    print(f"Reference    : {reference_time.isoformat()}")
    print(f"Transcripts  : {len(transcripts)}")

    results = []
    for transcript in transcripts:
        try:
            result = _run_one(transcript, reference_time)
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED on {transcript!r}: {exc}", file=sys.stderr)
            continue
        results.append(result)
        _print_row(result)

    print("-" * 72)
    successes = sum(1 for r in results if r["time"])
    print(f"Parsed {successes}/{len(results)} transcripts to a time.")
    return 0 if successes == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
