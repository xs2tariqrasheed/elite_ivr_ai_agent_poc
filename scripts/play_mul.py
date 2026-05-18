"""Local test playback that mirrors ``_speak``/``play_audio`` semantics.

This is a standalone helper for ad-hoc testing.  It accepts the same
``clips`` shape as ``phase_handlers.speak._speak`` — a list of clip
paths where each clip path is a list of directory-walk segments
relative to ``AUDIO_DIR`` — concatenates the underlying mp3 files,
and plays the merged result through the local speakers as a single
unit (no Twilio / WebSocket involved).

Usage:
    python scripts/play_mul.py
    python scripts/play_mul.py --clip verify_passenger_info_part_1 \
                               --clip passenger_names/9012 \
                               --clip verify_passenger_info_part_2 \
                               --clip phone_numbers/+923234251430
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Sequence

# Make the project root importable when run as ``python scripts/play_mul.py``.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pydub import AudioSegment
from pydub.playback import play

import config


def _resolve_clip_path(clip: Sequence[str]) -> str:
    """Resolve a clip path (segments relative to ``AUDIO_DIR``) to an mp3 file."""
    if not clip:
        raise ValueError("each clip must contain at least one path segment")

    parts = list(clip)
    parts[-1] = f"{parts[-1]}.mp3"
    full_path = os.path.join(config.AUDIO_DIR, *parts)

    if not os.path.isfile(full_path):
        raise FileNotFoundError(f"Audio clip not found: {full_path}")
    return full_path


def play_mul(clips: Sequence[Sequence[str]]) -> None:
    """Play one or more cached clips as a single utterance, locally.

    ``clips`` is a list of clip paths, where each clip path is itself a
    list of directory-walk segments relative to ``AUDIO_DIR``::

        play_mul([["rec_account_name"]])
        play_mul([
            ["verify_passenger_info_part_1"],
            ["passenger_names", "9012"],
            ["verify_passenger_info_part_2"],
            ["phone_numbers", "+923234251430"],
        ])

    All clips are concatenated into a single ``AudioSegment`` and played
    back-to-back, so this function returns only after the entire
    sequence has finished playing on the local speakers.
    """
    if not clips:
        raise ValueError("clips must contain at least one clip")

    merged = AudioSegment.empty()
    resolved: List[str] = []
    for clip in clips:
        path = _resolve_clip_path(clip)
        resolved.append(path)
        merged += AudioSegment.from_file(path, format="mp3")

    print(
        f"Playing {len(resolved)} clip(s) as a single unit "
        f"({len(merged)} ms total):"
    )
    for path in resolved:
        print(f"  - {os.path.relpath(path)}")

    play(merged)


def _parse_clip_arg(value: str) -> List[str]:
    """Parse a ``--clip dir/sub/name`` argument into path segments."""
    parts = [p for p in value.replace(os.sep, "/").split("/") if p]
    if not parts:
        raise argparse.ArgumentTypeError(f"empty clip path: {value!r}")
    return parts


_DEFAULT_CLIPS: List[List[str]] = [
    ["verify_passenger_info_part_1"],
    ["passenger_names", "9012"],
    ["verify_passenger_info_part_2"],
    ["phone_numbers", "+923234251430"],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locally play one or more cached audio clips back-to-back, "
            "mirroring the ``_speak`` helper."
        )
    )
    parser.add_argument(
        "--clip",
        dest="clips",
        action="append",
        type=_parse_clip_arg,
        help=(
            "A clip path relative to AUDIO_DIR (without .mp3 extension), "
            "with directories separated by ``/``. May be repeated to play "
            "multiple clips as a single unit. If omitted, a built-in "
            "passenger-info-verification example is played."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    clips = args.clips if args.clips else _DEFAULT_CLIPS
    play_mul(["my_runtime_clip"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
