"""Play cached mu-law clips locally (from ``agent_voice_service`` memory).

Uses the same ``--clip`` path format as ``scripts/play_mul.py``: each clip is
a list of segments relative to ``AUDIO_DIR`` (no ``.mp3`` suffix), e.g.::

    python scripts/play_cache.py --clip rec_account_name
    python scripts/play_cache.py --clip my_runtime_clip
    python scripts/play_cache.py \\
        --clip verify_passenger_info_part_1 \\
        --clip passenger_names/9012

Loads clips from the local process cache when available. If a clip is missing
(e.g. it was created via ``/gen-audio-in-memory`` on a running server), fetches
WAV audio from ``GET /audio-cache/clips/<path>`` (default base URL
``http://127.0.0.1:8000``).
"""

from __future__ import annotations

import argparse
import audioop
import base64
import logging
import os
import sys
import urllib.error
import urllib.request
from io import BytesIO
from typing import List, Optional, Sequence

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pydub import AudioSegment
from pydub.playback import play

from services import agent_voice_service as voice

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SERVER = "http://127.0.0.1:8000"


def _frames_to_segment(frames: Sequence[str]) -> AudioSegment:
    """Decode base64 mu-law @ 8 kHz frames to a pydub ``AudioSegment``."""
    mulaw = b"".join(base64.b64decode(frame) for frame in frames)
    pcm16 = audioop.ulaw2lin(mulaw, 2)
    return AudioSegment(
        data=pcm16,
        sample_width=2,
        frame_rate=8000,
        channels=1,
    )


def _normalize_clip_path(clip: Sequence[str]) -> List[str]:
    """Drop a leading cache root key if the user passed a full cache path."""
    parts = list(clip)
    if not parts:
        raise ValueError("each clip must contain at least one path segment")

    root_key = voice._audio_root_key()
    if parts[0] == root_key:
        parts = parts[1:]
    if not parts:
        raise ValueError(
            f"clip path must include at least one segment under {root_key!r}"
        )
    return parts


def _fetch_clip_from_server(
    path_parts: Sequence[str],
    server_base: str,
    timeout: int,
) -> AudioSegment:
    clip_path = "/".join(path_parts)
    url = f"{server_base.rstrip('/')}/audio-cache/clips/{clip_path}"
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            wav_bytes = response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise KeyError(f"server {exc.code} for {clip_path}: {body}") from exc
    except urllib.error.URLError as exc:
        raise KeyError(f"server unreachable at {server_base}: {exc.reason}") from exc

    return AudioSegment.from_file(BytesIO(wav_bytes), format="wav")


def _resolve_clip_segment(
    path_parts: List[str],
    server_base: Optional[str],
    timeout: int,
) -> AudioSegment:
    try:
        frames = voice._frames_for(*path_parts)
        return _frames_to_segment(frames)
    except KeyError:
        if not server_base:
            raise
        logger.info(
            "Clip %s not in local cache; fetching from %s",
            "/".join(path_parts),
            server_base,
        )
        return _fetch_clip_from_server(path_parts, server_base, timeout)


def play_cache(
    clips: Sequence[Sequence[str]],
    *,
    server_base: Optional[str] = DEFAULT_SERVER,
    timeout: int = 30,
) -> None:
    """Play one or more cached clips back-to-back."""
    if not clips:
        raise ValueError("clips must contain at least one clip")

    voice.load_audio_files()

    merged = AudioSegment.empty()
    labels: List[str] = []
    for clip in clips:
        path_parts = _normalize_clip_path(clip)
        merged += _resolve_clip_segment(path_parts, server_base, timeout)
        labels.append("/".join(path_parts))

    print(
        f"Playing {len(labels)} cached clip(s) as a single unit "
        f"({len(merged)} ms total):"
    )
    for label in labels:
        print(f"  - {label}")

    play(merged)


def _parse_clip_arg(value: str) -> List[str]:
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
            "Play audio from the mu-law cache (local load, with server fallback "
            "for in-memory-only clips)."
        )
    )
    parser.add_argument(
        "--clip",
        dest="clips",
        action="append",
        type=_parse_clip_arg,
        help=(
            "Clip path relative to AUDIO_DIR (without .mp3), directories "
            "separated by /. Repeat for multiple clips. If omitted, plays a "
            "built-in passenger-info-verification example."
        ),
    )
    parser.add_argument(
        "--server",
        default=DEFAULT_SERVER,
        help=(
            "Base URL of the running IVR API for clips not in the local process "
            f"(default: {DEFAULT_SERVER}). Use --no-server to disable."
        ),
    )
    parser.add_argument(
        "--no-server",
        action="store_true",
        help="Do not fetch missing clips from the running API server.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="HTTP timeout in seconds when fetching from the server.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    clips = args.clips if args.clips else _DEFAULT_CLIPS
    server_base = None if args.no_server else args.server
    try:
        play_cache(clips, server_base=server_base, timeout=args.timeout)
    except KeyError as exc:
        logger.error("%s", exc)
        return 1
    except ValueError as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
