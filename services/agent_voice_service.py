"""Agent voice service.

Responsibilities:
    * Load every pre-recorded mp3 from ``AUDIO_DIR`` (recursive)
      into memory at startup, transcoded once to the format Twilio
      Media Streams expects (mu-law @ 8 kHz mono).
    * Provide a helper to play a cached clip into a Twilio WebSocket.

Twilio Media Streams send/receive base64-encoded G.711 mu-law payloads
at 8 kHz mono.  We pre-slice each clip into 20 ms (160 byte) frames so
that streaming is just a tight loop of ``send_text``.

Cache layout
------------
The in-memory cache mirrors the on-disk directory structure rooted at
``AUDIO_DIR``.  The first key is the basename of ``AUDIO_DIR`` itself,
so e.g. ``audio_files/account_names/12345.mp3`` is reachable as::

    _AUDIO_CACHE["audio_files"]["account_names"]["12345"]

Top-level mp3s live directly under the root key, e.g.::

    _AUDIO_CACHE["audio_files"]["rec_account_name"]
"""
import audioop
import base64
import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional, Sequence

from pydub import AudioSegment

import config

logger = logging.getLogger(__name__)


# Nested cache: leaves are ``List[str]`` (base64-encoded mu-law frames);
# interior nodes are sub-directories keyed by directory name.
_AUDIO_CACHE: Dict[str, Any] = {}

# Twilio expects 20 ms frames @ 8 kHz, so 160 bytes of mu-law per frame.
_FRAME_BYTES = 160


def _audio_root_key() -> str:
    """Return the cache's top-level key (basename of ``AUDIO_DIR``)."""
    return os.path.basename(os.path.normpath(config.AUDIO_DIR)) or "audio_files"


def _mp3_to_mulaw_frames(path: str) -> List[str]:
    """Decode an mp3 file to base64-encoded mu-law @ 8 kHz frames."""
    seg = AudioSegment.from_file(path, format="mp3")
    # Force mono, 8 kHz, 16-bit PCM
    seg = seg.set_channels(1).set_frame_rate(8000).set_sample_width(2)
    pcm16 = seg.raw_data
    mulaw = audioop.lin2ulaw(pcm16, 2)

    frames: List[str] = []
    for i in range(0, len(mulaw), _FRAME_BYTES):
        chunk = mulaw[i:i + _FRAME_BYTES]
        if len(chunk) < _FRAME_BYTES:
            # Pad final frame with silence (mu-law silence = 0xFF)
            chunk = chunk + (b"\xff" * (_FRAME_BYTES - len(chunk)))
        frames.append(base64.b64encode(chunk).decode("ascii"))
    return frames


def _load_audio_tree(dir_path: str) -> Dict[str, Any]:
    """Recursively decode every ``*.mp3`` under ``dir_path`` into a tree."""
    tree: Dict[str, Any] = {}
    for entry in sorted(os.listdir(dir_path)):
        full = os.path.join(dir_path, entry)

        if os.path.isfile(full) and entry.lower().endswith(".mp3"):
            name = entry[:-4]
            if name in tree and isinstance(tree[name], dict):
                raise ValueError(
                    f"Audio name conflict in {dir_path}: '{name}' exists as both file and directory"
                )
            try:
                tree[name] = _mp3_to_mulaw_frames(full)
                logger.debug("Loaded %s (%d frames)", full, len(tree[name]))
            except Exception:
                logger.exception("Failed to load audio file: %s", full)
                raise
            continue

        if os.path.isdir(full):
            sub_tree = _load_audio_tree(full)
            if not sub_tree:
                continue
            if entry in tree and isinstance(tree[entry], list):
                raise ValueError(
                    f"Audio name conflict in {dir_path}: '{entry}' exists as both directory and file"
                )
            tree[entry] = sub_tree

    return tree


def _count_clips(node: Any) -> int:
    if isinstance(node, list):
        return 1
    if isinstance(node, dict):
        return sum(_count_clips(v) for v in node.values())
    return 0


def list_cached_audio_parts() -> List[List[str]]:
    """Return every cached clip path as path-parts.

    Each item includes the root key (usually ``audio_files``), followed by
    optional nested sub-directory names and the clip filename without extension.
    """
    parts_list: List[List[str]] = []

    def _walk(node: Any, prefix: List[str]) -> None:
        if isinstance(node, list):
            parts_list.append(prefix.copy())
            return
        if isinstance(node, dict):
            for key in sorted(node):
                _walk(node[key], prefix + [key])

    for root_key in sorted(_AUDIO_CACHE):
        _walk(_AUDIO_CACHE[root_key], [root_key])

    return parts_list


def load_audio_files() -> None:
    """Pre-load every mp3 under ``AUDIO_DIR`` recursively."""
    audio_dir = config.AUDIO_DIR
    logger.info("Loading audio files from %s", audio_dir)

    if not os.path.isdir(audio_dir):
        raise FileNotFoundError(f"AUDIO_DIR does not exist: {audio_dir}")

    root_cache = _load_audio_tree(audio_dir)

    _AUDIO_CACHE.clear()
    _AUDIO_CACHE[_audio_root_key()] = root_cache

    total = _count_clips(root_cache)
    logger.info("Loaded %d audio clips", total)


def _frames_for(*audio_path: str) -> List[str]:
    """Resolve an audio path (relative to ``AUDIO_DIR``) to its frames.

    Example::

        _frames_for("rec_account_name")
        _frames_for("account_names", "12345")
    """
    if not audio_path:
        raise ValueError("audio_path must contain at least one segment")

    root_key = _audio_root_key()
    node: Any = _AUDIO_CACHE.get(root_key)
    if node is None:
        raise KeyError(f"Audio root not loaded: {root_key}")

    for part in audio_path:
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"Audio not loaded: {'/'.join(audio_path)}")
        node = node[part]

    if not isinstance(node, list):
        raise KeyError(
            f"Audio path resolves to a directory, not a clip: {'/'.join(audio_path)}"
        )
    return node


async def play_audio(
    websocket,
    stream_sid: str,
    clips: Sequence[Sequence[str]],
    mark_name: Optional[str] = None,
) -> str:
    """Stream one or more cached clips to Twilio as a single utterance.

    ``clips`` is a sequence of clip paths, where each clip path is itself
    a sequence of directory-walk segments relative to ``AUDIO_DIR``::

        play_audio(ws, sid, [["rec_account_name"]])
        play_audio(ws, sid, [["account_names", "12345"], ["rec_greet_unknown"]])

    Frames from every clip are concatenated and sent back-to-back over
    the WebSocket, followed by a single ``mark`` event.  Twilio echoes
    that mark back once the entire merged sequence has finished playing
    on the caller's line, which is what enables non-interruptible flow.

    Returns the mark name that was sent so the caller can wait for it.
    """
    if not clips:
        raise ValueError("clips must contain at least one clip")

    all_frames: List[str] = []
    labels: List[str] = []
    for clip in clips:
        if not clip:
            raise ValueError("each clip must contain at least one path segment")
        all_frames.extend(_frames_for(*clip))
        labels.append("_".join(clip))

    if mark_name is None:
        mark_name = f"end_{'__'.join(labels)}_{uuid.uuid4().hex[:8]}"

    logger.info(
        "Playing audio sequence %s on stream %s (%d clips, %d frames)",
        " + ".join("/".join(c) for c in clips),
        stream_sid,
        len(clips),
        len(all_frames),
    )

    for frame_b64 in all_frames:
        msg = {
            "event": "media",
            "streamSid": stream_sid,
            "media": {"payload": frame_b64},
        }
        await websocket.send_text(json.dumps(msg))

    # Single mark for the entire merged sequence - Twilio will echo this
    # back after all buffered audio has finished playing.
    mark_msg = {
        "event": "mark",
        "streamSid": stream_sid,
        "mark": {"name": mark_name},
    }
    await websocket.send_text(json.dumps(mark_msg))
    return mark_name


async def send_clear(websocket, stream_sid: str) -> None:
    """Clear any audio Twilio has buffered (used before hangup)."""
    msg = {"event": "clear", "streamSid": stream_sid}
    try:
        await websocket.send_text(json.dumps(msg))
    except Exception:
        logger.debug("send_clear: websocket already closed")
