"""Agent voice service.

Responsibilities:
    * Load every pre-recorded mp3 from ``AUDIO_DIR`` into memory at
      startup, transcoded once to the format Twilio Media Streams
      expects (mu-law @ 8 kHz mono).
    * Expose a registry of audio constants (see ``constants.audio_files``).
    * Provide a helper to play a cached clip into a Twilio WebSocket.

Twilio Media Streams send/receive base64-encoded G.711 mu-law payloads
at 8 kHz mono.  We pre-slice each clip into 20 ms (160 byte) frames so
that streaming is just a tight loop of ``send_text``.
"""
import audioop
import base64
import json
import logging
import os
import uuid
from typing import Dict, List, Optional

from pydub import AudioSegment

import config
from constants import audio_files as audio_const

logger = logging.getLogger(__name__)


# Each clip is stored as a list of base64-encoded mu-law frames.
_AUDIO_CACHE: Dict[str, List[str]] = {}

# Twilio expects 20 ms frames @ 8 kHz, so 160 bytes of mu-law per frame.
_FRAME_BYTES = 160


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


def load_audio_files() -> None:
    """Pre-load every clip listed in ``audio_const.ALL_AUDIO_FILES``.

    Called once at app startup.  Missing files raise ``FileNotFoundError``.
    """
    audio_dir = config.AUDIO_DIR
    logger.info("Loading audio files from %s", audio_dir)

    for name in audio_const.ALL_AUDIO_FILES:
        path = os.path.join(audio_dir, name + ".mp3")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required audio file is missing: {path}")
        try:
            _AUDIO_CACHE[name] = _mp3_to_mulaw_frames(path)
            logger.debug("Loaded %s (%d frames)", name, len(_AUDIO_CACHE[name]))
        except Exception:
            logger.exception("Failed to load audio file: %s", path)
            raise

    logger.info("Loaded %d audio clips", len(_AUDIO_CACHE))


def _frames_for(audio_name: str) -> List[str]:
    if audio_name not in _AUDIO_CACHE:
        raise KeyError(f"Audio not loaded: {audio_name}")
    return _AUDIO_CACHE[audio_name]


async def play_audio(
    websocket,
    stream_sid: str,
    audio_name: str,
    mark_name: Optional[str] = None,
) -> str:
    """Stream a cached audio clip to Twilio.

    Sends every 20 ms frame followed by a ``mark`` event.  Twilio echoes
    that mark back when the audio has actually finished playing on the
    caller's line, which is what enables non-interruptible flow.

    Returns the mark name that was sent so the caller can wait for it.
    """
    frames = _frames_for(audio_name)
    if mark_name is None:
        mark_name = f"end_{audio_name}_{uuid.uuid4().hex[:8]}"

    logger.info("Playing audio %s on stream %s (%d frames)", audio_name, stream_sid, len(frames))

    for frame_b64 in frames:
        msg = {
            "event": "media",
            "streamSid": stream_sid,
            "media": {"payload": frame_b64},
        }
        await websocket.send_text(json.dumps(msg))

    # Mark event - Twilio will echo this back after the buffered audio
    # has finished playing.
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
