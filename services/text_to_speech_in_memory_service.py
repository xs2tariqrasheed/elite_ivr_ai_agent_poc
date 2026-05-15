"""In-memory text-to-speech service using ElevenLabs.

Synthesizes mp3 audio from text via the ``elevenlabs`` Python SDK,
re-encodes the result to IVR-friendly 64 kbps mono/16kHz mp3 with
ffmpeg, and stores the decoded mu-law frames directly into
``agent_voice_service``'s in-memory cache under the provided
``cache_key`` (no file is written to disk).
"""
import logging
import shutil
import subprocess
from functools import lru_cache

from elevenlabs.client import ElevenLabs

import config
from services import agent_voice_service as voice

logger = logging.getLogger(__name__)


class TextToSpeechError(Exception):
    """Raised when the ElevenLabs SDK call or ffmpeg conversion fails."""


@lru_cache(maxsize=1)
def _get_client() -> ElevenLabs:
    if not config.ELEVENLABS_API_KEY:
        raise TextToSpeechError("ELEVENLABS_API_KEY is not configured")
    return ElevenLabs(api_key=config.ELEVENLABS_API_KEY)


def text_to_speech_in_memory(input_text: str, cache_key: str) -> str:
    """Synthesize ``input_text`` and cache it in memory under ``cache_key``.

    After this returns, the clip is playable via the agent voice cache as
    ``[[cache_key]]`` (e.g. ``await _speak(ws, state, [[cache_key]])``).
    Returns ``cache_key`` for convenience.
    """
    if not input_text:
        raise ValueError("input_text must not be empty")
    if not cache_key:
        raise ValueError("cache_key must not be empty")

    client = _get_client()

    logger.info(
        "Requesting ElevenLabs TTS for cache_key %s (%d chars)",
        cache_key,
        len(input_text),
    )
    try:
        audio_stream = client.text_to_speech.convert(
            text=input_text,
            voice_id=config.ELEVENLABS_VOICE_ID,
            model_id=config.ELEVENLABS_MODEL_ID,
            output_format="mp3_44100_128",
        )
    except Exception as exc:
        raise TextToSpeechError(f"ElevenLabs SDK error: {exc}") from exc

    raw_audio = b"".join(chunk for chunk in audio_stream if chunk)
    if not raw_audio:
        raise TextToSpeechError("ElevenLabs returned an empty audio stream")

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise TextToSpeechError("ffmpeg executable not found on PATH")

    # Re-encode to IVR-friendly mono mp3 (16kHz, 64 kbps) via pipes.
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", "pipe:0",
        "-ac", "1",
        "-ar", "16000",
        "-b:a", "64k",
        "-f", "mp3",
        "pipe:1",
    ]
    try:
        result = subprocess.run(
            cmd,
            input=raw_audio,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
        raise TextToSpeechError(f"ffmpeg conversion failed: {stderr}") from exc

    voice.cache_clip(cache_key, result.stdout)

    logger.info(
        "Cached in-memory TTS audio for %s (%d bytes, 64kbps mono @ 16kHz)",
        cache_key,
        len(result.stdout),
    )
    return cache_key
