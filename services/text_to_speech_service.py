"""Text-to-speech service using ElevenLabs.

Synthesizes mp3 audio from text via the ``elevenlabs`` Python SDK,
re-encodes the result to 320 kbps mono mp3 with ffmpeg, and writes
it into ``config.AUDIO_DIR``.
"""
import logging
import os
import shutil
import subprocess
from functools import lru_cache

from elevenlabs.client import ElevenLabs

import config

logger = logging.getLogger(__name__)


class TextToSpeechError(Exception):
    """Raised when the ElevenLabs SDK call or ffmpeg conversion fails."""


@lru_cache(maxsize=1)
def _get_client() -> ElevenLabs:
    if not config.ELEVENLABS_API_KEY:
        raise TextToSpeechError("ELEVENLABS_API_KEY is not configured")
    return ElevenLabs(api_key=config.ELEVENLABS_API_KEY)


def text_to_speech(input_text: str, file_name: str) -> str:
    """Synthesize ``input_text`` to audio and save it as ``file_name``.

    The file is written under ``config.AUDIO_DIR``. Returns the absolute
    path to the saved audio file.
    """
    if not input_text:
        raise ValueError("input_text must not be empty")
    if not file_name:
        raise ValueError("file_name must not be empty")

    client = _get_client()

    logger.info(
        "Requesting ElevenLabs TTS for file %s (%d chars)", file_name, len(input_text)
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

    os.makedirs(config.AUDIO_DIR, exist_ok=True)
    out_path = os.path.join(config.AUDIO_DIR, file_name)

    raw_audio = b"".join(chunk for chunk in audio_stream if chunk)
    if not raw_audio:
        raise TextToSpeechError("ElevenLabs returned an empty audio stream")

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise TextToSpeechError("ffmpeg executable not found on PATH")

    # Re-encode to 320 kbps mono mp3 via ffmpeg stdin/stdout pipes.
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", "pipe:0",
        "-ac", "1",
        "-b:a", "320k",
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

    with open(out_path, "wb") as f:
        f.write(result.stdout)

    logger.info(
        "Saved TTS audio to %s (%d bytes, 320kbps mono)", out_path, len(result.stdout)
    )
    return out_path
