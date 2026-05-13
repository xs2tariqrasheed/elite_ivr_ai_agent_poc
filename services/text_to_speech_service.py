"""Text-to-speech service using ElevenLabs.

Synthesizes mp3 audio from text via the ``elevenlabs`` Python SDK and
writes the result into ``config.AUDIO_DIR``.
"""
import logging
import os
from functools import lru_cache

from elevenlabs.client import ElevenLabs

import config

logger = logging.getLogger(__name__)


class TextToSpeechError(Exception):
    """Raised when the ElevenLabs SDK call fails."""


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

    total_bytes = 0
    with open(out_path, "wb") as f:
        for chunk in audio_stream:
            if chunk:
                f.write(chunk)
                total_bytes += len(chunk)

    logger.info("Saved TTS audio to %s (%d bytes)", out_path, total_bytes)
    return out_path
