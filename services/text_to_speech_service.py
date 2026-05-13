"""Text-to-speech service using ElevenLabs.

Synthesizes mp3 audio from text via the ElevenLabs HTTP API and writes
the result into ``config.AUDIO_DIR``.
"""
import logging
import os

import requests

import config

logger = logging.getLogger(__name__)


_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"


class TextToSpeechError(Exception):
    """Raised when the ElevenLabs API call fails."""


def text_to_speech(input_text: str, file_name: str) -> str:
    """Synthesize ``input_text`` to audio and save it as ``file_name``.

    The file is written under ``config.AUDIO_DIR``. Returns the absolute
    path to the saved audio file.
    """
    if not input_text:
        raise ValueError("input_text must not be empty")
    if not file_name:
        raise ValueError("file_name must not be empty")
    if not config.ELEVENLABS_API_KEY:
        raise TextToSpeechError("ELEVENLABS_API_KEY is not configured")

    url = _TTS_URL.format(voice_id=config.ELEVENLABS_VOICE_ID)
    headers = {
        "xi-api-key": config.ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    payload = {
        "text": input_text,
        "model_id": config.ELEVENLABS_MODEL_ID,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
        },
    }

    logger.info("Requesting ElevenLabs TTS for file %s (%d chars)", file_name, len(input_text))
    response = requests.post(url, json=payload, headers=headers, timeout=60)
    if response.status_code != 200:
        raise TextToSpeechError(
            f"ElevenLabs API error {response.status_code}: {response.text}"
        )

    os.makedirs(config.AUDIO_DIR, exist_ok=True)
    out_path = os.path.join(config.AUDIO_DIR, file_name)
    with open(out_path, "wb") as f:
        f.write(response.content)

    logger.info("Saved TTS audio to %s (%d bytes)", out_path, len(response.content))
    return out_path
