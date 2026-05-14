"""Endpoint that synthesizes an account-name TTS clip via ElevenLabs."""

import logging
import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import config
from services import agent_voice_service as voice
from services.text_to_speech_service import TextToSpeechError, text_to_speech

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/audio-cache")
def list_audio_cache() -> dict:
    """List all cached audio clips as path parts from ``AUDIO_DIR`` root."""
    clip_parts = voice.list_cached_audio_parts()
    return {
        "count": len(clip_parts),
        "clips": [{"parts": parts} for parts in clip_parts],
    }


class GenAudioRequest(BaseModel):
    text: str = Field(..., min_length=1)
    file_name: str = Field(..., min_length=1)


@router.post("/gen-audio")
def gen_audio(payload: GenAudioRequest) -> dict:
    try:
        text_to_speech(payload.text, payload.file_name)
    except TextToSpeechError as exc:
        logger.exception("TTS failed for file %s", payload.file_name)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "file_name": payload.file_name,
        "text": payload.text,
    }


KNOWN_GREET_SUBDIR = "known_greet_hi"


class KnownGreetAudioRequest(BaseModel):
    name: str = Field(..., min_length=1)
    account: str = Field(..., min_length=1)


def get_known_greet_text(name: str) -> str:
    return f"[politely] Hi {name}. <break time='0.5s' />"


@router.post("/known-greet-audio")
def create_known_greet_audio(payload: KnownGreetAudioRequest) -> str:
    """Generate an ElevenLabs TTS clip of ``name`` and save it as
    ``<account>.mp3`` under ``audio_files/known_greet_hi``.

    Overwrites the file if one already exists for the same account number.
    """
    safe_account_number = os.path.basename(payload.account.strip())
    if not safe_account_number:
        raise HTTPException(status_code=400, detail="account is invalid")

    target_dir = os.path.join(config.AUDIO_DIR, KNOWN_GREET_SUBDIR)
    os.makedirs(target_dir, exist_ok=True)

    file_name = os.path.join(KNOWN_GREET_SUBDIR, f"{safe_account_number}.mp3")
    try:
        text_to_speech(get_known_greet_text(payload.name), file_name)
    except TextToSpeechError as exc:
        logger.exception("TTS failed for account %s", safe_account_number)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return "ok"
