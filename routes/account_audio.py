"""Endpoint that synthesizes an account-name TTS clip via ElevenLabs."""

import logging
import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import AliasChoices, BaseModel, Field

import config
from services import account_service
from services import agent_voice_service as voice
from services.text_to_speech_in_memory_service import (
    TextToSpeechError as InMemoryTextToSpeechError,
    text_to_speech_in_memory,
)
from services.text_to_speech_service import TextToSpeechError, text_to_speech

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/accounts")
def list_loaded_accounts() -> dict:
    """List all account records currently loaded in memory."""
    accounts = account_service.get_loaded_accounts()
    return {
        "count": len(accounts),
        "accounts": accounts,
    }


@router.get("/audio-cache")
def list_audio_cache() -> dict:
    """List all cached audio clips as path parts from ``AUDIO_DIR`` root."""
    clip_parts = voice.list_cached_audio_parts()
    return {
        "count": len(clip_parts),
        "clips": [{"parts": parts} for parts in clip_parts],
    }


@router.get("/audio-cache/clips/{clip_path:path}")
def get_audio_cache_clip(clip_path: str) -> Response:
    """Return a cached clip as WAV (path segments relative to ``AUDIO_DIR``)."""
    parts = [p for p in clip_path.replace("\\", "/").split("/") if p]
    root_key = os.path.basename(os.path.normpath(config.AUDIO_DIR)) or "audio_files"
    if parts and parts[0] == root_key:
        parts = parts[1:]
    if not parts:
        raise HTTPException(status_code=400, detail="clip path must not be empty")

    try:
        wav = voice.clip_wav_bytes(*parts)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return Response(content=wav, media_type="audio/wav")


class GenAudioRequest(BaseModel):
    text: str = Field(..., min_length=1)
    file_name: str = Field(..., min_length=1)


def _cache_key_from_file_name(file_name: str) -> str:
    """Map a gen-audio-style file name to an in-memory cache key (no ``.mp3``)."""
    key = file_name.strip().replace("\\", "/")
    if key.lower().endswith(".mp3"):
        key = key[:-4]
    if not key:
        raise ValueError("file_name must not be empty")
    return key


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


@router.post("/gen-audio-in-memory")
def gen_audio_in_memory(payload: GenAudioRequest) -> dict:
    """Synthesize TTS and cache in memory only (no file on disk)."""
    try:
        cache_key = _cache_key_from_file_name(payload.file_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        text_to_speech_in_memory(payload.text, cache_key)
    except InMemoryTextToSpeechError as exc:
        logger.exception("In-memory TTS failed for cache_key %s", cache_key)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    root_key = os.path.basename(os.path.normpath(config.AUDIO_DIR)) or "audio_files"
    return {
        "cache_key": cache_key,
        "parts": [root_key, cache_key],
        "text": payload.text,
    }


KNOWN_GREET_SUBDIR = "known_greet_hi"
VERIFICATIONS_SUBDIR = "verifications"


class KnownGreetAudioRequest(BaseModel):
    name: str = Field(..., min_length=1)
    account: str = Field(
        ...,
        min_length=1,
        validation_alias=AliasChoices("account", "account_number", "phone"),
    )


def get_known_greet_text(name: str) -> str:
    return f"[politely] Hi {name}. <break time='0.5s' />"


def get_verifications_text(account_number: str, phone: str) -> str:
    return f"[politely] Would this reservation be for {account_number} with call back number {phone} ?"


@router.post("/known-greet-audio")
def create_known_greet_audio(payload: KnownGreetAudioRequest) -> str:
    """Generate an ElevenLabs TTS clip of ``name`` and save it as
    ``<account_number>.mp3`` under ``audio_files/known_greet_hi``.

    Accepts either an account number or phone number for account lookup.
    Overwrites the file if one already exists for the same account number.
    """
    safe_account_identifier = os.path.basename(payload.account.strip())
    if not safe_account_identifier:
        raise HTTPException(status_code=400, detail="account is invalid")
    safe_name = payload.name.strip()
    if not safe_name:
        raise HTTPException(status_code=400, detail="name is invalid")

    account = account_service.get_account_by_account_number(safe_account_identifier)
    if account is None:
        account = account_service.get_account_by_phone(safe_account_identifier)
    if account is None:
        raise HTTPException(status_code=404, detail="account not found")
    account_service.update_account(account.id, name=safe_name)

    target_dir = os.path.join(config.AUDIO_DIR, KNOWN_GREET_SUBDIR)
    os.makedirs(target_dir, exist_ok=True)

    file_name = os.path.join(KNOWN_GREET_SUBDIR, f"{account.account_number}.mp3")
    try:
        text_to_speech(get_known_greet_text(safe_name), file_name)
    except TextToSpeechError as exc:
        logger.exception("TTS failed for account %s", account.account_number)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # verifications
    verification_file_name = os.path.join(
        VERIFICATIONS_SUBDIR, f"{account.account_number}.mp3"
    )
    phone_without_country_code = account.phone.replace("+", "")
    try:
        text_to_speech(
            (
                "[politely] Thanks for the new reservation."
                f"[asking] Would this reservation be for {safe_name} with call back number {phone_without_country_code} ?"
            ),
            verification_file_name,
        )
    except TextToSpeechError as exc:
        logger.exception("TTS failed for account %s", account.account_number)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # reload accounts with try except
    try:
        account_service.load_accounts()
    except Exception as exc:
        logger.exception("Failed to reload accounts")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # reload audio cache with try except
    try:
        voice.load_audio_files()
    except Exception as exc:
        logger.exception("Failed to reload audio cache")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return "ok"
