"""Endpoint that synthesizes an account-name TTS clip via ElevenLabs."""
import logging
import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import config
from services.text_to_speech_service import TextToSpeechError, text_to_speech

logger = logging.getLogger(__name__)

router = APIRouter()

ACCOUNT_NAMES_SUBDIR = "account_names"


class AccountNameAudioRequest(BaseModel):
    account_name: str = Field(..., min_length=1)
    account_number: str = Field(..., min_length=1)


@router.post("/account-name-audio")
def create_account_name_audio(payload: AccountNameAudioRequest) -> dict:
    """Generate an ElevenLabs TTS clip of ``account_name`` and save it as
    ``<account_number>.mp3`` under ``audio_files/account_names``.

    Overwrites the file if one already exists for the same account number.
    """
    safe_account_number = os.path.basename(payload.account_number.strip())
    if not safe_account_number:
        raise HTTPException(status_code=400, detail="account_number is invalid")

    target_dir = os.path.join(config.AUDIO_DIR, ACCOUNT_NAMES_SUBDIR)
    os.makedirs(target_dir, exist_ok=True)

    file_name = os.path.join(ACCOUNT_NAMES_SUBDIR, f"{safe_account_number}.mp3")
    try:
        out_path = text_to_speech(payload.account_name, file_name)
    except TextToSpeechError as exc:
        logger.exception("TTS failed for account %s", safe_account_number)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "account_number": safe_account_number,
        "account_name": payload.account_name,
        "audio_path": out_path,
    }
