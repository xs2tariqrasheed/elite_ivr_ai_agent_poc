"""Speech-to-text service backed by faster-whisper (tiny.en).

The model is loaded once at app startup.  Callers pass a numpy float32
array of mono 16 kHz PCM audio (as required by the implementation
guide) along with optional ``initial_prompt`` and ``hotwords`` strings
to bias decoding for the current call phase.
"""
import logging
import time
from typing import List, Optional

import numpy as np
from faster_whisper import WhisperModel

import config

logger = logging.getLogger(__name__)


_model: Optional[WhisperModel] = None


def load_model() -> None:
    """Load the tiny.en model into memory.  Call once at startup."""
    global _model
    if _model is not None:
        return

    logger.info(
        "Loading faster-whisper model path=%s device=%s compute_type=%s",
        config.TINY_EN_MODEL_PATH,
        config.WHISPER_DEVICE,
        config.WHISPER_COMPUTE_TYPE,
    )
    _model = WhisperModel(
        config.TINY_EN_MODEL_PATH,
        device=config.WHISPER_DEVICE,
        compute_type=config.WHISPER_COMPUTE_TYPE,
    )
    logger.info("faster-whisper model loaded")


def transcribe_audio(
    audio: np.ndarray,
    initial_prompt: Optional[str] = None,
    hotwords: Optional[str] = None,
) -> str:
    """Transcribe a mono 16 kHz float32 PCM signal and return joined text.

    ``audio`` must be a numpy float32 array in [-1.0, 1.0] sampled at
    16 kHz, mono.
    """
    if _model is None:
        raise RuntimeError("STT model not loaded. Call load_model() first.")

    if audio is None or audio.size == 0:
        logger.debug("transcribe_audio: empty audio buffer")
        return ""

    logger.debug(
        "Transcribing %.2fs of audio (initial_prompt=%r, hotwords=%r)",
        len(audio) / 16000.0,
        initial_prompt,
        hotwords,
    )

    start = time.time()
    try:
        segments, _info = _model.transcribe(
            audio,
            beam_size=1,
            best_of=1,
            temperature=0,
            condition_on_previous_text=False,
            language=config.WHISPER_LANGUAGE or None,
            initial_prompt=initial_prompt,
            hotwords=hotwords,
            vad_filter=True,
        )
        parts: List[str] = [seg.text for seg in segments]
    except Exception:
        logger.exception("Whisper transcription failed")
        return ""

    text = " ".join(p.strip() for p in parts if p and p.strip())
    elapsed = time.time() - start
    logger.info("STT done in %.2fs: %r", elapsed, text)
    return text
