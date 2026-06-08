"""ElevenLabs text-to-speech over the HTTP streaming endpoint.

Synthesizes raw PCM (16 kHz, 16-bit mono) audio for gapless playback in the
browser / Twilio bridge. Uses the HTTP `/stream` endpoint rather than the
realtime `stream-input` WebSocket because the expressive `eleven_v3` model is
only served over HTTP — the WebSocket endpoint rejects it (HTTP 403).
"""
import asyncio
import logging
from typing import AsyncIterator

import httpx

log = logging.getLogger("voice")

# Bound each synthesis request; v3 is slower than Flash, so allow generous time
# but still fail rather than hang the turn forever.
_TIMEOUT_SECONDS = 30.0

# Retry only transport-level failures (connection reset / timeout) and only
# before any audio has been emitted, so a retry can't duplicate playback.
_CONNECT_RETRIES = 2
_RETRY_BACKOFF_SECONDS = 0.3


def _url(voice_id: str, output_format: str) -> str:
    return (
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
        f"?output_format={output_format}"
    )


async def _synthesize(
    client: httpx.AsyncClient,
    text: str,
    api_key: str,
    voice_id: str,
    model_id: str,
    output_format: str,
) -> AsyncIterator[bytes]:
    """Stream PCM audio for one piece of text."""
    body = {
        "text": text,
        "model_id": model_id,
        # Lower stability makes eleven_v3 more responsive to audio tags
        # (e.g. [asking], [politely]); at higher values the tags get flattened.
        # 0.3 stays expressive without the hallucination risk of 0.0.
        "voice_settings": {"stability": 0.3, "similarity_boost": 0.8},
    }
    headers = {"xi-api-key": api_key, "content-type": "application/json"}
    url = _url(voice_id, output_format)

    last_exc: Exception | None = None
    for attempt in range(_CONNECT_RETRIES + 1):
        produced = False
        try:
            async with client.stream("POST", url, json=body, headers=headers) as resp:
                if resp.status_code != 200:
                    detail = (await resp.aread()).decode("utf-8", "replace")
                    # Deterministic API error (bad model/auth/quota) — don't retry.
                    raise RuntimeError(
                        f"ElevenLabs HTTP {resp.status_code}: {detail[:300]}"
                    )
                async for chunk in resp.aiter_bytes():
                    if chunk:
                        produced = True
                        yield chunk
                return
        except httpx.HTTPError as exc:
            last_exc = exc
            if produced:
                # Already emitted audio for this piece; a retry would replay it.
                raise
            log.warning(
                "ElevenLabs TTS attempt %d/%d failed: %s",
                attempt + 1, _CONNECT_RETRIES + 1, exc,
            )
            await asyncio.sleep(_RETRY_BACKOFF_SECONDS)
    raise last_exc  # type: ignore[misc]


async def stream_tts_input(
    text_chunks: AsyncIterator[str],
    api_key: str,
    voice_id: str,
    model_id: str = "eleven_v3",
    output_format: str = "pcm_16000",
) -> AsyncIterator[bytes]:
    """Synthesize the agent's reply to audio.

    `text_chunks` is an async iterator of reply fragments (e.g. sentences from a
    streaming LLM). They are buffered into the full reply and synthesized in a
    SINGLE request, because eleven_v3 needs the surrounding context to apply
    audio tags (e.g. [asking], [politely]) — on tiny per-sentence fragments the
    tags get flattened or ignored. ElevenLabs recommends giving v3 longer text
    rather than synthesizing one short fragment at a time.
    """
    pieces: list[str] = []
    async for piece in text_chunks:
        piece = (piece or "").strip()
        if not piece:
            continue
        log.info("*** llm text piece: %r", piece)
        pieces.append(piece)

    text = " ".join(pieces).strip()
    if not text:
        return

    # Safety net: eleven_v3 conveys emotion from inline tags like [politely]. The
    # agent prompt requires them, but the LLM occasionally omits them entirely,
    # which makes delivery flat. If a v3 reply has no tag at all, prepend a
    # sensible default ([asking] for a question, [politely] otherwise) so the
    # model always has expression to act on.
    if model_id.startswith("eleven_v3") and "[" not in text:
        default_tag = "[asking]" if text.rstrip().endswith("?") else "[politely]"
        text = f"{default_tag} {text}"
        log.info("No expression tag in reply; prepended %s", default_tag)

    log.info("TTS synthesizing (%s, %d chars): %r", model_id, len(text), text)
    async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
        async for audio in _synthesize(
            client, text, api_key, voice_id, model_id, output_format
        ):
            yield audio
