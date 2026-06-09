"""AssemblyAI Universal-Streaming (v3) WebSocket wrapper."""
import json
import logging

import websockets

log = logging.getLogger("stt")

# Silence (ms) AssemblyAI must hear before declaring end-of-turn when it's
# confident the turn is over. The default (~160 ms) ends a turn on the brief
# pause inside a single answer — e.g. "Terminal ... 4" splits into two turns,
# which desyncs the conversation. A larger value coalesces those pauses into one
# turn at the cost of a little extra response latency. Tune against real calls.
_MIN_EOT_SILENCE_MS = 700

# How "complete" AssemblyAI must judge a turn before it will finalize it (0–1).
# At the default (~0.7) a short, low-confidence reply like a bare "yes please"
# never clears the bar, so the model emits no transcript at all and the caller
# has to repeat themselves. Lowering it makes the model commit short utterances.
# Too low and it may finalize mid-sentence; tune against real calls.
_EOT_CONFIDENCE_THRESHOLD = 0.4


# format_turns gives punctuated final turns. speech_model is REQUIRED by the v3
# API (no default). encoding is pcm_s16le (browser) or pcm_mulaw (Twilio).
def _aai_url(
    encoding: str,
    sample_rate: int,
    min_eot_silence_ms: int,
    eot_confidence_threshold: float,
) -> str:
    return (
        "wss://streaming.assemblyai.com/v3/ws"
        "?speech_model=universal-streaming-english"
        f"&sample_rate={sample_rate}&encoding={encoding}&format_turns=true"
        f"&min_end_of_turn_silence_when_confident={min_eot_silence_ms}"
        f"&end_of_turn_confidence_threshold={eot_confidence_threshold}"
    )


class AssemblyAIStream:
    def __init__(
        self,
        api_key: str,
        encoding: str = "pcm_s16le",
        sample_rate: int = 16000,
        min_eot_silence_ms: int = _MIN_EOT_SILENCE_MS,
        eot_confidence_threshold: float = _EOT_CONFIDENCE_THRESHOLD,
    ):
        self.api_key = api_key
        self.encoding = encoding
        self.sample_rate = sample_rate
        self.min_eot_silence_ms = min_eot_silence_ms
        self.eot_confidence_threshold = eot_confidence_threshold
        self.ws = None

    async def connect(self):
        self.ws = await websockets.connect(
            _aai_url(
                self.encoding,
                self.sample_rate,
                self.min_eot_silence_ms,
                self.eot_confidence_threshold,
            ),
            extra_headers={"Authorization": self.api_key},
            max_size=None,
            ping_interval=5,
            ping_timeout=20,
        )
        return self

    async def send_audio(self, pcm: bytes):
        if self.ws is not None:
            await self.ws.send(pcm)

    def __aiter__(self):
        return self._events()

    async def _events(self):
        try:
            async for raw in self.ws:
                try:
                    yield json.loads(raw)
                except (ValueError, TypeError):
                    continue
        except websockets.exceptions.ConnectionClosed as exc:
            log.error(
                "AssemblyAI closed the stream: code=%s reason=%r",
                exc.code, exc.reason,
            )
            raise RuntimeError(
                f"AssemblyAI closed the stream (code {exc.code}): "
                f"{exc.reason or 'no reason given'}"
            ) from exc

    async def close(self):
        if self.ws is not None:
            try:
                await self.ws.send(json.dumps({"type": "Terminate"}))
            except Exception:
                pass
            try:
                await self.ws.close()
            except Exception:
                pass
            self.ws = None


def build_stt(settings, *, encoding: str, sample_rate: int):
    """Construct the STT stream for the configured provider.

    Both backends expose the same interface (connect / send_audio / async
    iteration of event dicts / close), so callers stay provider-agnostic. Select
    via Settings.stt_provider ("assemblyai" or "deepgram").
    """
    provider = (settings.stt_provider or "assemblyai").lower()
    if provider == "deepgram":
        # Imported lazily so the AssemblyAI path doesn't depend on it.
        from services.stt_deepgram import DeepgramStream

        return DeepgramStream(
            settings.deepgram_api_key, encoding=encoding, sample_rate=sample_rate
        )
    return AssemblyAIStream(
        settings.assemblyai_api_key, encoding=encoding, sample_rate=sample_rate
    )
