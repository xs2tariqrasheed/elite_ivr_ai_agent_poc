"""WebSocket voice pipeline orchestrator.

Wires AudioBridge and TurnHandler together for one connection.
"""
import asyncio
import logging
import traceback

from fastapi import WebSocket, WebSocketDisconnect

from agents.registry import build_agent
from configs.settings import BROWSER_AUDIO, AudioFormat, Settings
from services.audio_bridge import AudioBridge
from services.pipeline_state import PipelineState
from services.stt import build_stt
from services.turn_handler import TurnHandler

log = logging.getLogger("voice")


class VoiceSession:
    """Wires all pipeline components together for one browser connection."""

    def __init__(
        self,
        client: WebSocket,
        settings: Settings,
        agent_name: str | None = None,
        params: dict | None = None,
        audio_format: AudioFormat | None = None,
    ) -> None:
        self._client = client
        self._settings = settings
        fmt = audio_format or BROWSER_AUDIO
        self._stt = build_stt(
            settings,
            encoding=fmt.stt_encoding,
            sample_rate=fmt.stt_sample_rate,
        )

        self._agent = build_agent(agent_name or settings.agent, settings, params)
        state = PipelineState()

        self._turn_handler = TurnHandler(
            client, self._agent, settings, state,
            tts_output_format=fmt.tts_output_format,
        )
        self._bridge = AudioBridge(
            client, self._stt, state, self._turn_handler.handle_turn
        )
        self._state = state

    async def run(self) -> None:
        """Validate keys, connect STT, then run the pipeline until disconnect."""
        provider = (self._settings.stt_provider or "assemblyai").lower()
        stt_key = (
            ("DEEPGRAM_API_KEY", self._settings.deepgram_api_key)
            if provider == "deepgram"
            else ("ASSEMBLYAI_API_KEY", self._settings.assemblyai_api_key)
        )
        missing = [
            name
            for name, val in (
                stt_key,
                ("OPENAI_API_KEY", self._settings.openai_api_key),
                ("ELEVENLABS_API_KEY", self._settings.elevenlabs_api_key),
            )
            if not val
        ]
        if missing:
            await self._client.send_json({
                "type": "error",
                "text": f"Server missing API keys: {', '.join(missing)}",
            })
            await self._client.close()
            return

        try:
            await self._stt.connect()
            log.info("%s stream connected", provider)
        except Exception as exc:  # noqa: BLE001
            log.error("STT connect failed: %s", exc)
            traceback.print_exc()
            await self._client.send_json(
                {"type": "error", "text": f"STT connect failed: {exc}"}
            )
            await self._client.close()
            return

        try:
            await self._client.send_json({"type": "ready"})
            # Let the agent speak first (e.g. a greeting) by queuing one synthetic
            # turn ahead of any caller turns, so it runs through the same
            # one-at-a-time worker as everything else.
            opening = getattr(self._agent, "opening_trigger", None)
            if opening:
                self._bridge.enqueue_turn(opening)
            await asyncio.gather(
                self._bridge.browser_to_stt(),
                self._bridge.stt_to_agent(),
                self._bridge.turn_worker(),
            )
        except WebSocketDisconnect:
            log.info("Browser disconnected")
        except Exception as exc:  # noqa: BLE001
            log.error("Session error: %s", exc)
            traceback.print_exc()
            try:
                await self._client.send_json({"type": "error", "text": str(exc)})
            except Exception:
                pass
        finally:
            # Mark closed first so the in-flight turn treats teardown errors as
            # expected, then cancel it so it doesn't run on as an orphan task.
            self._state.closed = True
            t = self._state.turn_task
            if t is not None and not t.done():
                t.cancel()
                try:
                    await t
                except BaseException:
                    pass
            await self._stt.close()
            try:
                await self._client.close()
            except Exception:
                pass
