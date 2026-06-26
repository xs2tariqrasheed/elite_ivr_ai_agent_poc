"""Application settings loaded from environment / .env file."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    assemblyai_api_key: str = os.getenv("ASSEMBLYAI_API_KEY", "")
    deepgram_api_key: str = os.getenv("DEEPGRAM_API_KEY", "")
    # Which speech-to-text backend the pipeline uses: "assemblyai" or "deepgram".
    # Both expose the same stream interface (see services/stt.py:build_stt), so
    # swapping providers needs only this env var.
    stt_provider: str = os.getenv("STT_PROVIDER", "assemblyai")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    elevenlabs_api_key: str = os.getenv("ELEVENLABS_API_KEY", "")
    elevenlabs_voice_id: str = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    # Which agent the pipeline runs; see agents/registry.py.
    agent: str = os.getenv("AGENT", "quick_reservation")
    # Twilio. account_sid/auth_token are used by the REST helpers / request
    # validation; stream_ws_url is the public wss:// address Twilio dials into
    # for Media Streams (returned in the TwiML at /twilio/voice).
    twilio_account_sid: str = os.getenv("TWILIO_ACCOUNT_SID", "")
    twilio_auth_token: str = os.getenv("TWILIO_AUTH_TOKEN", "")
    twilio_stream_ws_url: str = os.getenv("TWILIO_STREAM_WS_URL", "")
    # Passcode that gates the /admin CRUD frontend, plus the secret used to sign
    # the login session cookie.
    passcode: str = os.getenv("PASSCODE", "")
    session_secret: str = os.getenv("SESSION_SECRET", "change-me-dev-secret")

    # ----- Barge-in -----------------------------------------------------------
    # When enabled, the caller can interrupt the agent mid-reply (or while it is
    # still composing): the instant sustained caller speech is detected, the
    # in-flight turn (LLM stream + TTS request + queued audio + the abandoned
    # reply in agent memory) is pruned and the caller's new utterance is
    # processed from scratch. When disabled, the pipeline stays strictly
    # half-duplex (the agent cannot be interrupted) — byte-for-byte the prior
    # behavior. Detection is energy-based (mean-abs PCM16 amplitude) so it never
    # has to un-mute STT into the agent's own echo; see services/audio_bridge.py.
    barge_in_enabled: bool = os.getenv("BARGE_IN_ENABLED", "true").lower() == "true"
    # Energy threshold (mean-abs PCM16) for a frame to count as caller speech
    # while the agent's audio is ON the wire. Set above steady-state line echo so
    # the agent's own voice can't trip a barge, but low enough to catch a normal
    # speaking caller (band-limited μ-law telephony speech reads lower than wide-
    # band mic audio). The "barge-watch" log line prints the live measured level
    # vs this threshold during a call — tune against it. Raise on echoey
    # speakerphone lines; lower if real barge-ins are missed.
    barge_in_voice_level: int = int(os.getenv("BARGE_IN_VOICE_LEVEL", "300"))
    # Energy threshold while the agent is composing (no audio on the wire, so no
    # echo). Lower than the playback threshold since there is nothing to reject,
    # but kept above the diagnostic line-presence floor (_VOICE_LEVEL=400 in
    # audio_bridge) so steady comfort noise / line hiss can't barge.
    barge_in_voice_level_idle: int = int(os.getenv("BARGE_IN_VOICE_LEVEL_IDLE", "250"))
    # Debounce: a barge requires this many consecutive voice frames AND at least
    # `barge_in_min_ms` of wall-clock (whichever is stricter), so a single echo
    # spike, click, or cough can never trip it. Twilio coalesces to ~100 ms
    # frames, so 2 frames / 120 ms cuts the agent off ~within a fifth of a second.
    barge_in_min_frames: int = int(os.getenv("BARGE_IN_MIN_FRAMES", "2"))
    barge_in_min_ms: int = int(os.getenv("BARGE_IN_MIN_MS", "120"))
    # Ignore inbound energy for this long after playback starts; echo is loudest
    # at a syllable's onset, so the guard avoids a self-trip on the first frames.
    barge_in_echo_guard_seconds: float = float(
        os.getenv("BARGE_IN_ECHO_GUARD_SECONDS", "0.25")
    )
    # Extra seconds STT stays muted after playback is projected to end, covering
    # the transport jitter buffer and echo tail. Replaces the former hardcoded
    # constant in audio_bridge so the mute tail is tunable per deployment.
    barge_in_echo_tail_seconds: float = float(
        os.getenv("BARGE_IN_ECHO_TAIL_SECONDS", "0.7")
    )

    # Play a short pre-recorded "one moment…" clip while the agent processes a
    # caller's turn, to mask LLM/TTS latency. Disabled for now (set
    # GAP_FILLER_ENABLED=true to re-enable).
    gap_filler_enabled: bool = os.getenv("GAP_FILLER_ENABLED", "false").lower() == "true"


@dataclass(frozen=True)
class AudioFormat:
    """Wire formats for one connection's STT input and TTS output.

    Browser connections speak PCM16@16k. Twilio Media Streams speak μ-law@8k:
    TTS is emitted as μ-law@8k so it plays back natively, but inbound caller
    audio is transcoded to PCM16@16k (see services.audio) before STT, because
    AssemblyAI's universal-streaming model only transcribes reliably at 16 kHz.
    """

    stt_encoding: str = "pcm_s16le"
    stt_sample_rate: int = 16000
    tts_output_format: str = "pcm_16000"


BROWSER_AUDIO = AudioFormat()
TWILIO_AUDIO = AudioFormat(
    stt_encoding="pcm_s16le",
    stt_sample_rate=16000,
    tts_output_format="ulaw_8000",
)


settings = Settings()
