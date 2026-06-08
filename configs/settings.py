"""Application settings loaded from environment / .env file."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    assemblyai_api_key: str = os.getenv("ASSEMBLYAI_API_KEY", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    elevenlabs_api_key: str = os.getenv("ELEVENLABS_API_KEY", "")
    elevenlabs_voice_id: str = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    # Which agent the pipeline runs; see agents/registry.py.
    agent: str = os.getenv("AGENT", "reservation")
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
