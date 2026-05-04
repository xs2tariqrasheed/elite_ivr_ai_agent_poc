"""Application configuration loaded from environment variables."""
import os
from dotenv import load_dotenv

load_dotenv()


def _get(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


# Twilio
TWILIO_ACCOUNT_SID = _get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = _get("TWILIO_AUTH_TOKEN")
TWILIO_STREAM_WS_URL = _get("TWILIO_STREAM_WS_URL")

# Public URL
PUBLIC_BASE_URL = _get("PUBLIC_BASE_URL")

# Ollama
OLLAMA_HOST = _get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = _get("OLLAMA_MODEL", "qwen2.5:1.5b")

# fastText intent
INTENT_MODEL_PATH = _get("INTENT_MODEL_PATH", "intent_model.bin")
INTENT_CONFIDENCE_THRESHOLD = float(_get("INTENT_CONFIDENCE_THRESHOLD", "0.6"))

# faster-whisper
TINY_EN_MODEL_PATH = _get("TINY_EN_MODEL_PATH", "tiny.en")
WHISPER_DEVICE = _get("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = _get("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_LANGUAGE = _get("WHISPER_LANGUAGE", "en")

# Audio
AUDIO_DIR = _get("AUDIO_DIR", "audio_files")

# Data
DUMMY_DATA_PATH = _get("DUMMY_DATA_PATH", "dummy_data.json")

# Server
HOST = _get("HOST", "0.0.0.0")
PORT = int(_get("PORT", "8000"))
LOG_LEVEL = _get("LOG_LEVEL", "INFO")
