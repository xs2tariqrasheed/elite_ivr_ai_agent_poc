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

# Database (Postgres via SQLAlchemy)
#
# DATABASE_URL is what the running app uses for queries. On Supabase this is
# normally the pooled (pgbouncer) URL on port 6543.
#
# DIRECT_URL is used by Alembic for migrations because DDL inside a transaction
# does not behave well through pgbouncer in transaction-pool mode. On Supabase
# this is the direct connection on port 5432. If unset, DATABASE_URL is reused.
DATABASE_URL = _get(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@localhost:5432/elite_ivr",
)
DIRECT_URL = _get("DIRECT_URL", DATABASE_URL)
DB_ECHO = _get("DB_ECHO", "false").lower() in ("1", "true", "yes")

# Server
HOST = _get("HOST", "0.0.0.0")
PORT = int(_get("PORT", "8000"))
LOG_LEVEL = _get("LOG_LEVEL", "INFO")
