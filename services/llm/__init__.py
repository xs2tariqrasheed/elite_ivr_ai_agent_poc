"""LLM service backed by a locally-running Ollama model (qwen2.5:1.5b).

Helpers exposed: ``extract_account_number``, ``extract_phone_number``,
``extract_pickup_date_time``, ``classify_intent``, and
``detect_yes_no_llm``. Each asks the LLM for a deterministic answer and
falls back gracefully whenever the model declines or returns junk.
"""

from .account_number import extract_account_number
from .intent import classify_intent
from .llm_client import get_ollama_client
from .phone_number import extract_phone_number
from .pickup_date_time import extract_pickup_date_time, extract_pickup_date_time_openai
from .warm_up import warm_up_model
from .yes_no import detect_yes_no_llm

__all__ = [
    "extract_account_number",
    "extract_phone_number",
    "extract_pickup_date_time",
    "extract_pickup_date_time_openai",
    "classify_intent",
    "detect_yes_no_llm",
    "warm_up_model",
    "get_ollama_client",
]
