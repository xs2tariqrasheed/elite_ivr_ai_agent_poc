"""LLM service backed by a locally-running Ollama model (qwen2.5:1.5b).

Helpers exposed: ``extract_account_number``, ``extract_phone_number``,
and ``classify_intent``. Each asks the LLM for a deterministic answer
and falls back gracefully whenever the model declines or returns junk.
"""

from .account_number import extract_account_number
from .intent import classify_intent
from .llm_client import get_ollama_client
from .phone_number import extract_phone_number
from .warm_up import warm_up_model

__all__ = [
    "extract_account_number",
    "extract_phone_number",
    "classify_intent",
    "warm_up_model",
    "get_ollama_client",
]
