"""Model warm-up for Ollama."""
import logging

import config

from .llm_client import get_ollama_client

logger = logging.getLogger(__name__)


def warm_up_model() -> None:
    """Issue a tiny request so the model is loaded into Ollama's RAM.

    Called once at app startup so the first real request isn't slow.
    """
    try:
        client = get_ollama_client()
        logger.info("Warming up Ollama model %s", config.OLLAMA_MODEL)
        client.generate(
            model=config.OLLAMA_MODEL,
            prompt="ok",
            options={"num_predict": 1},
        )
        logger.info("Ollama model %s warmed up", config.OLLAMA_MODEL)
    except Exception:
        logger.exception("Ollama warm-up failed (continuing anyway)")
