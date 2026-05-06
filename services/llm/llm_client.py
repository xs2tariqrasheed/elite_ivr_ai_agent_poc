"""Ollama client initialization."""
import logging
from typing import Optional

import ollama

import config

logger = logging.getLogger(__name__)


_client: Optional[ollama.Client] = None


def get_ollama_client() -> ollama.Client:
    """Return the lazily-initialised Ollama client."""
    global _client
    if _client is None:
        logger.info("Creating Ollama client at %s", config.OLLAMA_HOST)
        _client = ollama.Client(host=config.OLLAMA_HOST)
    return _client
