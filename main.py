"""FastAPI entry point for the Elite Limousine IVR agent.

Routes are defined under the ``routes`` package and registered here.
"""

import logging

from fastapi import FastAPI

import config
from logging_config import setup_logging
from routes import account_audio_router, health_router, twilio_router
from services import account_service
from services import agent_voice_service as voice
from services import intent_service as intent
from services import llm
from services import stt_service as stt


setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="Elite Limousine IVR")

app.include_router(health_router)
app.include_router(twilio_router)
app.include_router(account_audio_router)


@app.on_event("startup")
def on_startup() -> None:
    logger.info("Booting Elite IVR …")
    voice.load_audio_files()
    stt.load_model()
    intent.load_model()
    account_service.load_accounts()
    llm.warm_up_model()
    logger.info("Boot complete")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        log_level=config.LOG_LEVEL.lower(),
    )
