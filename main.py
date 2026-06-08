"""Voice ordering agent — FastAPI entry point.

Pipeline per browser connection:
  browser mic (PCM16@16k)  ->  AssemblyAI streaming STT
  final user turn          ->  LangGraph ReAct agent (GPT-4o-mini)
  agent reply text         ->  ElevenLabs streaming TTS  ->  browser playback
"""
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from websockets.exceptions import ConnectionClosedOK

from configs.settings import settings
from db import models  # noqa: F401 - ensure models register on Base before create_all
from db.database import Base, engine
from routes.accounts import router as accounts_router
from routes.admin import router as admin_router
from routes.health import router as health_router
from routes.twilio import router as twilio_router
from routes.websocket import router as ws_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

log = logging.getLogger("voice")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Quiet the benign websocket-close noise at caller hang-up.

    When the caller hangs up, the `websockets` library's internal keepalive-ping
    task can surface a *clean* close (ConnectionClosedOK) that nothing awaits, so
    asyncio dumps it via the default handler at ERROR level even though the call
    ended normally. Downgrade just that case to debug; everything else falls
    through to the default handler untouched.
    """
    loop = asyncio.get_running_loop()
    previous = loop.get_exception_handler()

    def handler(loop_, context):
        if isinstance(context.get("exception"), ConnectionClosedOK):
            log.debug("Ignored benign websocket close: %s", context.get("message"))
            return
        (previous or loop_.default_exception_handler)(context)

    loop.set_exception_handler(handler)
    Base.metadata.create_all(bind=engine)
    yield
    loop.set_exception_handler(previous)


app = FastAPI(title="Pizza Voice Agent", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# Signs the login-session cookie used by the /admin frontend.
app.add_middleware(SessionMiddleware, secret_key=settings.session_secret)

app.include_router(health_router)
app.include_router(accounts_router)
app.include_router(admin_router)
app.include_router(ws_router)
app.include_router(twilio_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
