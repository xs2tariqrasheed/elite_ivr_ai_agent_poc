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
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK

from configs.settings import settings
from db import models  # noqa: F401 - ensure models register on Base before create_all
from db.database import Base, engine
from services.gap_filler import load as load_gap_fillers
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
    ended normally. The same happens as ConnectionClosedError when *we* send the
    normal close (code 1000) at teardown and the peer (Deepgram) drops the TCP
    connection without replying with its own close frame — websockets labels the
    missing reply an "error" but the session is already over. Downgrade just
    those cases to debug; everything else falls through to the default handler
    untouched.
    """
    loop = asyncio.get_running_loop()
    previous = loop.get_exception_handler()

    def _is_benign_close(exc) -> bool:
        if isinstance(exc, ConnectionClosedOK):
            return True
        return (
            isinstance(exc, ConnectionClosed)
            and exc.rcvd is None
            and exc.sent is not None
            and exc.sent.code == 1000
        )

    def handler(loop_, context):
        if _is_benign_close(context.get("exception")):
            log.debug("Ignored benign websocket close: %s", context.get("message"))
            return
        (previous or loop_.default_exception_handler)(context)

    loop.set_exception_handler(handler)
    Base.metadata.create_all(bind=engine)
    # Pre-decode the gap-filler clips into the wire formats once, so they're ready
    # to play the instant a caller's turn ends (see services.gap_filler).
    await asyncio.to_thread(load_gap_fillers)
    yield
    loop.set_exception_handler(previous)


app = FastAPI(title="Quick Reservation Agent 1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ivrdemo.eliteny.com"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
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
