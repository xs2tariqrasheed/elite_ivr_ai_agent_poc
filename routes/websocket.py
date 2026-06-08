"""WebSocket route — one VoiceSession per browser connection."""
from fastapi import APIRouter, WebSocket

from configs.settings import settings
from db import get_account
from services.voice_session import VoiceSession

router = APIRouter()


@router.websocket("/ws")
async def ws_endpoint(client: WebSocket):
    await client.accept()

    agent_name = client.query_params.get("agent") or settings.agent
    params: dict = {}
    raw_account = client.query_params.get("account")
    if raw_account is not None:
        try:
            account = get_account(int(raw_account))
        except ValueError:
            account = None
        if account is not None:
            params["account"] = account

    await VoiceSession(client, settings, agent_name=agent_name, params=params).run()
