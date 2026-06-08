"""Standalone connectivity check for the three external services.

Run:  python diagnose.py
It connects to each provider and prints the exact success/failure, including
any WebSocket close code + reason, so we can see why a stream is rejected.
"""
import asyncio
import os

import httpx
import websockets
from dotenv import load_dotenv

load_dotenv()

AAI = os.getenv("ASSEMBLYAI_API_KEY", "")
OPENAI = os.getenv("OPENAI_API_KEY", "")
ELEVEN = os.getenv("ELEVENLABS_API_KEY", "")
VOICE = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")


def mask(k: str) -> str:
    return f"{k[:4]}…{k[-4:]} (len {len(k)})" if k else "MISSING"


async def check_assemblyai():
    print("\n=== AssemblyAI ===")
    print("key:", mask(AAI))
    if not AAI:
        return
    url = (
        "wss://streaming.assemblyai.com/v3/ws"
        "?speech_model=universal-streaming-english"
        "&sample_rate=16000&encoding=pcm_s16le&format_turns=true"
    )
    try:
        async with websockets.connect(
            url, extra_headers={"Authorization": AAI}, ping_interval=5
        ) as ws:
            print("handshake OK — waiting for first message…")
            # Send 300ms of silence so the server has audio to react to.
            await ws.send(b"\x00\x00" * 4800)
            try:
                for _ in range(3):
                    msg = await asyncio.wait_for(ws.recv(), timeout=8)
                    print("  <-", msg[:200])
            except asyncio.TimeoutError:
                print("  (no further messages within timeout — likely fine)")
    except websockets.exceptions.InvalidStatusCode as e:
        print(f"  HANDSHAKE REJECTED: HTTP {e.status_code}")
    except websockets.exceptions.ConnectionClosed as e:
        print(f"  CLOSED by server: code={e.code} reason={e.reason!r}")
    except Exception as e:  # noqa: BLE001
        print(f"  ERROR: {type(e).__name__}: {e}")


async def check_elevenlabs():
    print("\n=== ElevenLabs ===")
    print("key:", mask(ELEVEN), "| voice:", VOICE)
    if not ELEVEN:
        return
    # eleven_v3 is served over HTTP only (the realtime stream-input WS rejects
    # it with 403), so exercise the same HTTP /stream endpoint the app uses.
    url = (
        f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE}/stream"
        "?output_format=pcm_16000"
    )
    body = {
        "text": "Hello there.",
        "model_id": "eleven_v3",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.8},
    }
    headers = {"xi-api-key": ELEVEN, "content-type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            async with client.stream("POST", url, json=body, headers=headers) as resp:
                if resp.status_code != 200:
                    detail = (await resp.aread()).decode("utf-8", "replace")
                    print(f"  REJECTED: HTTP {resp.status_code}: {detail[:200]}")
                    return
                audio_bytes = 0
                async for chunk in resp.aiter_bytes():
                    audio_bytes += len(chunk)
                print("  audio received:", audio_bytes > 0, f"({audio_bytes} bytes)")
    except Exception as e:  # noqa: BLE001
        print(f"  ERROR: {type(e).__name__}: {e}")


async def check_openai():
    print("\n=== OpenAI ===")
    print("key:", mask(OPENAI))
    if not OPENAI:
        return
    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        r = await llm.ainvoke("Say 'ok'.")
        print("  reply:", (r.content or "")[:60])
    except Exception as e:  # noqa: BLE001
        print(f"  ERROR: {type(e).__name__}: {e}")


async def main():
    await check_assemblyai()
    await check_elevenlabs()
    await check_openai()


if __name__ == "__main__":
    asyncio.run(main())
