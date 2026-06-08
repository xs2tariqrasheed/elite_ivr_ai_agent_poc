"""Health-check route."""
from fastapi import APIRouter

from configs.settings import settings

router = APIRouter()


@router.get("/health")
async def health():
    return {
        "ok": True,
        "assemblyai": bool(settings.assemblyai_api_key),
        "openai": bool(settings.openai_api_key),
        "elevenlabs": bool(settings.elevenlabs_api_key),
    }
