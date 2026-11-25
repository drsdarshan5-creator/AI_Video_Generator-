from fastapi import APIRouter

from ..config import settings

router = APIRouter()


@router.get("", response_model=dict)
async def get_configured_providers():
    """Expose configured provider identifiers without runtime or secret details."""
    return {
        "image": settings.IMAGE_PROVIDER,
        "voice": settings.VOICE_PROVIDER,
        "video": settings.VIDEO_PROVIDER,
        "audio": settings.AUDIO_PROVIDER,
    }
