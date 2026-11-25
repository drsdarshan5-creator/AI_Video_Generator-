from fastapi import APIRouter
from .story import router as story_router
from .characters import router as characters_router
from .scenes import router as scenes_router
from .movie import router as movie_router
from .providers import router as providers_router

api_router = APIRouter()
api_router.include_router(story_router, prefix="/story", tags=["Story"])
api_router.include_router(characters_router, prefix="/characters", tags=["Characters"])
api_router.include_router(scenes_router, prefix="/scenes", tags=["Scenes"])
api_router.include_router(movie_router, prefix="/movie", tags=["Movie"])
api_router.include_router(providers_router, prefix="/providers", tags=["Providers"])

__all__ = ["api_router"]
