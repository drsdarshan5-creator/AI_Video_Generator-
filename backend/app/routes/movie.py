import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models import MovieRenderRequest, MovieRenderResponse, Scene
from ..models.db_models import Scene as DBScene
from ..models.db_models import Story as DBStory
from ..services.video import VideoService

router = APIRouter()
video_service = VideoService()
logger = logging.getLogger(__name__)


@router.post("/render", response_model=MovieRenderResponse)
async def render_movie(
    request: MovieRenderRequest,
    db: AsyncSession = Depends(get_db),
):
    """Trigger final movie generation and video assembly pipeline."""
    try:
        try:
            story_result = await db.execute(
                select(DBStory).where(DBStory.id == request.story_id)
            )
            story = story_result.scalar_one_or_none()
        except SQLAlchemyError as exc:
            logger.exception(
                "Could not retrieve story %s for movie rendering",
                request.story_id,
            )
            raise HTTPException(
                status_code=503,
                detail="Story data is unavailable for movie assembly.",
            ) from exc

        if story is None:
            raise HTTPException(
                status_code=404,
                detail=f"Story '{request.story_id}' was not found.",
            )

        try:
            result = await db.execute(
                select(DBScene)
                .where(DBScene.story_id == request.story_id)
                .order_by(DBScene.scene_number, DBScene.id)
            )
            scenes = [
                Scene.model_validate(scene, from_attributes=True)
                for scene in result.scalars().all()
            ]
        except SQLAlchemyError as exc:
            logger.exception(
                "Could not load persisted scenes for movie story %s",
                request.story_id,
            )
            raise HTTPException(
                status_code=503,
                detail="Scene data is unavailable for movie assembly.",
            ) from exc

        if not scenes:
            from .scenes import _SCENES_STORE

            scenes = [
                scene
                for scene in _SCENES_STORE
                if scene.story_id == request.story_id
            ]
        response = await video_service.start_movie_render(request, scenes=scenes)
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to start movie rendering for story %s", request.story_id)
        raise HTTPException(
            status_code=500,
            detail="Failed to start movie rendering.",
        ) from e


@router.get("/status/{movie_id}", response_model=MovieRenderResponse)
async def get_render_status(movie_id: str):
    """Poll rendering progress and retrieve generated video URL when complete."""
    try:
        return await video_service.get_job_status(movie_id)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail="Movie render status is temporarily unavailable.",
        ) from exc


@router.get("/audio/{movie_id}", response_model=List[Dict[str, Any]])
async def get_movie_audio_results(movie_id: str):
    """Retrieve per-scene audio results without storing audio binaries in the database."""
    results = video_service.get_movie_audio_results(movie_id)
    if results is None:
        raise HTTPException(status_code=404, detail="Movie render job not found.")
    return results
