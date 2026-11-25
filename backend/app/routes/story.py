import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models import StoryAnalysisRequest, StoryAnalysisResponse
from ..services.story_analyzer import StoryAnalyzerService
from ..services.ai import StoryAnalysisError, ProviderNotAvailableError, InvalidProviderError

logger = logging.getLogger("uvicorn")

router = APIRouter()
story_service = StoryAnalyzerService()


@router.post("/analyze", response_model=StoryAnalysisResponse, status_code=status.HTTP_201_CREATED)
async def analyze_story(
    request: StoryAnalysisRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Analyze narrative premise and decompose it into characters, logline, and scenes.
    Persists the structured story, characters, and scenes directly into PostgreSQL.
    """
    try:
        response = await story_service.analyze(request, db=db)
        return response
    except ProviderNotAvailableError as pne:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(pne),
        )
    except InvalidProviderError as ipe:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ipe),
        )
    except StoryAnalysisError as sae:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(sae),
        )
    except Exception as e:
        logger.error(f"Error during story analysis: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze story due to an unexpected application error.",
        )


@router.get("/sample", response_model=dict)
async def get_sample_story():
    """Retrieve sample story prompts to kickstart video creation."""
    return {
        "sample_prompts": [
            {
                "title": "Neon Odyssey",
                "genre": "Cyberpunk Sci-Fi",
                "prompt": "In the rain-drenched streets of Neo-Kyoto 2142, detective Alex Vance discovers an encrypted neural transmission pointing to an ancient monolith hidden beneath the city foundation.",
            },
            {
                "title": "The Last Archivist",
                "genre": "Mystery / Fantasy",
                "prompt": "Deep within an underground library that stores the forgotten memories of civilization, an archivist discovers a golden book whose pages write themselves in real time.",
            },
        ]
    }


@router.get("/{story_id}", response_model=StoryAnalysisResponse)
async def get_story(story_id: str, db: AsyncSession = Depends(get_db)):
    """Retrieve an analyzed story by ID from PostgreSQL."""
    story = await story_service.get_story_by_id(story_id, db=db)
    if not story:
        raise HTTPException(status_code=404, detail=f"Story with ID '{story_id}' not found.")
    return story


@router.get("", response_model=List[StoryAnalysisResponse])
@router.get("/", response_model=List[StoryAnalysisResponse])
async def list_stories(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """List analyzed stories stored in PostgreSQL."""
    return await story_service.list_stories(db=db, limit=limit, offset=offset)


@router.delete("/{story_id}", response_model=dict)
async def delete_story(story_id: str, db: AsyncSession = Depends(get_db)):
    """Delete an analyzed story from PostgreSQL."""
    deleted = await story_service.delete_story(story_id, db=db)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Story with ID '{story_id}' not found.")
    return {"status": "success", "message": f"Story '{story_id}' deleted successfully."}
