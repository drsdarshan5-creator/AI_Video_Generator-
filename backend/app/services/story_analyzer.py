import uuid
import logging
from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from ..models import (
    Character as SchemaCharacter,
    Scene as SchemaScene,
    DialogueLine as SchemaDialogueLine,
    StoryAnalysisRequest,
    StoryAnalysisResponse,
)
from ..models.db_models import (
    Story as DBStory,
    Character as DBCharacter,
    Scene as DBScene,
)
from .ai import (
    get_provider,
    BaseStoryLLMProvider,
    StoryAnalysisError,
    ProviderNotAvailableError,
    InvalidProviderError,
    ProviderExecutionError,
    validate_structured_output,
)
from ..config import settings

logger = logging.getLogger("uvicorn")


class StoryAnalyzerService:
    """Service to coordinate narrative decomposition, provider execution, and database persistence."""

    def __init__(self):
        self.default_provider_name = settings.STORY_ANALYZER_PROVIDER

    async def analyze(
        self,
        request: StoryAnalysisRequest,
        db: Optional[AsyncSession] = None,
    ) -> StoryAnalysisResponse:
        """
        Analyze story premise into structured film units and persist them into PostgreSQL.
        Dispatches to the configured AI / rule-based provider.
        """
        provider_name = request.provider or self.default_provider_name
        provider = get_provider(provider_name)

        # Run narrative decomposition through the selected provider
        try:
            raw_result = await provider.analyze(
                prompt=request.prompt,
                genre=request.genre or settings.DEFAULT_GENRE,
                target_duration_minutes=request.target_duration_minutes or 1,
            )
            raw_result = validate_structured_output(raw_result)
        except (StoryAnalysisError, ProviderNotAvailableError, InvalidProviderError):
            raise
        except Exception as provider_err:
            logger.error(
                "Provider '%s' failed during story analysis",
                provider.provider_name,
                exc_info=True,
            )
            raise ProviderExecutionError(
                f"Provider '{provider.provider_name}' failed while generating story analysis."
            ) from provider_err

        story_id = str(uuid.uuid4())[:8]

        # Convert provider characters into validated Pydantic schemas
        characters: List[SchemaCharacter] = []
        for idx, char_dict in enumerate(raw_result.get("characters", [])):
            raw_id = char_dict.get("id", f"char-{idx + 1}")
            char_id = f"{raw_id}-{story_id}"
            characters.append(
                SchemaCharacter(
                    id=char_id,
                    story_id=story_id,
                    name=char_dict.get("name", f"Character {idx + 1}"),
                    role=char_dict.get("role", "Protagonist"),
                    description=char_dict.get("description", ""),
                    personality=char_dict.get("personality"),
                    appearance_description=char_dict.get("appearance_description"),
                    age_range=char_dict.get("age_range"),
                    clothing_style=char_dict.get("clothing_style"),
                    visual_prompt=char_dict.get("visual_prompt", ""),
                    voice_name=char_dict.get("voice_name", "Default Narrator"),
                    voice_id=char_dict.get("voice_id"),
                )
            )

        # Convert provider scenes into validated Pydantic schemas
        scenes: List[SchemaScene] = []
        for s_idx, scene_dict in enumerate(raw_result.get("scenes", [])):
            dialogues: List[SchemaDialogueLine] = []
            for d in scene_dict.get("dialogues", []):
                dialogues.append(
                    SchemaDialogueLine(
                        character_name=d.get("character_name", "Speaker"),
                        text=d.get("text", ""),
                        audio_url=d.get("audio_url"),
                    )
                )

            scenes.append(
                SchemaScene(
                    scene_number=scene_dict.get("scene_number", s_idx + 1),
                    story_id=story_id,
                    heading=scene_dict.get("heading", f"SCENE {s_idx + 1}"),
                    location=scene_dict.get("location"),
                    time_of_day=scene_dict.get("time_of_day"),
                    mood=scene_dict.get("mood"),
                    lighting=scene_dict.get("lighting"),
                    weather=scene_dict.get("weather"),
                    shot_type=scene_dict.get("shot_type", "Medium shot"),
                    action_description=scene_dict.get("action_description", ""),
                    visual_prompt=scene_dict.get("visual_prompt", ""),
                    camera_movement=scene_dict.get("camera_movement", "Static shot"),
                    duration_seconds=scene_dict.get("duration_seconds", 5),
                    dialogues=dialogues,
                    image_url=scene_dict.get("image_url"),
                    video_url=scene_dict.get("video_url"),
                )
            )

        # Build response schema
        response = StoryAnalysisResponse(
            story_id=story_id,
            project_id=request.project_id,
            title=raw_result.get("title", "Untitled Story"),
            logline=raw_result.get("logline", ""),
            synopsis=raw_result.get("synopsis"),
            theme=raw_result.get("theme"),
            tone=raw_result.get("tone"),
            genre=request.genre or raw_result.get("genre", settings.DEFAULT_GENRE),
            status="analyzed",
            provider=provider.provider_name,
            characters=characters,
            scenes=scenes,
        )

        # Persist into PostgreSQL when session is provided
        if db is not None:
            try:
                db_story = DBStory(
                    id=story_id,
                    project_id=request.project_id,
                    title=response.title,
                    logline=response.logline,
                    synopsis=response.synopsis,
                    theme=response.theme,
                    tone=response.tone,
                    genre=response.genre,
                    prompt=request.prompt,
                    target_duration_minutes=request.target_duration_minutes or 1,
                    status=response.status,
                    provider=response.provider,
                )
                db.add(db_story)

                for char in characters:
                    db_char = DBCharacter(
                        id=char.id,
                        story_id=story_id,
                        name=char.name,
                        role=char.role,
                        description=char.description,
                        personality=char.personality,
                        visual_prompt=char.visual_prompt,
                        voice_name=char.voice_name,
                        voice_id=char.voice_id,
                    )
                    db.add(db_char)

                for scn in scenes:
                    db_scene = DBScene(
                        story_id=story_id,
                        scene_number=scn.scene_number,
                        heading=scn.heading,
                        location=scn.location,
                        time_of_day=scn.time_of_day,
                        mood=scn.mood,
                        lighting=scn.lighting,
                        action_description=scn.action_description,
                        visual_prompt=scn.visual_prompt,
                        camera_movement=scn.camera_movement,
                        duration_seconds=scn.duration_seconds,
                        dialogues=[d.model_dump() for d in scn.dialogues] if scn.dialogues else [],
                        image_url=scn.image_url,
                        video_url=scn.video_url,
                    )
                    db.add(db_scene)

                await db.commit()
                logger.info(f"Story '{story_id}' and entities persisted to PostgreSQL.")
            except Exception as db_err:
                await db.rollback()
                logger.error(f"Failed to persist story to database: {db_err}", exc_info=True)
                raise db_err

        # Keep legacy in-memory stores in sync for backwards-compatible standalone endpoints
        try:
            from ..routes.characters import _CHARACTERS_STORE
            from ..routes.scenes import _SCENES_STORE
            _CHARACTERS_STORE.clear()
            _CHARACTERS_STORE.extend(characters)
            _SCENES_STORE.clear()
            _SCENES_STORE.extend(scenes)
        except Exception:
            pass

        return response

    async def get_story_by_id(self, story_id: str, db: AsyncSession) -> Optional[StoryAnalysisResponse]:
        """Fetch persisted story and all attached characters and scenes by story ID."""
        stmt = select(DBStory).where(DBStory.id == story_id)
        result = await db.execute(stmt)
        story = result.scalar_one_or_none()
        if not story:
            return None

        # Build characters list
        characters = [
            SchemaCharacter(
                id=c.id,
                story_id=c.story_id,
                name=c.name,
                role=c.role,
                description=c.description or "",
                personality=c.personality,
                appearance_description=c.description or None,
                visual_prompt=c.visual_prompt,
                voice_name=c.voice_name,
                voice_id=c.voice_id,
                created_at=c.created_at,
            )
            for c in (story.characters or [])
        ]

        # Build scenes list
        scenes = []
        for s in (story.scenes or []):
            raw_dialogues = s.dialogues or []
            dialogues = [
                SchemaDialogueLine(
                    character_name=d.get("character_name", ""),
                    text=d.get("text", ""),
                    audio_url=d.get("audio_url"),
                )
                for d in raw_dialogues
            ]
            scenes.append(
                SchemaScene(
                    id=s.id,
                    story_id=s.story_id,
                    scene_number=s.scene_number,
                    heading=s.heading,
                    location=s.location,
                    time_of_day=s.time_of_day,
                    mood=s.mood,
                    lighting=s.lighting,
                    weather=getattr(s, "weather", None),
                    shot_type=getattr(s, "shot_type", "Medium shot"),
                    action_description=s.action_description,
                    visual_prompt=s.visual_prompt,
                    camera_movement=s.camera_movement,
                    duration_seconds=s.duration_seconds,
                    dialogues=dialogues,
                    image_url=s.image_url,
                    video_url=s.video_url,
                    created_at=s.created_at,
                )
            )

        return StoryAnalysisResponse(
            story_id=story.id,
            project_id=story.project_id,
            title=story.title,
            logline=story.logline or "",
            synopsis=story.synopsis,
            theme=story.theme,
            tone=story.tone,
            genre=story.genre,
            status=story.status,
            provider=story.provider,
            characters=characters,
            scenes=scenes,
            created_at=story.created_at,
            updated_at=story.updated_at,
        )

    async def list_stories(
        self,
        db: AsyncSession,
        limit: int = 50,
        offset: int = 0,
    ) -> List[StoryAnalysisResponse]:
        """Fetch list of all analyzed stories from PostgreSQL."""
        stmt = select(DBStory).order_by(DBStory.created_at.desc()).offset(offset).limit(limit)
        result = await db.execute(stmt)
        stories = result.scalars().all()

        output = []
        for story in stories:
            characters = [
                SchemaCharacter(
                    id=c.id,
                    story_id=c.story_id,
                    name=c.name,
                    role=c.role,
                    description=c.description or "",
                    personality=c.personality,
                    appearance_description=c.description or None,
                    visual_prompt=c.visual_prompt,
                    voice_name=c.voice_name,
                    voice_id=c.voice_id,
                    created_at=c.created_at,
                )
                for c in (story.characters or [])
            ]
            scenes = [
                SchemaScene(
                    id=s.id,
                    story_id=s.story_id,
                    scene_number=s.scene_number,
                    heading=s.heading,
                    location=s.location,
                    time_of_day=s.time_of_day,
                    mood=s.mood,
                    lighting=s.lighting,
                    weather=getattr(s, "weather", None),
                    shot_type=getattr(s, "shot_type", "Medium shot"),
                    action_description=s.action_description,
                    visual_prompt=s.visual_prompt,
                    camera_movement=s.camera_movement,
                    duration_seconds=s.duration_seconds,
                    dialogues=[
                        SchemaDialogueLine(
                            character_name=d.get("character_name", ""),
                            text=d.get("text", ""),
                            audio_url=d.get("audio_url"),
                        )
                        for d in (s.dialogues or [])
                    ],
                    image_url=s.image_url,
                    video_url=s.video_url,
                    created_at=s.created_at,
                )
                for s in (story.scenes or [])
            ]
            output.append(
                StoryAnalysisResponse(
                    story_id=story.id,
                    project_id=story.project_id,
                    title=story.title,
                    logline=story.logline or "",
                    synopsis=story.synopsis,
                    theme=story.theme,
                    tone=story.tone,
                    genre=story.genre,
                    status=story.status,
                    provider=story.provider,
                    characters=characters,
                    scenes=scenes,
                    created_at=story.created_at,
                    updated_at=story.updated_at,
                )
            )
        return output

    async def delete_story(self, story_id: str, db: AsyncSession) -> bool:
        """Delete story and associated entities from database."""
        stmt = delete(DBStory).where(DBStory.id == story_id)
        result = await db.execute(stmt)
        await db.commit()
        return result.rowcount > 0
