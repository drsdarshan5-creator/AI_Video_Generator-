import logging
from typing import List, Optional
from fastapi import APIRouter, Body, HTTPException, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from ..config import settings
from ..database import get_db
from ..models import (
    Scene,
    Character,
    StoryboardGenerateRequest,
    DialogueLine,
)
from ..models.db_models import (
    Character as DBCharacter,
    Scene as DBScene,
    Story as DBStory,
)
from ..services.storyboard import StoryboardService
from ..services.storyboard import StoryboardGenerationError
from ..services.audio import AudioService, SceneAudioOptions
from ..services.video import VideoService
from ..services.video_providers import VideoGenerationRequest
from ..services.screenplay import (
    ScreenplayService,
    ScreenplayGenerationError,
)
from ..services.cinematic_intelligence import (
    CinematicMetadataError,
    build_scene_visual_prompt,
)

router = APIRouter()
storyboard_service = StoryboardService()
audio_service = AudioService()
video_service = VideoService()
logger = logging.getLogger(__name__)

# In-memory storage fallback for backwards compatibility
_SCENES_STORE: List[Scene] = []


@router.get("", response_model=List[Scene])
@router.get("/", response_model=List[Scene])
async def list_scenes(
    story_id: Optional[str] = Query(default=None, description="Filter scenes by story ID"),
    db: AsyncSession = Depends(get_db),
):
    """Retrieve screenplay scenes from PostgreSQL, falling back to memory store."""
    try:
        stmt = select(DBScene).order_by(DBScene.scene_number)
        if story_id:
            stmt = stmt.where(DBScene.story_id == story_id)
        result = await db.execute(stmt)
        db_scenes = result.scalars().all()

        if db_scenes:
            return [
                Scene(
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
                        DialogueLine(
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
                for s in db_scenes
            ]
    except Exception:
        pass

    # Fallback to in-memory store
    if story_id:
        return [s for s in _SCENES_STORE if s.story_id == story_id]
    return _SCENES_STORE


@router.get("/{scene_number}", response_model=Scene)
async def get_scene(
    scene_number: int,
    story_id: Optional[str] = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    """Retrieve one scene with its visual-generation metadata."""
    try:
        stmt = select(DBScene).where(DBScene.scene_number == scene_number)
        if story_id:
            stmt = stmt.where(DBScene.story_id == story_id)
        result = await db.execute(stmt)
        scene = result.scalar_one_or_none()
        if scene:
            return Scene(
                id=scene.id,
                story_id=scene.story_id,
                scene_number=scene.scene_number,
                heading=scene.heading,
                location=scene.location,
                time_of_day=scene.time_of_day,
                mood=scene.mood,
                lighting=scene.lighting,
                weather=getattr(scene, "weather", None),
                shot_type=getattr(scene, "shot_type", "Medium shot"),
                action_description=scene.action_description,
                visual_prompt=scene.visual_prompt,
                camera_movement=scene.camera_movement,
                duration_seconds=scene.duration_seconds,
                dialogues=[
                    DialogueLine(
                        character_name=d.get("character_name", ""),
                        text=d.get("text", ""),
                        audio_url=d.get("audio_url"),
                    )
                    for d in (scene.dialogues or [])
                ],
                image_url=scene.image_url,
                video_url=scene.video_url,
                created_at=scene.created_at,
            )
    except Exception:
        pass

    for scene in _SCENES_STORE:
        if scene.scene_number == scene_number:
            if not story_id or scene.story_id == story_id:
                return scene
    raise HTTPException(
        status_code=404,
        detail=f"Scene {scene_number} was not found.",
    )


@router.post("/{scene_number}/audio", response_model=dict)
async def generate_scene_audio(
    scene_number: int,
    options: Optional[SceneAudioOptions] = Body(default=None),
    db: AsyncSession = Depends(get_db),
):
    """Generate non-dialogue audio for one stored screenplay scene."""
    scene_options = options or SceneAudioOptions()
    database_unavailable = False
    try:
        stmt = select(DBScene).where(DBScene.scene_number == scene_number)
        if scene_options.story_id:
            stmt = stmt.where(DBScene.story_id == scene_options.story_id)
        result = await db.execute(stmt.order_by(DBScene.id))
        stored_scenes = result.scalars().all()
    except SQLAlchemyError:
        logger.exception(
            "Could not retrieve scene %s for audio generation",
            scene_number,
        )
        stored_scenes = []
        database_unavailable = True

    if not scene_options.story_id and len(stored_scenes) > 1:
        raise HTTPException(
            status_code=422,
            detail="story_id is required when the scene number is ambiguous.",
        )

    if stored_scenes:
        scene = Scene.model_validate(stored_scenes[0], from_attributes=True)
    else:
        matching_scenes = [
            item
            for item in _SCENES_STORE
            if item.scene_number == scene_number
            and (
                not scene_options.story_id
                or item.story_id == scene_options.story_id
            )
        ]
        if len(matching_scenes) > 1 and not scene_options.story_id:
            raise HTTPException(
                status_code=422,
                detail="story_id is required when the scene number is ambiguous.",
            )
        if not matching_scenes:
            if database_unavailable:
                raise HTTPException(
                    status_code=503,
                    detail="Scene lookup is unavailable.",
                )
            raise HTTPException(
                status_code=404,
                detail=f"Scene {scene_number} was not found.",
            )
        scene = matching_scenes[0]

    try:
        result = await audio_service.generate_scene_audio(
            scene,
            audio_type=scene_options.audio_type,
            duration_seconds=scene_options.duration_seconds,
            seed=scene_options.seed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    response = result.model_dump(mode="json")
    response["audio_url"] = None
    if result.output_path:
        try:
            relative_path = result.output_path.resolve().relative_to(
                settings.MEDIA_ROOT.resolve()
            )
        except ValueError:
            logger.warning(
                "Generated scene audio is outside the public media directory: %s",
                result.output_path,
            )
        else:
            response["audio_url"] = f"/generated/{relative_path.as_posix()}"
    return response


@router.post("/{scene_number}/video", response_model=dict)
async def generate_scene_video(
    scene_number: int,
    request: VideoGenerationRequest,
    db: AsyncSession = Depends(get_db),
):
    """Generate one scene clip and persist its public media reference."""
    database_unavailable = False
    try:
        stmt = select(DBScene).where(DBScene.scene_number == scene_number)
        if request.story_id:
            stmt = stmt.where(DBScene.story_id == request.story_id)
        result = await db.execute(stmt.order_by(DBScene.id))
        stored_scenes = result.scalars().all()
    except SQLAlchemyError:
        logger.exception(
            "Could not retrieve scene %s for video generation",
            scene_number,
        )
        stored_scenes = []
        database_unavailable = True

    if not request.story_id and len(stored_scenes) > 1:
        raise HTTPException(
            status_code=422,
            detail="story_id is required when the scene number is ambiguous.",
        )

    stored_scene = stored_scenes[0] if stored_scenes else None
    if stored_scene is not None:
        scene = Scene.model_validate(stored_scene, from_attributes=True)
    else:
        matching_scenes = [
            item
            for item in _SCENES_STORE
            if item.scene_number == scene_number
            and (
                not request.story_id
                or item.story_id == request.story_id
            )
        ]
        if len(matching_scenes) > 1 and not request.story_id:
            raise HTTPException(
                status_code=422,
                detail="story_id is required when the scene number is ambiguous.",
            )
        if not matching_scenes:
            if database_unavailable:
                raise HTTPException(
                    status_code=503,
                    detail="Scene lookup is unavailable.",
                )
            raise HTTPException(
                status_code=404,
                detail=f"Scene {scene_number} was not found.",
            )
        scene = matching_scenes[0]

    generation_request = request.model_copy(
        update={
            "scene_id": request.scene_id
            or str(scene.id or f"{scene.story_id}:{scene.scene_number}"),
            "story_id": request.story_id or scene.story_id,
            "metadata": {
                **request.metadata,
                "scene_number": scene.scene_number,
                "scene_heading": scene.heading,
            },
        }
    )
    generation_result = await video_service.generate_scene_video(
        generation_request
    )
    video_url = None

    if generation_result.status == "generated":
        output_path = generation_result.output_path
        if output_path is None or not output_path.is_file():
            generation_result = generation_result.model_copy(
                update={
                    "status": "failed",
                    "error": "Video provider reported success without an output file.",
                }
            )
        else:
            try:
                relative_path = output_path.resolve().relative_to(
                    settings.MEDIA_ROOT.resolve()
                )
            except ValueError:
                generation_result = generation_result.model_copy(
                    update={
                        "status": "failed",
                        "error": "Generated video is outside the public media directory.",
                    }
                )
            else:
                video_url = f"/generated/{relative_path.as_posix()}"
                if stored_scene is not None:
                    stored_scene.video_url = video_url
                    try:
                        await db.commit()
                    except SQLAlchemyError as exc:
                        await db.rollback()
                        logger.exception(
                            "Could not persist generated video for scene %s",
                            scene_number,
                        )
                        raise HTTPException(
                            status_code=500,
                            detail="Video was generated but its scene reference could not be saved.",
                        ) from exc
                for index, item in enumerate(_SCENES_STORE):
                    if (
                        item.scene_number == scene_number
                        and item.story_id == scene.story_id
                    ):
                        _SCENES_STORE[index] = item.model_copy(
                            update={"video_url": video_url}
                        )
                        break

    response = generation_result.model_dump(mode="json")
    response["video_url"] = video_url
    return response


@router.post("", response_model=Scene)
@router.post("/", response_model=Scene)
async def add_scene(scene: Scene, db: AsyncSession = Depends(get_db)):
    """Add or update a scene in PostgreSQL and memory store."""
    try:
        from ..routes.characters import _CHARACTERS_STORE
        scene.visual_prompt = build_scene_visual_prompt(
            scene,
            _CHARACTERS_STORE,
        )
    except CinematicMetadataError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        stmt = select(DBScene).where(
            DBScene.story_id == scene.story_id,
            DBScene.scene_number == scene.scene_number,
        )
        result = await db.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            existing.heading = scene.heading
            existing.location = scene.location
            existing.time_of_day = scene.time_of_day
            existing.mood = scene.mood
            existing.lighting = scene.lighting
            existing.action_description = scene.action_description
            existing.visual_prompt = scene.visual_prompt
            existing.camera_movement = scene.camera_movement
            existing.duration_seconds = scene.duration_seconds
            existing.dialogues = [d.model_dump() for d in scene.dialogues] if scene.dialogues else []
            existing.image_url = scene.image_url
            existing.video_url = scene.video_url
        else:
            db_scene = DBScene(
                story_id=scene.story_id,
                scene_number=scene.scene_number,
                heading=scene.heading,
                location=scene.location,
                time_of_day=scene.time_of_day,
                mood=scene.mood,
                lighting=scene.lighting,
                action_description=scene.action_description,
                visual_prompt=scene.visual_prompt,
                camera_movement=scene.camera_movement,
                duration_seconds=scene.duration_seconds,
                dialogues=[d.model_dump() for d in scene.dialogues] if scene.dialogues else [],
                image_url=scene.image_url,
                video_url=scene.video_url,
            )
            db.add(db_scene)
        await db.commit()
    except Exception:
        await db.rollback()

    # Update memory store
    for idx, s in enumerate(_SCENES_STORE):
        if s.scene_number == scene.scene_number:
            _SCENES_STORE[idx] = scene
            return scene
    _SCENES_STORE.append(scene)
    return scene


@router.post("/storyboard", response_model=dict)
async def generate_scene_storyboard(
    request: StoryboardGenerateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Generate a storyboard frame specification and local placeholder image."""
    from ..routes.characters import _CHARACTERS_STORE

    scene = next(
        (
            item for item in _SCENES_STORE
            if item.scene_number == request.scene_number
            and (not request.story_id or item.story_id == request.story_id)
        ),
        None,
    )
    characters = list(_CHARACTERS_STORE)

    try:
        stmt = select(DBScene).where(DBScene.scene_number == request.scene_number)
        if request.story_id:
            stmt = stmt.where(DBScene.story_id == request.story_id)
        db_scene_result = await db.execute(stmt.order_by(DBScene.id).limit(1))
        db_scene = db_scene_result.scalar_one_or_none()
        if db_scene:
            scene = Scene(
                id=db_scene.id,
                story_id=db_scene.story_id,
                scene_number=db_scene.scene_number,
                heading=db_scene.heading,
                location=db_scene.location,
                time_of_day=db_scene.time_of_day,
                mood=db_scene.mood,
                lighting=db_scene.lighting,
                action_description=db_scene.action_description,
                visual_prompt=db_scene.visual_prompt,
                camera_movement=db_scene.camera_movement,
                duration_seconds=db_scene.duration_seconds,
                dialogues=[
                    DialogueLine(
                        character_name=item.get("character_name", ""),
                        text=item.get("text", ""),
                        audio_url=item.get("audio_url"),
                    )
                    for item in (db_scene.dialogues or [])
                ],
                image_url=db_scene.image_url,
                video_url=db_scene.video_url,
            )
            character_result = await db.execute(
                select(DBCharacter).where(DBCharacter.story_id == db_scene.story_id)
            )
            db_characters = character_result.scalars().all()
            if db_characters:
                characters = [
                    Character(
                        id=item.id,
                        story_id=item.story_id,
                        name=item.name,
                        role=item.role,
                        description=item.description or "",
                        personality=item.personality,
                        appearance_description=item.description,
                        visual_prompt=item.visual_prompt,
                        voice_name=item.voice_name,
                        voice_id=item.voice_id,
                    )
                    for item in db_characters
                ]
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve scene data for storyboard generation.",
        ) from exc

    if scene is None:
        prompt = request.prompt_override or f"Cinematic visual for scene {request.scene_number}"
        result = await storyboard_service.generate_image(
            prompt,
            {"scene_number": request.scene_number},
        )
        return {
            "scene_number": request.scene_number,
            "visual_prompt": prompt,
            "image_url": result.image_url,
            "status": result.status,
            "provider_name": result.provider_name,
            "error": result.error,
        }

    try:
        frame = storyboard_service.build_frame(scene, characters)
    except StoryboardGenerationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if request.prompt_override:
        frame.visual_prompt = request.prompt_override
    frame = (await storyboard_service.render_frames([frame]))[0]

    # Update in database if scene exists
    try:
        stmt = select(DBScene).where(DBScene.scene_number == request.scene_number)
        if request.story_id:
            stmt = stmt.where(DBScene.story_id == request.story_id)
        result = await db.execute(stmt)
        scenes = result.scalars().all()
        for scn in scenes:
            scn.image_url = frame.image_url
        await db.commit()
    except Exception:
        await db.rollback()

    # Update memory store
    for stored_scene in _SCENES_STORE:
        if (
            stored_scene.scene_number == request.scene_number
            and (not request.story_id or stored_scene.story_id == request.story_id)
        ):
            stored_scene.image_url = frame.image_url

    return {
        "scene_number": frame.scene_number,
        "visual_prompt": frame.visual_prompt,
        "image_url": frame.image_url,
        "status": frame.status,
        "frame": frame.model_dump(),
    }


@router.get("/storyboard/{story_id}", response_model=dict)
async def generate_story_storyboard(
    story_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Build and render placeholder storyboard frames for every scene in a story."""
    from ..routes.characters import _CHARACTERS_STORE

    try:
        result = await db.execute(select(DBStory).where(DBStory.id == story_id))
        story = result.scalar_one_or_none()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve story data for storyboard generation.",
        ) from exc

    if story:
        scenes = [
            Scene(
                id=item.id,
                story_id=item.story_id,
                scene_number=item.scene_number,
                heading=item.heading,
                location=item.location,
                time_of_day=item.time_of_day,
                mood=item.mood,
                lighting=item.lighting,
                action_description=item.action_description,
                visual_prompt=item.visual_prompt,
                camera_movement=item.camera_movement,
                duration_seconds=item.duration_seconds,
                dialogues=[
                    DialogueLine(
                        character_name=line.get("character_name", ""),
                        text=line.get("text", ""),
                        audio_url=line.get("audio_url"),
                    )
                    for line in (item.dialogues or [])
                ],
                image_url=item.image_url,
                video_url=item.video_url,
            )
            for item in (story.scenes or [])
        ]
        characters = [
            Character(
                id=item.id,
                story_id=item.story_id,
                name=item.name,
                role=item.role,
                description=item.description or "",
                personality=item.personality,
                appearance_description=item.description,
                visual_prompt=item.visual_prompt,
                voice_name=item.voice_name,
                voice_id=item.voice_id,
            )
            for item in (story.characters or [])
        ]
    else:
        scenes = [
            scene for scene in _SCENES_STORE
            if scene.story_id == story_id
        ]
        characters = [
            character for character in _CHARACTERS_STORE
            if character.story_id == story_id
        ]
        if not scenes:
            raise HTTPException(
                status_code=404,
                detail=f"Story '{story_id}' was not found or has no scenes.",
            )

    try:
        frames = storyboard_service.build_storyboard(scenes, characters)
        frames = await storyboard_service.render_placeholders(frames)
    except StoryboardGenerationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "story_id": story_id,
        "frames": [frame.model_dump() for frame in frames],
    }


@router.get("/screenplay/export", response_model=dict)
async def export_screenplay(
    title: str = "My Movie",
    story_id: Optional[str] = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    """Generate structured and formatted screenplay output for an analyzed story."""
    story = None

    if story_id:
        try:
            result = await db.execute(select(DBStory).where(DBStory.id == story_id))
            story = result.scalar_one_or_none()
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve the analyzed story.",
            ) from exc

        if story is None:
            raise HTTPException(
                status_code=404,
                detail=f"Analyzed story '{story_id}' was not found.",
            )

        story_scenes = [
            Scene(
                id=scn.id,
                story_id=scn.story_id,
                scene_number=scn.scene_number,
                heading=scn.heading,
                location=scn.location,
                time_of_day=scn.time_of_day,
                mood=scn.mood,
                lighting=scn.lighting,
                weather=getattr(scn, "weather", None),
                shot_type=getattr(scn, "shot_type", "Medium shot"),
                action_description=scn.action_description,
                visual_prompt=scn.visual_prompt,
                camera_movement=scn.camera_movement,
                duration_seconds=scn.duration_seconds,
                dialogues=[
                    DialogueLine(
                        character_name=d.get("character_name", ""),
                        text=d.get("text", ""),
                        audio_url=d.get("audio_url"),
                    )
                    for d in (scn.dialogues or [])
                ],
                image_url=scn.image_url,
                video_url=scn.video_url,
            )
            for scn in (story.scenes or [])
        ]
        story_characters = [
            Character(
                id=char.id,
                story_id=char.story_id,
                name=char.name,
                role=char.role,
                description=char.description or "",
                personality=char.personality,
                visual_prompt=char.visual_prompt,
                appearance_description=char.description or None,
                voice_name=char.voice_name,
                voice_id=char.voice_id,
            )
            for char in (story.characters or [])
        ]

        story_data = {
            "title": story.title,
            "genre": story.genre,
            "logline": story.logline,
            "theme": story.theme,
            "tone": story.tone,
            "provider": story.provider,
            "characters": story_characters,
            "scenes": story_scenes,
        }
    else:
        story_data = {
            "title": title,
            "genre": "Cinematic Drama",
            "logline": "A cinematic story awaiting full analysis.",
            "theme": None,
            "tone": None,
            "provider": "rule_based",
            "characters": [],
            "scenes": await list_scenes(story_id=None, db=db),
        }

    try:
        screenplay = ScreenplayService.generate_from_story(story_data)
    except ScreenplayGenerationError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    response = screenplay.model_dump()
    response["screenplay"] = screenplay.formatted_text
    return response
