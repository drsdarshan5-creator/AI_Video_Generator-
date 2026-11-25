from typing import List, Optional
import re
import logging
from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from ..database import get_db
from ..config import settings
from ..models import Character, DialogueLine
from ..models.db_models import Character as DBCharacter, Scene as DBScene
from ..services.voice import VoiceService
from ..services.cinematic_intelligence import (
    CinematicMetadataError,
    build_character_visual_prompt,
)

router = APIRouter()
voice_service = VoiceService()
logger = logging.getLogger(__name__)

# In-memory storage fallback for backwards compatibility
_CHARACTERS_STORE: List[Character] = []


@router.get("", response_model=List[Character])
@router.get("/", response_model=List[Character])
async def list_characters(
    story_id: Optional[str] = Query(default=None, description="Filter characters by story ID"),
    db: AsyncSession = Depends(get_db),
):
    """List all extracted or configured characters from PostgreSQL, falling back to memory store."""
    try:
        stmt = select(DBCharacter)
        if story_id:
            stmt = stmt.where(DBCharacter.story_id == story_id)
        result = await db.execute(stmt)
        db_chars = result.scalars().all()

        if db_chars:
            return [
                Character(
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
                for c in db_chars
            ]
    except Exception:
        pass

    # Fallback to in-memory store
    if story_id:
        return [c for c in _CHARACTERS_STORE if c.story_id == story_id]
    return _CHARACTERS_STORE


@router.get("/{character_id}", response_model=Character)
async def get_character(
    character_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Retrieve one character with its visual-generation metadata."""
    try:
        result = await db.execute(
            select(DBCharacter).where(DBCharacter.id == character_id)
        )
        character = result.scalar_one_or_none()
        if character:
            return Character(
                id=character.id,
                story_id=character.story_id,
                name=character.name,
                role=character.role,
                description=character.description or "",
                personality=character.personality,
                appearance_description=character.description or None,
                age_range=(
                    re.search(r"(mid-\d+s|late-\d+s|early-\d+s|\d+-\d+)", character.visual_prompt).group(1)
                    if re.search(r"(mid-\d+s|late-\d+s|early-\d+s|\d+-\d+)", character.visual_prompt)
                    else None
                ),
                visual_prompt=character.visual_prompt,
                voice_name=character.voice_name,
                voice_id=character.voice_id,
                created_at=character.created_at,
            )
    except Exception:
        pass

    for character in _CHARACTERS_STORE:
        if character.id == character_id:
            return character
    raise HTTPException(
        status_code=404,
        detail=f"Character '{character_id}' was not found.",
    )


@router.post("", response_model=Character)
@router.post("/", response_model=Character)
async def create_character(
    character: Character,
    db: AsyncSession = Depends(get_db),
):
    """Add or update a character in the project."""
    try:
        character.visual_prompt = build_character_visual_prompt(character)
    except CinematicMetadataError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        # Check if character exists in DB
        stmt = select(DBCharacter).where(DBCharacter.id == character.id)
        result = await db.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            existing.name = character.name
            existing.role = character.role or existing.role
            existing.description = character.description
            existing.personality = character.personality
            existing.visual_prompt = character.visual_prompt
            existing.voice_name = character.voice_name
            existing.voice_id = character.voice_id
        else:
            db_char = DBCharacter(
                id=character.id,
                story_id=character.story_id,
                name=character.name,
                role=character.role,
                description=character.description,
                personality=character.personality,
                visual_prompt=character.visual_prompt,
                voice_name=character.voice_name,
                voice_id=character.voice_id,
            )
            db.add(db_char)
        await db.commit()
    except Exception:
        await db.rollback()

    # Update in-memory fallback store
    for idx, c in enumerate(_CHARACTERS_STORE):
        if c.id == character.id:
            _CHARACTERS_STORE[idx] = character
            return character

    _CHARACTERS_STORE.append(character)
    return character


@router.post("/{character_id}/voice", response_model=dict)
async def test_character_voice(
    character_id: str,
    text: str = "This is a voice preview for testing.",
    db: AsyncSession = Depends(get_db),
):
    """Generate sample voice audio preview for a character."""
    character = next((c for c in _CHARACTERS_STORE if c.id == character_id), None)
    db_character = None
    try:
        result = await db.execute(
            select(DBCharacter).where(DBCharacter.id == character_id)
        )
        db_character = result.scalar_one_or_none()
    except SQLAlchemyError as exc:
        logger.exception(
            "Could not retrieve character %s for voice generation",
            character_id,
        )
        raise HTTPException(
            status_code=503,
            detail="Character data is unavailable for voice generation.",
        ) from exc

    if db_character is not None:
        character = Character(
            id=db_character.id,
            story_id=db_character.story_id,
            name=db_character.name,
            role=db_character.role,
            description=db_character.description or "",
            personality=db_character.personality,
            appearance_description=db_character.description or None,
            visual_prompt=db_character.visual_prompt,
            voice_name=db_character.voice_name,
            voice_id=db_character.voice_id,
        )

    char_name = character.name if character else "Narrator"

    result = await voice_service.generate_speech(
        character_name=char_name,
        dialogue_text=text,
        character_id=character.id if character else None,
        voice_name=character.voice_name if character else None,
        voice_id=character.voice_id if character else None,
    )
    audio_url = None
    if result.status == "generated":
        if result.output_path is None:
            raise HTTPException(
                status_code=500,
                detail="Voice provider reported success without an audio file.",
            )
        output_path = result.output_path.resolve()
        try:
            relative_path = output_path.relative_to(settings.MEDIA_ROOT.resolve())
        except ValueError as exc:
            logger.error(
                "Generated voice preview is outside the public media directory: %s",
                output_path,
            )
            raise HTTPException(
                status_code=500,
                detail="Generated voice audio is outside the public media directory.",
            ) from exc
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise HTTPException(
                status_code=500,
                detail="Voice provider reported success without a valid audio file.",
            )
        audio_url = f"/generated/{relative_path.as_posix()}"

        if db_character is not None and db_character.story_id:
            try:
                scene_result = await db.execute(
                    select(DBScene).where(DBScene.story_id == db_character.story_id)
                )
                stored_scenes = scene_result.scalars().all()
                for stored_scene in stored_scenes:
                    dialogue_lines = list(stored_scene.dialogues or [])
                    for index, line in enumerate(dialogue_lines):
                        if (
                            line.get("character_name", "").casefold()
                            == char_name.casefold()
                            and line.get("text", "").strip() == text.strip()
                        ):
                            dialogue_lines[index] = {
                                **line,
                                "audio_url": audio_url,
                            }
                            stored_scene.dialogues = dialogue_lines
                            await db.commit()
                            break
                    else:
                        continue
                    break
            except SQLAlchemyError as exc:
                await db.rollback()
                try:
                    output_path.unlink()
                except OSError:
                    logger.exception(
                        "Could not remove voice preview after dialogue persistence failed: %s",
                        output_path,
                    )
                logger.exception(
                    "Could not persist generated voice for character %s",
                    character_id,
                )
                raise HTTPException(
                    status_code=503,
                    detail="Generated voice audio could not be attached to its dialogue.",
                ) from exc

        from .scenes import _SCENES_STORE

        for scene in _SCENES_STORE:
            if scene.story_id != (character.story_id if character else None):
                continue
            for index, dialogue in enumerate(scene.dialogues):
                if (
                    dialogue.character_name.casefold() == char_name.casefold()
                    and dialogue.text.strip() == text.strip()
                ):
                    scene.dialogues[index] = DialogueLine(
                        character_name=dialogue.character_name,
                        text=dialogue.text,
                        audio_url=audio_url,
                    )
                    break

    return {
        "character_name": char_name,
        "preview_text": text,
        "audio_url": audio_url,
        "status": result.status,
        "provider": result.provider,
        "message": result.message,
    }
