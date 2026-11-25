from typing import Iterable, Any


class CinematicMetadataError(ValueError):
    """Raised when cinematic metadata is incomplete or malformed."""

    pass


def _field(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def build_character_visual_prompt(character: Any) -> str:
    """Build a reusable character visual prompt without calling an AI model."""
    name = _field(character, "name")
    if not name or not str(name).strip():
        raise CinematicMetadataError("Character name is required.")
    description = _field(character, "description")
    appearance = _field(character, "appearance_description")
    clothing = _field(character, "clothing_style")
    age_range = _field(character, "age_range")
    personality = _field(character, "personality")

    parts = [
        "cinematic character portrait",
        appearance or description,
        clothing,
        f"{age_range} age range" if age_range else None,
        personality,
        "realistic film lighting",
        "consistent character design",
        "photorealistic",
    ]
    return ", ".join(part.strip() for part in parts if part and part.strip())


def build_scene_visual_prompt(
    scene: Any,
    characters: Iterable[Any] = (),
) -> str:
    """Build a reusable scene prompt from deterministic metadata."""
    heading = _field(scene, "heading")
    duration = _field(scene, "duration_seconds")
    action = _field(scene, "action_description")
    dialogues = _field(scene, "dialogues") or []

    if not heading or not str(heading).strip():
        raise CinematicMetadataError("Scene heading is required.")
    if not isinstance(duration, int) or not 1 <= duration <= 60:
        raise CinematicMetadataError("Scene duration must be between 1 and 60 seconds.")
    if not action or not str(action).strip():
        raise CinematicMetadataError("Scene action description is required.")

    characters = list(characters)
    action_lower = str(action).lower()
    dialogue_names = set()
    for dialogue in dialogues:
        dialogue_name = (
            dialogue.get("character_name")
            if isinstance(dialogue, dict)
            else getattr(dialogue, "character_name", None)
        )
        if dialogue_name:
            dialogue_names.add(str(dialogue_name).lower())

    referenced_characters = []
    for character in characters:
        character_name = _field(character, "name")
        if not character_name:
            continue
        name_lower = str(character_name).lower()
        if name_lower in action_lower or name_lower in dialogue_names:
            description_parts = [
                _field(character, "appearance_description")
                or _field(character, "description"),
                _field(character, "clothing_style"),
                f"{_field(character, 'age_range')} age range"
                if _field(character, "age_range")
                else None,
                _field(character, "personality"),
            ]
            description = ", ".join(
                str(part).strip()
                for part in description_parts
                if part and str(part).strip()
            )
            if not description:
                description = _field(character, "visual_prompt")
            if description:
                referenced_characters.append(f"{character_name}: {description}")

    parts = [
        "cinematic production frame",
        _field(scene, "location"),
        _field(scene, "time_of_day"),
        _field(scene, "mood"),
        _field(scene, "lighting"),
        _field(scene, "weather"),
        action,
        *referenced_characters,
        _field(scene, "shot_type"),
        _field(scene, "camera_movement"),
        "coherent scene continuity",
        "realistic cinematic composition",
    ]
    return ", ".join(part.strip() for part in parts if part and part.strip())
