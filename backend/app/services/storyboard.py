import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from ..config import settings
from ..models import (
    Character,
    Scene,
    StoryboardCharacterReference,
    StoryboardFrame,
)
from .cinematic_intelligence import (
    CinematicMetadataError,
    build_character_visual_prompt,
    build_scene_visual_prompt,
)
from .image_providers import (
    BaseImageProvider,
    ImageGenerationResult,
    get_image_provider,
)

logger = logging.getLogger(__name__)


class StoryboardGenerationError(ValueError):
    """Raised when scene data cannot be converted into a storyboard frame."""

    pass


class StoryboardService:
    """Build storyboard specifications and render local placeholder images."""

    def __init__(self, image_provider: Optional[BaseImageProvider] = None):
        self.image_provider = image_provider or get_image_provider()

    @property
    def output_dir(self) -> Path:
        """Retain the previous output_dir hook for placeholder-focused callers."""
        return getattr(self.image_provider, "output_dir", settings.IMAGE_OUTPUT_DIR)

    @output_dir.setter
    def output_dir(self, value: Path) -> None:
        if hasattr(self.image_provider, "output_dir"):
            self.image_provider.output_dir = Path(value)

    @staticmethod
    def _field(value: Any, key: str, default: Any = None) -> Any:
        if isinstance(value, dict):
            return value.get(key, default)
        return getattr(value, key, default)

    @classmethod
    def _normalize_scene(cls, scene: Any) -> Scene:
        if isinstance(scene, Scene):
            return scene
        try:
            return Scene.model_validate(scene)
        except Exception as exc:
            scene_number = cls._field(scene, "scene_number", "unknown")
            raise StoryboardGenerationError(
                f"Scene {scene_number} is invalid and cannot be storyboarded."
            ) from exc

    @staticmethod
    def _referenced_characters(
        scene: Scene,
        characters: Iterable[Character],
    ) -> List[StoryboardCharacterReference]:
        action = scene.action_description.lower()
        speaker_names = {
            dialogue.character_name.casefold()
            for dialogue in scene.dialogues
        }
        references = []
        for character in characters:
            if (
                character.name.casefold() in action
                or character.name.casefold() in speaker_names
            ):
                visual_description = (
                    character.visual_prompt
                    or build_character_visual_prompt(character)
                )
                references.append(
                    StoryboardCharacterReference(
                        character_id=character.id,
                        name=character.name,
                        visual_description=visual_description,
                    )
                )
        return references

    def build_frame(
        self,
        scene: Any,
        characters: Iterable[Any] = (),
    ) -> StoryboardFrame:
        """Create one validated storyboard specification without rendering an image."""
        normalized_scene = self._normalize_scene(scene)
        normalized_characters = []
        try:
            for character in characters:
                normalized_characters.append(
                    character if isinstance(character, Character)
                    else Character.model_validate(character)
                )
            references = self._referenced_characters(
                normalized_scene,
                normalized_characters,
            )
            prompt = build_scene_visual_prompt(
                normalized_scene,
                normalized_characters,
            )
        except (CinematicMetadataError, ValueError) as exc:
            raise StoryboardGenerationError(str(exc)) from exc

        return StoryboardFrame(
            scene_number=normalized_scene.scene_number,
            scene_heading=normalized_scene.heading,
            location=normalized_scene.location,
            time_of_day=normalized_scene.time_of_day,
            mood=normalized_scene.mood,
            lighting=normalized_scene.lighting,
            weather=normalized_scene.weather,
            shot_type=normalized_scene.shot_type,
            camera_movement=normalized_scene.camera_movement,
            duration_seconds=normalized_scene.duration_seconds,
            action_description=normalized_scene.action_description,
            character_references=references,
            visual_prompt=prompt,
            dialogue_summary=normalized_scene.dialogues,
        )

    def build_storyboard(
        self,
        scenes: Iterable[Any],
        characters: Iterable[Any] = (),
    ) -> List[StoryboardFrame]:
        """Create deterministic storyboard frame specifications for scenes."""
        character_list = list(characters)
        try:
            normalized_scenes = [self._normalize_scene(scene) for scene in scenes]
        except StoryboardGenerationError:
            raise
        if not normalized_scenes:
            raise StoryboardGenerationError("No scenes were provided for storyboard generation.")

        scene_numbers = [scene.scene_number for scene in normalized_scenes]
        seen_numbers = set()
        duplicates = []
        for number in scene_numbers:
            if number in seen_numbers and number not in duplicates:
                duplicates.append(number)
            seen_numbers.add(number)
        if duplicates:
            duplicate_list = ", ".join(str(number) for number in sorted(duplicates))
            raise StoryboardGenerationError(
                f"Duplicate scene numbers found: {duplicate_list}."
            )

        expected_numbers = set(range(1, max(scene_numbers) + 1))
        missing_numbers = sorted(expected_numbers - seen_numbers)
        if missing_numbers:
            missing_list = ", ".join(str(number) for number in missing_numbers)
            raise StoryboardGenerationError(
                f"Missing scene numbers: {missing_list}."
            )

        frames = [
            self.build_frame(scene, character_list)
            for scene in sorted(normalized_scenes, key=lambda item: item.scene_number)
        ]
        if len(frames) != len(normalized_scenes):
            raise StoryboardGenerationError(
                "Each scene must produce exactly one storyboard frame."
            )

        character_counts = {}
        for frame in frames:
            for reference in frame.character_references:
                character_counts[reference.character_id] = (
                    character_counts.get(reference.character_id, 0) + 1
                )

        for index, frame in enumerate(frames):
            if index:
                previous = frames[index - 1]
                frame.previous_scene_reference = previous.scene_number
                frame.continuity_notes.append(
                    f"Follows scene {previous.scene_number}: {previous.scene_heading}"
                )
            frame.recurring_character_references = [
                reference
                for reference in frame.character_references
                if character_counts.get(reference.character_id, 0) > 1
            ]
            if frame.recurring_character_references:
                names = ", ".join(
                    reference.name for reference in frame.recurring_character_references
                )
                frame.continuity_notes.append(
                    f"Preserve established character designs for: {names}."
                )
        return frames

    async def render_frames(
        self,
        frames: Iterable[StoryboardFrame],
    ) -> List[StoryboardFrame]:
        """Render frames through the configured provider, retaining failures per frame."""
        rendered = []
        for frame in frames:
            metadata = frame.model_dump(
                exclude={"visual_prompt", "image_url", "image_path", "provider_name",
                         "generation_metadata", "generation_error", "status"}
            )
            metadata["character_names"] = [
                reference.name for reference in frame.character_references
            ]
            result = await self.generate_image(frame.visual_prompt, metadata)
            frame.image_path = result.image_path
            frame.image_url = result.image_url
            frame.provider_name = result.provider_name
            frame.status = result.status
            frame.visual_prompt = result.prompt_used
            frame.generation_metadata = result.metadata
            frame.generation_error = result.error
            rendered.append(frame)
        return rendered

    async def render_placeholders(
        self,
        frames: Iterable[StoryboardFrame],
    ) -> List[StoryboardFrame]:
        """Backward-compatible alias for rendering with the configured image provider."""
        return await self.render_frames(frames)

    async def generate_image(
        self,
        prompt: str,
        metadata: Dict[str, Any],
    ) -> ImageGenerationResult:
        """Call the configured provider and convert provider failures into frame results."""
        provider_name = getattr(self.image_provider, "provider_name", "unknown")
        try:
            if not self.image_provider.is_available():
                return ImageGenerationResult(
                    provider_name=provider_name,
                    status="failed",
                    prompt_used=prompt,
                    metadata=metadata,
                    error=f"Image provider '{provider_name}' is unavailable.",
                )
            result = await self.image_provider.generate_image(prompt, metadata)
            if not isinstance(result, ImageGenerationResult):
                result = ImageGenerationResult.model_validate(result)
            return result
        except Exception as exc:
            logger.exception(
                "Image provider '%s' failed to generate a storyboard frame",
                provider_name,
            )
            return ImageGenerationResult(
                provider_name=provider_name,
                status="failed",
                prompt_used=prompt,
                metadata=metadata,
                error=f"Image provider '{provider_name}' failed: {exc}",
            )

    async def generate_frame(
        self,
        scene_number: int,
        visual_prompt: str,
        scene_heading: Optional[str] = None,
        shot_type: Optional[str] = None,
        duration_seconds: Optional[int] = None,
        action_description: Optional[str] = None,
        character_names: Iterable[str] = (),
    ) -> str:
        """Compatibility wrapper returning only the provider image URL."""
        result = await self.generate_image(
            visual_prompt,
            {
                "scene_number": scene_number,
                "scene_heading": scene_heading,
                "shot_type": shot_type,
                "duration_seconds": duration_seconds,
                "action_description": action_description,
                "character_names": list(character_names),
                "width": settings.IMAGE_WIDTH,
                "height": settings.IMAGE_HEIGHT,
            },
        )
        return result.image_url or ""
