import math
import logging
import re
from typing import Any, Dict, Literal, Optional, Union

from pydantic import BaseModel, Field, ValidationError, field_validator

from ..models import Scene
from ..config import settings
from .audio_providers import (
    AudioGenerationRequest,
    AudioGenerationResult,
    BaseAudioProvider,
    get_audio_provider,
)

logger = logging.getLogger(__name__)

SceneAudioType = Literal["ambience", "sound_effect", "music"]
_SCENE_AUDIO_TYPES = frozenset(("ambience", "sound_effect", "music"))
_SOUND_EFFECT_CUES = re.compile(
    r"\b(?:footsteps?|steps?|running|run|door|slam|explosion|blast|"
    r"vehicle|car|truck|engine|impact|crash|gunfire|knock|splash)\b",
    re.IGNORECASE,
)
_AMBIENCE_CUES = re.compile(
    r"\b(?:rain|rainfall|wind|storm|city|urban|street|forest|woods|"
    r"crowd|market|ocean|sea|waves|river|birds|traffic|room tone)\b",
    re.IGNORECASE,
)
_MUSIC_CUES = re.compile(
    r"\b(?:dramatic|tense|suspense|romantic|emotional|hopeful|"
    r"melancholy|eerie|mysterious|uplifting|joyful)\b",
    re.IGNORECASE,
)


class SceneAudioOptions(BaseModel):
    """Optional controls for generating non-dialogue audio from a stored scene."""

    story_id: Optional[str] = None
    audio_type: Optional[SceneAudioType] = None
    duration_seconds: Optional[float] = Field(default=None, gt=0)
    seed: Optional[int] = Field(default=None, ge=0)

    @field_validator("duration_seconds")
    @classmethod
    def validate_finite_duration(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and not math.isfinite(value):
            raise ValueError("Scene audio duration must be finite.")
        return value


class AudioService:
    """Orchestrate scene-audio requests through the configured provider."""

    def __init__(self, provider: Optional[BaseAudioProvider] = None):
        self.provider = provider if provider is not None else get_audio_provider()

    async def generate_audio(
        self,
        request: Union[AudioGenerationRequest, Dict[str, Any]],
    ) -> AudioGenerationResult:
        """Validate and delegate one scene-audio request."""
        validated_request = AudioGenerationRequest.model_validate(request)
        request_values = validated_request.model_dump()
        request_values.update(
            {
                "output_dir": (
                    validated_request.output_dir or settings.AUDIO_OUTPUT_DIR
                ),
                "sample_rate": (
                    validated_request.sample_rate
                    or settings.AUDIO_DEFAULT_SAMPLE_RATE
                ),
                "channels": (
                    validated_request.channels or settings.AUDIO_DEFAULT_CHANNELS
                ),
                "output_format": (
                    validated_request.output_format or settings.AUDIO_DEFAULT_FORMAT
                ),
            }
        )
        validated_request = AudioGenerationRequest.model_validate(request_values)
        provider_name = self.provider.name

        try:
            if not self.provider.is_available():
                return AudioGenerationResult(
                    provider=provider_name,
                    status="failed",
                    audio_type=validated_request.audio_type,
                    metadata=validated_request.metadata.copy(),
                    error=f"Audio provider '{provider_name}' is unavailable.",
                )

            result = await self.provider.generate_audio(validated_request)
            if not isinstance(result, AudioGenerationResult):
                result = AudioGenerationResult.model_validate(result)
            return result
        except Exception as exc:
            logger.exception(
                "Audio provider '%s' failed to generate scene audio",
                provider_name,
            )
            return AudioGenerationResult(
                provider=provider_name,
                status="failed",
                audio_type=validated_request.audio_type,
                metadata=validated_request.metadata.copy(),
                error=f"Audio provider '{provider_name}' failed: {exc}",
            )

    @staticmethod
    def select_scene_audio_type(
        scene: Union[Scene, Dict[str, Any]],
        audio_type: Optional[str] = None,
    ) -> SceneAudioType:
        normalized_scene = AudioService._normalize_scene(scene)
        if audio_type is not None:
            if audio_type == "ambience":
                return "ambience"
            if audio_type == "sound_effect":
                return "sound_effect"
            if audio_type == "music":
                return "music"
            supported = ", ".join(sorted(_SCENE_AUDIO_TYPES))
            raise ValueError(f"Scene audio type must be one of [{supported}].")

        action = normalized_scene.action_description
        if _SOUND_EFFECT_CUES.search(action):
            return "sound_effect"

        environment = " ".join(
            filter(
                None,
                (
                    normalized_scene.heading,
                    normalized_scene.location,
                    normalized_scene.weather,
                    action,
                ),
            )
        )
        if _AMBIENCE_CUES.search(environment):
            return "ambience"
        if _MUSIC_CUES.search(normalized_scene.mood or ""):
            return "music"
        return "ambience"

    @staticmethod
    def build_scene_audio_prompt(
        scene: Union[Scene, Dict[str, Any]],
        audio_type: Optional[str] = None,
    ) -> str:
        normalized_scene = AudioService._normalize_scene(scene)
        selected_type = AudioService.select_scene_audio_type(
            normalized_scene,
            audio_type,
        )
        parts = ["cinematic"]
        if normalized_scene.mood:
            parts.append(f"{normalized_scene.mood.strip().lower()} atmosphere")
        if normalized_scene.location:
            parts.append(normalized_scene.location.strip())
        if normalized_scene.time_of_day:
            parts.append(f"at {normalized_scene.time_of_day.strip().lower()}")
        if normalized_scene.weather:
            parts.append(normalized_scene.weather.strip())
        if normalized_scene.action_description:
            parts.append(normalized_scene.action_description.strip())
        if normalized_scene.dialogues:
            parts.append("subtle background audio beneath dialogue")
        parts.append(
            {
                "ambience": "environmental ambience",
                "sound_effect": "cinematic scene sound effects",
                "music": "cinematic musical atmosphere",
            }[selected_type]
        )
        return ", ".join(parts)

    async def generate_scene_audio(
        self,
        scene: Union[Scene, Dict[str, Any]],
        *,
        audio_type: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> AudioGenerationResult:
        """Build a deterministic prompt and generate audio for one validated scene."""
        normalized_scene = self._normalize_scene(scene)
        selected_type = self.select_scene_audio_type(normalized_scene, audio_type)
        prompt = self.build_scene_audio_prompt(normalized_scene, selected_type)
        duration = (
            duration_seconds
            if duration_seconds is not None
            else normalized_scene.duration_seconds
        )
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Scene audio duration must be finite and greater than zero.")
        if seed is not None and seed < 0:
            raise ValueError("Scene audio seed must be non-negative.")
        selected_seed = seed if seed is not None else normalized_scene.scene_number
        request = AudioGenerationRequest(
            scene_id=(
                str(normalized_scene.id)
                if normalized_scene.id is not None
                else f"{normalized_scene.story_id or 'scene'}:"
                f"{normalized_scene.scene_number}"
            ),
            prompt=prompt,
            audio_type=selected_type,
            duration_seconds=duration,
            seed=selected_seed,
            metadata={
                "scene_number": normalized_scene.scene_number,
                "story_id": normalized_scene.story_id,
                "scene_location": normalized_scene.location,
                "scene_time_of_day": normalized_scene.time_of_day,
                "scene_weather": normalized_scene.weather,
                "scene_mood": normalized_scene.mood,
                "dialogue_count": len(normalized_scene.dialogues),
            },
        )
        result = await self.generate_audio(request)
        metadata = dict(result.metadata)
        metadata.setdefault("prompt", prompt)
        metadata.setdefault("seed", selected_seed)
        metadata.setdefault("ai_generated", False)
        metadata.update(request.metadata)
        return result.model_copy(update={"metadata": metadata})

    @staticmethod
    def _normalize_scene(scene: Union[Scene, Dict[str, Any]]) -> Scene:
        if isinstance(scene, Scene):
            return scene
        try:
            return Scene.model_validate(scene)
        except ValidationError as exc:
            raise ValueError("Invalid scene data for audio generation.") from exc
