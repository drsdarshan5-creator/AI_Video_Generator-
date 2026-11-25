from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Type

from pydantic import BaseModel, Field, model_validator


AudioType = Literal[
    "dialogue",
    "narration",
    "music",
    "procedural_music",
    "ambience",
    "sound_effect",
    "mixed",
]
AudioStatus = Literal["generated", "placeholder", "failed"]


class AudioGenerationRequest(BaseModel):
    """Provider-neutral request for generating audio associated with a scene."""

    scene_id: Optional[str] = None
    prompt: Optional[str] = None
    text: Optional[str] = None
    description: Optional[str] = None
    duration_seconds: Optional[float] = Field(default=None, gt=0)
    audio_type: AudioType = "mixed"
    voice_name: Optional[str] = None
    voice_id: Optional[str] = None
    reference_audio_path: Optional[Path] = None
    reference_audio_url: Optional[str] = None
    sample_rate: Optional[int] = Field(default=None, gt=0)
    channels: Optional[int] = Field(default=None, gt=0)
    music_mood: Optional[str] = None
    intensity: Optional[float] = Field(default=None, ge=0, le=1)
    tempo_bpm: Optional[float] = Field(default=None, gt=0, le=300)
    seed: Optional[int] = None
    output_format: Optional[str] = None
    output_dir: Optional[Path] = None
    output_path: Optional[Path] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_content(self) -> "AudioGenerationRequest":
        if not any(
            value and value.strip()
            for value in (self.prompt, self.text, self.description)
        ):
            raise ValueError(
                "Audio request must include text or a description; a prompt is also accepted."
            )
        if self.output_format is not None and not self.output_format.strip():
            raise ValueError("Audio output format cannot be blank.")
        return self


class AudioGenerationResult(BaseModel):
    """Structured result and provenance for one audio-generation request."""

    provider: str
    status: AudioStatus
    audio_type: Optional[AudioType] = None
    model: Optional[str] = None
    output_path: Optional[Path] = None
    duration_seconds: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    format: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class AudioProviderError(Exception):
    """Raised for audio-provider configuration or execution errors."""


class InvalidAudioProviderError(AudioProviderError):
    """Raised when an audio provider name is unknown or invalid."""


class AudioProviderExecutionError(AudioProviderError):
    """Raised when an audio provider cannot execute a generation request."""


class BaseAudioProvider(ABC):
    """Runtime-neutral interface for scene-audio generation backends."""

    name = "base"

    @abstractmethod
    def is_available(self) -> bool:
        """Return whether this provider can handle a request."""

    def get_status(self) -> Dict[str, Any]:
        available = self.is_available()
        return {
            "provider": self.name,
            "available": available,
            "status": "available" if available else "unavailable",
        }

    @abstractmethod
    async def generate_audio(
        self,
        request: AudioGenerationRequest,
    ) -> AudioGenerationResult:
        """Generate scene audio and return structured provenance."""


_AUDIO_PROVIDER_REGISTRY: Dict[str, Type[BaseAudioProvider]] = {}


def register_audio_provider(
    name: str,
    provider_cls: Type[BaseAudioProvider],
) -> None:
    """Register an audio provider class by its configuration name."""
    normalized_name = name.strip().lower()
    if not normalized_name:
        raise InvalidAudioProviderError("Audio provider name cannot be empty.")
    if not isinstance(provider_cls, type) or not issubclass(
        provider_cls,
        BaseAudioProvider,
    ):
        raise InvalidAudioProviderError(
            f"Audio provider '{normalized_name}' must implement BaseAudioProvider."
        )
    _AUDIO_PROVIDER_REGISTRY[normalized_name] = provider_cls


def get_audio_provider(name: Optional[str] = None) -> BaseAudioProvider:
    """Instantiate a registered provider, defaulting to configured AUDIO_PROVIDER."""
    from ...config import settings

    target_name = (name or settings.AUDIO_PROVIDER).strip().lower()
    if target_name not in _AUDIO_PROVIDER_REGISTRY:
        available = ", ".join(sorted(_AUDIO_PROVIDER_REGISTRY))
        raise InvalidAudioProviderError(
            f"Unknown audio provider '{target_name}'. "
            f"Available audio providers: [{available}]"
        )
    return _AUDIO_PROVIDER_REGISTRY[target_name]()


def list_audio_providers() -> list[str]:
    """Return registered audio provider names."""
    return sorted(_AUDIO_PROVIDER_REGISTRY)
