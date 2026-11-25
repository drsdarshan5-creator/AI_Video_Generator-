from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Type

from pydantic import BaseModel, Field, field_validator


VideoStatus = Literal["generated", "placeholder", "failed"]


class VideoGenerationRequest(BaseModel):
    """Provider-neutral input for generating a video segment for one scene."""

    scene_id: Optional[str] = None
    story_id: Optional[str] = None
    prompt: str = Field(..., min_length=1)
    image_path: Optional[Path] = None
    image_url: Optional[str] = None
    duration_seconds: Optional[float] = Field(default=None, gt=0)
    width: Optional[int] = Field(default=None, gt=0)
    height: Optional[int] = Field(default=None, gt=0)
    fps: Optional[float] = Field(default=None, gt=0)
    output_path: Optional[Path] = None
    seed: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Video prompt cannot be blank.")
        return value.strip()


class VideoGenerationResult(BaseModel):
    """Structured result and provenance for one video-generation request."""

    provider: str
    status: VideoStatus
    output_path: Optional[Path] = None
    duration_seconds: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    format: Optional[str] = None
    runtime_provider: Optional[str] = None
    model: Optional[str] = None
    inference_time_seconds: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class VideoProviderError(Exception):
    """Raised for video-provider configuration or execution errors."""


class InvalidVideoProviderError(VideoProviderError):
    """Raised when a configured video provider is unknown or invalid."""


class ProviderExecutionError(VideoProviderError):
    """Raised when a video provider cannot execute a generation request."""


class BaseVideoProvider(ABC):
    """Runtime-neutral interface for scene-video generation backends."""

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
    async def generate_video(
        self,
        request: VideoGenerationRequest,
    ) -> VideoGenerationResult:
        """Generate one scene video and return structured provenance."""


_VIDEO_PROVIDER_REGISTRY: Dict[str, Type[BaseVideoProvider]] = {}


def register_video_provider(
    name: str,
    provider_cls: Type[BaseVideoProvider],
) -> None:
    """Register a video provider class by its configuration name."""
    normalized_name = name.strip().lower()
    if not normalized_name:
        raise InvalidVideoProviderError("Video provider name cannot be empty.")
    if not isinstance(provider_cls, type) or not issubclass(
        provider_cls,
        BaseVideoProvider,
    ):
        raise InvalidVideoProviderError(
            f"Video provider '{normalized_name}' must implement BaseVideoProvider."
        )
    _VIDEO_PROVIDER_REGISTRY[normalized_name] = provider_cls


def get_video_provider(name: Optional[str] = None) -> BaseVideoProvider:
    """Instantiate a registered provider, defaulting to configured VIDEO_PROVIDER."""
    from ...config import settings

    target_name = (name or settings.VIDEO_PROVIDER).strip().lower()
    if target_name not in _VIDEO_PROVIDER_REGISTRY:
        available = ", ".join(sorted(_VIDEO_PROVIDER_REGISTRY))
        raise InvalidVideoProviderError(
            f"Unknown video provider '{target_name}'. "
            f"Available video providers: [{available}]"
        )
    return _VIDEO_PROVIDER_REGISTRY[target_name]()


def list_video_providers() -> list[str]:
    """Return registered video provider names."""
    return sorted(_VIDEO_PROVIDER_REGISTRY)
