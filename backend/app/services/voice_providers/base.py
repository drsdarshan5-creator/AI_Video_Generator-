from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Type

from pydantic import BaseModel, Field, field_validator


VoiceStatus = Literal["generated", "placeholder", "failed"]


class VoiceGenerationRequest(BaseModel):
    """Provider-neutral input for synthesizing one voice segment."""

    text: str = Field(..., min_length=1)
    character_id: Optional[str] = None
    character_name: Optional[str] = None
    voice_name: Optional[str] = None
    voice_id: Optional[str] = None
    language: Optional[str] = None
    speaking_style: Optional[str] = None
    output_path: Optional[Path] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Speech text cannot be blank.")
        return value.strip()


class VoiceGenerationResult(BaseModel):
    """Structured result for one voice-generation request."""

    provider: str
    status: VoiceStatus
    output_path: Optional[Path] = None
    duration_seconds: Optional[float] = None
    sample_rate: Optional[int] = None
    format: Optional[str] = None
    voice_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    message: Optional[str] = None


class VoiceProviderError(Exception):
    """Raised for voice-provider configuration or execution errors."""


class InvalidVoiceProviderError(VoiceProviderError):
    """Raised when a configured voice provider is unknown or invalid."""


class BaseVoiceProvider(ABC):
    """Runtime-neutral interface for voice-generation backends."""

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
    async def generate_speech(
        self,
        request: VoiceGenerationRequest,
    ) -> VoiceGenerationResult:
        """Generate speech and return structured provenance."""


_VOICE_PROVIDER_REGISTRY: Dict[str, Type[BaseVoiceProvider]] = {}


def register_voice_provider(
    name: str,
    provider_cls: Type[BaseVoiceProvider],
) -> None:
    """Register a provider class by its configuration name."""
    normalized_name = name.strip().lower()
    if not normalized_name:
        raise InvalidVoiceProviderError("Voice provider name cannot be empty.")
    if not isinstance(provider_cls, type) or not issubclass(
        provider_cls,
        BaseVoiceProvider,
    ):
        raise InvalidVoiceProviderError(
            f"Voice provider '{normalized_name}' must implement BaseVoiceProvider."
        )
    _VOICE_PROVIDER_REGISTRY[normalized_name] = provider_cls


def get_voice_provider(name: Optional[str] = None) -> BaseVoiceProvider:
    """Instantiate a registered provider, defaulting to configured VOICE_PROVIDER."""
    from ...config import settings

    target_name = (name or settings.VOICE_PROVIDER).strip().lower()
    if target_name not in _VOICE_PROVIDER_REGISTRY:
        available = ", ".join(sorted(_VOICE_PROVIDER_REGISTRY))
        raise InvalidVoiceProviderError(
            f"Unknown voice provider '{target_name}'. "
            f"Available voice providers: [{available}]"
        )
    return _VOICE_PROVIDER_REGISTRY[target_name]()


def list_voice_providers() -> list:
    return sorted(_VOICE_PROVIDER_REGISTRY)
