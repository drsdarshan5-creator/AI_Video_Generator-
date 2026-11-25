from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, Optional, Type

from pydantic import BaseModel, Field


ImageStatus = Literal["generated", "placeholder", "failed"]


class ImageGenerationResult(BaseModel):
    """Provider-neutral result for one storyboard image request."""

    image_path: Optional[str] = None
    image_url: Optional[str] = None
    provider_name: str
    status: ImageStatus
    prompt_used: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class ImageProviderError(Exception):
    """Raised for image-provider configuration or execution errors."""

    pass


class InvalidImageProviderError(ImageProviderError):
    """Raised when an image provider name is unknown or invalid."""

    pass


class BaseImageProvider(ABC):
    """Runtime-neutral interface for storyboard image generation backends."""

    provider_name: str = "base"

    @abstractmethod
    def is_available(self) -> bool:
        """Return whether this provider is ready to generate images."""

        pass

    def get_status(self) -> Dict[str, Any]:
        """Return a structured availability summary."""
        return {
            "provider": self.provider_name,
            "available": self.is_available(),
            "status": "available" if self.is_available() else "unavailable",
        }

    @abstractmethod
    async def generate_image(
        self,
        prompt: str,
        metadata: Dict[str, Any],
    ) -> ImageGenerationResult:
        """Generate one image and return structured status and provenance."""

        pass


_IMAGE_PROVIDER_REGISTRY: Dict[str, Type[BaseImageProvider]] = {}


def register_image_provider(
    name: str,
    provider_cls: Type[BaseImageProvider],
) -> None:
    """Register a provider class by its configuration name."""
    normalized_name = name.strip().lower()
    if not normalized_name:
        raise InvalidImageProviderError("Image provider name cannot be empty.")
    if not isinstance(provider_cls, type) or not issubclass(
        provider_cls,
        BaseImageProvider,
    ):
        raise InvalidImageProviderError(
            f"Image provider '{normalized_name}' must implement BaseImageProvider."
        )
    _IMAGE_PROVIDER_REGISTRY[normalized_name] = provider_cls


def get_image_provider(name: Optional[str] = None) -> BaseImageProvider:
    """Instantiate a registered provider, defaulting to configured IMAGE_PROVIDER."""
    from ...config import settings

    target_name = (name or settings.IMAGE_PROVIDER).strip().lower()
    if target_name not in _IMAGE_PROVIDER_REGISTRY:
        available = ", ".join(sorted(_IMAGE_PROVIDER_REGISTRY))
        raise InvalidImageProviderError(
            f"Unknown image provider '{target_name}'. "
            f"Available image providers: [{available}]"
        )
    return _IMAGE_PROVIDER_REGISTRY[target_name]()


def list_image_providers() -> list:
    """Return registered image provider names."""
    return sorted(_IMAGE_PROVIDER_REGISTRY)
