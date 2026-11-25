from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypedDict, Type


class StructuredDialogueOutput(TypedDict, total=False):
    character_name: str
    text: str
    audio_url: Optional[str]


class StructuredCharacterOutput(TypedDict, total=False):
    id: str
    name: str
    role: str
    description: str
    personality: Optional[str]
    appearance_description: Optional[str]
    age_range: Optional[str]
    clothing_style: Optional[str]
    visual_prompt: str
    voice_name: str
    voice_id: Optional[str]


class StructuredSceneOutput(TypedDict, total=False):
    scene_number: int
    heading: str
    location: Optional[str]
    time_of_day: Optional[str]
    mood: Optional[str]
    lighting: Optional[str]
    weather: Optional[str]
    action_description: str
    visual_prompt: str
    camera_movement: str
    shot_type: str
    duration_seconds: int
    dialogues: List[StructuredDialogueOutput]
    image_url: Optional[str]
    video_url: Optional[str]


class StructuredStoryOutput(TypedDict, total=False):
    title: str
    logline: str
    synopsis: Optional[str]
    theme: Optional[str]
    tone: Optional[str]
    genre: str
    provider: str
    characters: List[StructuredCharacterOutput]
    scenes: List[StructuredSceneOutput]


class StoryAnalysisError(Exception):
    """Base exception for story analysis failures."""

    pass


class ProviderNotAvailableError(StoryAnalysisError):
    """Raised when the requested AI provider is not installed or ready."""

    pass


class InvalidProviderError(StoryAnalysisError):
    """Raised when an unknown provider name is requested."""

    pass


class ProviderExecutionError(StoryAnalysisError):
    """Raised when an available provider fails while producing structured output."""

    pass


class BaseStoryLLMProvider(ABC):
    """Runtime-neutral interface for story analysis providers."""

    provider_name: str = "base"

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is ready to process requests."""

        pass

    @abstractmethod
    async def analyze(
        self,
        prompt: str,
        genre: str = "Cinematic Drama",
        target_duration_minutes: int = 1,
        **kwargs: Any,
    ) -> StructuredStoryOutput:
        """Analyze a premise into a structured story breakdown."""

        pass

    @abstractmethod
    def extract_characters(
        self,
        prompt: str,
        genre: str,
    ) -> List[StructuredCharacterOutput]:
        """Extract or generate the structured character list."""

        pass

    @abstractmethod
    def generate_scenes(
        self,
        prompt: str,
        characters: List[StructuredCharacterOutput],
        duration_minutes: int,
        genre: str,
    ) -> List[StructuredSceneOutput]:
        """Generate the structured screenplay scene breakdown."""

        pass

    @abstractmethod
    def get_structured_output_schema(self) -> Dict[str, Any]:
        """Return the provider-neutral structured output contract."""

        pass


_PROVIDER_REGISTRY: Dict[str, Type[BaseStoryLLMProvider]] = {}


def register_provider(name: str, provider_cls: Type[BaseStoryLLMProvider]) -> None:
    """Register a story analyzer provider class."""
    normalized_name = name.lower().strip()
    if not normalized_name:
        raise InvalidProviderError("Provider name cannot be empty.")
    if not issubclass(provider_cls, BaseStoryLLMProvider):
        raise InvalidProviderError(
            f"Provider '{normalized_name}' must implement BaseStoryLLMProvider."
        )
    _PROVIDER_REGISTRY[normalized_name] = provider_cls


def get_provider(name: Optional[str] = None) -> BaseStoryLLMProvider:
    """Retrieve a provider instance, defaulting to rule_based."""
    target_name = (name or "rule_based").lower().strip()
    if target_name not in _PROVIDER_REGISTRY:
        available = ", ".join(list_available_providers())
        raise InvalidProviderError(
            f"Unknown story analyzer provider '{target_name}'. "
            f"Available providers: [{available}]"
        )
    return _PROVIDER_REGISTRY[target_name]()


def list_available_providers() -> List[str]:
    """Return registered provider names."""
    return list(_PROVIDER_REGISTRY.keys())


def validate_structured_output(output: Any) -> StructuredStoryOutput:
    """Validate fields required by the application from every provider."""
    if not isinstance(output, dict):
        raise ProviderExecutionError("Provider returned invalid structured output.")

    missing = [
        field
        for field in ("title", "logline", "characters", "scenes")
        if field not in output
    ]
    if missing:
        raise ProviderExecutionError(
            f"Provider output is missing required fields: {', '.join(missing)}"
        )
    if not isinstance(output["characters"], list) or not isinstance(output["scenes"], list):
        raise ProviderExecutionError("Provider characters and scenes must be lists.")

    return output
