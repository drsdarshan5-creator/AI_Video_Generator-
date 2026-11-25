"""AI Model Provider module for Story Analysis and Generation."""
from .base import (
    BaseStoryLLMProvider,
    StoryAnalysisError,
    ProviderNotAvailableError,
    InvalidProviderError,
    ProviderExecutionError,
    StructuredStoryOutput,
    StructuredCharacterOutput,
    StructuredSceneOutput,
    StructuredDialogueOutput,
    register_provider,
    get_provider,
    list_available_providers,
    validate_structured_output,
)
from .rule_based_provider import RuleBasedStoryProvider
from .local_llm_provider import LocalLLMStoryProvider

# Register standard providers
register_provider("rule_based", RuleBasedStoryProvider)
register_provider("local_llm", LocalLLMStoryProvider)

__all__ = [
    "BaseStoryLLMProvider",
    "RuleBasedStoryProvider",
    "LocalLLMStoryProvider",
    "StoryAnalysisError",
    "ProviderNotAvailableError",
    "InvalidProviderError",
    "ProviderExecutionError",
    "StructuredStoryOutput",
    "StructuredCharacterOutput",
    "StructuredSceneOutput",
    "StructuredDialogueOutput",
    "register_provider",
    "get_provider",
    "list_available_providers",
    "validate_structured_output",
]
