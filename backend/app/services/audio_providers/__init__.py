from .base import (
    AudioGenerationRequest,
    AudioGenerationResult,
    AudioProviderError,
    AudioProviderExecutionError,
    AudioStatus,
    AudioType,
    BaseAudioProvider,
    InvalidAudioProviderError,
    get_audio_provider,
    list_audio_providers,
    register_audio_provider,
)
from .placeholder import PlaceholderAudioProvider
from .local_audio import LocalAudioProvider
from .ai_audio import AIAudioProvider

register_audio_provider("placeholder", PlaceholderAudioProvider)
register_audio_provider("local_audio", LocalAudioProvider)
register_audio_provider("ai_audio", AIAudioProvider)

__all__ = [
    "AudioGenerationRequest",
    "AudioGenerationResult",
    "AIAudioProvider",
    "AudioProviderError",
    "AudioProviderExecutionError",
    "AudioStatus",
    "AudioType",
    "BaseAudioProvider",
    "InvalidAudioProviderError",
    "LocalAudioProvider",
    "PlaceholderAudioProvider",
    "get_audio_provider",
    "list_audio_providers",
    "register_audio_provider",
]
