from .base import (
    BaseVoiceProvider,
    InvalidVoiceProviderError,
    VoiceGenerationRequest,
    VoiceGenerationResult,
    VoiceProviderError,
    get_voice_provider,
    list_voice_providers,
    register_voice_provider,
)
from .placeholder import PlaceholderVoiceProvider
from .piper_onnx import PiperONNXVoiceProvider

register_voice_provider("placeholder", PlaceholderVoiceProvider)
register_voice_provider("piper_onnx", PiperONNXVoiceProvider)

__all__ = [
    "BaseVoiceProvider",
    "InvalidVoiceProviderError",
    "PlaceholderVoiceProvider",
    "PiperONNXVoiceProvider",
    "VoiceGenerationRequest",
    "VoiceGenerationResult",
    "VoiceProviderError",
    "get_voice_provider",
    "list_voice_providers",
    "register_voice_provider",
]
