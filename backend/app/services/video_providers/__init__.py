from .base import (
    BaseVideoProvider,
    InvalidVideoProviderError,
    ProviderExecutionError,
    VideoGenerationRequest,
    VideoGenerationResult,
    VideoProviderError,
    get_video_provider,
    list_video_providers,
    register_video_provider,
)
from .cogvideox import CogVideoXVideoProvider
from .placeholder import PlaceholderVideoProvider

register_video_provider("placeholder", PlaceholderVideoProvider)
register_video_provider("cogvideox", CogVideoXVideoProvider)

__all__ = [
    "BaseVideoProvider",
    "CogVideoXVideoProvider",
    "InvalidVideoProviderError",
    "PlaceholderVideoProvider",
    "ProviderExecutionError",
    "VideoGenerationRequest",
    "VideoGenerationResult",
    "VideoProviderError",
    "get_video_provider",
    "list_video_providers",
    "register_video_provider",
]
