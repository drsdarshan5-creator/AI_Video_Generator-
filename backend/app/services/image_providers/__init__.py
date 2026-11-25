from .base import (
    BaseImageProvider,
    ImageGenerationResult,
    ImageProviderError,
    InvalidImageProviderError,
    get_image_provider,
    list_image_providers,
    register_image_provider,
)
from .placeholder import PlaceholderImageProvider
from .onnx_directml import OnnxDirectMLImageProvider

register_image_provider("placeholder", PlaceholderImageProvider)
register_image_provider("onnx_directml", OnnxDirectMLImageProvider)

__all__ = [
    "BaseImageProvider",
    "ImageGenerationResult",
    "ImageProviderError",
    "InvalidImageProviderError",
    "OnnxDirectMLImageProvider",
    "PlaceholderImageProvider",
    "get_image_provider",
    "list_image_providers",
    "register_image_provider",
]
