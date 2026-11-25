from .base import (
    BaseVideoProvider,
    VideoGenerationRequest,
    VideoGenerationResult,
)


class PlaceholderVideoProvider(BaseVideoProvider):
    """Safe no-video provider used until a real generation backend is configured."""

    name = "placeholder"

    def is_available(self) -> bool:
        return True

    async def generate_video(
        self,
        request: VideoGenerationRequest,
    ) -> VideoGenerationResult:
        return VideoGenerationResult(
            provider=self.name,
            status="placeholder",
            metadata=request.metadata.copy(),
            error="Video generation provider is not configured.",
        )
