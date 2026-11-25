from .base import (
    AudioGenerationRequest,
    AudioGenerationResult,
    BaseAudioProvider,
)


class PlaceholderAudioProvider(BaseAudioProvider):
    """Safe no-audio provider used until a real generation backend is configured."""

    name = "placeholder"

    def is_available(self) -> bool:
        return True

    async def generate_audio(
        self,
        request: AudioGenerationRequest,
    ) -> AudioGenerationResult:
        return AudioGenerationResult(
            provider=self.name,
            status="placeholder",
            audio_type=request.audio_type,
            metadata={
                "generation_performed": False,
                "external_api_used": False,
            },
            error="Audio generation provider is not configured; no audio file was created.",
        )
