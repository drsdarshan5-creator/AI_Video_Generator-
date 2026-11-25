from .base import (
    BaseVoiceProvider,
    VoiceGenerationRequest,
    VoiceGenerationResult,
)


class PlaceholderVoiceProvider(BaseVoiceProvider):
    """Safe no-audio provider used until a real TTS backend is configured."""

    name = "placeholder"

    def is_available(self) -> bool:
        return True

    async def generate_speech(
        self,
        request: VoiceGenerationRequest,
    ) -> VoiceGenerationResult:
        return VoiceGenerationResult(
            provider=self.name,
            status="placeholder",
            voice_id=request.voice_id,
            message=(
                "Real TTS generation is not configured; no audio file was created."
            ),
        )
