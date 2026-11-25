from pathlib import Path
from typing import Any, Dict, Optional

from .voice_providers import (
    BaseVoiceProvider,
    VoiceGenerationRequest,
    VoiceGenerationResult,
    get_voice_provider,
)


class VoiceService:
    """Orchestrate voice requests through the configured provider."""

    def __init__(self, provider: Optional[BaseVoiceProvider] = None):
        self.provider = provider or get_voice_provider()

    async def generate_speech(
        self,
        character_name: str,
        dialogue_text: str,
        voice_id: Optional[str] = None,
        *,
        character_id: Optional[str] = None,
        voice_name: Optional[str] = None,
        language: Optional[str] = None,
        speaking_style: Optional[str] = None,
        output_path: Optional[Path] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> VoiceGenerationResult:
        request = VoiceGenerationRequest(
            text=dialogue_text,
            character_id=character_id,
            character_name=character_name,
            voice_name=voice_name,
            voice_id=voice_id,
            language=language,
            speaking_style=speaking_style,
            output_path=output_path,
            metadata=metadata or {},
        )
        if not self.provider.is_available():
            return VoiceGenerationResult(
                provider=self.provider.name,
                status="failed",
                voice_id=voice_id,
                message=f"Voice provider '{self.provider.name}' is unavailable.",
            )
        return await self.provider.generate_speech(request)
