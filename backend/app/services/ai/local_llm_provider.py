from typing import Dict, Any, Optional
from .base import BaseStoryLLMProvider, ProviderNotAvailableError


class LocalLLMStoryProvider(BaseStoryLLMProvider):
    """
    Local LLM Provider interface and stub.
    Prepared for a future runtime-independent local inference implementation.
    Currently inactive by design; raises ProviderNotAvailableError when triggered.
    """

    provider_name: str = "local_llm"

    def __init__(self, endpoint_url: Optional[str] = None, model_name: Optional[str] = None):
        self.endpoint_url = endpoint_url
        self.model_name = model_name
        self._is_active = False

    def is_available(self) -> bool:
        """
        Check if local LLM service is running and accessible.
        Remains False until Step 2 activation to enforce offline rule-based default.
        """
        return self._is_active

    async def analyze(
        self,
        prompt: str,
        genre: str = "Cinematic Drama",
        target_duration_minutes: int = 1,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Execute story analysis via local LLM.
        Raises ProviderNotAvailableError until Step 2 local LLM setup is approved.
        """
        if not self.is_available():
            raise ProviderNotAvailableError(
                "Local LLM provider ('local_llm') is not yet activated. "
                "Use the 'rule_based' provider until local inference is explicitly enabled."
            )

        # Stub implementation ready for Step 2 activation
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(prompt, genre, target_duration_minutes)
        return await self._call_local_llm_inference(system_prompt, user_prompt)

    def extract_characters(self, prompt: str, genre: str) -> list:
        self._raise_unavailable()

    def generate_scenes(
        self,
        prompt: str,
        characters: list,
        duration_minutes: int,
        genre: str,
    ) -> list:
        self._raise_unavailable()

    def get_structured_output_schema(self) -> Dict[str, Any]:
        return {
            "title": "string",
            "logline": "string",
            "synopsis": "string",
            "theme": "string",
            "tone": "string",
            "genre": "string",
            "characters": "list",
            "scenes": "list",
        }

    def _raise_unavailable(self) -> None:
        raise ProviderNotAvailableError(
            "Local LLM provider ('local_llm') is not activated. "
            "Use the 'rule_based' provider."
        )

    def _build_system_prompt(self) -> str:
        """Build structured cinematic analysis system prompt for local LLM."""
        return (
            "You are an expert Hollywood Screenwriter and Film Director AI. "
            "Analyze story premises and return valid JSON adhering strictly to the schema: "
            "title, logline, synopsis, theme, tone, characters (list), scenes (list)."
        )

    def _build_user_prompt(self, premise: str, genre: str, duration: int) -> str:
        """Format the narrative input prompt for the model."""
        return (
            f"GENRE: {genre}\n"
            f"TARGET DURATION: {duration} minute(s)\n"
            f"PREMISE:\n{premise}\n\n"
            "Produce structured cinematic JSON breakdown."
        )

    async def _call_local_llm_inference(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Future runtime-independent inference hook; no runtime is invoked by this provider."""
        self._raise_unavailable()
