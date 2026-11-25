import pytest

from app.config import settings
from app.services.ai import (
    BaseStoryLLMProvider,
    InvalidProviderError,
    LocalLLMStoryProvider,
    ProviderExecutionError,
    ProviderNotAvailableError,
    RuleBasedStoryProvider,
    get_provider,
    list_available_providers,
    validate_structured_output,
)


def test_default_provider_remains_rule_based():
    assert settings.STORY_ANALYZER_PROVIDER == "rule_based"
    assert get_provider().provider_name == "rule_based"


def test_provider_registry_contains_expected_providers():
    providers = list_available_providers()
    assert "rule_based" in providers
    assert "local_llm" in providers


def test_rule_based_provider_satisfies_runtime_neutral_interface():
    provider = RuleBasedStoryProvider()
    assert isinstance(provider, BaseStoryLLMProvider)
    assert not getattr(RuleBasedStoryProvider, "__abstractmethods__", set())
    assert provider.is_available() is True
    assert isinstance(provider.get_structured_output_schema(), dict)


@pytest.mark.anyio
async def test_rule_based_provider_returns_structured_analysis():
    provider = RuleBasedStoryProvider()
    result = await provider.analyze(
        prompt="A cybernetic detective searches a flooded city for a missing inventor.",
        genre="Cyberpunk Mystery",
        target_duration_minutes=2,
    )

    assert validate_structured_output(result)["provider"] == "rule_based"
    assert len(result["characters"]) >= 2
    assert len(result["scenes"]) >= 3
    assert result["title"]
    assert result["logline"]


def test_rule_based_character_and_scene_interface_methods():
    provider = RuleBasedStoryProvider()
    characters = provider.extract_characters(
        "A scientist discovers an impossible signal.",
        "Sci-Fi",
    )
    scenes = provider.generate_scenes(
        "A scientist discovers an impossible signal.",
        characters,
        2,
        "Sci-Fi",
    )

    assert len(characters) >= 1
    assert len(scenes) >= 1
    assert scenes[0]["scene_number"] == 1


@pytest.mark.anyio
async def test_local_provider_is_cleanly_unavailable():
    provider = LocalLLMStoryProvider()
    assert provider.is_available() is False

    with pytest.raises(ProviderNotAvailableError, match="rule_based"):
        await provider.analyze(prompt="A valid story premise")

    with pytest.raises(ProviderNotAvailableError, match="rule_based"):
        provider.extract_characters("A valid story premise", "Drama")


def test_invalid_provider_configuration_is_rejected():
    with pytest.raises(InvalidProviderError, match="Unknown story analyzer provider"):
        get_provider("non_existent_provider")


def test_structured_output_validation_rejects_malformed_provider_output():
    with pytest.raises(ProviderExecutionError, match="missing required fields"):
        validate_structured_output({"title": "Incomplete"})

