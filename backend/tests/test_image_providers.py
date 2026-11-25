import pytest
from pathlib import Path

from app.models import Scene
from app.services.image_providers import (
    BaseImageProvider,
    InvalidImageProviderError,
    PlaceholderImageProvider,
    get_image_provider,
    list_image_providers,
    register_image_provider,
)
from app.services.storyboard import StoryboardService


def make_scene():
    return Scene(
        scene_number=1,
        heading="INT. ARCHIVE - NIGHT",
        location="Archive",
        mood="Quiet suspense",
        lighting="Low amber light",
        action_description="An archivist opens a sealed case.",
        visual_prompt="Stored prompt",
        duration_seconds=7,
    )


def test_image_provider_interface_and_default_registry():
    provider = get_image_provider()

    assert isinstance(provider, BaseImageProvider)
    assert isinstance(provider, PlaceholderImageProvider)
    assert provider.is_available() is True
    assert provider.get_status() == {
        "provider": "placeholder",
        "available": True,
        "status": "available",
    }
    assert "placeholder" in list_image_providers()


def test_invalid_image_provider_name_is_rejected(monkeypatch):
    with pytest.raises(InvalidImageProviderError, match="Unknown image provider"):
        get_image_provider("does-not-exist")

    from app.config import settings

    monkeypatch.setattr(settings, "IMAGE_PROVIDER", "not-registered")
    with pytest.raises(InvalidImageProviderError, match="not-registered"):
        get_image_provider()


@pytest.mark.anyio
async def test_placeholder_provider_returns_structured_result(tmp_path):
    provider = PlaceholderImageProvider()
    provider.output_dir = tmp_path

    result = await provider.generate_image(
        "A deterministic test prompt.",
        {
            "scene_number": 1,
            "scene_heading": "INT. ARCHIVE - NIGHT",
            "shot_type": "Wide shot",
            "duration_seconds": 7,
            "action_description": "An archivist opens a sealed case.",
            "character_names": ["Archivist"],
        },
    )

    assert result.provider_name == "placeholder"
    assert result.status == "placeholder"
    assert result.prompt_used == "A deterministic test prompt."
    assert result.image_url.startswith("/generated/images/scene_1_")
    assert result.image_path
    assert (tmp_path / Path(result.image_path).name).exists()
    assert result.metadata["width"] > 0


@pytest.mark.anyio
async def test_invalid_placeholder_request_returns_failed_result():
    provider = PlaceholderImageProvider()

    result = await provider.generate_image("Prompt", {"scene_number": 0})

    assert result.status == "failed"
    assert result.image_url is None
    assert result.error and "positive scene_number" in result.error


class RaisingImageProvider(BaseImageProvider):
    provider_name = "raising-test"

    def is_available(self):
        return True

    async def generate_image(self, prompt, metadata):
        raise RuntimeError("simulated provider failure")


@pytest.mark.anyio
async def test_storyboard_service_contains_provider_failure_per_frame():
    service = StoryboardService(image_provider=RaisingImageProvider())
    frames = service.build_storyboard([make_scene()])

    rendered = await service.render_frames(frames)

    assert len(rendered) == 1
    assert rendered[0].status == "failed"
    assert rendered[0].image_url is None
    assert rendered[0].provider_name == "raising-test"
    assert "simulated provider failure" in rendered[0].generation_error


@pytest.mark.anyio
async def test_storyboard_service_integrates_placeholder_provider(tmp_path):
    provider = PlaceholderImageProvider()
    provider.output_dir = tmp_path
    service = StoryboardService(image_provider=provider)

    frames = await service.render_frames(service.build_storyboard([make_scene()]))

    assert frames[0].status == "placeholder"
    assert frames[0].provider_name == "placeholder"
    assert frames[0].image_path
    assert frames[0].image_url
    assert frames[0].generation_metadata["scene_heading"] == "INT. ARCHIVE - NIGHT"


def test_registry_rejects_non_provider_classes():
    with pytest.raises(InvalidImageProviderError, match="must implement"):
        register_image_provider("invalid-test", object)

