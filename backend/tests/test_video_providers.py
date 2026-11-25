import pytest

from app.config import settings
from app.services.video import VideoService
from app.services.video_providers import (
    BaseVideoProvider,
    InvalidVideoProviderError,
    PlaceholderVideoProvider,
    VideoGenerationRequest,
    VideoGenerationResult,
    get_video_provider,
    list_video_providers,
)


def make_request(**overrides):
    values = {
        "scene_id": "scene-1",
        "story_id": "story-1",
        "prompt": "A quiet scene at sunrise.",
        "duration_seconds": 5,
        "width": 1280,
        "height": 720,
        "fps": 24,
        "seed": 12,
        "metadata": {"source": "test"},
    }
    values.update(overrides)
    return VideoGenerationRequest(**values)


def test_placeholder_provider_is_registered_and_defaulted():
    assert "placeholder" in list_video_providers()
    assert settings.VIDEO_PROVIDER == "placeholder"
    assert isinstance(get_video_provider(), PlaceholderVideoProvider)
    assert get_video_provider("placeholder").name == "placeholder"


def test_unknown_video_provider_raises_application_error():
    with pytest.raises(InvalidVideoProviderError, match="Unknown video provider"):
        get_video_provider("not-registered")


@pytest.mark.anyio
async def test_placeholder_returns_structured_result_without_creating_video(tmp_path):
    provider = PlaceholderVideoProvider()
    result = await provider.generate_video(
        make_request(metadata={"test_dir": str(tmp_path)})
    )

    assert result.provider == "placeholder"
    assert result.status == "placeholder"
    assert result.output_path is None
    assert result.duration_seconds is None
    assert result.width is None
    assert result.height is None
    assert result.fps is None
    assert result.format is None
    assert result.runtime_provider is None
    assert result.model is None
    assert result.inference_time_seconds is None
    assert result.metadata == {"test_dir": str(tmp_path)}
    assert result.error == "Video generation provider is not configured."
    assert list(tmp_path.iterdir()) == []


def test_video_service_resolves_the_configured_placeholder():
    service = VideoService()

    assert isinstance(service.video_provider, PlaceholderVideoProvider)


@pytest.mark.anyio
async def test_video_service_delegates_to_placeholder_provider():
    service = VideoService()
    request = make_request()

    result = await service.generate_scene_video(request)

    assert result.provider == "placeholder"
    assert result.status == "placeholder"
    assert result.metadata == request.metadata
    assert result.error == "Video generation provider is not configured."


class UnavailableVideoProvider(BaseVideoProvider):
    name = "unavailable"

    def is_available(self) -> bool:
        return False

    async def generate_video(
        self,
        request: VideoGenerationRequest,
    ) -> VideoGenerationResult:
        raise AssertionError("Unavailable providers must not be called.")


class FailingVideoProvider(BaseVideoProvider):
    name = "failing"

    def is_available(self) -> bool:
        return True

    async def generate_video(
        self,
        request: VideoGenerationRequest,
    ) -> VideoGenerationResult:
        raise RuntimeError("test generation failure")


@pytest.mark.anyio
async def test_video_service_returns_structured_result_for_unavailable_provider():
    service = VideoService(video_provider=UnavailableVideoProvider())

    result = await service.generate_scene_video(make_request())

    assert result.provider == "unavailable"
    assert result.status == "failed"
    assert "is unavailable" in result.error


@pytest.mark.anyio
async def test_video_service_converts_provider_errors_to_structured_failure():
    service = VideoService(video_provider=FailingVideoProvider())

    result = await service.generate_scene_video(make_request())

    assert result.provider == "failing"
    assert result.status == "failed"
    assert "test generation failure" in result.error
