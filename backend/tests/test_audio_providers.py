import sys
import wave

import pytest
from pydantic import ValidationError

from app.config import settings
from app.services.audio import AudioService
from app.services.audio_providers import (
    AudioGenerationRequest,
    AudioGenerationResult,
    BaseAudioProvider,
    InvalidAudioProviderError,
    PlaceholderAudioProvider,
    get_audio_provider,
    list_audio_providers,
    register_audio_provider,
)


def make_request(**overrides):
    values = {
        "scene_id": "scene-1",
        "description": "Gentle rain over a city at night.",
        "duration_seconds": 4,
        "audio_type": "ambience",
        "output_format": "wav",
        "music_mood": "calm",
        "intensity": 0.3,
        "tempo_bpm": 80,
        "seed": 12,
        "metadata": {"source": "test"},
    }
    values.update(overrides)
    return AudioGenerationRequest(**values)


def test_audio_provider_interface_and_default_registry():
    provider = get_audio_provider()

    assert isinstance(provider, BaseAudioProvider)
    assert isinstance(provider, PlaceholderAudioProvider)
    assert provider.is_available()
    assert provider.get_status() == {
        "provider": "placeholder",
        "available": True,
        "status": "available",
    }
    assert "placeholder" in list_audio_providers()
    assert "local_audio" in list_audio_providers()
    assert settings.AUDIO_PROVIDER == "placeholder"


def test_registry_registration_and_lookup():
    class TestAudioProvider(PlaceholderAudioProvider):
        name = "test-audio"

    register_audio_provider("test-audio", TestAudioProvider)

    assert isinstance(get_audio_provider(" TEST-AUDIO "), TestAudioProvider)
    assert "test-audio" in list_audio_providers()


def test_unknown_audio_provider_raises_clear_error():
    with pytest.raises(InvalidAudioProviderError, match="Unknown audio provider"):
        get_audio_provider("not-registered")


@pytest.mark.parametrize(
    "values",
    [
        {"description": "   ", "text": None},
        {"description": None, "text": "  "},
        {"description": None, "text": None},
    ],
)
def test_request_requires_nonblank_text_or_description(values):
    with pytest.raises(ValidationError, match="text or a description"):
        make_request(**values)


@pytest.mark.parametrize(
    "values",
    [
        {"duration_seconds": 0},
        {"intensity": 1.1},
        {"tempo_bpm": 301},
        {"output_format": " "},
        {"audio_type": "unsupported"},
    ],
)
def test_request_rejects_invalid_values(values):
    with pytest.raises(ValidationError):
        make_request(**values)


@pytest.mark.anyio
async def test_placeholder_is_deterministic_and_does_not_create_audio(tmp_path):
    provider = PlaceholderAudioProvider()
    request = make_request(output_dir=tmp_path)

    first = await provider.generate_audio(request)
    second = await provider.generate_audio(request)

    assert first == second
    assert first.provider == "placeholder"
    assert first.status == "placeholder"
    assert first.output_path is None
    assert first.duration_seconds is None
    assert first.sample_rate is None
    assert first.channels is None
    assert first.format is None
    assert first.metadata == {
        "generation_performed": False,
        "external_api_used": False,
    }
    assert "no audio file was created" in first.error
    assert list(tmp_path.iterdir()) == []


def test_audio_service_resolves_configured_placeholder():
    assert isinstance(AudioService().provider, PlaceholderAudioProvider)


@pytest.mark.anyio
async def test_audio_service_validates_and_delegates_to_placeholder():
    request = make_request()
    result = await AudioService().generate_audio(request.model_dump())

    assert result.status == "placeholder"
    assert result.provider == "placeholder"
    assert result.error
    assert result.metadata["generation_performed"] is False


@pytest.mark.anyio
async def test_audio_service_applies_configured_output_defaults():
    class CapturingAudioProvider(PlaceholderAudioProvider):
        async def generate_audio(self, request):
            self.received_request = request
            return await super().generate_audio(request)

    provider = CapturingAudioProvider()
    await AudioService(provider).generate_audio(make_request())

    assert provider.received_request.output_dir == settings.AUDIO_OUTPUT_DIR
    assert provider.received_request.sample_rate == settings.AUDIO_DEFAULT_SAMPLE_RATE
    assert provider.received_request.channels == settings.AUDIO_DEFAULT_CHANNELS
    assert provider.received_request.output_format == settings.AUDIO_DEFAULT_FORMAT


class UnavailableAudioProvider(BaseAudioProvider):
    name = "unavailable"

    def is_available(self):
        return False

    async def generate_audio(self, request):
        raise AssertionError("Unavailable providers must not be called.")


class FailingAudioProvider(BaseAudioProvider):
    name = "failing"

    def is_available(self):
        return True

    async def generate_audio(self, request):
        raise RuntimeError("test generation failure")


@pytest.mark.anyio
async def test_audio_service_returns_structured_failure_for_unavailable_provider():
    result = await AudioService(UnavailableAudioProvider()).generate_audio(
        make_request()
    )

    assert result.provider == "unavailable"
    assert result.status == "failed"
    assert "is unavailable" in result.error


@pytest.mark.anyio
async def test_audio_service_converts_provider_errors_to_structured_failure():
    result = await AudioService(FailingAudioProvider()).generate_audio(make_request())

    assert result.provider == "failing"
    assert result.status == "failed"
    assert "test generation failure" in result.error


def test_audio_provider_import_does_not_load_heavy_runtime():
    assert "torch" not in sys.modules
    assert "diffusers" not in sys.modules
    assert "transformers" not in sys.modules
    assert "onnxruntime" not in sys.modules
    assert "piper" not in sys.modules


def test_local_audio_provider_is_registered_and_uses_safe_configuration():
    from app.services.audio_providers import LocalAudioProvider

    provider = get_audio_provider("local_audio")

    assert isinstance(provider, LocalAudioProvider)
    assert provider.is_available()
    assert settings.AUDIO_PROVIDER == "placeholder"
    assert provider.output_dir == settings.AUDIO_OUTPUT_DIR


@pytest.mark.anyio
async def test_local_audio_generates_deterministic_valid_wav(tmp_path):
    from app.services.audio_providers import LocalAudioProvider

    provider = LocalAudioProvider()
    request = make_request(
        duration_seconds=0.25,
        sample_rate=8000,
        channels=1,
        output_dir=tmp_path,
    )

    first = await provider.generate_audio(request)
    second = await provider.generate_audio(request)

    assert first.status == "generated"
    assert first.provider == "local_audio"
    assert first.audio_type == "ambience"
    assert first.output_path.exists()
    assert first.output_path.stat().st_size > 44
    assert first.sample_rate == 8000
    assert first.channels == 1
    assert first.format == "wav"
    assert first.duration_seconds == pytest.approx(0.25)
    assert first.metadata["ai_generated"] is False
    with wave.open(str(first.output_path), "rb") as wav_file:
        assert wav_file.getframerate() == 8000
        assert wav_file.getnchannels() == 1
        assert wav_file.getnframes() > 0
        first_frames = wav_file.readframes(wav_file.getnframes())
    with wave.open(str(second.output_path), "rb") as wav_file:
        second_frames = wav_file.readframes(wav_file.getnframes())
    assert first_frames == second_frames


@pytest.mark.anyio
async def test_local_audio_supports_stereo_output(tmp_path):
    from app.services.audio_providers import LocalAudioProvider

    result = await LocalAudioProvider().generate_audio(
        make_request(
            duration_seconds=0.1,
            sample_rate=8000,
            channels=2,
            output_dir=tmp_path,
        )
    )

    assert result.status == "generated"
    assert result.channels == 2
    with wave.open(str(result.output_path), "rb") as wav_file:
        assert wav_file.getnchannels() == 2
        assert wav_file.getnframes() > 0


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("audio_type", "generation_method"),
    [
        ("ambience", "filtered_deterministic_noise"),
        ("sound_effect", "deterministic_sine_sweep"),
        ("procedural_music", "deterministic_procedural_music"),
    ],
)
async def test_local_audio_generates_supported_types(audio_type, generation_method, tmp_path):
    from app.services.audio_providers import LocalAudioProvider

    result = await LocalAudioProvider().generate_audio(
        make_request(
            audio_type=audio_type,
            duration_seconds=0.1,
            output_dir=tmp_path,
        )
    )

    assert result.status == "generated"
    assert result.audio_type == audio_type
    assert result.metadata["generation_method"] == generation_method
    assert result.metadata["ai_generated"] is False
    assert result.output_path.parent == tmp_path.resolve()


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"audio_type": "dialogue"}, "does not generate 'dialogue'"),
        ({"duration_seconds": None}, "requires duration_seconds"),
        ({"duration_seconds": 61}, "cannot exceed 60 seconds"),
        ({"output_format": "mp3"}, "supports WAV output only"),
        ({"channels": 3}, "supports one or two channels"),
    ],
)
async def test_local_audio_rejects_unsupported_requests(overrides, error, tmp_path):
    from app.services.audio_providers import LocalAudioProvider

    result = await LocalAudioProvider().generate_audio(
        make_request(output_dir=tmp_path, **overrides)
    )

    assert result.status == "failed"
    assert error in result.error
    assert list(tmp_path.iterdir()) == []


@pytest.mark.anyio
async def test_local_audio_reports_missing_wav_runtime(tmp_path, monkeypatch):
    from app.services.audio_providers import LocalAudioProvider

    provider = LocalAudioProvider()

    def missing_runtime():
        raise ImportError("wave is unavailable")

    monkeypatch.setattr(provider, "_load_wave_module", missing_runtime)
    result = await provider.generate_audio(
        make_request(duration_seconds=0.1, output_dir=tmp_path)
    )

    assert result.status == "failed"
    assert "WAV runtime unavailable: wave is unavailable" in result.error
    assert list(tmp_path.iterdir()) == []


@pytest.mark.anyio
async def test_local_audio_reports_output_path_errors(tmp_path):
    from app.services.audio_providers import LocalAudioProvider

    invalid_output_dir = tmp_path / "not-a-directory"
    invalid_output_dir.write_text("file", encoding="utf-8")
    result = await LocalAudioProvider().generate_audio(
        make_request(duration_seconds=0.1, output_dir=invalid_output_dir)
    )

    assert result.status == "failed"
    assert result.error
