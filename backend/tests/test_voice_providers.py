import struct
import sys
import wave
from pathlib import Path
from types import SimpleNamespace
from types import SimpleNamespace

import pytest

from app.config import settings
from app.routes import characters as character_routes
from app.services.voice import VoiceService
from app.services.voice_providers import (
    InvalidVoiceProviderError,
    PlaceholderVoiceProvider,
    PiperONNXVoiceProvider,
    VoiceGenerationRequest,
    VoiceGenerationResult,
    get_voice_provider,
    list_voice_providers,
)


def test_placeholder_provider_can_initialize():
    provider = PlaceholderVoiceProvider()

    assert provider.name == "placeholder"
    assert provider.is_available()


def test_registry_contains_placeholder_and_default_is_placeholder():
    assert "placeholder" in list_voice_providers()
    assert "piper_onnx" in list_voice_providers()
    assert settings.VOICE_PROVIDER == "placeholder"
    assert isinstance(get_voice_provider(), PlaceholderVoiceProvider)
    assert isinstance(get_voice_provider("piper_onnx"), PiperONNXVoiceProvider)


def test_unknown_provider_raises_application_error():
    with pytest.raises(InvalidVoiceProviderError, match="Unknown voice provider"):
        get_voice_provider("unknown-provider")


@pytest.mark.anyio
async def test_voice_service_returns_placeholder_without_creating_audio(tmp_path):
    output_path = tmp_path / "speech.wav"

    result = await VoiceService().generate_speech(
        character_name="Mara",
        dialogue_text="A short line of dialogue.",
        character_id="mara-1",
        voice_name="Default Narrator",
        voice_id="voice-1",
        language="en",
        speaking_style="calm",
        output_path=output_path,
    )

    assert result.provider == "placeholder"
    assert result.status == "placeholder"
    assert result.voice_id == "voice-1"
    assert result.output_path is None
    assert result.duration_seconds is None
    assert result.sample_rate is None
    assert result.format is None
    assert result.message
    assert "no audio file was created" in result.message
    assert not output_path.exists()


@pytest.mark.anyio
async def test_piper_provider_returns_structured_failure_for_missing_model(tmp_path):
    provider = PiperONNXVoiceProvider(
        model_path=tmp_path / "missing.onnx",
        output_dir=tmp_path / "output",
    )

    result = await provider.generate_speech(
        VoiceGenerationRequest(text="A short test.", voice_id="voice-1")
    )

    assert result.provider == "piper_onnx"
    assert result.status == "failed"
    assert result.voice_id == "voice-1"
    assert "model files are missing" in result.message
    assert result.output_path is None


@pytest.mark.anyio
async def test_piper_provider_returns_structured_failure_when_runtime_is_missing(
    tmp_path,
    monkeypatch,
):
    model_path = tmp_path / "voice.onnx"
    model_path.touch()
    Path(f"{model_path}.json").write_text("{}", encoding="utf-8")
    provider = PiperONNXVoiceProvider(model_path, tmp_path / "output")
    monkeypatch.setattr(provider, "_runtime_installed", lambda: False)

    result = await provider.generate_speech(
        VoiceGenerationRequest(text="A short test.")
    )

    assert result.provider == "piper_onnx"
    assert result.status == "failed"
    assert "must be installed" in result.message
    assert result.metadata["model"] == "voice.onnx"


@pytest.mark.anyio
async def test_piper_provider_returns_valid_metadata_for_directml_generation(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "app.services.voice_providers.piper_onnx.gc.collect",
        lambda: 0,
    )
    model_path = tmp_path / "voice.onnx"
    model_path.touch()
    Path(f"{model_path}.json").write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "output"
    provider = PiperONNXVoiceProvider(model_path, output_dir)

    class FakeSession:
        def get_providers(self):
            return ["DmlExecutionProvider", "CPUExecutionProvider"]

    class FakeRuntime:
        class ExecutionMode:
            ORT_SEQUENTIAL = "sequential"

        class SessionOptions:
            execution_mode = None
            enable_mem_pattern = True

        @staticmethod
        def get_available_providers():
            return ["DmlExecutionProvider", "CPUExecutionProvider"]

        @staticmethod
        def InferenceSession(model, sess_options, providers):
            assert model == str(model_path.resolve())
            assert providers == [
                "DmlExecutionProvider",
                "CPUExecutionProvider",
            ]
            assert sess_options.execution_mode == "sequential"
            assert sess_options.enable_mem_pattern is False
            return FakeSession()

    class FakeVoice:
        def __init__(self, session, config, download_dir):
            self.session = session

        @staticmethod
        def synthesize_wav(text, wav_file):
            assert text == "CineAI integration test."
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(22050)
            wav_file.writeframes(struct.pack("<h", 2500) * 1024)

    fake_piper = SimpleNamespace(
        PiperConfig=SimpleNamespace(
            from_dict=lambda config: SimpleNamespace(sample_rate=22050)
        ),
        PiperVoice=FakeVoice,
    )
    monkeypatch.setattr(provider, "_runtime_installed", lambda: True)
    monkeypatch.setattr(
        provider,
        "_load_runtime",
        lambda: (fake_piper, FakeRuntime),
    )
    output_path = output_dir / "integration.wav"

    result = await provider.generate_speech(
        VoiceGenerationRequest(
            text="CineAI integration test.",
            character_id="character-1",
            voice_id="voice-1",
            output_path=output_path,
        )
    )

    assert result.provider == "piper_onnx"
    assert result.status == "generated"
    assert result.output_path == output_path
    assert result.duration_seconds == pytest.approx(1024 / 22050)
    assert result.sample_rate == 22050
    assert result.format == "wav"
    assert result.voice_id == "voice-1"
    assert result.metadata["runtime_provider"] == "DmlExecutionProvider"
    assert result.metadata["session_providers"] == [
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert result.metadata["model"] == "voice.onnx"
    assert result.metadata["inference_time_seconds"] >= 0
    assert result.metadata["channels"] == 1
    with wave.open(str(output_path), "rb") as wav_file:
        assert wav_file.getnframes() > 0


@pytest.mark.anyio
async def test_piper_provider_does_not_use_cpu_only_without_directml(
    tmp_path,
    monkeypatch,
):
    model_path = tmp_path / "voice.onnx"
    model_path.touch()
    Path(f"{model_path}.json").write_text("{}", encoding="utf-8")
    provider = PiperONNXVoiceProvider(model_path, tmp_path / "output")

    class CpuOnlyRuntime:
        @staticmethod
        def get_available_providers():
            return ["CPUExecutionProvider"]

    monkeypatch.setattr(provider, "_runtime_installed", lambda: True)
    monkeypatch.setattr(
        provider,
        "_load_runtime",
        lambda: (SimpleNamespace(), CpuOnlyRuntime),
    )

    result = await provider.generate_speech(
        VoiceGenerationRequest(text="A short test.")
    )

    assert result.status == "failed"
    assert "CPU fallback is disabled" in result.message
    assert result.metadata["runtime_provider"] is None


@pytest.mark.anyio
async def test_character_voice_route_returns_generated_voice_url(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "MEDIA_ROOT", tmp_path)
    output_path = tmp_path / "voice" / "preview.wav"
    output_path.parent.mkdir(parents=True)
    output_path.write_bytes(b"valid generated audio")

    class FakeResult:
        @staticmethod
        def scalar_one_or_none():
            return None

    class FakeSession:
        async def execute(self, _statement):
            return FakeResult()

    async def generated_speech(**kwargs):
        return VoiceGenerationResult(
            provider="piper_onnx",
            status="generated",
            output_path=output_path,
        )

    monkeypatch.setattr(
        character_routes.voice_service,
        "generate_speech",
        generated_speech,
    )

    response = await character_routes.test_character_voice(
        "not-in-memory",
        text="A route preview.",
        db=FakeSession(),
    )

    assert response["audio_url"] == "/generated/voice/preview.wav"
    assert response["status"] == "generated"


@pytest.mark.anyio
async def test_character_voice_preview_attaches_audio_to_persisted_dialogue(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(settings, "MEDIA_ROOT", tmp_path)
    output_path = tmp_path / "voice" / "preview.wav"
    output_path.parent.mkdir(parents=True)
    output_path.write_bytes(b"generated audio")
    db_character = SimpleNamespace(
        id="character-1",
        story_id="story-1",
        name="Mara",
        role="Lead",
        description="A focused archivist",
        personality="Persistent",
        visual_prompt="Mara in an archive",
        voice_name="Narrator",
        voice_id="voice-1",
    )
    db_scene = SimpleNamespace(
        story_id="story-1",
        dialogues=[
            {"character_name": "Mara", "text": "The signal is fading."},
            {"character_name": "Mara", "text": "We should leave."},
        ],
    )

    class FakeResult:
        def __init__(self, *, character=None, scenes=None):
            self.character = character
            self.scenes = scenes or []

        def scalar_one_or_none(self):
            return self.character

        def scalars(self):
            return self

        def all(self):
            return self.scenes

    class FakeSession:
        def __init__(self):
            self.query_count = 0
            self.committed = False

        async def execute(self, _statement):
            self.query_count += 1
            if self.query_count == 1:
                return FakeResult(character=db_character)
            return FakeResult(scenes=[db_scene])

        async def commit(self):
            self.committed = True

        async def rollback(self):
            raise AssertionError("Successful dialogue persistence must not roll back.")

    async def generated_speech(**kwargs):
        assert kwargs["character_name"] == "Mara"
        assert kwargs["character_id"] == "character-1"
        return VoiceGenerationResult(
            provider="piper_onnx",
            status="generated",
            output_path=output_path,
        )

    session = FakeSession()
    monkeypatch.setattr(
        character_routes.voice_service,
        "generate_speech",
        generated_speech,
    )

    response = await character_routes.test_character_voice(
        "character-1",
        text="The signal is fading.",
        db=session,
    )

    assert response["status"] == "generated"
    assert response["audio_url"] == "/generated/voice/preview.wav"
    assert db_scene.dialogues[0]["audio_url"] == "/generated/voice/preview.wav"
    assert "audio_url" not in db_scene.dialogues[1]
    assert session.committed


def test_voice_provider_import_does_not_load_tts_runtime():
    assert "piper" not in sys.modules
    assert "onnxruntime" not in sys.modules
    assert not any(
        module == "TTS" or module.startswith("TTS.")
        for module in sys.modules
    )
