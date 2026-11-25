import asyncio
import json
import sys
import wave

import pytest

from app.config import BASE_DIR, Settings, settings
from app.services.audio_providers import (
    AIAudioProvider,
    AudioGenerationRequest,
    AudioProviderExecutionError,
    BaseAudioProvider,
    get_audio_provider,
    list_audio_providers,
)
from app.services.audio_providers.audio_validation import validate_pcm_wav


def build_provider(tmp_path, *, create_model=True, create_python=True):
    model_path = tmp_path / "model"
    if create_model:
        model_path.mkdir()
        (model_path / "model_index.json").write_text("{}", encoding="utf-8")
    python_path = tmp_path / "audio-python.exe"
    if create_python:
        python_path.touch()
    config = Settings(
        AI_AUDIO_MODEL_PATH=model_path,
        AI_AUDIO_PYTHON_PATH=python_path,
        AUDIO_OUTPUT_DIR=tmp_path / "generated",
        AI_AUDIO_MAX_DURATION_SECONDS=5.0,
        AI_AUDIO_STEPS=3,
    )
    return AIAudioProvider(config)


def valid_request(**overrides):
    values = {
        "prompt": "A soft rainfall on a quiet city street.",
        "audio_type": "ambience",
        "duration_seconds": 0.5,
        "sample_rate": 22050,
        "channels": 1,
        "seed": 17,
        "output_format": "wav",
    }
    values.update(overrides)
    return AudioGenerationRequest(**values)


def test_ai_audio_provider_is_registered_and_implements_interface():
    provider = get_audio_provider("ai_audio")

    assert isinstance(provider, BaseAudioProvider)
    assert isinstance(provider, AIAudioProvider)
    assert {"placeholder", "local_audio", "ai_audio"} <= set(list_audio_providers())
    assert settings.AUDIO_PROVIDER == "placeholder"


def test_importing_ai_audio_provider_does_not_load_ml_runtime():
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules
    assert "diffusers" not in sys.modules
    assert "torchaudio" not in sys.modules
    assert "stable_audio_tools" not in sys.modules


def test_ai_audio_settings_have_bounded_safe_defaults():
    assert settings.AI_AUDIO_MODEL_ID == "cvssp/audioldm2"
    assert settings.AI_AUDIO_DEVICE == "cuda"
    assert settings.AI_AUDIO_MAX_DURATION_SECONDS == 5.0
    assert settings.AI_AUDIO_STEPS == 20
    assert settings.AI_AUDIO_TIMEOUT_SECONDS >= 30


def test_ai_audio_rejects_unsupported_audio_type(tmp_path):
    provider = build_provider(tmp_path)

    with pytest.raises(ValueError, match="supports only"):
        provider._generation_parameters(
            valid_request(audio_type="procedural_music")
        )


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"duration_seconds": 5.5}, "duration must be between"),
        ({"duration_seconds": float("inf")}, "finite duration"),
        ({"sample_rate": 6000}, "sample rate must be between"),
        ({"channels": 4}, "supports one or two"),
        ({"output_format": "mp3"}, "WAV output only"),
    ],
)
def test_ai_audio_validates_generation_request(tmp_path, overrides, error):
    provider = build_provider(tmp_path)

    with pytest.raises(ValueError, match=error):
        provider._generation_parameters(valid_request(**overrides))


def test_ai_audio_requires_cuda_device(tmp_path):
    provider = build_provider(tmp_path)
    provider.config.AI_AUDIO_DEVICE = "cpu"

    with pytest.raises(ValueError, match="CPU fallback is disabled"):
        provider._generation_parameters(valid_request())


def test_ai_audio_rejects_output_path_outside_output_directory(tmp_path):
    provider = build_provider(tmp_path)

    with pytest.raises(ValueError, match="must be inside"):
        provider._generation_parameters(
            valid_request(
                output_path=tmp_path / "outside.wav",
                output_dir=tmp_path / "generated",
            )
        )


def test_ai_audio_rejects_non_wav_output_path(tmp_path):
    provider = build_provider(tmp_path)

    with pytest.raises(ValueError, match="must use the .wav extension"):
        provider._generation_parameters(
            valid_request(
                output_path=tmp_path / "generated" / "output.mp3",
                output_dir=tmp_path / "generated",
            )
        )


def test_ai_audio_rejects_existing_output_path(tmp_path):
    provider = build_provider(tmp_path)
    output_dir = tmp_path / "generated"
    output_dir.mkdir()
    existing_output = output_dir / "existing.wav"
    existing_output.touch()

    with pytest.raises(ValueError, match="already exists"):
        provider._generation_parameters(
            valid_request(output_path=existing_output, output_dir=output_dir)
        )


@pytest.mark.anyio
async def test_ai_audio_reports_missing_isolated_runtime(tmp_path):
    provider = build_provider(tmp_path, create_python=False)

    with pytest.raises(AudioProviderExecutionError, match="environment is missing"):
        await provider.generate_audio(valid_request())


@pytest.mark.anyio
async def test_ai_audio_reports_missing_local_model(tmp_path):
    provider = build_provider(tmp_path, create_model=False)

    with pytest.raises(AudioProviderExecutionError, match="model files are missing"):
        await provider.generate_audio(valid_request())


@pytest.mark.anyio
async def test_ai_audio_propagates_worker_errors(tmp_path, monkeypatch):
    provider = build_provider(tmp_path)

    async def fail_worker(_payload):
        raise AudioProviderExecutionError("CUDA out of memory during test.")

    monkeypatch.setattr(provider, "_run_worker", fail_worker)
    with pytest.raises(AudioProviderExecutionError, match="CUDA out of memory"):
        await provider.generate_audio(valid_request())


@pytest.mark.anyio
async def test_ai_audio_worker_decodes_structured_oom_error(tmp_path, monkeypatch):
    provider = build_provider(tmp_path)

    class FailedProcess:
        returncode = 1

        async def communicate(self, _input=None):
            return (
                b'{"ok": false, "error": "CUDA out of memory", "error_code": "cuda_oom"}',
                b"",
            )

    async def create_process(*args, **kwargs):
        return FailedProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    with pytest.raises(AudioProviderExecutionError, match="CUDA out of memory"):
        await provider._run_worker({"test": True})


@pytest.mark.anyio
async def test_ai_audio_worker_uses_isolated_offline_runtime(tmp_path, monkeypatch):
    provider = build_provider(tmp_path)
    captured = {}

    class SuccessfulProcess:
        returncode = 0

        async def communicate(self, input=None):
            captured["payload"] = json.loads(input.decode("utf-8"))
            return (b'{"ok": true, "result": {}}', b"")

    async def create_process(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return SuccessfulProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    payload = {"model_path": "local-model"}

    response = await provider._run_worker(payload)

    assert response == {"ok": True, "result": {}}
    assert captured["args"] == (
        str(provider.python_path),
        str(provider.worker_path),
    )
    assert captured["kwargs"]["cwd"] == str(BASE_DIR)
    assert captured["kwargs"]["env"]["HF_HUB_OFFLINE"] == "1"
    assert captured["kwargs"]["env"]["TRANSFORMERS_OFFLINE"] == "1"
    assert captured["payload"] == payload


@pytest.mark.anyio
async def test_ai_audio_returns_validated_generation_metadata(tmp_path, monkeypatch):
    provider = build_provider(tmp_path)

    async def fake_worker(payload):
        output_path = payload["output_path"]
        with wave.open(output_path, "wb") as wav_file:
            wav_file.setnchannels(payload["channels"])
            wav_file.setsampwidth(2)
            wav_file.setframerate(payload["sample_rate"])
            wav_file.writeframes(b"\x10\x00" * int(payload["duration_seconds"] * payload["sample_rate"]))
        return {
            "ok": True,
            "result": {
                "generation_time_seconds": 1.25,
                "device": "cuda:0",
                "gpu": "NVIDIA GeForce RTX 5050 Laptop GPU",
                "gpu_peak_allocated_bytes": 2048,
                "source_sample_rate": 16000,
            },
        }

    monkeypatch.setattr(provider, "_run_worker", fake_worker)
    request = valid_request(output_dir=tmp_path / "custom-output")

    result = await provider.generate_audio(request)

    assert result.status == "generated"
    assert result.provider == "ai_audio"
    assert result.model == "cvssp/audioldm2"
    assert result.output_path.is_file()
    assert result.duration_seconds == pytest.approx(0.5)
    assert result.sample_rate == 22050
    assert result.channels == 1
    assert result.format == "wav"
    assert result.metadata["prompt"] == request.prompt
    assert result.metadata["seed"] == 17
    assert result.metadata["generation_time_seconds"] == 1.25
    assert result.metadata["ai_generated"] is True
    info = validate_pcm_wav(
        result.output_path,
        expected_sample_rate=22050,
        expected_channels=1,
        expected_duration_seconds=0.5,
    )
    assert info.peak_sample > 0


def test_pcm_validation_rejects_silence_and_truncated_wav(tmp_path):
    silent_path = tmp_path / "silent.wav"
    with wave.open(str(silent_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 160)

    with pytest.raises(ValueError, match="silent"):
        validate_pcm_wav(
            silent_path,
            expected_sample_rate=16000,
            expected_channels=1,
        )

    truncated_path = tmp_path / "truncated.wav"
    truncated_path.write_bytes(b"RIFF" + b"\x00" * 64)
    with pytest.raises(ValueError, match="valid WAV"):
        validate_pcm_wav(
            truncated_path,
            expected_sample_rate=16000,
            expected_channels=1,
        )
