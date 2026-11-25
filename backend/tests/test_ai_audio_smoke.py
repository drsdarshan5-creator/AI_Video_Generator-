import json
import os
import pytest

from app.config import Settings
from app.services.audio_providers import AIAudioProvider, AudioGenerationRequest
from app.services.audio_providers.audio_validation import validate_pcm_wav

pytestmark = pytest.mark.skipif(
    os.environ.get("CINEAI_AI_AUDIO_SMOKE") != "1",
    reason="Set CINEAI_AI_AUDIO_SMOKE=1 to run the real AI audio smoke test.",
)


@pytest.mark.anyio
async def test_ai_audio_smoke_generates_and_validates_real_wav(tmp_path):
    provider = AIAudioProvider(
        Settings(
            AI_AUDIO_MAX_DURATION_SECONDS=3.0,
            AI_AUDIO_STEPS=5,
            AUDIO_DEFAULT_SAMPLE_RATE=16000,
        )
    )
    assert provider.python_path.is_file(), (
        f"Audio runtime is missing: {provider.python_path}. "
        "Install backend/requirements-audio.txt into .venv-audio first."
    )
    assert (provider.model_path / "model_index.json").is_file(), (
        f"AudioLDM 2 model is missing from {provider.model_path}. "
        "Run tools/download_audioldm2.py explicitly first."
    )
    with (provider.model_path / "model_index.json").open(
        encoding="utf-8"
    ) as model_index_file:
        model_index = json.load(model_index_file)
    assert model_index["language_model"] == [
        "transformers",
        "GPT2LMHeadModel",
    ]

    result = await provider.generate_audio(
        AudioGenerationRequest(
            prompt="A short gentle rain ambience on a quiet city street at night.",
            audio_type="ambience",
            duration_seconds=3.0,
            sample_rate=16000,
            channels=1,
            seed=42,
            output_format="wav",
            output_dir=tmp_path,
        )
    )

    assert result.status == "generated"
    assert result.model == "cvssp/audioldm2"
    assert result.metadata["ai_generated"] is True
    assert result.metadata["generation_time_seconds"] > 0
    assert result.metadata["inference_steps"] == 5
    assert result.metadata["source_sample_rate"] == 16000
    assert result.metadata["device"].startswith("cuda")
    assert result.sample_rate == 16000
    validate_pcm_wav(
        result.output_path,
        expected_sample_rate=16000,
        expected_channels=1,
        expected_duration_seconds=3.0,
    )
