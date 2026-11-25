import os
import wave

import pytest

from app.services.audio_providers import AudioGenerationRequest, LocalAudioProvider

pytestmark = pytest.mark.skipif(
    os.environ.get("CINEAI_LOCAL_AUDIO_SMOKE") != "1",
    reason="Set CINEAI_LOCAL_AUDIO_SMOKE=1 to run the local audio smoke test.",
)


@pytest.mark.anyio
async def test_local_audio_smoke_generates_non_silent_wav_assets(tmp_path):
    provider = LocalAudioProvider(output_dir=tmp_path)
    for audio_type in ("ambience", "sound_effect", "procedural_music"):
        result = await provider.generate_audio(
            AudioGenerationRequest(
                description=f"Smoke test {audio_type}.",
                duration_seconds=0.5,
                audio_type=audio_type,
                seed=101,
                sample_rate=22050,
                channels=1,
                output_format="wav",
            )
        )

        assert result.status == "generated"
        assert result.output_path.is_file()
        assert result.output_path.stat().st_size > 44
        with wave.open(str(result.output_path), "rb") as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            assert wav_file.getframerate() == 22050
            assert wav_file.getnchannels() == 1
            assert wav_file.getnframes() > 0
            assert wav_file.getnframes() / wav_file.getframerate() > 0
            assert any(frames)
