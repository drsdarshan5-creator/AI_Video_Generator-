import wave
from types import SimpleNamespace

import pytest

from app.config import Settings, settings
from app.models import DialogueLine, Scene
from app.services.audio import AudioService, SceneAudioOptions
from app.services.audio_providers import (
    AIAudioProvider,
    AudioGenerationResult,
)
from app.services.audio_providers.audio_validation import validate_pcm_wav
from app.services.audio_providers.local_audio import LocalAudioProvider
from app.services.audio_providers.placeholder import PlaceholderAudioProvider


def make_scene(**overrides):
    values = {
        "id": 4,
        "story_id": "story-audio-test",
        "scene_number": 2,
        "heading": "EXT. CITY STREET - NIGHT",
        "location": "Rainy city street",
        "time_of_day": "NIGHT",
        "mood": "Tense",
        "weather": "Heavy rain",
        "action_description": "A runner hurries along the wet pavement.",
        "visual_prompt": "A cinematic rainy street at night.",
        "duration_seconds": 3,
        "dialogues": [
            DialogueLine(character_name="Mara", text="Keep moving.")
        ],
    }
    values.update(overrides)
    return Scene(**values)


def test_scene_audio_prompt_is_deterministic_and_uses_scene_context():
    scene = make_scene()
    first = AudioService.build_scene_audio_prompt(scene)
    second = AudioService.build_scene_audio_prompt(scene)

    assert first == second
    assert "tense atmosphere" in first
    assert "Rainy city street" in first
    assert "at night" in first
    assert "Heavy rain" in first
    assert "runner hurries" in first
    assert "subtle background audio beneath dialogue" in first
    assert "Keep moving" not in first


@pytest.mark.parametrize(
    ("scene", "expected"),
    [
        (make_scene(action_description="The leaves sway softly."), "ambience"),
        (
            make_scene(action_description="Footsteps pound across the street."),
            "sound_effect",
        ),
        (
            make_scene(
                heading="INT. EMPTY STUDIO - NIGHT",
                location="Empty studio",
                weather=None,
                mood="Romantic",
                action_description="Two people share a quiet look.",
                dialogues=[],
            ),
            "music",
        ),
    ],
)
def test_scene_audio_type_is_selected_deterministically(scene, expected):
    assert AudioService.select_scene_audio_type(scene) == expected


def test_scene_audio_type_can_be_explicitly_selected():
    assert (
        AudioService.select_scene_audio_type(
            make_scene(),
            audio_type="music",
        )
        == "music"
    )

    with pytest.raises(ValueError, match="Scene audio type"):
        AudioService.select_scene_audio_type(make_scene(), audio_type="narration")


def test_scene_audio_rejects_invalid_scene_data():
    with pytest.raises(ValueError, match="Invalid scene data"):
        AudioService.build_scene_audio_prompt({"scene_number": 0})


def test_scene_audio_options_reject_non_finite_duration():
    with pytest.raises(ValueError, match="duration must be finite"):
        SceneAudioOptions(duration_seconds=float("inf"))


@pytest.mark.anyio
async def test_placeholder_provider_integrates_with_scene_audio_service():
    result = await AudioService(PlaceholderAudioProvider()).generate_scene_audio(
        make_scene()
    )

    assert result.status == "placeholder"
    assert result.output_path is None
    assert result.metadata["scene_number"] == 2
    assert result.metadata["prompt"] == AudioService.build_scene_audio_prompt(
        make_scene()
    )
    assert result.metadata["ai_generated"] is not True


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("duration", "seed", "message"),
    [
        (float("inf"), None, "duration must be finite"),
        (None, -1, "seed must be non-negative"),
    ],
)
async def test_scene_audio_rejects_invalid_generation_options(
    duration,
    seed,
    message,
):
    with pytest.raises(ValueError, match=message):
        await AudioService(PlaceholderAudioProvider()).generate_scene_audio(
            make_scene(),
            duration_seconds=duration,
            seed=seed,
        )


@pytest.mark.anyio
async def test_local_audio_provider_integrates_and_returns_valid_wav(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(settings, "AUDIO_OUTPUT_DIR", tmp_path)
    result = await AudioService(
        LocalAudioProvider(output_dir=tmp_path)
    ).generate_scene_audio(
        make_scene(weather=None, dialogues=[]),
        audio_type="ambience",
        seed=23,
    )

    assert result.status == "generated"
    assert result.metadata["scene_number"] == 2
    assert result.metadata["seed"] == 23
    assert result.metadata["ai_generated"] is False
    assert result.output_path.is_file()
    info = validate_pcm_wav(
        result.output_path,
        expected_sample_rate=result.sample_rate,
        expected_channels=1,
        expected_duration_seconds=3,
    )
    assert info.peak_sample > 0


@pytest.mark.anyio
async def test_ai_audio_provider_integrates_through_mocked_worker(
    tmp_path,
    monkeypatch,
):
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "model_index.json").write_text("{}", encoding="utf-8")
    python_path = tmp_path / "audio-python.exe"
    python_path.touch()
    output_dir = tmp_path / "audio"
    monkeypatch.setattr(settings, "AUDIO_OUTPUT_DIR", output_dir)
    monkeypatch.setattr(settings, "AUDIO_DEFAULT_SAMPLE_RATE", 16000)
    provider = AIAudioProvider(
        Settings(
            AI_AUDIO_MODEL_PATH=model_path,
            AI_AUDIO_PYTHON_PATH=python_path,
            AUDIO_OUTPUT_DIR=output_dir,
            AI_AUDIO_MAX_DURATION_SECONDS=5,
            AI_AUDIO_STEPS=5,
        )
    )
    captured = {}

    async def fake_worker(payload):
        captured.update(payload)
        with wave.open(payload["output_path"], "wb") as wav_file:
            wav_file.setnchannels(payload["channels"])
            wav_file.setsampwidth(2)
            wav_file.setframerate(payload["sample_rate"])
            wav_file.writeframes(
                b"\x10\x00" * int(
                    payload["duration_seconds"] * payload["sample_rate"]
                )
            )
        return {
            "ok": True,
            "result": {
                "generation_time_seconds": 1.6,
                "device": "cuda:0",
                "gpu": "NVIDIA GeForce RTX 5050 Laptop GPU",
                "gpu_peak_allocated_bytes": 2_340_000_000,
                "source_sample_rate": 16000,
            },
        }

    monkeypatch.setattr(provider, "_run_worker", fake_worker)
    result = await AudioService(provider).generate_scene_audio(
        make_scene(weather=None, dialogues=[]),
        audio_type="music",
        seed=87,
    )

    assert result.provider == "ai_audio"
    assert result.model == "cvssp/audioldm2"
    assert result.status == "generated"
    assert result.metadata["scene_number"] == 2
    assert result.metadata["seed"] == 87
    assert result.metadata["inference_steps"] == 5
    assert result.metadata["generation_time_seconds"] == 1.6
    assert result.metadata["ai_generated"] is True
    assert captured["audio_type"] == "music"
    assert captured["duration_seconds"] == 3
    assert captured["sample_rate"] == 16000
    assert captured["seed"] == 87


@pytest.mark.anyio
async def test_scene_audio_endpoint_uses_stored_scene_and_placeholder(
    monkeypatch,
):
    from app.routes import scenes as scenes_route

    scene = make_scene()

    class FakeResult:
        def scalars(self):
            return self

        def all(self):
            return [scene]

    class FakeSession:
        async def execute(self, _statement):
            return FakeResult()

    monkeypatch.setattr(
        scenes_route,
        "audio_service",
        AudioService(PlaceholderAudioProvider()),
    )
    result = await scenes_route.generate_scene_audio(
        scene_number=2,
        options=SceneAudioOptions(story_id=scene.story_id),
        db=FakeSession(),
    )

    assert result["status"] == "placeholder"
    assert result["metadata"]["scene_number"] == 2


@pytest.mark.anyio
async def test_scene_audio_endpoint_falls_back_to_memory_when_database_is_unavailable(
    monkeypatch,
):
    from sqlalchemy.exc import SQLAlchemyError

    from app.routes import scenes as scenes_route

    scene = make_scene()

    class UnavailableSession:
        async def execute(self, _statement):
            raise SQLAlchemyError("database unavailable")

    monkeypatch.setattr(
        scenes_route,
        "audio_service",
        AudioService(PlaceholderAudioProvider()),
    )
    monkeypatch.setattr(scenes_route, "_SCENES_STORE", [scene])
    result = await scenes_route.generate_scene_audio(
        scene_number=2,
        options=SceneAudioOptions(story_id=scene.story_id),
        db=UnavailableSession(),
    )

    assert result["status"] == "placeholder"
    assert result["metadata"]["scene_number"] == scene.scene_number
