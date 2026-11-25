from types import SimpleNamespace

import pytest

from app.config import settings
from app.models import Scene
from app.routes import providers as providers_route
from app.routes import scenes as scenes_route
from app.services.audio_providers import AudioGenerationResult
from app.services.video import VideoService
from app.services.video_providers import (
    BaseVideoProvider,
    VideoGenerationRequest,
    VideoGenerationResult,
)


def make_scene(**overrides):
    values = {
        "id": 11,
        "story_id": "frontend-story",
        "scene_number": 1,
        "heading": "EXT. HARBOR - NIGHT",
        "location": "Harbor",
        "time_of_day": "Night",
        "mood": "Tense",
        "action_description": "A signal flashes across the water.",
        "visual_prompt": "A cinematic harbor at night.",
        "duration_seconds": 5,
        "dialogues": [],
    }
    values.update(overrides)
    return Scene(**values)


class FakeResult:
    def __init__(self, rows):
        self.rows = rows

    def scalars(self):
        return self

    def all(self):
        return self.rows


class FakeSession:
    def __init__(self, rows):
        self.rows = rows
        self.committed = False

    async def execute(self, _statement):
        return FakeResult(self.rows)

    async def commit(self):
        self.committed = True

    async def rollback(self):
        self.committed = False


class GeneratedVideoProvider(BaseVideoProvider):
    name = "test-video"

    def __init__(self, output_path):
        self.output_path = output_path

    def is_available(self):
        return True

    async def generate_video(self, request):
        return VideoGenerationResult(
            provider=self.name,
            status="generated",
            output_path=self.output_path,
            metadata=request.metadata,
        )


@pytest.mark.anyio
async def test_provider_configuration_exposes_only_provider_names(monkeypatch):
    for field, value in (
        ("IMAGE_PROVIDER", "placeholder"),
        ("VOICE_PROVIDER", "piper_onnx"),
        ("VIDEO_PROVIDER", "cogvideox"),
        ("AUDIO_PROVIDER", "local_audio"),
    ):
        monkeypatch.setattr(settings, field, value)

    assert await providers_route.get_configured_providers() == {
        "image": "placeholder",
        "voice": "piper_onnx",
        "video": "cogvideox",
        "audio": "local_audio",
    }


@pytest.mark.anyio
async def test_scene_video_route_returns_generated_media_url_and_persists_reference(
    monkeypatch,
    tmp_path,
):
    media_root = tmp_path / "generated"
    output_path = media_root / "videos" / "scene.mp4"
    output_path.parent.mkdir(parents=True)
    output_path.write_bytes(b"mp4")
    monkeypatch.setattr(settings, "MEDIA_ROOT", media_root)

    db_scene = SimpleNamespace(
        id=11,
        story_id="frontend-story",
        scene_number=1,
        heading="EXT. HARBOR - NIGHT",
        location="Harbor",
        time_of_day="Night",
        mood="Tense",
        lighting=None,
        weather=None,
        shot_type="Medium shot",
        action_description="A signal flashes across the water.",
        visual_prompt="A cinematic harbor at night.",
        camera_movement="Static shot",
        duration_seconds=5,
        dialogues=[],
        image_url=None,
        video_url=None,
        created_at=None,
    )
    db = FakeSession([db_scene])
    service = VideoService(video_provider=GeneratedVideoProvider(output_path))
    monkeypatch.setattr(scenes_route, "video_service", service)
    monkeypatch.setattr(scenes_route, "_SCENES_STORE", [])

    response = await scenes_route.generate_scene_video(
        1,
        VideoGenerationRequest(
            story_id="frontend-story",
            prompt="A cinematic harbor at night.",
            duration_seconds=5,
        ),
        db,
    )

    assert response["status"] == "generated"
    assert response["video_url"] == "/generated/videos/scene.mp4"
    assert db_scene.video_url == response["video_url"]
    assert db.committed


@pytest.mark.anyio
async def test_scene_audio_route_adds_public_url_without_changing_result_fields(
    monkeypatch,
    tmp_path,
):
    media_root = tmp_path / "generated"
    output_path = media_root / "audio" / "scene.wav"
    output_path.parent.mkdir(parents=True)
    output_path.write_bytes(b"wav")
    monkeypatch.setattr(settings, "MEDIA_ROOT", media_root)

    class FakeAudioService:
        async def generate_scene_audio(self, *_args, **_kwargs):
            return AudioGenerationResult(
                provider="local_audio",
                status="generated",
                output_path=output_path,
            )

    monkeypatch.setattr(scenes_route, "audio_service", FakeAudioService())
    response = await scenes_route.generate_scene_audio(
        1,
        scenes_route.SceneAudioOptions(story_id="frontend-story"),
        FakeSession([make_scene()]),
    )

    assert response["status"] == "generated"
    assert response["provider"] == "local_audio"
    assert response["output_path"] == str(output_path)
    assert response["audio_url"] == "/generated/audio/scene.wav"
