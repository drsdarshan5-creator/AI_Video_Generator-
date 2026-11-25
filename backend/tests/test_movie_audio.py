from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

from fastapi import HTTPException
import pytest
from PIL import Image

import imageio_ffmpeg
from app.config import settings
from app.models import MovieRenderRequest, Scene
from app.services.audio_providers import AudioGenerationResult
from app.services.movie_assembler import MovieAssembler, MovieAssemblyResult
from app.services.video import VideoService


def make_scene():
    return Scene(
        scene_number=1,
        story_id="movie-audio-test",
        heading="EXT. FOREST - DAWN",
        location="Forest",
        time_of_day="DAWN",
        mood="Calm",
        weather="Light wind",
        action_description="A traveler watches the morning light.",
        visual_prompt="A forest at dawn.",
        duration_seconds=3,
        image_url="/generated/images/scene.png",
    )


class FakeAudioService:
    def __init__(self, provider_name):
        self.provider = SimpleNamespace(name=provider_name)
        self.generated_scenes = []

    async def generate_scene_audio(self, scene):
        self.generated_scenes.append(scene)
        return AudioGenerationResult(
            provider=self.provider.name,
            status="generated",
            audio_type="ambience",
            output_path=Path("/generated/audio/scene.wav"),
            duration_seconds=3,
            sample_rate=16000,
            channels=1,
            format="wav",
            metadata={
                "prompt": "cinematic forest at dawn",
                "seed": scene.scene_number,
                "ai_generated": self.provider.name == "ai_audio",
            },
        )


class FakeMovieAssembler:
    def __init__(self):
        self.requests = []
        self.preflight_requests = []

    def validate_assets(self, request):
        self.preflight_requests.append(request)

    def assemble(self, request):
        self.requests.append(request)
        output_path = Path(request.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"test movie")
        return MovieAssemblyResult(
            output_path=output_path,
            duration_seconds=sum(
                scene.duration_seconds for scene in request.scenes
            ),
            fps=request.fps,
            width=1280,
            height=720,
            scene_count=len(request.scenes),
            audio_track_count=sum(
                len(scene.voice_tracks)
                + len(scene.ambience_tracks)
                + len(scene.sound_effect_tracks)
                + len(scene.music_tracks)
                for scene in request.scenes
            ),
        )


@pytest.mark.anyio
async def test_movie_pipeline_generates_and_exposes_audio_only_when_opted_in(
    tmp_path,
    monkeypatch,
):
    import app.services.video as video_module

    async def no_wait(_seconds):
        raise AssertionError("Movie rendering must not use artificial stage delays.")

    monkeypatch.setattr(settings, "MEDIA_ROOT", tmp_path)
    monkeypatch.setattr(video_module.asyncio, "sleep", no_wait)
    audio_service = FakeAudioService("ai_audio")
    assembler = FakeMovieAssembler()
    service = VideoService(
        audio_service=audio_service,
        movie_assembler=assembler,
    )
    service.movies_dir = tmp_path / "movies"
    movie_id = "movie-audio-job"
    service._jobs[movie_id] = {"movie_id": movie_id, "status": "processing"}
    service._movie_audio_results[movie_id] = []

    await service._process_render(
        movie_id,
        "Test Movie",
        [make_scene()],
    )

    results = service.get_movie_audio_results(movie_id)
    assert audio_service.generated_scenes[0].scene_number == 1
    assert results[0]["scene_number"] == 1
    assert results[0]["audio"]["metadata"]["prompt"] == "cinematic forest at dawn"
    assert str(
        assembler.requests[0].scenes[0].ambience_tracks[0].path
    ).replace("\\", "/") == "/generated/audio/scene.wav"
    assert service._jobs[movie_id]["status"] == "completed"
    assert (tmp_path / "movies" / f"{movie_id}.mp4").is_file()


@pytest.mark.anyio
async def test_movie_pipeline_surfaces_scene_audio_failures(tmp_path, monkeypatch):
    import app.services.video as video_module

    class FailedAudioService(FakeAudioService):
        async def generate_scene_audio(self, scene):
            self.generated_scenes.append(scene)
            return AudioGenerationResult(
                provider=self.provider.name,
                status="failed",
                audio_type="ambience",
                metadata={"scene_number": scene.scene_number},
                error="test generation failure",
            )

    async def no_wait(_seconds):
        return None

    monkeypatch.setattr(settings, "MEDIA_ROOT", tmp_path)
    monkeypatch.setattr(video_module.asyncio, "sleep", no_wait)
    service = VideoService(
        audio_service=FailedAudioService("ai_audio"),
        movie_assembler=FakeMovieAssembler(),
    )
    service.movies_dir = tmp_path / "movies"
    movie_id = "movie-audio-failure"
    service._jobs[movie_id] = {"movie_id": movie_id, "status": "processing"}
    service._movie_audio_results[movie_id] = []

    await service._process_render(movie_id, "Test Movie", [make_scene()])

    results = service.get_movie_audio_results(movie_id)
    assert results[0]["audio"]["status"] == "failed"
    assert "scene audio failures" in service._jobs[movie_id]["message"]


@pytest.mark.anyio
async def test_placeholder_movie_pipeline_skips_scene_audio(tmp_path, monkeypatch):
    import app.services.video as video_module

    async def no_wait(_seconds):
        return None

    monkeypatch.setattr(settings, "MEDIA_ROOT", tmp_path)
    monkeypatch.setattr(video_module.asyncio, "sleep", no_wait)
    audio_service = FakeAudioService("placeholder")
    service = VideoService(
        audio_service=audio_service,
        movie_assembler=FakeMovieAssembler(),
    )
    service.movies_dir = tmp_path / "movies"
    movie_id = "movie-placeholder-job"
    service._jobs[movie_id] = {"movie_id": movie_id, "status": "processing"}
    service._movie_audio_results[movie_id] = []

    await service._process_render(
        movie_id,
        "Test Movie",
        [make_scene()],
    )

    assert audio_service.generated_scenes == []
    assert service.get_movie_audio_results(movie_id) == []
    assert service._jobs[movie_id]["status"] == "completed"


@pytest.mark.anyio
async def test_movie_pipeline_assembles_a_real_mp4_from_scene_assets(
    tmp_path,
    monkeypatch,
):
    import app.services.video as video_module

    async def no_wait(_seconds):
        return None

    monkeypatch.setattr(settings, "MEDIA_ROOT", tmp_path)
    monkeypatch.setattr(video_module.asyncio, "sleep", no_wait)
    image_path = tmp_path / "images" / "scene.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (320, 240), (25, 45, 70)).save(image_path)

    movies_dir = tmp_path / "movies"
    assembler = MovieAssembler(
        media_root=tmp_path,
        movies_dir=movies_dir,
        ffmpeg_executable=imageio_ffmpeg.get_ffmpeg_exe(),
        timeout_seconds=30,
    )
    service = VideoService(
        audio_service=FakeAudioService("placeholder"),
        movie_assembler=assembler,
    )
    service.movies_dir = movies_dir
    movie_id = "movie-real-assembly"
    service._jobs[movie_id] = {"movie_id": movie_id, "status": "processing"}

    await service._process_render(
        movie_id,
        "Test Movie",
        [make_scene()],
        fps=8,
        resolution="320x240",
    )

    output_path = movies_dir / f"{movie_id}.mp4"
    assert output_path.is_file()
    assert service._jobs[movie_id]["status"] == "completed"
    assert service._jobs[movie_id]["progress"] == 100
    assert service._jobs[movie_id]["video_url"] == (
        f"/generated/movies/{movie_id}.mp4"
    )


@pytest.mark.anyio
async def test_movie_pipeline_fails_without_visual_assets(tmp_path, monkeypatch):
    import app.services.video as video_module

    async def no_wait(_seconds):
        return None

    monkeypatch.setattr(settings, "MEDIA_ROOT", tmp_path)
    monkeypatch.setattr(video_module.asyncio, "sleep", no_wait)
    audio_service = FakeAudioService("ai_audio")
    movies_dir = tmp_path / "movies"
    service = VideoService(
        audio_service=audio_service,
        movie_assembler=MovieAssembler(
            media_root=tmp_path,
            movies_dir=movies_dir,
            ffmpeg_executable=imageio_ffmpeg.get_ffmpeg_exe(),
            timeout_seconds=30,
        ),
    )
    service.movies_dir = movies_dir
    movie_id = "movie-missing-assets"
    service._jobs[movie_id] = {"movie_id": movie_id, "status": "processing"}
    scene = make_scene()
    scene.image_url = "/generated/images/missing.png"

    await service._process_render(movie_id, "Test Movie", [scene])

    assert service._jobs[movie_id]["status"] == "failed"
    assert service._jobs[movie_id]["video_url"] is None
    assert not (movies_dir / f"{movie_id}.mp4").exists()
    assert audio_service.generated_scenes == []


@pytest.mark.anyio
async def test_movie_route_loads_scenes_for_opted_in_audio_provider(
    tmp_path,
    monkeypatch,
):
    video_module = import_module("app.services.video")
    movie_route = import_module("app.routes.movie")
    scene = make_scene()
    audio_service = FakeAudioService("ai_audio")
    monkeypatch.setattr(settings, "MEDIA_ROOT", tmp_path)
    service = VideoService(
        audio_service=audio_service,
        movie_assembler=FakeMovieAssembler(),
    )
    service.movies_dir = tmp_path / "movies"
    scheduled = []

    class FakeResult:
        def scalars(self):
            return self

        def all(self):
            return [scene]

        def scalar_one_or_none(self):
            return SimpleNamespace(id=scene.story_id)

    class FakeSession:
        async def execute(self, _statement):
            return FakeResult()

    class FakeTask:
        def add_done_callback(self, callback):
            self.done_callback = callback

    with monkeypatch.context() as context:
        context.setattr(movie_route, "video_service", service)
        async def persist_movie(_job):
            return None

        context.setattr(service, "_create_persisted_movie", persist_movie)
        context.setattr(
            video_module.asyncio,
            "create_task",
            lambda coroutine: (scheduled.append(coroutine), FakeTask())[1],
        )
        response = await movie_route.render_movie(
            MovieRenderRequest(story_id=scene.story_id, title="Test Movie"),
            db=FakeSession(),
        )

    assert response.status == "processing"
    assert response.progress == 0
    assert len(scheduled) == 1

    async def no_wait(_seconds):
        return None

    with monkeypatch.context() as context:
        context.setattr(video_module.asyncio, "sleep", no_wait)
        await scheduled[0]

    assert audio_service.generated_scenes[0].scene_number == 1
    assert (
        service.get_movie_audio_results(response.movie_id)[0]["audio"]["status"]
        == "generated"
    )


@pytest.mark.anyio
async def test_movie_route_preserves_render_response_when_story_has_no_scenes(
    monkeypatch,
):
    movie_route = import_module("app.routes.movie")
    service = VideoService(audio_service=FakeAudioService("ai_audio"))
    started = {}

    class FakeResult:
        def scalars(self):
            return self

        def all(self):
            return []

        def scalar_one_or_none(self):
            return SimpleNamespace(id="story-empty")

    class FakeSession:
        async def execute(self, _statement):
            return FakeResult()

    async def start_movie_render(request, scenes=None):
        started["scenes"] = scenes
        return movie_route.MovieRenderResponse(
            movie_id="movie-empty",
            story_id=request.story_id,
            title=request.title,
            status="processing",
        )

    monkeypatch.setattr(movie_route, "video_service", service)
    monkeypatch.setattr(service, "start_movie_render", start_movie_render)
    response = await movie_route.render_movie(
        MovieRenderRequest(story_id="story-empty", title="Empty Movie"),
        db=FakeSession(),
    )

    assert response.status == "processing"
    assert started["scenes"] == []


def test_movie_request_schema_remains_unchanged():
    request = MovieRenderRequest(story_id="story-1", title="A Film")

    assert request.fps == 24
    assert request.resolution == "1080p"
