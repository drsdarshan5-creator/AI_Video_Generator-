import uuid
import asyncio
import logging
from typing import Dict, Any, List, Optional, Sequence
from sqlalchemy.exc import SQLAlchemyError
from ..config import settings
from ..database import AsyncSessionLocal
from ..models import MovieRenderRequest, MovieRenderResponse, Scene
from ..models.db_models import Movie as DBMovie
from .audio import AudioService
from .audio_providers import AudioGenerationResult
from .movie_assembler import (
    MovieAssembler,
    MovieAssemblyError,
    MovieAssemblyRequest,
    MovieAudioAsset,
    MovieSceneAssets,
)
from .video_providers import (
    BaseVideoProvider,
    VideoGenerationRequest,
    VideoGenerationResult,
    get_video_provider,
)

logger = logging.getLogger(__name__)


class VideoService:
    """Service to generate video clips and stitch them into a complete movie."""

    def __init__(
        self,
        video_provider: Optional[BaseVideoProvider] = None,
        audio_service: Optional[AudioService] = None,
        movie_assembler: Optional[MovieAssembler] = None,
    ):
        self.videos_dir = settings.VIDEOS_DIR
        self.movies_dir = settings.MOVIES_DIR
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._movie_audio_results: Dict[str, List[Dict[str, Any]]] = {}
        self._render_tasks: Dict[str, asyncio.Task] = {}
        self._persisted_movie_ids: set[str] = set()
        self.video_provider = (
            video_provider if video_provider is not None else get_video_provider()
        )
        self.audio_service = (
            audio_service if audio_service is not None else AudioService()
        )
        self.movie_assembler = (
            movie_assembler
            if movie_assembler is not None
            else MovieAssembler()
        )

    async def generate_scene_video(
        self,
        request: VideoGenerationRequest,
    ) -> VideoGenerationResult:
        """Delegate one scene-video request to the configured provider."""
        provider_name = self.video_provider.name
        try:
            if not self.video_provider.is_available():
                return VideoGenerationResult(
                    provider=provider_name,
                    status="failed",
                    metadata=request.metadata.copy(),
                    error=f"Video provider '{provider_name}' is unavailable.",
                )

            result = await self.video_provider.generate_video(request)
            if not isinstance(result, VideoGenerationResult):
                result = VideoGenerationResult.model_validate(result)
            return result
        except Exception as exc:
            logger.exception(
                "Video provider '%s' failed to generate scene video",
                provider_name,
            )
            return VideoGenerationResult(
                provider=provider_name,
                status="failed",
                metadata=request.metadata.copy(),
                error=f"Video provider '{provider_name}' failed: {exc}",
            )

    async def start_movie_render(
        self,
        request: MovieRenderRequest,
        scenes: Optional[Sequence[Scene]] = None,
    ) -> MovieRenderResponse:
        """Initiate asynchronous rendering job for full movie."""
        movie_id = f"mov_{uuid.uuid4().hex[:8]}"
        movie_scenes = list(scenes or [])

        job_info = {
            "movie_id": movie_id,
            "story_id": request.story_id,
            "title": request.title,
            "status": "processing",
            "progress": 0,
            "video_url": None,
            "message": "Queued for scene asset validation.",
        }
        await self._create_persisted_movie(job_info)
        self._jobs[movie_id] = job_info
        self._movie_audio_results[movie_id] = []

        task = asyncio.create_task(
            self._process_render(
                movie_id,
                request.title,
                movie_scenes,
                story_id=request.story_id,
                fps=request.fps,
                resolution=request.resolution,
            )
        )
        self._render_tasks[movie_id] = task
        task.add_done_callback(
            lambda _task, current_id=movie_id: self._render_tasks.pop(
                current_id,
                None,
            )
        )

        return MovieRenderResponse(**job_info)

    async def get_job_status(self, movie_id: str) -> MovieRenderResponse:
        """Get status of an existing movie rendering job."""
        job = self._jobs.get(movie_id)
        if job is not None:
            return MovieRenderResponse(**job)

        try:
            async with AsyncSessionLocal() as db:
                movie = await db.get(DBMovie, movie_id)
                if movie is not None and movie.status == "processing":
                    movie.status = "failed"
                    movie.progress = 100
                    movie.video_url = None
                    movie.message = (
                        "Movie rendering was interrupted by a backend restart."
                    )
                    await db.commit()
        except SQLAlchemyError as exc:
            logger.exception("Could not retrieve movie job %s", movie_id)
            raise RuntimeError("Movie render status is temporarily unavailable.") from exc

        if movie is None:
            return MovieRenderResponse(
                movie_id=movie_id,
                story_id="",
                title="",
                status="failed",
                progress=0,
                message="Movie render job not found",
            )

        return MovieRenderResponse(
            movie_id=movie.id,
            story_id=movie.story_id or "",
            title=movie.title,
            status=movie.status,
            progress=movie.progress,
            video_url=movie.video_url,
            message=movie.message,
            created_at=movie.created_at,
        )

    def get_movie_audio_results(
        self,
        movie_id: str,
    ) -> Optional[List[Dict[str, Any]]]:
        """Return generated scene-audio metadata for an existing movie job."""
        if movie_id not in self._jobs:
            return None
        return list(self._movie_audio_results.get(movie_id, []))

    async def _create_persisted_movie(self, job: Dict[str, Any]) -> None:
        try:
            async with AsyncSessionLocal() as db:
                db.add(
                    DBMovie(
                        id=job["movie_id"],
                        story_id=job["story_id"],
                        title=job["title"],
                        status=job["status"],
                        progress=job["progress"],
                        video_url=job["video_url"],
                        message=job["message"],
                    )
                )
                await db.commit()
        except SQLAlchemyError:
            logger.exception(
                "Could not persist movie job %s",
                job["movie_id"],
            )
            raise
        self._persisted_movie_ids.add(job["movie_id"])

    async def _update_job(self, movie_id: str, **updates: Any) -> None:
        job = self._jobs.get(movie_id)
        if movie_id not in self._persisted_movie_ids:
            if job is not None:
                job.update(updates)
            return

        try:
            async with AsyncSessionLocal() as db:
                movie = await db.get(DBMovie, movie_id)
                if movie is None:
                    self._persisted_movie_ids.discard(movie_id)
                else:
                    for field in ("status", "progress", "video_url", "message"):
                        if field in updates:
                            setattr(movie, field, updates[field])
                    await db.commit()
        except SQLAlchemyError:
            logger.exception("Could not persist movie job status for %s", movie_id)
            if job is not None:
                job.update(updates)
                job["message"] = (
                    f"{job.get('message', '')} "
                    "Movie status could not be saved to PostgreSQL."
                ).strip()
            return

        if job is not None:
            job.update(updates)

    async def _process_render(
        self,
        movie_id: str,
        title: str,
        scenes: Optional[Sequence[Scene]] = None,
        *,
        story_id: Optional[str] = None,
        fps: int = 24,
        resolution: str = "1080p",
    ):
        """Assemble local scene assets into the final movie."""
        movie_scenes = list(scenes or [])
        audio_enabled = (
            bool(movie_scenes)
            and self.audio_service.provider.name != "placeholder"
        )
        try:
            preflight_request = self._build_assembly_request(
                movie_id=movie_id,
                story_id=story_id or (
                    movie_scenes[0].story_id if movie_scenes else None
                ),
                title=title,
                scenes=movie_scenes,
                fps=fps,
                resolution=resolution,
            )
            await asyncio.to_thread(
                self.movie_assembler.validate_assets,
                preflight_request,
            )
        except Exception as exc:
            logger.exception("Movie asset validation failed for movie %s", movie_id)
            await self._update_job(
                movie_id,
                status="failed",
                progress=100,
                video_url=None,
                message=f"Movie asset validation failed: {exc}",
            )
            return

        await self._update_job(
            movie_id,
            message="Scene assets validated; preparing the ordered timeline.",
        )
        if audio_enabled:
            await self._update_job(
                movie_id,
                message="Generating scene audio assets.",
            )
            await self._generate_movie_scene_audio(movie_id, movie_scenes)
        await self._update_job(
            movie_id,
            message="Assembling the final movie.",
        )

        try:
            assembly_request = self._build_assembly_request(
                movie_id=movie_id,
                story_id=story_id or (
                    movie_scenes[0].story_id if movie_scenes else None
                ),
                title=title,
                scenes=movie_scenes,
                fps=fps,
                resolution=resolution,
            )
            result = await asyncio.to_thread(
                self.movie_assembler.assemble,
                assembly_request,
            )
            media_root = settings.MEDIA_ROOT.resolve()
            relative_output = result.output_path.resolve().relative_to(media_root)
            video_url = f"/generated/{relative_output.as_posix()}"
            failed_audio = sum(
                item["audio"]["status"] == "failed"
                for item in self._movie_audio_results.get(movie_id, [])
            )
            await self._update_job(
                movie_id,
                status="completed",
                progress=100,
                video_url=video_url,
                message=(
                    "Movie assembled successfully with scene audio "
                    "failures; see the movie audio results."
                    if failed_audio
                    else "Movie assembled successfully."
                ),
            )
        except Exception as exc:
            logger.exception("Movie assembly failed for movie %s", movie_id)
            await self._update_job(
                movie_id,
                status="failed",
                progress=100,
                video_url=None,
                message=f"Movie assembly failed: {exc}",
            )

    def _build_assembly_request(
        self,
        *,
        movie_id: str,
        story_id: Optional[str],
        title: str,
        scenes: Sequence[Scene],
        fps: int,
        resolution: str,
    ) -> MovieAssemblyRequest:
        if not story_id:
            raise MovieAssemblyError("A story ID is required for movie assembly.")

        audio_by_scene = {
            int(item["scene_number"]): item["audio"]
            for item in self._movie_audio_results.get(movie_id, [])
        }
        assembly_scenes = []
        for scene in scenes:
            voice_tracks = [
                MovieAudioAsset(path=dialogue.audio_url)
                for dialogue in scene.dialogues
                if dialogue.audio_url
            ]
            ambience_tracks = []
            sound_effect_tracks = []
            music_tracks = []
            generated_audio = audio_by_scene.get(scene.scene_number, {})
            audio_path = generated_audio.get("output_path")
            if generated_audio.get("status") == "generated" and audio_path:
                audio_asset = MovieAudioAsset(path=audio_path)
                audio_type = generated_audio.get("audio_type")
                if audio_type == "ambience":
                    ambience_tracks.append(audio_asset)
                elif audio_type == "sound_effect":
                    sound_effect_tracks.append(audio_asset)
                elif audio_type in {"music", "procedural_music"}:
                    music_tracks.append(audio_asset)

            assembly_scenes.append(
                MovieSceneAssets(
                    scene_number=scene.scene_number,
                    duration_seconds=scene.duration_seconds,
                    video_path=scene.video_url,
                    image_path=scene.image_url,
                    voice_tracks=voice_tracks,
                    ambience_tracks=ambience_tracks,
                    sound_effect_tracks=sound_effect_tracks,
                    music_tracks=music_tracks,
                    transition_to_next=getattr(
                        scene,
                        "transition_to_next",
                        "cut",
                    ),
                )
            )

        return MovieAssemblyRequest(
            movie_id=movie_id,
            story_id=story_id,
            title=title,
            scenes=assembly_scenes,
            fps=fps,
            resolution=resolution,
            output_path=self.movies_dir / f"{movie_id}.mp4",
        )

    async def _generate_movie_scene_audio(
        self,
        movie_id: str,
        scenes: Sequence[Scene],
    ) -> None:
        results = self._movie_audio_results.setdefault(movie_id, [])
        for scene in scenes:
            try:
                result = await self.audio_service.generate_scene_audio(scene)
            except Exception as exc:
                logger.exception(
                    "Scene audio generation failed for movie %s scene %s",
                    movie_id,
                    scene.scene_number,
                )
                result = AudioGenerationResult(
                    provider=self.audio_service.provider.name,
                    status="failed",
                    audio_type=None,
                    metadata={"scene_number": scene.scene_number},
                    error=f"Scene audio generation failed: {exc}",
                )
            results.append(
                {
                    "scene_number": scene.scene_number,
                    "scene_id": scene.id,
                    "audio": result.model_dump(mode="json"),
                }
            )
