import logging
import math
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from ..config import settings

logger = logging.getLogger(__name__)

MediaReference = Union[str, Path]
TransitionType = Literal["cut", "fade"]


class MovieAudioAsset(BaseModel):
    """A local audio file placed at an offset within its scene."""

    path: MediaReference
    offset_seconds: float = Field(default=0, ge=0)

    @field_validator("offset_seconds")
    @classmethod
    def validate_finite_offset(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Audio offset must be finite.")
        return value


class MovieSceneAssets(BaseModel):
    """Media references and timing for one scene in a movie timeline."""

    scene_number: int = Field(ge=1)
    duration_seconds: float = Field(gt=0, le=600)
    video_path: Optional[MediaReference] = None
    image_path: Optional[MediaReference] = None
    voice_tracks: List[MovieAudioAsset] = Field(default_factory=list)
    ambience_tracks: List[MovieAudioAsset] = Field(default_factory=list)
    sound_effect_tracks: List[MovieAudioAsset] = Field(default_factory=list)
    music_tracks: List[MovieAudioAsset] = Field(default_factory=list)
    transition_to_next: TransitionType = "cut"

    @field_validator("duration_seconds")
    @classmethod
    def validate_finite_duration(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Scene duration must be finite.")
        return value

    @model_validator(mode="after")
    def validate_visual_source(self) -> "MovieSceneAssets":
        if not self.video_path and not self.image_path:
            raise ValueError(
                f"Scene {self.scene_number} needs a video or storyboard image."
            )
        for tracks in (
            self.voice_tracks,
            self.ambience_tracks,
            self.sound_effect_tracks,
            self.music_tracks,
        ):
            if any(track.offset_seconds >= self.duration_seconds for track in tracks):
                raise ValueError(
                    f"Scene {self.scene_number} audio offsets must start "
                    "before the scene ends."
                )
        return self


class MovieAssemblyRequest(BaseModel):
    """Provider-neutral specification for assembling a complete MP4 movie."""

    movie_id: str = Field(min_length=1, max_length=64)
    story_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    scenes: List[MovieSceneAssets] = Field(min_length=1)
    fps: int = Field(default=24, ge=1, le=60)
    resolution: str = "1080p"
    output_format: Literal["mp4"] = "mp4"
    output_path: Optional[MediaReference] = None
    transition_duration_seconds: float = Field(default=0.5, gt=0, le=2)

    @field_validator("movie_id")
    @classmethod
    def validate_movie_id(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
            raise ValueError("Movie ID may contain only letters, numbers, '_' or '-'.")
        return value

    @field_validator("title", "story_id")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Movie title and story ID cannot be blank.")
        return value.strip()

    @field_validator("transition_duration_seconds")
    @classmethod
    def validate_finite_transition(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Transition duration must be finite.")
        return value

    @model_validator(mode="after")
    def validate_scene_order(self) -> "MovieAssemblyRequest":
        scene_numbers = [scene.scene_number for scene in self.scenes]
        if scene_numbers != sorted(scene_numbers) or len(set(scene_numbers)) != len(
            scene_numbers
        ):
            raise ValueError("Movie scenes must be ordered with unique scene numbers.")
        return self


class MovieAssemblyResult(BaseModel):
    """Validated output and media metadata from one assembly run."""

    status: Literal["generated"] = "generated"
    output_path: Path
    duration_seconds: float
    fps: int
    width: int
    height: int
    scene_count: int
    audio_track_count: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MovieAssemblyError(RuntimeError):
    """Raised when source media cannot be combined into a valid movie."""


class MovieAssembler:
    """Assemble local media files with FFmpeg without depending on ML runtimes."""

    _RESOLUTION_PRESETS = {
        "480p": (854, 480),
        "720p": (1280, 720),
        "1080p": (1920, 1080),
    }

    def __init__(
        self,
        *,
        media_root: Optional[Path] = None,
        movies_dir: Optional[Path] = None,
        ffmpeg_executable: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
    ) -> None:
        self.media_root = Path(media_root or settings.MEDIA_ROOT).resolve()
        self.movies_dir = Path(movies_dir or settings.MOVIES_DIR).resolve()
        self.ffmpeg_executable = ffmpeg_executable
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else settings.MOVIE_ASSEMBLY_TIMEOUT_SECONDS
        )

    def assemble(
        self,
        request: Union[MovieAssemblyRequest, Dict[str, Any]],
    ) -> MovieAssemblyResult:
        """Create a synchronized H.264/AAC MP4 from the supplied local assets."""
        assembly_request = self._validate_request(request)
        width, height = self._resolve_resolution(assembly_request.resolution)
        output_path = self._resolve_output_path(assembly_request)
        if output_path.exists():
            raise MovieAssemblyError(
                f"Movie output already exists and will not be overwritten: {output_path}"
            )
        visual_sources = self._resolve_visual_sources(assembly_request)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = self._build_command(
            assembly_request,
            visual_sources,
            output_path,
            width,
            height,
        )
        succeeded = False
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            if completed.returncode != 0:
                detail = completed.stderr.strip()[-4000:] or (
                    f"FFmpeg exited with code {completed.returncode}."
                )
                raise MovieAssemblyError(f"FFmpeg movie assembly failed: {detail}")
            if not output_path.is_file() or output_path.stat().st_size == 0:
                raise MovieAssemblyError(
                    "FFmpeg completed without creating a non-empty MP4 file."
                )
            succeeded = True
        except subprocess.TimeoutExpired as exc:
            raise MovieAssemblyError(
                f"Movie assembly exceeded the {self.timeout_seconds}-second timeout."
            ) from exc
        except OSError as exc:
            raise MovieAssemblyError(f"Could not run FFmpeg: {exc}") from exc
        except MovieAssemblyError:
            raise
        finally:
            if not succeeded and output_path.exists():
                try:
                    output_path.unlink()
                except OSError:
                    logger.exception(
                        "Could not remove incomplete movie output %s",
                        output_path,
                    )

        duration, track_count, transition_count = (
            self._timeline_metadata(assembly_request)
        )
        fallback_count = sum(
            image_path is not None
            for _, _, image_path in visual_sources
        )
        return MovieAssemblyResult(
            output_path=output_path,
            duration_seconds=duration,
            fps=assembly_request.fps,
            width=width,
            height=height,
            scene_count=len(assembly_request.scenes),
            audio_track_count=track_count,
            metadata={
                "fallback_image_scenes": fallback_count,
                "fade_transitions": transition_count,
                "video_codec": "libx264",
                "audio_codec": "aac" if track_count else None,
            },
        )

    def validate_assets(
        self,
        request: Union[MovieAssemblyRequest, Dict[str, Any]],
    ) -> None:
        """Validate references and output settings before optional GPU work starts."""
        assembly_request = self._validate_request(request)
        self._resolve_resolution(assembly_request.resolution)
        output_path = self._resolve_output_path(assembly_request)
        if output_path.exists():
            raise MovieAssemblyError(
                f"Movie output already exists and will not be overwritten: {output_path}"
            )
        self._resolve_visual_sources(assembly_request)
        self._resolve_audio_paths(assembly_request)

    @staticmethod
    def _validate_request(
        request: Union[MovieAssemblyRequest, Dict[str, Any]],
    ) -> MovieAssemblyRequest:
        try:
            return MovieAssemblyRequest.model_validate(request)
        except ValidationError as exc:
            raise MovieAssemblyError(f"Invalid movie assembly request: {exc}") from exc

    def _resolve_visual_sources(
        self,
        request: MovieAssemblyRequest,
    ) -> List[Tuple[MovieSceneAssets, Optional[Path], Optional[Path]]]:
        visual_sources: List[
            Tuple[MovieSceneAssets, Optional[Path], Optional[Path]]
        ] = []
        for scene in request.scenes:
            video_path = None
            image_path = None
            if scene.video_path:
                try:
                    video_path = self._resolve_media_path(scene.video_path)
                except FileNotFoundError:
                    logger.warning(
                        "Scene %s video is missing; trying its storyboard image.",
                        scene.scene_number,
                    )
            if video_path is None and scene.image_path:
                try:
                    image_path = self._resolve_media_path(scene.image_path)
                except FileNotFoundError as exc:
                    raise MovieAssemblyError(
                        f"Scene {scene.scene_number} storyboard image is missing: {exc}"
                    ) from exc
            if video_path is None and image_path is None:
                raise MovieAssemblyError(
                    f"Scene {scene.scene_number} has no usable video or image asset."
                )
            visual_sources.append((scene, video_path, image_path))
        return visual_sources

    def _resolve_audio_paths(self, request: MovieAssemblyRequest) -> None:
        for scene in request.scenes:
            for track in (
                *scene.voice_tracks,
                *scene.ambience_tracks,
                *scene.sound_effect_tracks,
                *scene.music_tracks,
            ):
                try:
                    self._resolve_media_path(track.path)
                except FileNotFoundError as exc:
                    raise MovieAssemblyError(
                        f"Scene {scene.scene_number} audio asset is missing: {exc}"
                    ) from exc

    def _resolve_ffmpeg(self) -> str:
        if self.ffmpeg_executable:
            return self.ffmpeg_executable
        try:
            import imageio_ffmpeg

            return imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError) as exc:
            raise MovieAssemblyError(
                "FFmpeg is unavailable; install the backend requirements."
            ) from exc

    def _resolve_resolution(self, value: str) -> tuple[int, int]:
        preset = self._RESOLUTION_PRESETS.get(value.strip().lower())
        if preset:
            return preset
        match = re.fullmatch(r"(\d{2,4})[xX](\d{2,4})", value.strip())
        if not match:
            supported = ", ".join(self._RESOLUTION_PRESETS)
            raise MovieAssemblyError(
                f"Unsupported resolution '{value}'; use one of [{supported}] "
                "or WIDTHxHEIGHT."
            )
        width, height = (int(match.group(1)), int(match.group(2)))
        if (
            width < 320
            or height < 240
            or width > 3840
            or height > 2160
            or width % 2
            or height % 2
        ):
            raise MovieAssemblyError(
                "Custom resolution must use even dimensions between 320x240 "
                "and 3840x2160."
            )
        return width, height

    def _resolve_media_path(self, reference: MediaReference) -> Path:
        raw = str(reference).strip()
        if not raw:
            raise MovieAssemblyError("Media file references cannot be blank.")

        normalized = raw.replace("\\", "/")
        if normalized.startswith("/generated/"):
            normalized = normalized[len("/generated/"):]
        elif normalized.startswith("generated/"):
            normalized = normalized[len("generated/"):]
        elif "://" in normalized:
            raise MovieAssemblyError(
                "Movie assembly accepts local files and /generated/ references only."
            )

        candidate = Path(normalized).expanduser()
        if not candidate.is_absolute():
            candidate = self.media_root / candidate
        resolved = candidate.resolve()
        if not resolved.is_relative_to(self.media_root):
            raise MovieAssemblyError(
                f"Media path is outside the generated-media directory: {reference}"
            )
        if not resolved.is_file():
            raise FileNotFoundError(f"Movie media file was not found: {resolved}")
        return resolved

    def _resolve_output_path(self, request: MovieAssemblyRequest) -> Path:
        if request.output_path is None:
            output_path = self.movies_dir / f"{request.movie_id}.mp4"
        else:
            output_path = Path(str(request.output_path)).expanduser()
            if not output_path.is_absolute():
                output_path = self.movies_dir / output_path
            output_path = output_path.resolve()

        output_path = output_path.resolve()
        if not output_path.is_relative_to(self.media_root):
            raise MovieAssemblyError(
                f"Movie output must be under the generated-media directory "
                f"{self.media_root}."
            )
        if output_path.parent != self.movies_dir:
            raise MovieAssemblyError(
                f"Movie output must be written directly under {self.movies_dir}."
            )
        if output_path.suffix.lower() != ".mp4":
            raise MovieAssemblyError("Movie output path must use the .mp4 extension.")
        return output_path

    def _build_command(
        self,
        request: MovieAssemblyRequest,
        visual_sources: List[
            Tuple[MovieSceneAssets, Optional[Path], Optional[Path]]
        ],
        output_path: Path,
        width: int,
        height: int,
    ) -> List[str]:
        command = [
            self._resolve_ffmpeg(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-n",
        ]
        video_input_indexes = []
        for _, video_path, image_path in visual_sources:
            input_index = len(video_input_indexes)
            if image_path is not None:
                command.extend(
                    [
                        "-loop",
                        "1",
                        "-framerate",
                        str(request.fps),
                        "-i",
                        str(image_path),
                    ]
                )
            else:
                command.extend(["-i", str(video_path)])
            video_input_indexes.append(input_index)

        audio_tracks = []
        for scene_index, (scene, _, _) in enumerate(visual_sources):
            track_groups = (
                scene.voice_tracks,
                scene.ambience_tracks,
                scene.sound_effect_tracks,
                scene.music_tracks,
            )
            for tracks in track_groups:
                for track in tracks:
                    input_index = len(video_input_indexes) + len(audio_tracks)
                    try:
                        audio_path = self._resolve_media_path(track.path)
                    except FileNotFoundError as exc:
                        raise MovieAssemblyError(
                            f"Scene {scene.scene_number} audio asset is missing: {exc}"
                        ) from exc
                    command.extend(["-i", str(audio_path)])
                    audio_tracks.append(
                        (
                            input_index,
                            scene_index,
                            track.offset_seconds,
                        )
                    )

        filter_parts = []
        for index, (scene, _, _) in enumerate(visual_sources):
            duration = self._format_seconds(scene.duration_seconds)
            input_index = video_input_indexes[index]
            filter_parts.append(
                f"[{input_index}:v:0]"
                f"fps={request.fps},"
                f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
                f"setsar=1,format=yuv420p,"
                f"tpad=stop_mode=clone:stop_duration={duration},"
                f"trim=duration={duration},setpts=PTS-STARTPTS,settb=AVTB"
                f"[scene{index}]"
            )

        scene_starts = [0.0]
        timeline_duration = request.scenes[0].duration_seconds
        previous_label = "scene0"
        for index in range(1, len(request.scenes)):
            previous_scene = request.scenes[index - 1]
            current_scene = request.scenes[index]
            transition_duration = 0.0
            if previous_scene.transition_to_next == "fade":
                transition_duration = min(
                    request.transition_duration_seconds,
                    previous_scene.duration_seconds / 2,
                    current_scene.duration_seconds / 2,
                )
            scene_starts.append(timeline_duration - transition_duration)
            next_label = f"joined{index}"
            if transition_duration:
                offset = timeline_duration - transition_duration
                filter_parts.append(
                    f"[{previous_label}][scene{index}]"
                    f"xfade=transition=fade:"
                    f"duration={self._format_seconds(transition_duration)}:"
                    f"offset={self._format_seconds(offset)}[{next_label}]"
                )
            else:
                filter_parts.append(
                    f"[{previous_label}][scene{index}]"
                    f"concat=n=2:v=1:a=0[{next_label}]"
                )
            timeline_duration += (
                current_scene.duration_seconds - transition_duration
            )
            previous_label = next_label

        final_video_label = previous_label
        if audio_tracks:
            for index, (input_index, scene_index, offset) in enumerate(audio_tracks):
                start = scene_starts[scene_index] + offset
                delay_ms = max(0, round(start * 1000))
                filter_parts.append(
                    f"[{input_index}:a:0]"
                    f"aresample=48000,aformat=channel_layouts=stereo,"
                    f"asetpts=PTS-STARTPTS,adelay={delay_ms}:all=1,"
                    f"apad,atrim=duration="
                    f"{self._format_seconds(timeline_duration)},"
                    f"asetpts=PTS-STARTPTS[audio{index}]"
                )
            audio_labels = "".join(
                f"[audio{index}]" for index in range(len(audio_tracks))
            )
            filter_parts.append(
                f"{audio_labels}amix=inputs={len(audio_tracks)}:"
                f"duration=longest:dropout_transition=0,"
                f"atrim=duration={self._format_seconds(timeline_duration)},"
                "asetpts=PTS-STARTPTS[audiomix]"
            )

        command.extend(["-filter_complex", ";".join(filter_parts)])
        command.extend(["-map", f"[{final_video_label}]"])
        if audio_tracks:
            command.extend(["-map", "[audiomix]"])
        command.extend(
            [
                "-t",
                self._format_seconds(timeline_duration),
                "-r",
                str(request.fps),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
            ]
        )
        if audio_tracks:
            command.extend(
                [
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-ar",
                    "48000",
                    "-ac",
                    "2",
                ]
            )
        command.extend(["-movflags", "+faststart", str(output_path)])
        return command

    @staticmethod
    def _format_seconds(value: float) -> str:
        return f"{value:.6f}".rstrip("0").rstrip(".")

    @staticmethod
    def _timeline_metadata(
        request: MovieAssemblyRequest,
    ) -> tuple[float, int, int]:
        duration = 0.0
        audio_track_count = 0
        fade_count = 0
        for index, scene in enumerate(request.scenes):
            if index:
                previous = request.scenes[index - 1]
                if previous.transition_to_next == "fade":
                    duration -= min(
                        request.transition_duration_seconds,
                        previous.duration_seconds / 2,
                        scene.duration_seconds / 2,
                    )
                    fade_count += 1
            duration += scene.duration_seconds
            audio_track_count += sum(
                len(tracks)
                for tracks in (
                    scene.voice_tracks,
                    scene.ambience_tracks,
                    scene.sound_effect_tracks,
                    scene.music_tracks,
                )
            )
        return duration, audio_track_count, fade_count
