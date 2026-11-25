import subprocess
import wave
from pathlib import Path

from PIL import Image
import pytest

import imageio_ffmpeg

from app.services.movie_assembler import (
    MovieAssembler,
    MovieAssemblyError,
    MovieAssemblyRequest,
    MovieAudioAsset,
    MovieSceneAssets,
)


def create_wav(path: Path, duration_seconds: float = 0.4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 16000
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(
            b"\x10\x00" * int(sample_rate * duration_seconds)
        )


def create_image(path: Path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (320, 240), color).save(path)


def make_request(output_path=None):
    return MovieAssemblyRequest(
        movie_id="assembly-test",
        story_id="story-test",
        title="Assembly Test",
        fps=8,
        resolution="320x240",
        output_path=output_path,
        transition_duration_seconds=0.25,
        scenes=[
            MovieSceneAssets(
                scene_number=1,
                duration_seconds=1.5,
                video_path="/generated/videos/missing.mp4",
                image_path="/generated/images/scene-1.png",
                transition_to_next="fade",
                ambience_tracks=[
                    MovieAudioAsset(path="/generated/audio/ambience.wav")
                ],
            ),
            MovieSceneAssets(
                scene_number=2,
                duration_seconds=1.5,
                video_path="/generated/videos/scene-2.mp4",
                image_path="/generated/images/scene-2.png",
                voice_tracks=[
                    MovieAudioAsset(
                        path="/generated/audio/voice.wav",
                        offset_seconds=0.1,
                    )
                ],
            ),
        ],
    )


def test_assembler_renders_mp4_with_fade_and_synchronized_audio(tmp_path):
    scene1_image = tmp_path / "images" / "scene-1.png"
    scene2_image = tmp_path / "images" / "scene-2.png"
    create_image(scene1_image, (15, 25, 45))
    create_image(scene2_image, (75, 35, 25))
    video_path = tmp_path / "videos" / "scene-2.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    encoded_scene = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-loop",
            "1",
            "-framerate",
            "8",
            "-i",
            str(scene2_image),
            "-frames:v",
            "12",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert encoded_scene.returncode == 0, encoded_scene.stderr
    create_wav(tmp_path / "audio" / "ambience.wav")
    create_wav(tmp_path / "audio" / "voice.wav")
    assembler = MovieAssembler(
        media_root=tmp_path,
        movies_dir=tmp_path / "movies",
        ffmpeg_executable=imageio_ffmpeg.get_ffmpeg_exe(),
        timeout_seconds=30,
    )

    result = assembler.assemble(make_request())

    assert result.status == "generated"
    assert result.output_path.is_file()
    assert result.output_path.stat().st_size > 1000
    assert result.duration_seconds == pytest.approx(2.75)
    assert result.audio_track_count == 2
    assert result.metadata["fade_transitions"] == 1
    assert result.metadata["fallback_image_scenes"] == 1
    decode = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(result.output_path),
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert decode.returncode == 0, decode.stderr
    audio_decode = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(result.output_path),
            "-map",
            "0:a:0",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert audio_decode.returncode == 0, audio_decode.stderr


def test_assembler_rejects_media_outside_generated_root(tmp_path):
    media_root = tmp_path / "generated"
    outside_image = tmp_path / "outside.png"
    create_image(outside_image, (0, 0, 0))
    request = MovieAssemblyRequest(
        movie_id="unsafe-path",
        story_id="story-test",
        title="Unsafe Path",
        resolution="320x240",
        scenes=[
            MovieSceneAssets(
                scene_number=1,
                duration_seconds=1,
                image_path=outside_image,
            )
        ],
    )
    assembler = MovieAssembler(
        media_root=media_root,
        movies_dir=media_root / "movies",
        ffmpeg_executable=imageio_ffmpeg.get_ffmpeg_exe(),
    )

    with pytest.raises(MovieAssemblyError, match="outside the generated-media"):
        assembler.assemble(request)


def test_movie_request_requires_ordered_scenes_and_visual_sources():
    request = make_request()
    request_values = request.model_dump()
    request_values["scenes"] = [
        request.scenes[1],
        request.scenes[0],
    ]
    with pytest.raises(ValueError, match="ordered with unique"):
        MovieAssemblyRequest(**request_values)

    with pytest.raises(ValueError, match="needs a video or storyboard image"):
        MovieSceneAssets(scene_number=1, duration_seconds=1)
