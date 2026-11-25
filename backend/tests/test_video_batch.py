import asyncio
import json
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path

import imageio_ffmpeg
import pytest

from app.services.video import VideoService
from app.services.video_batch import (
    VideoBatchRunner,
    VideoBatchShotRequest,
    VideoShotAttempt,
    validate_mp4,
)
from app.services.video_providers import (
    BaseVideoProvider,
    VideoGenerationRequest,
    VideoGenerationResult,
)


def write_test_mp4(path: Path, *, width: int = 16, height: int = 16, fps: int = 8):
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s={width}x{height}:r={fps}:d=1",
            "-frames:v",
            str(fps),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        capture_output=True,
        check=True,
    )


class FakeVideoProvider(BaseVideoProvider):
    name = "fake"

    def __init__(self, fail_once_for=None):
        self.fail_once_for = set(fail_once_for or [])
        self.calls = []
        self.warm_starts = 0
        self.warm_stops = 0
        self.warm_active = False
        self.warm_flags = []

    def is_available(self):
        return True

    @asynccontextmanager
    async def warm_session(self):
        self.warm_starts += 1
        self.warm_active = True
        try:
            yield self
        finally:
            self.warm_active = False
            self.warm_stops += 1

    async def generate_video(self, request: VideoGenerationRequest):
        shot_number = request.metadata["shot_number"]
        attempt_number = request.metadata["attempt_number"]
        self.calls.append((shot_number, attempt_number))
        self.warm_flags.append(self.warm_active)
        if (shot_number, attempt_number) in self.fail_once_for:
            return VideoGenerationResult(
                provider=self.name,
                status="failed",
                metadata={"simulated_failure": True},
                error="Simulated shot failure.",
            )

        assert request.output_path is not None
        await asyncio.to_thread(write_test_mp4, request.output_path)
        return VideoGenerationResult(
            provider=self.name,
            status="generated",
            output_path=request.output_path,
            duration_seconds=1,
            width=16,
            height=16,
            fps=8,
            format="mp4",
            metadata={"frames": 8, "synthetic_test": True},
        )


def make_shots():
    return [
        VideoBatchShotRequest(
            shot_number=number,
            request=VideoGenerationRequest(
                prompt=f"Synthetic test scene {number}.",
                duration_seconds=1,
                width=16,
                height=16,
                fps=8,
                metadata={"test_scene": number},
            ),
        )
        for number in (1, 2, 3)
    ]


def make_runner(tmp_path, provider):
    return VideoBatchRunner(
        manifest_root=tmp_path / "manifests",
        video_service=VideoService(video_provider=provider),
    )


@pytest.mark.anyio
async def test_batch_resume_reuses_success_and_retries_only_failed_shot(tmp_path):
    provider = FakeVideoProvider(fail_once_for={(2, 1)})
    runner = make_runner(tmp_path, provider)
    initial = runner.create_batch("synthetic-three", make_shots())

    assert initial.status == "pending"
    manifest_path = (
        tmp_path / "manifests" / "synthetic-three" / "manifest.json"
    )
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert saved["shots"][0]["request"]["prompt"] == "Synthetic test scene 1."
    assert saved["shots"][0]["request"]["metadata"] == {"test_scene": 1}

    failed = await runner.run_batch("synthetic-three")
    assert failed.status == "failed"
    assert [shot.status for shot in failed.shots] == [
        "completed",
        "failed",
        "pending",
    ]
    first_output = failed.shots[0].output_path
    first_bytes = first_output.read_bytes()
    assert failed.shots[0].validation_result["valid"] is True
    assert failed.shots[0].generation_time_seconds is not None
    assert provider.calls == [(1, 1), (2, 1)]

    retried = await runner.retry_shot("synthetic-three", 2)
    assert retried.shots[1].status == "completed"
    assert len(retried.shots[1].attempts) == 2
    assert retried.shots[1].attempts[0].status == "failed"
    assert retried.shots[1].attempts[1].validation_result["valid"] is True
    assert provider.calls[-1] == (2, 2)

    completed = await runner.run_batch("synthetic-three")
    assert completed.status == "completed"
    assert [shot.status for shot in completed.shots] == [
        "completed",
        "completed",
        "completed",
    ]
    assert provider.calls == [(1, 1), (2, 1), (2, 2), (3, 1)]
    assert first_output.read_bytes() == first_bytes
    assert all(shot.validation_result["valid"] for shot in completed.shots)
    assert all(shot.output_path.is_file() for shot in completed.shots)


@pytest.mark.anyio
async def test_resume_recovers_valid_output_from_interrupted_running_shot(tmp_path):
    provider = FakeVideoProvider()
    runner = make_runner(tmp_path, provider)
    manifest = runner.create_batch("interrupted-three", make_shots())
    shot = manifest.shots[0]
    output_path = (
        tmp_path
        / "manifests"
        / "interrupted-three"
        / "shot_001_attempt_001.mp4"
    )
    await asyncio.to_thread(write_test_mp4, output_path)
    shot.status = "running"
    shot.output_path = output_path
    shot.attempts.append(
        VideoShotAttempt(
            attempt_number=1,
            status="running",
            output_path=output_path,
            started_at=manifest.created_at,
            expected_output={"width": 16, "height": 16, "fps": 8, "frames": 8},
        )
    )
    with runner._locked(manifest.batch_id):
        runner._write_manifest(manifest)

    resumed = await runner.run_batch("interrupted-three")

    assert resumed.status == "completed"
    assert resumed.shots[0].status == "completed"
    assert resumed.shots[0].attempts[0].validation_result["valid"] is True
    assert provider.calls == [(2, 1), (3, 1)]
    assert output_path.is_file()


@pytest.mark.anyio
async def test_warm_worker_mode_wraps_all_shots_in_one_warm_session(tmp_path):
    provider = FakeVideoProvider()
    runner = make_runner(tmp_path, provider)
    runner.create_batch("warm-three", make_shots())

    manifest = await runner.run_batch("warm-three", warm_worker=True)

    assert manifest.status == "completed"
    assert manifest.warm_worker_enabled is True
    assert provider.warm_starts == 1
    assert provider.warm_stops == 1
    assert provider.warm_flags == [True, True, True]


def test_mp4_validator_decodes_video_and_reports_corrupt_files(tmp_path):
    valid_path = tmp_path / "valid.mp4"
    write_test_mp4(valid_path)

    valid = validate_mp4(
        valid_path,
        {"width": 16, "height": 16, "fps": 8, "frames": 8},
    )
    assert valid["valid"] is True
    assert valid["frames"] == 8
    assert valid["width"] == 16
    assert valid["height"] == 16

    invalid_path = tmp_path / "invalid.mp4"
    invalid_path.write_bytes(b"not an mp4")
    invalid = validate_mp4(invalid_path)
    assert invalid["valid"] is False
    assert invalid["errors"]
