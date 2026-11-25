"""Persistent, resumable batches of independently generated video shots."""

import argparse
import asyncio
import contextlib
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, Sequence

import imageio_ffmpeg
from pydantic import BaseModel, Field

from ..config import settings
from .video import VideoService
from .video_providers import VideoGenerationRequest, VideoGenerationResult

ShotStatus = Literal["pending", "running", "completed", "failed"]
BatchStatus = Literal["pending", "running", "completed", "failed"]


class VideoBatchError(RuntimeError):
    """Raised when a batch manifest cannot be safely created or executed."""


class VideoBatchShotRequest(BaseModel):
    shot_number: int = Field(gt=0)
    request: VideoGenerationRequest


class VideoShotAttempt(BaseModel):
    attempt_number: int
    status: Literal["running", "completed", "failed"]
    output_path: Path
    started_at: datetime
    completed_at: Optional[datetime] = None
    generation_time_seconds: Optional[float] = None
    expected_output: Dict[str, Any] = Field(default_factory=dict)
    validation_result: Dict[str, Any] = Field(default_factory=dict)
    generation_metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class VideoBatchShot(BaseModel):
    shot_number: int
    request: VideoGenerationRequest
    status: ShotStatus = "pending"
    output_path: Optional[Path] = None
    generation_time_seconds: Optional[float] = None
    validation_result: Dict[str, Any] = Field(default_factory=dict)
    attempts: List[VideoShotAttempt] = Field(default_factory=list)
    error: Optional[str] = None


class VideoBatchManifest(BaseModel):
    version: Literal[1] = 1
    batch_id: str
    created_at: datetime
    updated_at: datetime
    status: BatchStatus = "pending"
    warm_worker_enabled: bool = False
    shots: List[VideoBatchShot]
    error: Optional[str] = None


def validate_mp4(
    output_path: Path,
    expected_output: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Decode an MP4 and validate its dimensions, frame rate, and frame count."""
    path = Path(output_path)
    expected = expected_output or {}
    errors: List[str] = []
    result: Dict[str, Any] = {
        "valid": False,
        "path": str(path),
        "file_size_bytes": 0,
        "width": None,
        "height": None,
        "fps": None,
        "frames": 0,
        "errors": errors,
    }

    if path.suffix.lower() != ".mp4":
        errors.append("Output filename must have an .mp4 extension.")
    if not path.is_file():
        errors.append("Output MP4 does not exist.")
        return result

    result["file_size_bytes"] = path.stat().st_size
    if result["file_size_bytes"] <= 0:
        errors.append("Output MP4 is empty.")
        return result

    reader = None
    try:
        reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
        metadata = next(reader)
        width, height = metadata["size"]
        fps = float(metadata["fps"])
        frame_count = sum(1 for _ in reader)
        result.update(
            {
                "width": int(width),
                "height": int(height),
                "fps": fps,
                "frames": frame_count,
            }
        )
        if expected.get("width") is not None and width != expected["width"]:
            errors.append(
                f"Expected width {expected['width']}, decoded width was {width}."
            )
        if expected.get("height") is not None and height != expected["height"]:
            errors.append(
                f"Expected height {expected['height']}, decoded height was {height}."
            )
        if expected.get("fps") is not None and abs(fps - expected["fps"]) > 0.1:
            errors.append(f"Expected {expected['fps']} FPS, decoded {fps} FPS.")
        if (
            expected.get("frames") is not None
            and frame_count != expected["frames"]
        ):
            errors.append(
                f"Expected {expected['frames']} frames, decoded {frame_count}."
            )
    except Exception as exc:
        errors.append(f"Could not decode MP4: {exc}")
    finally:
        if reader is not None:
            reader.close()

    result["valid"] = not errors
    return result


@contextlib.contextmanager
def _exclusive_file_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_file:
        if os.name == "nt":
            import msvcrt

            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write(b"\0")
                lock_file.flush()
            lock_file.seek(0)
            try:
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise VideoBatchError(
                    f"Batch is already being modified: {lock_path.parent.name}"
                ) from exc
            try:
                yield
            finally:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise VideoBatchError(
                    f"Batch is already being modified: {lock_path.parent.name}"
                ) from exc
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


class VideoBatchRunner:
    """Create, resume, and retry durable per-shot generation batches."""

    def __init__(
        self,
        manifest_root: Optional[Path] = None,
        video_service: Optional[VideoService] = None,
    ) -> None:
        root = manifest_root or (settings.COGVIDEOX_OUTPUT_DIR / "batches")
        self.manifest_root = Path(root).expanduser().resolve()
        self.video_service = video_service or VideoService()

    @staticmethod
    def _validate_batch_id(batch_id: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", batch_id):
            raise VideoBatchError(
                "Batch ID must be 1-64 letters, numbers, underscores, or hyphens."
            )
        return batch_id

    def _batch_dir(self, batch_id: str) -> Path:
        return self.manifest_root / self._validate_batch_id(batch_id)

    def _manifest_path(self, batch_id: str) -> Path:
        return self._batch_dir(batch_id) / "manifest.json"

    def _locked(self, batch_id: str):
        return _exclusive_file_lock(self._batch_dir(batch_id) / ".manifest.lock")

    def create_batch(
        self,
        batch_id: str,
        shots: Sequence[VideoBatchShotRequest],
    ) -> VideoBatchManifest:
        batch_id = self._validate_batch_id(batch_id)
        if not shots:
            raise VideoBatchError("A batch must contain at least one shot.")
        shot_numbers = [shot.shot_number for shot in shots]
        if len(shot_numbers) != len(set(shot_numbers)):
            raise VideoBatchError("Shot numbers must be unique within a batch.")

        with self._locked(batch_id):
            manifest_path = self._manifest_path(batch_id)
            if manifest_path.exists():
                raise VideoBatchError(f"Batch already exists: {batch_id}")

            now = datetime.now(timezone.utc)
            manifest = VideoBatchManifest(
                batch_id=batch_id,
                created_at=now,
                updated_at=now,
                shots=[
                    VideoBatchShot(
                        shot_number=shot.shot_number,
                        request=shot.request.model_copy(
                            update={"output_path": None}
                        ),
                    )
                    for shot in sorted(shots, key=lambda item: item.shot_number)
                ],
            )
            self._write_manifest(manifest)
            return manifest

    def load_batch(self, batch_id: str) -> VideoBatchManifest:
        manifest_path = self._manifest_path(batch_id)
        try:
            with manifest_path.open("r", encoding="utf-8") as manifest_file:
                return VideoBatchManifest.model_validate_json(manifest_file.read())
        except FileNotFoundError as exc:
            raise VideoBatchError(f"Batch manifest not found: {batch_id}") from exc
        except (OSError, ValueError) as exc:
            raise VideoBatchError(
                f"Could not read batch manifest for {batch_id}: {exc}"
            ) from exc

    def _write_manifest(self, manifest: VideoBatchManifest) -> None:
        manifest.updated_at = datetime.now(timezone.utc)
        manifest_path = self._manifest_path(manifest.batch_id)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = manifest_path.with_name(
            f".{manifest_path.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            with temporary_path.open("w", encoding="utf-8") as manifest_file:
                manifest_file.write(
                    json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n"
                )
                manifest_file.flush()
                os.fsync(manifest_file.fileno())
            os.replace(temporary_path, manifest_path)
        except OSError as exc:
            raise VideoBatchError(
                f"Could not persist batch manifest {manifest_path}: {exc}"
            ) from exc
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    @staticmethod
    def _expected_output(request: VideoGenerationRequest) -> Dict[str, Any]:
        fps = request.fps or settings.COGVIDEOX_FPS
        frames = (
            round(request.duration_seconds * fps)
            if request.duration_seconds is not None
            else settings.COGVIDEOX_FRAMES
        )
        return {
            "width": request.width or settings.COGVIDEOX_WIDTH,
            "height": request.height or settings.COGVIDEOX_HEIGHT,
            "fps": fps,
            "frames": frames,
        }

    @staticmethod
    def _expected_from_result(
        result: VideoGenerationResult,
        fallback: Dict[str, Any],
    ) -> Dict[str, Any]:
        frames = result.metadata.get("frames")
        if frames is None and result.duration_seconds is not None and result.fps:
            frames = round(result.duration_seconds * result.fps)
        return {
            "width": result.width if result.width is not None else fallback["width"],
            "height": result.height if result.height is not None else fallback["height"],
            "fps": result.fps if result.fps is not None else fallback["fps"],
            "frames": frames if frames is not None else fallback["frames"],
        }

    def _recover_interrupted(self, manifest: VideoBatchManifest) -> None:
        for shot in manifest.shots:
            if shot.status != "running":
                continue
            if not shot.attempts or shot.attempts[-1].status != "running":
                shot.status = "failed"
                shot.error = "Interrupted shot has no running attempt checkpoint."
                self._write_manifest(manifest)
                continue

            attempt = shot.attempts[-1]
            validation = validate_mp4(
                attempt.output_path,
                attempt.expected_output,
            )
            elapsed = max(
                0.0,
                (datetime.now(timezone.utc) - attempt.started_at).total_seconds(),
            )
            attempt.completed_at = datetime.now(timezone.utc)
            attempt.generation_time_seconds = elapsed
            attempt.validation_result = validation
            shot.generation_time_seconds = elapsed
            shot.validation_result = validation
            shot.output_path = attempt.output_path
            if validation["valid"]:
                attempt.status = "completed"
                shot.status = "completed"
                attempt.error = None
                shot.error = None
            else:
                attempt.status = "failed"
                shot.status = "failed"
                attempt.error = (
                    "Generation was interrupted before checkpoint completion; "
                    "the output did not pass MP4 validation."
                )
                shot.error = attempt.error
            self._write_manifest(manifest)

    def _revalidate_completed(self, manifest: VideoBatchManifest) -> None:
        for shot in manifest.shots:
            if shot.status != "completed" or shot.output_path is None:
                continue
            expected = (
                shot.attempts[-1].expected_output
                if shot.attempts
                else self._expected_output(shot.request)
            )
            validation = validate_mp4(shot.output_path, expected)
            shot.validation_result = validation
            if not validation["valid"]:
                shot.status = "failed"
                shot.error = (
                    "Previously completed output no longer passes validation: "
                    + "; ".join(validation["errors"])
                )
            self._write_manifest(manifest)

    async def run_batch(
        self,
        batch_id: str,
        *,
        warm_worker: bool = False,
    ) -> VideoBatchManifest:
        with self._locked(batch_id):
            manifest = self.load_batch(batch_id)
            self._recover_interrupted(manifest)
            self._revalidate_completed(manifest)
            manifest.warm_worker_enabled = warm_worker
            manifest.error = None
            manifest.status = "running"
            self._write_manifest(manifest)
            try:
                has_incomplete_shots = any(
                    shot.status in ("pending", "failed") for shot in manifest.shots
                )
                if warm_worker and has_incomplete_shots:
                    warm_session = getattr(
                        self.video_service.video_provider,
                        "warm_session",
                        None,
                    )
                    if not callable(warm_session):
                        raise VideoBatchError(
                            "The configured video provider does not support warm-worker mode."
                        )
                    async with warm_session():
                        await self._run_incomplete_shots(manifest)
                else:
                    await self._run_incomplete_shots(manifest)
            except asyncio.CancelledError:
                manifest.error = "Batch execution was cancelled."
                self._refresh_batch_status(manifest)
                self._write_manifest(manifest)
                raise
            except Exception as exc:
                manifest.error = str(exc)
                manifest.status = "failed"
                self._write_manifest(manifest)
                raise
            self._refresh_batch_status(manifest)
            self._write_manifest(manifest)
            return manifest

    async def retry_shot(
        self,
        batch_id: str,
        shot_number: int,
        *,
        warm_worker: bool = False,
    ) -> VideoBatchManifest:
        with self._locked(batch_id):
            manifest = self.load_batch(batch_id)
            self._recover_interrupted(manifest)
            self._revalidate_completed(manifest)
            shot = next(
                (item for item in manifest.shots if item.shot_number == shot_number),
                None,
            )
            if shot is None:
                raise VideoBatchError(
                    f"Shot {shot_number} is not present in batch {batch_id}."
                )
            if shot.status != "failed":
                raise VideoBatchError(
                    f"Only failed shots can be retried; shot {shot_number} is "
                    f"{shot.status}."
                )

            manifest.warm_worker_enabled = warm_worker
            manifest.error = None
            manifest.status = "running"
            self._write_manifest(manifest)
            try:
                if warm_worker:
                    warm_session = getattr(
                        self.video_service.video_provider,
                        "warm_session",
                        None,
                    )
                    if not callable(warm_session):
                        raise VideoBatchError(
                            "The configured video provider does not support warm-worker mode."
                        )
                    async with warm_session():
                        await self._run_one_shot(manifest, shot)
                else:
                    await self._run_one_shot(manifest, shot)
            except asyncio.CancelledError:
                manifest.error = "Shot retry was cancelled."
                self._refresh_batch_status(manifest)
                self._write_manifest(manifest)
                raise
            except Exception as exc:
                manifest.error = str(exc)
                manifest.status = "failed"
                self._write_manifest(manifest)
                raise
            self._refresh_batch_status(manifest)
            self._write_manifest(manifest)
            return manifest

    async def _run_incomplete_shots(
        self,
        manifest: VideoBatchManifest,
    ) -> None:
        for shot in manifest.shots:
            if shot.status in ("pending", "failed"):
                await self._run_one_shot(manifest, shot)
            if shot.status == "failed":
                break

    async def _run_one_shot(
        self,
        manifest: VideoBatchManifest,
        shot: VideoBatchShot,
    ) -> None:
        attempt_number = len(shot.attempts) + 1
        output_path = (
            self._batch_dir(manifest.batch_id)
            / f"shot_{shot.shot_number:03d}_attempt_{attempt_number:03d}.mp4"
        )
        expected = self._expected_output(shot.request)
        attempt = VideoShotAttempt(
            attempt_number=attempt_number,
            status="running",
            output_path=output_path,
            started_at=datetime.now(timezone.utc),
            expected_output=expected,
        )
        shot.status = "running"
        shot.output_path = output_path
        shot.generation_time_seconds = None
        shot.validation_result = {}
        shot.error = None
        shot.attempts.append(attempt)
        self._write_manifest(manifest)
        started = time.monotonic()

        request_metadata = {
            **shot.request.metadata,
            "batch_id": manifest.batch_id,
            "shot_number": shot.shot_number,
            "attempt_number": attempt_number,
        }
        request = shot.request.model_copy(
            update={
                "output_path": output_path,
                "metadata": request_metadata,
            }
        )
        try:
            result = await self.video_service.generate_scene_video(request)
            attempt.generation_metadata = dict(result.metadata)
            if result.status != "generated":
                raise VideoBatchError(
                    result.error
                    or f"Provider returned status '{result.status}' instead of generated."
                )
            if result.output_path is None:
                raise VideoBatchError(
                    "Provider reported generation success without an output path."
                )

            actual_expected = self._expected_from_result(result, expected)
            validation = validate_mp4(result.output_path, actual_expected)
            attempt.output_path = result.output_path
            attempt.validation_result = validation
            shot.output_path = result.output_path
            shot.validation_result = validation
            if not validation["valid"]:
                raise VideoBatchError(
                    "Generated MP4 failed validation: "
                    + "; ".join(validation["errors"])
                )
            attempt.status = "completed"
            shot.status = "completed"
        except asyncio.CancelledError:
            attempt.status = "failed"
            attempt.error = "Shot generation was cancelled."
            shot.status = "failed"
            shot.error = attempt.error
            raise
        except Exception as exc:
            attempt.status = "failed"
            attempt.error = str(exc)
            shot.status = "failed"
            shot.error = attempt.error
        finally:
            elapsed = max(0.0, time.monotonic() - started)
            attempt.completed_at = datetime.now(timezone.utc)
            attempt.generation_time_seconds = elapsed
            shot.generation_time_seconds = elapsed
            self._write_manifest(manifest)

    @staticmethod
    def _refresh_batch_status(manifest: VideoBatchManifest) -> None:
        if all(shot.status == "completed" for shot in manifest.shots):
            manifest.status = "completed"
        elif any(shot.status == "failed" for shot in manifest.shots):
            manifest.status = "failed"
        elif any(shot.status == "running" for shot in manifest.shots):
            manifest.status = "running"
        else:
            manifest.status = "pending"


def _read_shots(path: Path) -> List[VideoBatchShotRequest]:
    try:
        with path.open("r", encoding="utf-8") as shots_file:
            raw = json.load(shots_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise VideoBatchError(f"Could not read shot requests from {path}: {exc}") from exc
    if isinstance(raw, dict):
        raw = raw.get("shots")
    if not isinstance(raw, list):
        raise VideoBatchError("Shot request JSON must be a list or an object with 'shots'.")
    return [VideoBatchShotRequest.model_validate(item) for item in raw]


def _print_manifest(manifest: VideoBatchManifest) -> None:
    print(
        json.dumps(
            {
                "batch_id": manifest.batch_id,
                "status": manifest.status,
                "warm_worker_enabled": manifest.warm_worker_enabled,
                "shots": [
                    {
                        "shot_number": shot.shot_number,
                        "status": shot.status,
                        "output_path": (
                            str(shot.output_path) if shot.output_path else None
                        ),
                        "generation_time_seconds": shot.generation_time_seconds,
                        "validation_result": shot.validation_result,
                        "error": shot.error,
                    }
                    for shot in manifest.shots
                ],
            },
            indent=2,
        )
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run resumable video shot batches.")
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=settings.COGVIDEOX_OUTPUT_DIR / "batches",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    create_parser = commands.add_parser("create", help="Create a batch manifest.")
    create_parser.add_argument("--batch-id", required=True)
    create_parser.add_argument("--shots-json", required=True, type=Path)
    run_parser = commands.add_parser("run", help="Run or resume an incomplete batch.")
    run_parser.add_argument("--batch-id", required=True)
    run_parser.add_argument("--warm-worker", action="store_true")
    retry_parser = commands.add_parser("retry", help="Retry one failed shot.")
    retry_parser.add_argument("--batch-id", required=True)
    retry_parser.add_argument("--shot-number", required=True, type=int)
    retry_parser.add_argument("--warm-worker", action="store_true")
    args = parser.parse_args(argv)
    runner = VideoBatchRunner(manifest_root=args.manifest_root)

    try:
        if args.command == "create":
            manifest = runner.create_batch(
                args.batch_id,
                _read_shots(args.shots_json),
            )
        elif args.command == "run":
            manifest = asyncio.run(
                runner.run_batch(args.batch_id, warm_worker=args.warm_worker)
            )
        else:
            manifest = asyncio.run(
                runner.retry_shot(
                    args.batch_id,
                    args.shot_number,
                    warm_worker=args.warm_worker,
                )
            )
    except Exception as exc:
        parser.exit(1, f"error: {exc}\n")

    _print_manifest(manifest)
    return 1 if manifest.status == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
