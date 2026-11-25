import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from app.config import BASE_DIR, Settings, settings
from app.services.video import VideoService
from app.services.video_providers import (
    BaseVideoProvider,
    CogVideoXVideoProvider,
    ProviderExecutionError,
    VideoGenerationRequest,
    get_video_provider,
    list_video_providers,
)
from app.services.video_providers import cogvideox


def make_request(**overrides):
    values = {
        "prompt": "A cinematic night street in the rain.",
        "width": 720,
        "height": 480,
        "fps": 8,
        "seed": 42,
        "metadata": {"source": "test"},
    }
    values.update(overrides)
    return VideoGenerationRequest(**values)


def make_settings(tmp_path, **overrides):
    values = {
        "_env_file": None,
        "COGVIDEOX_PYTHON_PATH": Path(sys.executable),
        "COGVIDEOX_OUTPUT_DIR": tmp_path,
    }
    values.update(overrides)
    return Settings(**values)


def test_cogvideox_provider_is_registered_and_implements_interface():
    assert "placeholder" in list_video_providers()
    assert "cogvideox" in list_video_providers()
    assert isinstance(get_video_provider("cogvideox"), CogVideoXVideoProvider)
    assert issubclass(CogVideoXVideoProvider, BaseVideoProvider)
    assert Settings(_env_file=None).VIDEO_PROVIDER == "placeholder"


def test_importing_provider_does_not_load_ml_runtime():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BASE_DIR / "backend")
    code = (
        "import sys; "
        "from app.services.video_providers import get_video_provider; "
        "get_video_provider('cogvideox'); "
        "blocked={'torch','diffusers','transformers'}; "
        "assert blocked.isdisjoint(sys.modules), blocked.intersection(sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=BASE_DIR,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_cogvideox_configuration_validation():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, COGVIDEOX_WIDTH=721)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, COGVIDEOX_FRAMES=6)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, COGVIDEOX_FPS=0)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, COGVIDEOX_SCHEDULER="euler")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, COGVIDEOX_CPU_OFFLOAD="invalid")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, COGVIDEOX_DTYPE="float32")


def test_cogvideox_optimized_settings_are_configurable(tmp_path):
    cfg = make_settings(
        tmp_path,
        COGVIDEOX_FRAMES=24,
        COGVIDEOX_FPS=8,
        COGVIDEOX_STEPS=20,
        COGVIDEOX_GUIDANCE_SCALE=6.0,
        COGVIDEOX_SCHEDULER="DPM",
        COGVIDEOX_CPU_OFFLOAD="HYBRID",
        COGVIDEOX_USE_DYNAMIC_CFG=False,
        COGVIDEOX_ATTENTION_SLICING=True,
        COGVIDEOX_VAE_SLICING=True,
        COGVIDEOX_VAE_TILING=True,
        COGVIDEOX_DTYPE="float16",
        COGVIDEOX_LOCAL_FILES_ONLY=True,
    )
    provider = CogVideoXVideoProvider(cfg)
    payload = provider._generation_parameters(
        VideoGenerationRequest(prompt="A cinematic street scene.")
    )
    assert payload["frames"] == 24
    assert payload["fps"] == 8.0
    assert payload["inference_steps"] == 20
    assert payload["guidance_scale"] == 6.0
    assert payload["scheduler"] == "dpm"
    assert payload["cpu_offload"] == "hybrid"
    assert payload["use_dynamic_cfg"] is False
    assert payload["attention_slicing"] is True
    assert payload["vae_slicing"] is True
    assert payload["vae_tiling"] is True
    assert payload["dtype"] == "float16"
    assert payload["local_files_only"] is True


@pytest.mark.anyio
async def test_invalid_resolution_is_rejected_before_worker_start(tmp_path, monkeypatch):
    spawn = AsyncMock()
    monkeypatch.setattr(cogvideox.asyncio, "create_subprocess_exec", spawn)
    provider = CogVideoXVideoProvider(make_settings(tmp_path))

    with pytest.raises(ValueError, match="only the verified 720x480"):
        await provider.generate_video(make_request(width=640))

    spawn.assert_not_awaited()


@pytest.mark.anyio
async def test_invalid_request_frame_count_is_rejected_before_worker_start(
    tmp_path,
    monkeypatch,
):
    spawn = AsyncMock()
    monkeypatch.setattr(cogvideox.asyncio, "create_subprocess_exec", spawn)
    provider = CogVideoXVideoProvider(make_settings(tmp_path))

    with pytest.raises(ValueError, match="frame count"):
        await provider.generate_video(make_request(duration_seconds=0.75))

    spawn.assert_not_awaited()


@pytest.mark.anyio
async def test_invalid_fps_is_rejected_before_worker_start(tmp_path, monkeypatch):
    spawn = AsyncMock()
    monkeypatch.setattr(cogvideox.asyncio, "create_subprocess_exec", spawn)
    provider = CogVideoXVideoProvider(make_settings(tmp_path))

    with pytest.raises(ValueError, match="FPS must be between"):
        await provider.generate_video(make_request(fps=0.5))

    spawn.assert_not_awaited()


@pytest.mark.anyio
async def test_image_input_is_rejected_for_text_to_video_model(tmp_path, monkeypatch):
    spawn = AsyncMock()
    monkeypatch.setattr(cogvideox.asyncio, "create_subprocess_exec", spawn)
    provider = CogVideoXVideoProvider(make_settings(tmp_path))

    with pytest.raises(ValueError, match="text-to-video only"):
        await provider.generate_video(make_request(image_url="https://example.test/frame.png"))

    spawn.assert_not_awaited()


@pytest.mark.anyio
async def test_missing_isolated_runtime_is_unavailable(tmp_path):
    provider = CogVideoXVideoProvider(
        make_settings(tmp_path, COGVIDEOX_PYTHON_PATH=tmp_path / "missing-python.exe")
    )

    assert not provider.is_available()
    with pytest.raises(ProviderExecutionError, match="isolated CogVideoX environment"):
        await provider.generate_video(make_request())


class SuccessfulWorker:
    returncode = 0

    async def communicate(self, input=None):
        payload = json.loads(input)
        Path(payload["output_path"]).write_bytes(b"test-mp4-content")
        response = {
            "ok": True,
            "result": {
                "width": 720,
                "height": 480,
                "frames": 8,
                "fps": 8,
                "inference_time_seconds": 20.0,
                "device": "cuda:0",
                "dtype": "float16",
                "gpu": "NVIDIA GeForce RTX 5050 Laptop GPU",
                "gpu_vram_free_before_inference_bytes": 7_000_000_000,
                "gpu_peak_allocated_bytes": 3_000_000_000,
                "gpu_peak_reserved_bytes": 3_400_000_000,
            },
        }
        return json.dumps(response).encode("utf-8"), b""


@pytest.mark.anyio
async def test_provider_returns_structured_video_metadata(tmp_path, monkeypatch):
    worker = SuccessfulWorker()
    spawn = AsyncMock(return_value=worker)
    monkeypatch.setattr(cogvideox.asyncio, "create_subprocess_exec", spawn)
    provider = CogVideoXVideoProvider(make_settings(tmp_path))

    result = await provider.generate_video(make_request())

    assert result.provider == "cogvideox"
    assert result.status == "generated"
    assert result.model == "zai-org/CogVideoX-2b"
    assert result.model is not None
    assert result.output_path is not None and result.output_path.is_file()
    assert result.output_path.parent == tmp_path
    assert result.width == 720
    assert result.height == 480
    assert result.fps == 8
    assert result.duration_seconds == 1
    assert result.format == "mp4"
    assert result.runtime_provider == "cuda"
    assert result.inference_time_seconds == 20
    assert result.metadata["model_revision"] == settings.COGVIDEOX_MODEL_REVISION
    assert result.metadata["frames"] == 8
    assert result.metadata["inference_steps"] == 2
    assert result.metadata["seed"] == 42
    assert result.metadata["device"] == "cuda:0"
    assert result.metadata["dtype"] == "float16"
    assert result.metadata["cpu_offload"] == "sequential"
    assert result.metadata["source"] == "test"
    spawn.assert_awaited_once()


@pytest.mark.anyio
async def test_provider_uses_requested_batch_output_path_and_refuses_overwrite(
    tmp_path,
    monkeypatch,
):
    worker = SuccessfulWorker()
    spawn = AsyncMock(return_value=worker)
    monkeypatch.setattr(cogvideox.asyncio, "create_subprocess_exec", spawn)
    provider = CogVideoXVideoProvider(make_settings(tmp_path))
    requested_path = tmp_path / "batch-one" / "shot_001_attempt_001.mp4"

    result = await provider.generate_video(
        make_request(output_path=requested_path)
    )

    assert result.output_path == requested_path
    assert requested_path.is_file()
    assert spawn.await_count == 1
    with pytest.raises(ProviderExecutionError, match="already exists"):
        await provider.generate_video(make_request(output_path=requested_path))
    assert spawn.await_count == 1


@pytest.mark.anyio
async def test_provider_rejects_batch_output_paths_outside_output_directory(
    tmp_path,
    monkeypatch,
):
    spawn = AsyncMock()
    monkeypatch.setattr(cogvideox.asyncio, "create_subprocess_exec", spawn)
    provider = CogVideoXVideoProvider(make_settings(tmp_path))

    with pytest.raises(ValueError, match="inside its configured output directory"):
        await provider.generate_video(
            make_request(output_path=tmp_path.parent / "outside.mp4")
        )

    spawn.assert_not_awaited()


@pytest.mark.anyio
async def test_worker_runtime_errors_are_reported_and_video_service_wraps_them(
    tmp_path,
    monkeypatch,
):
    class FailedWorker:
        returncode = 1

        async def communicate(self, input=None):
            del input
            return (
                json.dumps(
                    {
                        "ok": False,
                        "error": "RuntimeError: CUDA is unavailable; CPU-only inference is disabled.",
                    }
                ).encode("utf-8"),
                b"worker traceback",
            )

    monkeypatch.setattr(
        cogvideox.asyncio,
        "create_subprocess_exec",
        AsyncMock(return_value=FailedWorker()),
    )
    provider = CogVideoXVideoProvider(make_settings(tmp_path))

    with pytest.raises(ProviderExecutionError, match="CUDA is unavailable"):
        await provider.generate_video(make_request())

    result = await VideoService(video_provider=provider).generate_scene_video(
        make_request()
    )
    assert result.status == "failed"
    assert "CUDA is unavailable" in result.error
    assert list(tmp_path.iterdir()) == []


@pytest.mark.anyio
async def test_warm_session_reuses_one_worker_and_shuts_down(tmp_path, monkeypatch):
    class WarmStdin:
        def __init__(self, worker):
            self.worker = worker
            self.closed = False

        def write(self, data: bytes):
            line = data.decode("utf-8").strip()
            payload = json.loads(line)
            if payload.get("command") == "shutdown":
                self.worker.shutdown_received = True
                return
            self.worker.requests.append(payload)
            Path(payload["output_path"]).write_bytes(b"warm-mp4")
            resp = {
                "ok": True,
                "result": {
                    "width": 720,
                    "height": 480,
                    "frames": 8,
                    "fps": 8,
                    "inference_time_seconds": 12.5,
                    "device": "cuda:0",
                    "dtype": "float16",
                    "cpu_offload": "hybrid",
                    "scheduler": "dpm",
                    "gpu": "NVIDIA GeForce RTX 5050 Laptop GPU",
                    "gpu_vram_free_before_inference_bytes": 7_000_000_000,
                    "gpu_peak_allocated_bytes": 4_000_000_000,
                    "gpu_peak_reserved_bytes": 4_900_000_000,
                },
            }
            self.worker.responses.append((json.dumps(resp) + "\n").encode("utf-8"))

        async def drain(self):
            return None

        def close(self):
            self.closed = True

    class WarmStdout:
        def __init__(self, worker):
            self.worker = worker

        async def readline(self):
            return self.worker.responses.pop(0) if self.worker.responses else b""

    class WarmStderr:
        async def read(self):
            return b""

    class FakeWarmWorker:
        def __init__(self):
            self.returncode = None
            self.requests = []
            self.responses = []
            self.shutdown_received = False
            self.stdin = WarmStdin(self)
            self.stdout = WarmStdout(self)
            self.stderr = WarmStderr()

        async def wait(self):
            self.returncode = 0
            return 0

        def kill(self):
            self.returncode = -9

    warm_worker = FakeWarmWorker()
    spawn = AsyncMock(return_value=warm_worker)
    monkeypatch.setattr(cogvideox.asyncio, "create_subprocess_exec", spawn)
    provider = CogVideoXVideoProvider(make_settings(tmp_path))

    async with provider.warm_session():
        r1 = await provider.generate_video(make_request(seed=1))
        r2 = await provider.generate_video(make_request(seed=2))

    assert spawn.await_count == 1
    assert "--batch-stdio" in spawn.call_args.args
    assert len(warm_worker.requests) == 2
    assert warm_worker.shutdown_received is True
    assert r1.status == "generated" and r1.output_path.is_file()
    assert r2.status == "generated" and r2.output_path.is_file()
    assert r1.metadata["cpu_offload"] == "hybrid"
    assert r1.metadata["scheduler"] == "dpm"
