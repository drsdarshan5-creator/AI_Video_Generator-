import asyncio
import contextlib
import json
import logging
import math
import os
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional

from ...config import BASE_DIR, Settings, settings
from .base import (
    BaseVideoProvider,
    ProviderExecutionError,
    VideoGenerationRequest,
    VideoGenerationResult,
)

logger = logging.getLogger(__name__)


class CogVideoXVideoProvider(BaseVideoProvider):
    """Run CogVideoX in its isolated Python environment on explicit requests."""

    name = "cogvideox"
    SUPPORTED_WIDTH = 720
    SUPPORTED_HEIGHT = 480
    MAX_FRAMES = 49
    _generation_lock = asyncio.Lock()

    def __init__(self, config: Optional[Settings] = None) -> None:
        self.config = config if config is not None else settings
        self.python_path = self._resolve_path(self.config.COGVIDEOX_PYTHON_PATH)
        self.worker_path = Path(__file__).with_name("cogvideox_worker.py").resolve()
        self.output_dir = self._resolve_path(self.config.COGVIDEOX_OUTPUT_DIR)
        self.model_path = (
            self._resolve_path(self.config.COGVIDEOX_MODEL_PATH)
            if self.config.COGVIDEOX_MODEL_PATH is not None
            else None
        )
        self.cache_dir = (
            self._resolve_path(self.config.COGVIDEOX_CACHE_DIR)
            if self.config.COGVIDEOX_CACHE_DIR is not None
            else None
        )
        self._warm_process: Optional[asyncio.subprocess.Process] = None

    @staticmethod
    def _resolve_path(path: Path) -> Path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = BASE_DIR / candidate
        return candidate.resolve()

    def is_available(self) -> bool:
        """Check only the isolated interpreter and worker, never load ML modules."""
        return self.python_path.is_file() and self.worker_path.is_file()

    def _generation_parameters(
        self,
        request: VideoGenerationRequest,
    ) -> Dict[str, Any]:
        if request.image_path is not None or request.image_url is not None:
            raise ValueError("CogVideoX-2B is configured for text-to-video only.")

        width = request.width if request.width is not None else self.config.COGVIDEOX_WIDTH
        height = (
            request.height if request.height is not None else self.config.COGVIDEOX_HEIGHT
        )
        if (width, height) != (self.SUPPORTED_WIDTH, self.SUPPORTED_HEIGHT):
            raise ValueError(
                "CogVideoX-2B currently supports only the verified 720x480 resolution."
            )

        fps = request.fps if request.fps is not None else self.config.COGVIDEOX_FPS
        if not math.isfinite(fps) or not 1 <= fps <= 30:
            raise ValueError("CogVideoX FPS must be between 1 and 30.")

        frames = self.config.COGVIDEOX_FRAMES
        if request.duration_seconds is not None:
            if not math.isfinite(request.duration_seconds):
                raise ValueError("CogVideoX duration must be finite.")
            frames = round(request.duration_seconds * fps)
        if (
            frames < 4
            or frames > self.MAX_FRAMES
            or (frames != 49 and frames % 4 != 0)
        ):
            raise ValueError(
                "CogVideoX frame count must be 4-49 and divisible by 4, "
                "or exactly 49."
            )

        seed = request.seed if request.seed is not None else self.config.COGVIDEOX_SEED
        if seed < 0:
            raise ValueError("CogVideoX seed must be non-negative.")

        model_id = self.config.COGVIDEOX_MODEL_ID.strip()
        revision = self.config.COGVIDEOX_MODEL_REVISION.strip()
        if not model_id or not revision:
            raise ValueError("CogVideoX model ID and revision must be configured.")
        if self.model_path is not None and not self.model_path.is_dir():
            raise ProviderExecutionError(
                f"Configured CogVideoX model path does not exist: {self.model_path}"
            )

        return {
            "model_id": model_id,
            "model_revision": revision,
            "model_path": str(self.model_path) if self.model_path is not None else None,
            "cache_dir": str(self.cache_dir) if self.cache_dir is not None else None,
            "prompt": request.prompt,
            "width": width,
            "height": height,
            "frames": frames,
            "fps": fps,
            "inference_steps": self.config.COGVIDEOX_STEPS,
            "guidance_scale": self.config.COGVIDEOX_GUIDANCE_SCALE,
            "scheduler": self.config.COGVIDEOX_SCHEDULER,
            "cpu_offload": self.config.COGVIDEOX_CPU_OFFLOAD,
            "use_dynamic_cfg": self.config.COGVIDEOX_USE_DYNAMIC_CFG,
            "attention_slicing": self.config.COGVIDEOX_ATTENTION_SLICING,
            "vae_slicing": self.config.COGVIDEOX_VAE_SLICING,
            "vae_tiling": self.config.COGVIDEOX_VAE_TILING,
            "dtype": self.config.COGVIDEOX_DTYPE,
            "local_files_only": self.config.COGVIDEOX_LOCAL_FILES_ONLY,
            "seed": seed,
        }

    @staticmethod
    def _remove_incomplete_output(output_path: Path) -> None:
        if output_path.exists():
            try:
                output_path.unlink()
            except OSError:
                logger.exception(
                    "Could not remove incomplete CogVideoX output %s.",
                    output_path,
                )

    async def _run_worker(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            process = await asyncio.create_subprocess_exec(
                str(self.python_path),
                str(self.worker_path),
                cwd=str(BASE_DIR),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._worker_env(),
            )
        except OSError as exc:
            raise ProviderExecutionError(
                f"Could not start the isolated CogVideoX runtime at "
                f"{self.python_path}: {exc}"
            ) from exc

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(json.dumps(payload).encode("utf-8")),
                timeout=self.config.COGVIDEOX_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as exc:
            if process.returncode is None:
                process.kill()
            await process.communicate()
            raise ProviderExecutionError(
                "CogVideoX exceeded the configured "
                f"{self.config.COGVIDEOX_TIMEOUT_SECONDS}-second timeout."
            ) from exc
        except asyncio.CancelledError:
            if process.returncode is None:
                process.kill()
            await process.communicate()
            raise

        stdout_text = stdout.decode("utf-8", errors="replace").strip()
        stderr_text = stderr.decode("utf-8", errors="replace").strip()
        try:
            response = json.loads(stdout_text)
        except json.JSONDecodeError as exc:
            detail = stderr_text[-4000:] or stdout_text[-1000:] or "No worker output."
            raise ProviderExecutionError(
                f"Isolated CogVideoX worker returned invalid output: {detail}"
            ) from exc

        if process.returncode != 0 or not response.get("ok"):
            message = response.get("error") or stderr_text[-4000:]
            raise ProviderExecutionError(
                f"Isolated CogVideoX generation failed: {message or 'unknown worker error'}"
            )
        return response

    def _worker_env(self) -> Dict[str, str]:
        return {
            **os.environ,
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "TQDM_DISABLE": "1",
        }

    async def _start_warm_worker(self) -> asyncio.subprocess.Process:
        try:
            return await asyncio.create_subprocess_exec(
                str(self.python_path),
                str(self.worker_path),
                "--batch-stdio",
                cwd=str(BASE_DIR),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._worker_env(),
            )
        except OSError as exc:
            raise ProviderExecutionError(
                f"Could not start the warm CogVideoX runtime at "
                f"{self.python_path}: {exc}"
            ) from exc

    async def _stop_warm_worker(self) -> None:
        process = self._warm_process
        self._warm_process = None
        if process is None:
            return
        try:
            if process.returncode is None and process.stdin is not None:
                try:
                    process.stdin.write(b'{"command": "shutdown"}\n')
                    await process.stdin.drain()
                    process.stdin.close()
                except OSError:
                    pass
                try:
                    await asyncio.wait_for(process.wait(), timeout=15)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()

    @contextlib.asynccontextmanager
    async def warm_session(self) -> AsyncIterator["CogVideoXVideoProvider"]:
        """Keep one isolated CogVideoX worker alive across multiple sequential shots."""
        if not self.is_available():
            raise ProviderExecutionError(
                "The isolated CogVideoX environment is unavailable. Expected Python "
                f"at {self.python_path} and worker at {self.worker_path}."
            )
        async with self._generation_lock:
            self._warm_process = await self._start_warm_worker()
        try:
            yield self
        finally:
            async with self._generation_lock:
                await self._stop_warm_worker()

    async def _run_warm_worker(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        process = self._warm_process
        if process is None or process.returncode is not None:
            process = await self._start_warm_worker()
            self._warm_process = process

        assert process.stdin is not None and process.stdout is not None
        try:
            process.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
            await process.stdin.drain()
            raw_line = await asyncio.wait_for(
                process.stdout.readline(),
                timeout=self.config.COGVIDEOX_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as exc:
            await self._stop_warm_worker()
            raise ProviderExecutionError(
                "CogVideoX warm worker exceeded the configured "
                f"{self.config.COGVIDEOX_TIMEOUT_SECONDS}-second timeout."
            ) from exc
        except asyncio.CancelledError:
            await self._stop_warm_worker()
            raise
        except OSError as exc:
            await self._stop_warm_worker()
            raise ProviderExecutionError(
                f"Lost connection to the warm CogVideoX worker: {exc}"
            ) from exc

        line_text = raw_line.decode("utf-8", errors="replace").strip()
        if not line_text:
            stderr_text = ""
            if process.stderr is not None:
                stderr_data = await process.stderr.read()
                stderr_text = stderr_data.decode("utf-8", errors="replace").strip()
            await self._stop_warm_worker()
            detail = stderr_text[-4000:] or "Warm worker exited without output."
            raise ProviderExecutionError(
                f"Isolated warm CogVideoX worker terminated unexpectedly: {detail}"
            )

        try:
            response = json.loads(line_text)
        except json.JSONDecodeError as exc:
            await self._stop_warm_worker()
            raise ProviderExecutionError(
                f"Isolated warm CogVideoX worker returned invalid output: {line_text[-1000:]}"
            ) from exc

        if not response.get("ok"):
            message = response.get("error") or "unknown warm worker error"
            raise ProviderExecutionError(
                f"Isolated CogVideoX generation failed: {message}"
            )
        return response

    async def generate_video(
        self,
        request: VideoGenerationRequest,
    ) -> VideoGenerationResult:
        payload = self._generation_parameters(request)
        if not self.is_available():
            raise ProviderExecutionError(
                "The isolated CogVideoX environment is unavailable. Expected Python "
                f"at {self.python_path} and worker at {self.worker_path}."
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        if request.output_path is None:
            output_path = self.output_dir / f"cogvideox_{uuid.uuid4().hex}.mp4"
        else:
            output_path = self._resolve_path(request.output_path)
            try:
                output_path.relative_to(self.output_dir)
            except ValueError as exc:
                raise ValueError(
                    "CogVideoX output path must be inside its configured output directory."
                ) from exc
            if output_path.suffix.lower() != ".mp4":
                raise ValueError("CogVideoX output path must use the .mp4 extension.")
            if output_path.exists():
                raise ProviderExecutionError(
                    f"CogVideoX output path already exists: {output_path}"
                )
            output_path.parent.mkdir(parents=True, exist_ok=True)
        payload["output_path"] = str(output_path)

        try:
            async with self._generation_lock:
                if self._warm_process is not None:
                    response = await self._run_warm_worker(payload)
                else:
                    response = await self._run_worker(payload)

            if not output_path.is_file() or output_path.stat().st_size <= 0:
                raise ProviderExecutionError(
                    "CogVideoX worker reported success but produced no non-empty MP4."
                )

            generated = response["result"]
            frames = int(generated["frames"])
            fps = float(generated["fps"])
            metadata = dict(request.metadata)
            metadata.update(
                {
                    "model_revision": payload["model_revision"],
                    "frames": frames,
                    "inference_steps": payload["inference_steps"],
                    "guidance_scale": payload["guidance_scale"],
                    "scheduler": generated.get("scheduler", payload["scheduler"]),
                    "use_dynamic_cfg": payload["use_dynamic_cfg"],
                    "attention_slicing": payload["attention_slicing"],
                    "vae_slicing": payload["vae_slicing"],
                    "vae_tiling": payload["vae_tiling"],
                    "local_files_only": payload["local_files_only"],
                    "seed": payload["seed"],
                    "device": generated["device"],
                    "dtype": generated["dtype"],
                    "cpu_offload": generated.get("cpu_offload", payload["cpu_offload"]),
                    "gpu": generated["gpu"],
                    "gpu_vram_free_before_inference_bytes": generated[
                        "gpu_vram_free_before_inference_bytes"
                    ],
                    "gpu_peak_allocated_bytes": generated["gpu_peak_allocated_bytes"],
                    "gpu_peak_reserved_bytes": generated["gpu_peak_reserved_bytes"],
                    "file_size_bytes": output_path.stat().st_size,
                }
            )
            return VideoGenerationResult(
                provider=self.name,
                status="generated",
                output_path=output_path,
                duration_seconds=frames / fps,
                width=int(generated["width"]),
                height=int(generated["height"]),
                fps=fps,
                format="mp4",
                runtime_provider="cuda",
                model=payload["model_id"],
                inference_time_seconds=float(generated["inference_time_seconds"]),
                metadata=metadata,
            )
        except asyncio.CancelledError:
            self._remove_incomplete_output(output_path)
            raise
        except Exception:
            self._remove_incomplete_output(output_path)
            raise
