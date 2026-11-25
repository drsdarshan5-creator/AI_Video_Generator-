import asyncio
import json
import logging
import math
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ...config import BASE_DIR, Settings, settings
from .audio_validation import validate_pcm_wav
from .base import (
    AudioGenerationRequest,
    AudioGenerationResult,
    AudioProviderExecutionError,
    BaseAudioProvider,
)

logger = logging.getLogger(__name__)

SUPPORTED_AUDIO_TYPES = {"ambience", "sound_effect", "music"}


class AIAudioProvider(BaseAudioProvider):
    """Run AudioLDM 2 in a separate CUDA-enabled Python environment."""

    name = "ai_audio"
    _generation_lock = asyncio.Lock()

    def __init__(self, config: Optional[Settings] = None) -> None:
        self.config = config if config is not None else settings
        self.python_path = self._resolve_path(self.config.AI_AUDIO_PYTHON_PATH)
        self.worker_path = Path(__file__).with_name("ai_audio_worker.py").resolve()
        self.model_path = self._resolve_path(self.config.AI_AUDIO_MODEL_PATH)
        self.output_dir = self._resolve_path(self.config.AUDIO_OUTPUT_DIR)

    @staticmethod
    def _resolve_path(path: Path) -> Path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = BASE_DIR / candidate
        return candidate.resolve()

    def is_available(self) -> bool:
        """Check configured files only; never import the audio ML runtime."""
        return (
            self.python_path.is_file()
            and self.worker_path.is_file()
            and (self.model_path / "model_index.json").is_file()
        )

    def _generation_parameters(
        self,
        request: AudioGenerationRequest,
    ) -> Tuple[Dict[str, Any], Path]:
        if request.audio_type not in SUPPORTED_AUDIO_TYPES:
            supported = ", ".join(sorted(SUPPORTED_AUDIO_TYPES))
            raise ValueError(
                f"ai_audio supports only [{supported}], not '{request.audio_type}'."
            )

        prompt = request.prompt or request.text or request.description
        if prompt is None or not prompt.strip():
            raise ValueError("AI audio generation requires a non-blank prompt.")
        prompt = prompt.strip()

        duration = request.duration_seconds
        if duration is None or not math.isfinite(duration):
            raise ValueError("AI audio generation requires a finite duration.")
        if not 0.25 <= duration <= self.config.AI_AUDIO_MAX_DURATION_SECONDS:
            raise ValueError(
                "AI audio duration must be between 0.25 and "
                f"{self.config.AI_AUDIO_MAX_DURATION_SECONDS:g} seconds."
            )

        sample_rate = request.sample_rate or self.config.AUDIO_DEFAULT_SAMPLE_RATE
        if not 8000 <= sample_rate <= 48000:
            raise ValueError("AI audio sample rate must be between 8000 and 48000 Hz.")
        channels = request.channels or self.config.AUDIO_DEFAULT_CHANNELS
        if channels not in (1, 2):
            raise ValueError("AI audio supports one or two output channels.")
        if (request.output_format or "wav").strip().lower() != "wav":
            raise ValueError("AI audio provider supports WAV output only.")
        if self.config.AI_AUDIO_DEVICE.strip().lower() not in ("cuda", "cuda:0"):
            raise ValueError("AI audio requires CUDA; CPU fallback is disabled.")

        output_dir = self._resolve_path(request.output_dir or self.output_dir)
        if request.output_path is None:
            output_path = output_dir / f"ai_audio_{uuid.uuid4().hex}.wav"
        else:
            output_path = self._resolve_path(request.output_path)
            if output_path.parent != output_dir:
                raise ValueError(
                    f"AI audio output_path must be inside {output_dir}."
                )
            if output_path.suffix.lower() != ".wav":
                raise ValueError("AI audio output_path must use the .wav extension.")
            if output_path.exists():
                raise ValueError(f"AI audio output_path already exists: {output_path}")

        seed = request.seed if request.seed is not None else self.config.AI_AUDIO_SEED
        if seed < 0:
            raise ValueError("AI audio seed must be non-negative.")

        payload = {
            "model_id": self.config.AI_AUDIO_MODEL_ID,
            "model_revision": self.config.AI_AUDIO_MODEL_REVISION,
            "model_path": str(self.model_path),
            "device": self.config.AI_AUDIO_DEVICE.strip().lower(),
            "prompt": prompt,
            "audio_type": request.audio_type,
            "duration_seconds": duration,
            "sample_rate": sample_rate,
            "channels": channels,
            "seed": seed,
            "inference_steps": self.config.AI_AUDIO_STEPS,
            "output_path": str(output_path),
        }
        return payload, output_path

    async def _run_worker(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            process = await asyncio.create_subprocess_exec(
                str(self.python_path),
                str(self.worker_path),
                cwd=str(BASE_DIR),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={
                    **os.environ,
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
                    "TQDM_DISABLE": "1",
                },
            )
        except OSError as exc:
            raise AudioProviderExecutionError(
                f"Could not start the isolated audio runtime at {self.python_path}: {exc}"
            ) from exc

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(json.dumps(payload).encode("utf-8")),
                timeout=self.config.AI_AUDIO_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as exc:
            if process.returncode is None:
                process.kill()
            await process.communicate()
            raise AudioProviderExecutionError(
                "AI audio generation exceeded the configured "
                f"{self.config.AI_AUDIO_TIMEOUT_SECONDS}-second timeout."
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
            raise AudioProviderExecutionError(
                f"Isolated audio worker returned invalid output: {detail}"
            ) from exc

        if process.returncode != 0 or not response.get("ok"):
            error = response.get("error") or stderr_text[-4000:]
            code = response.get("error_code")
            if code == "cuda_oom":
                raise AudioProviderExecutionError(
                    f"CUDA out of memory during AudioLDM 2 inference: {error}"
                )
            raise AudioProviderExecutionError(
                f"Isolated AI audio generation failed: {error or 'unknown worker error'}"
            )
        return response

    async def generate_audio(
        self,
        request: AudioGenerationRequest,
    ) -> AudioGenerationResult:
        payload, output_path = self._generation_parameters(request)
        if not self.python_path.is_file():
            raise AudioProviderExecutionError(
                f"Isolated audio Python environment is missing: {self.python_path}"
            )
        if not self.worker_path.is_file():
            raise AudioProviderExecutionError(
                f"Isolated audio worker is missing: {self.worker_path}"
            )
        if not (self.model_path / "model_index.json").is_file():
            raise AudioProviderExecutionError(
                f"AudioLDM 2 model files are missing from {self.model_path}. "
                "Use the documented model download command; request-time downloads are disabled."
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            async with self._generation_lock:
                response = await self._run_worker(payload)

            if not output_path.is_file():
                raise AudioProviderExecutionError(
                    "AI audio worker reported success but produced no WAV file."
                )

            info = validate_pcm_wav(
                output_path,
                expected_sample_rate=payload["sample_rate"],
                expected_channels=payload["channels"],
                expected_duration_seconds=payload["duration_seconds"],
            )
            worker_result = response["result"]
            metadata = dict(request.metadata)
            metadata.update(
                {
                    "model_revision": payload["model_revision"],
                    "prompt": payload["prompt"],
                    "seed": payload["seed"],
                    "inference_steps": payload["inference_steps"],
                    "generation_time_seconds": float(
                        worker_result["generation_time_seconds"]
                    ),
                    "device": worker_result["device"],
                    "gpu": worker_result["gpu"],
                    "gpu_peak_allocated_bytes": worker_result[
                        "gpu_peak_allocated_bytes"
                    ],
                    "source_sample_rate": worker_result["source_sample_rate"],
                    "ai_generated": True,
                    "encoding": "PCM_16",
                    "peak_sample": info.peak_sample,
                    "file_size_bytes": info.file_size_bytes,
                }
            )
            return AudioGenerationResult(
                provider=self.name,
                status="generated",
                audio_type=request.audio_type,
                model=payload["model_id"],
                output_path=output_path,
                duration_seconds=info.duration_seconds,
                sample_rate=info.sample_rate,
                channels=info.channels,
                format="wav",
                metadata=metadata,
            )
        except asyncio.CancelledError:
            self._remove_incomplete_output(output_path)
            raise
        except Exception:
            self._remove_incomplete_output(output_path)
            raise

    @staticmethod
    def _remove_incomplete_output(output_path: Path) -> None:
        if output_path.exists():
            try:
                output_path.unlink()
            except OSError:
                logger.exception("Could not remove incomplete AI audio output %s.", output_path)
