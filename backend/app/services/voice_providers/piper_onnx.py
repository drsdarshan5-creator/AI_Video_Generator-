import asyncio
import gc
import importlib
import importlib.util
import json
import logging
import re
import threading
import time
import uuid
import wave
from pathlib import Path
from typing import Any, Optional, Tuple

from ...config import BASE_DIR, settings
from .base import BaseVoiceProvider, VoiceGenerationRequest, VoiceGenerationResult

logger = logging.getLogger(__name__)


class PiperONNXVoiceProvider(BaseVoiceProvider):
    """Generate local WAV speech with Piper and ONNX Runtime DirectML."""

    name = "piper_onnx"
    _generation_lock = threading.Lock()

    def __init__(
        self,
        model_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ) -> None:
        self.model_path = self._resolve_path(
            model_path if model_path is not None else settings.PIPER_MODEL_PATH
        )
        self.config_path = Path(f"{self.model_path}.json")
        self.output_dir = self._resolve_path(
            output_dir if output_dir is not None else BASE_DIR / "generated" / "voice"
        )

    @staticmethod
    def _resolve_path(path: Path) -> Path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = BASE_DIR / candidate
        return candidate.resolve()

    def _runtime_installed(self) -> bool:
        try:
            return all(
                importlib.util.find_spec(module_name) is not None
                for module_name in ("piper", "onnxruntime")
            )
        except (ImportError, ValueError):
            return False

    def is_available(self) -> bool:
        return (
            self.model_path.is_file()
            and self.config_path.is_file()
            and self._runtime_installed()
        )

    def _load_runtime(self) -> Tuple[Any, Any]:
        try:
            return (
                importlib.import_module("piper"),
                importlib.import_module("onnxruntime"),
            )
        except (ImportError, OSError) as exc:
            raise RuntimeError(f"Piper ONNX runtime could not be loaded: {exc}") from exc

    def _create_session(self, runtime: Any) -> Any:
        if "DmlExecutionProvider" not in runtime.get_available_providers():
            raise RuntimeError(
                "ONNX Runtime does not provide DmlExecutionProvider; "
                "CPU fallback is disabled."
            )

        options = runtime.SessionOptions()
        options.execution_mode = runtime.ExecutionMode.ORT_SEQUENTIAL
        options.enable_mem_pattern = False
        session = runtime.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["DmlExecutionProvider", "CPUExecutionProvider"],
        )
        if "DmlExecutionProvider" not in session.get_providers():
            raise RuntimeError("DmlExecutionProvider was not enabled for the model.")
        return session

    def _output_path(self, request: VoiceGenerationRequest) -> Path:
        if request.output_path is not None:
            output_path = self._resolve_path(request.output_path)
            if output_path.parent != self.output_dir:
                raise ValueError(
                    f"Voice output must be written under {self.output_dir}."
                )
            if output_path.suffix.lower() != ".wav":
                raise ValueError("Piper output_path must use the .wav extension.")
            return output_path

        character_id = re.sub(
            r"[^A-Za-z0-9_-]+",
            "_",
            request.character_id or "narrator",
        ).strip("_")
        character_id = character_id[:80] or "narrator"
        return self.output_dir / f"voice_{character_id}_{uuid.uuid4().hex}.wav"

    def _failed_result(
        self,
        request: VoiceGenerationRequest,
        message: str,
        *,
        runtime_provider: Optional[str] = None,
        inference_time_seconds: Optional[float] = None,
    ) -> VoiceGenerationResult:
        return VoiceGenerationResult(
            provider=self.name,
            status="failed",
            voice_id=request.voice_id,
            metadata={
                "runtime_provider": runtime_provider,
                "model": self.model_path.name,
                "inference_time_seconds": inference_time_seconds,
            },
            message=message,
        )

    def _generate_sync(
        self,
        request: VoiceGenerationRequest,
    ) -> VoiceGenerationResult:
        session = None
        voice = None
        output_path = None
        output_created = False
        succeeded = False
        runtime_provider = None
        inference_time_seconds: Optional[float] = None

        try:
            missing_files = [
                path for path in (self.model_path, self.config_path) if not path.is_file()
            ]
            if missing_files:
                missing = ", ".join(str(path) for path in missing_files)
                return self._failed_result(
                    request,
                    f"Piper model files are missing: {missing}",
                )

            if not self._runtime_installed():
                return self._failed_result(
                    request,
                    "Piper and ONNX Runtime must be installed in the active "
                    "Python environment.",
                )

            piper, runtime = self._load_runtime()
            session = self._create_session(runtime)
            session_providers = session.get_providers()
            runtime_provider = session_providers[0]

            with self.config_path.open("r", encoding="utf-8") as config_file:
                voice_config = piper.PiperConfig.from_dict(json.load(config_file))
            voice = piper.PiperVoice(
                session=session,
                config=voice_config,
                download_dir=BASE_DIR,
            )

            output_path = self._output_path(request)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with output_path.open("xb") as output_file:
                output_created = True
                inference_started = time.perf_counter()
                with wave.open(output_file, "wb") as wav_file:
                    voice.synthesize_wav(request.text, wav_file)
                inference_time_seconds = time.perf_counter() - inference_started

            with wave.open(str(output_path), "rb") as wav_file:
                channels = wav_file.getnchannels()
                sample_rate = wav_file.getframerate()
                frames = wav_file.getnframes()
                sample_width = wav_file.getsampwidth()
                compression = wav_file.getcomptype()
                duration_seconds = frames / sample_rate if sample_rate else 0

            file_size = output_path.stat().st_size
            if (
                channels < 1
                or sample_rate < 1
                or frames < 1
                or sample_width != 2
                or compression != "NONE"
                or duration_seconds <= 0
                or file_size <= 0
            ):
                raise ValueError("Piper produced an invalid or empty PCM WAV file.")

            succeeded = True
            return VoiceGenerationResult(
                provider=self.name,
                status="generated",
                output_path=output_path,
                duration_seconds=duration_seconds,
                sample_rate=sample_rate,
                format="wav",
                voice_id=request.voice_id,
                metadata={
                    "runtime_provider": runtime_provider,
                    "session_providers": session_providers,
                    "model": self.model_path.name,
                    "inference_time_seconds": inference_time_seconds,
                    "channels": channels,
                    "encoding": "PCM_16",
                    "file_size_bytes": file_size,
                },
            )
        except Exception as exc:
            logger.exception("Piper ONNX speech generation failed.")
            return self._failed_result(
                request,
                str(exc),
                runtime_provider=runtime_provider,
                inference_time_seconds=inference_time_seconds,
            )
        finally:
            had_runtime_objects = voice is not None or session is not None
            voice = None
            session = None
            if had_runtime_objects:
                gc.collect()
            if output_created and not succeeded and output_path is not None:
                try:
                    output_path.unlink(missing_ok=True)
                except OSError:
                    logger.exception(
                        "Could not remove incomplete Piper output %s.",
                        output_path,
                    )

    async def generate_speech(
        self,
        request: VoiceGenerationRequest,
    ) -> VoiceGenerationResult:
        return await asyncio.to_thread(self._generate_serialized, request)

    def _generate_serialized(
        self,
        request: VoiceGenerationRequest,
    ) -> VoiceGenerationResult:
        with self._generation_lock:
            return self._generate_sync(request)
