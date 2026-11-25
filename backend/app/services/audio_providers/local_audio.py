import asyncio
import importlib
import importlib.util
import logging
import math
import random
import struct
import uuid
from pathlib import Path
from typing import Optional

from ...config import settings
from .base import (
    AudioGenerationRequest,
    AudioGenerationResult,
    BaseAudioProvider,
)

logger = logging.getLogger(__name__)

_MAX_DURATION_SECONDS = 60.0
_MIN_SAMPLE_RATE = 8000
_MAX_SAMPLE_RATE = 48000


class LocalAudioProvider(BaseAudioProvider):
    """Create simple deterministic WAV assets using only Python's standard library."""

    name = "local_audio"

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self.output_dir = Path(
            output_dir if output_dir is not None else settings.AUDIO_OUTPUT_DIR
        ).expanduser()

    def is_available(self) -> bool:
        return importlib.util.find_spec("wave") is not None

    async def generate_audio(
        self,
        request: AudioGenerationRequest,
    ) -> AudioGenerationResult:
        return await asyncio.to_thread(self._generate_audio_sync, request)

    def _generate_audio_sync(
        self,
        request: AudioGenerationRequest,
    ) -> AudioGenerationResult:
        output_path: Optional[Path] = None
        succeeded = False
        try:
            sample_rate, channels, duration, audio_type = self._validate_request(
                request
            )
            wave_module = self._load_wave_module()
            output_dir = Path(request.output_dir or self.output_dir).expanduser()
            output_dir = output_dir.resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            if request.output_path is None:
                output_path = output_dir / f"local_audio_{uuid.uuid4().hex}.wav"
            else:
                output_path = Path(request.output_path).expanduser().resolve()
                if output_path.parent != output_dir:
                    raise ValueError(
                        f"Local audio output_path must be inside {output_dir}."
                    )
                if output_path.suffix.lower() != ".wav":
                    raise ValueError("Local audio output_path must use the .wav extension.")

            samples = self._generate_samples(
                audio_type=audio_type,
                duration=duration,
                sample_rate=sample_rate,
                seed=request.seed if request.seed is not None else 0,
                intensity=request.intensity if request.intensity is not None else 0.35,
                tempo_bpm=request.tempo_bpm if request.tempo_bpm is not None else 90.0,
                channels=channels,
            )
            if not any(samples):
                raise ValueError("Generated audio was silent.")

            with output_path.open("xb") as output_file:
                with wave_module.open(output_file, "wb") as wav_file:
                    wav_file.setnchannels(channels)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(samples)

            with output_path.open("rb") as output_file:
                if output_file.read(4) != b"RIFF":
                    raise ValueError("Generated output is not a valid WAV file.")

            result = AudioGenerationResult(
                provider=self.name,
                status="generated",
                audio_type=request.audio_type,
                output_path=output_path,
                duration_seconds=duration,
                sample_rate=sample_rate,
                channels=channels,
                format="wav",
                metadata={
                    "generation_performed": True,
                    "generation_method": self._generation_method(audio_type),
                    "runtime": "python_standard_library",
                    "ai_generated": False,
                    "seed": request.seed if request.seed is not None else 0,
                    "intensity": (
                        request.intensity if request.intensity is not None else 0.35
                    ),
                    "tempo_bpm": request.tempo_bpm,
                    "file_size_bytes": output_path.stat().st_size,
                },
            )
            succeeded = True
            return result
        except ImportError as exc:
            return self._failed_result(request, f"WAV runtime unavailable: {exc}")
        except (OSError, ValueError) as exc:
            logger.exception("Local audio generation failed.")
            return self._failed_result(request, str(exc))
        finally:
            if not succeeded and output_path is not None and output_path.exists():
                try:
                    output_path.unlink()
                except OSError:
                    logger.exception(
                        "Could not remove incomplete local audio output %s.",
                        output_path,
                    )

    @staticmethod
    def _load_wave_module():
        return importlib.import_module("wave")

    @staticmethod
    def _validate_request(
        request: AudioGenerationRequest,
    ) -> tuple[int, int, float, str]:
        if request.audio_type not in (
            "ambience",
            "sound_effect",
            "music",
            "procedural_music",
        ):
            raise ValueError(
                f"local_audio does not generate '{request.audio_type}' audio."
            )
        if request.duration_seconds is None:
            raise ValueError("Local audio generation requires duration_seconds.")
        if request.duration_seconds > _MAX_DURATION_SECONDS:
            raise ValueError(
                f"Local audio duration cannot exceed {_MAX_DURATION_SECONDS:g} seconds."
            )
        sample_rate = request.sample_rate or settings.AUDIO_DEFAULT_SAMPLE_RATE
        if not _MIN_SAMPLE_RATE <= sample_rate <= _MAX_SAMPLE_RATE:
            raise ValueError(
                f"Sample rate must be between {_MIN_SAMPLE_RATE} and "
                f"{_MAX_SAMPLE_RATE} Hz."
            )
        channels = request.channels or settings.AUDIO_DEFAULT_CHANNELS
        if channels not in (1, 2):
            raise ValueError("Local audio supports one or two channels.")
        if (request.output_format or settings.AUDIO_DEFAULT_FORMAT).lower() != "wav":
            raise ValueError("local_audio supports WAV output only.")
        return sample_rate, channels, request.duration_seconds, request.audio_type

    @staticmethod
    def _generation_method(audio_type: str) -> str:
        if audio_type == "ambience":
            return "filtered_deterministic_noise"
        if audio_type == "sound_effect":
            return "deterministic_sine_sweep"
        return "deterministic_procedural_music"

    @staticmethod
    def _generate_samples(
        *,
        audio_type: str,
        duration: float,
        sample_rate: int,
        seed: int,
        intensity: float,
        tempo_bpm: float,
        channels: int,
    ) -> bytes:
        random_source = random.Random(seed)
        frame_count = max(1, round(duration * sample_rate))
        frames = bytearray()
        filtered_noise = 0.0
        notes = (220.0, 261.63, 293.66, 329.63, 392.0, 440.0)
        melody = [random_source.choice(notes) for _ in range(8)]
        beat_seconds = 60.0 / tempo_bpm

        for frame in range(frame_count):
            time_seconds = frame / sample_rate
            progress = frame / frame_count
            if audio_type == "ambience":
                filtered_noise = (
                    0.965 * filtered_noise
                    + 0.035 * random_source.uniform(-1.0, 1.0)
                )
                value = filtered_noise * intensity * 1.8
            elif audio_type == "sound_effect":
                frequency = 880.0 - 660.0 * progress
                envelope = min(1.0, progress * 18.0) * max(
                    0.0, 1.0 - progress
                ) ** 1.4
                value = math.sin(2.0 * math.pi * frequency * time_seconds)
                value *= envelope * intensity * 0.8
            else:
                beat_position = time_seconds / beat_seconds
                note_index = int(beat_position) % len(melody)
                note_frequency = melody[note_index]
                note_phase = time_seconds - int(beat_position) * beat_seconds
                envelope = min(1.0, note_phase * 12.0) * 0.7
                bass = math.sin(
                    2.0 * math.pi * (note_frequency / 2.0) * time_seconds
                )
                lead = math.sin(2.0 * math.pi * note_frequency * time_seconds)
                value = (0.55 * lead + 0.25 * bass) * envelope * intensity

            sample = max(-32767, min(32767, round(value * 32767)))
            if channels == 1:
                frames.extend(struct.pack("<h", sample))
            else:
                frames.extend(struct.pack("<hh", sample, sample))
        return bytes(frames)

    def _failed_result(
        self,
        request: AudioGenerationRequest,
        error: str,
    ) -> AudioGenerationResult:
        return AudioGenerationResult(
            provider=self.name,
            status="failed",
            audio_type=request.audio_type,
            metadata={
                "generation_performed": False,
                "ai_generated": False,
            },
            error=error,
        )


