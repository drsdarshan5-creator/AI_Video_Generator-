import struct
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class PCMFileInfo:
    duration_seconds: float
    sample_rate: int
    channels: int
    frames: int
    peak_sample: int
    file_size_bytes: int


def validate_pcm_wav(
    path: Path,
    *,
    expected_sample_rate: int,
    expected_channels: int,
    expected_duration_seconds: Optional[float] = None,
    require_non_silent: bool = True,
) -> PCMFileInfo:
    """Validate a complete PCM16 WAV and return measured file properties."""
    candidate = Path(path)
    if not candidate.is_file() or candidate.stat().st_size <= 44:
        raise ValueError(f"Audio output is missing or empty: {candidate}")

    try:
        with wave.open(str(candidate), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            sample_width = wav_file.getsampwidth()
            compression = wav_file.getcomptype()
            duration_seconds = frame_count / sample_rate if sample_rate else 0.0
            frames_data = wav_file.readframes(frame_count)
    except (EOFError, OSError, wave.Error) as exc:
        raise ValueError(f"Audio output is not a valid WAV file: {exc}") from exc

    if sample_width != 2 or compression != "NONE":
        raise ValueError("Audio output must be uncompressed PCM16 WAV.")
    if channels != expected_channels:
        raise ValueError(
            f"Audio output has {channels} channels; expected {expected_channels}."
        )
    if sample_rate != expected_sample_rate:
        raise ValueError(
            f"Audio output has sample rate {sample_rate}; "
            f"expected {expected_sample_rate}."
        )
    if frame_count < 1 or duration_seconds <= 0:
        raise ValueError("Audio output duration must be greater than zero.")
    expected_bytes = frame_count * channels * sample_width
    if len(frames_data) != expected_bytes:
        raise ValueError("Audio output is truncated or corrupted.")
    if len(frames_data) % 2:
        raise ValueError("PCM16 audio data has an invalid byte count.")

    samples = struct.iter_unpack("<h", frames_data)
    peak_sample = max((abs(sample[0]) for sample in samples), default=0)
    if require_non_silent and peak_sample == 0:
        raise ValueError("Audio output is silent.")
    if expected_duration_seconds is not None:
        duration_tolerance = max(0.1, 2 / sample_rate)
        if abs(duration_seconds - expected_duration_seconds) > duration_tolerance:
            raise ValueError(
                f"Audio output duration {duration_seconds:.3f}s does not match "
                f"requested {expected_duration_seconds:.3f}s."
            )

    return PCMFileInfo(
        duration_seconds=duration_seconds,
        sample_rate=sample_rate,
        channels=channels,
        frames=frame_count,
        peak_sample=peak_sample,
        file_size_bytes=candidate.stat().st_size,
    )
