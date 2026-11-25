import contextlib
import importlib
import json
import sys
import time
import traceback
import wave
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict

from audio_validation import validate_pcm_wav


def _generate(payload: Dict[str, Any]) -> Dict[str, Any]:
    torch: Any = importlib.import_module("torch")
    numpy: Any = importlib.import_module("numpy")
    scipy_signal: Any = importlib.import_module("scipy.signal")
    diffusers: Any = importlib.import_module("diffusers")

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA is unavailable; CPU-only audio inference is disabled.")
    device = torch.device(payload["device"])
    gpu_name = torch.cuda.get_device_name(device)
    if "nvidia" not in gpu_name.lower():
        raise RuntimeError(f"An NVIDIA CUDA GPU is required; found '{gpu_name}'.")

    model_path = Path(payload["model_path"])
    if not (model_path / "model_index.json").is_file():
        raise FileNotFoundError(f"AudioLDM 2 model files are missing: {model_path}")

    pipeline = diffusers.AudioLDM2Pipeline.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16,
        use_safetensors=True,
        local_files_only=True,
    )
    pipeline.enable_model_cpu_offload(gpu_id=0)
    unet_devices = []

    def capture_unet_device(_module, args, kwargs):
        del _module
        sample = kwargs.get("sample")
        if sample is None and args:
            sample = args[0]
        unet_devices.append(
            str(sample.device) if isinstance(sample, torch.Tensor) else None
        )

    hook = pipeline.unet.register_forward_pre_hook(
        capture_unet_device,
        with_kwargs=True,
    )
    try:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        generated = pipeline(
            prompt=payload["prompt"],
            audio_length_in_s=payload["duration_seconds"],
            num_inference_steps=payload["inference_steps"],
            num_waveforms_per_prompt=1,
            generator=torch.Generator(device=device).manual_seed(payload["seed"]),
        )
        torch.cuda.synchronize(device)
        generation_time = time.perf_counter() - started
    finally:
        hook.remove()

    if not any(item == "cuda:0" for item in unet_devices):
        raise RuntimeError(
            "AudioLDM 2 UNet did not run on CUDA; refusing CPU fallback."
        )

    waveform = numpy.asarray(generated.audios[0], dtype=numpy.float32).squeeze()
    if waveform.ndim != 1 or waveform.size == 0:
        raise ValueError(f"AudioLDM 2 returned an invalid waveform shape: {waveform.shape}")
    if not numpy.isfinite(waveform).all():
        raise ValueError("AudioLDM 2 output contains non-finite sample values.")
    if not numpy.any(numpy.abs(waveform) > 0):
        raise ValueError("AudioLDM 2 generated silent audio.")

    source_sample_rate = int(pipeline.vocoder.config.sampling_rate)
    target_sample_rate = int(payload["sample_rate"])
    if source_sample_rate != target_sample_rate:
        ratio = Fraction(target_sample_rate, source_sample_rate)
        waveform = scipy_signal.resample_poly(
            waveform,
            ratio.numerator,
            ratio.denominator,
        ).astype(numpy.float32, copy=False)

    if not numpy.isfinite(waveform).all():
        raise ValueError("Resampled audio contains non-finite sample values.")

    pcm = numpy.rint(numpy.clip(waveform, -1.0, 1.0) * 32767).astype("<i2")
    if payload["channels"] == 2:
        pcm = numpy.repeat(pcm[:, numpy.newaxis], 2, axis=1)

    output_path = Path(payload["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as output_file:
        with wave.open(output_file, "wb") as wav_file:
            wav_file.setnchannels(payload["channels"])
            wav_file.setsampwidth(2)
            wav_file.setframerate(target_sample_rate)
            wav_file.writeframes(pcm.tobytes())

    info = validate_pcm_wav(
        output_path,
        expected_sample_rate=target_sample_rate,
        expected_channels=payload["channels"],
        expected_duration_seconds=payload["duration_seconds"],
    )
    return {
        "generation_time_seconds": generation_time,
        "device": str(device),
        "gpu": gpu_name,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "source_sample_rate": source_sample_rate,
        "duration_seconds": info.duration_seconds,
        "sample_rate": info.sample_rate,
        "channels": info.channels,
    }


def main() -> int:
    output_path = None
    try:
        payload = json.load(sys.stdin)
        output_path = Path(payload["output_path"])
        with contextlib.redirect_stdout(sys.stderr):
            result = _generate(payload)
        json.dump({"ok": True, "result": result}, sys.stdout)
        sys.stdout.write("\n")
        return 0
    except Exception as exc:
        code = (
            "cuda_oom"
            if "out of memory" in str(exc).lower()
            or type(exc).__name__ == "OutOfMemoryError"
            else "runtime_error"
        )
        if output_path is not None and output_path.exists():
            try:
                output_path.unlink()
            except OSError:
                traceback.print_exc(file=sys.stderr)
        json.dump(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "error_code": code,
            },
            sys.stdout,
        )
        sys.stdout.write("\n")
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
