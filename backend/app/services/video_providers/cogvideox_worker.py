import contextlib
import gc
import importlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _configure_scheduler(
    pipe: Any,
    diffusers: Any,
    scheduler_name: str,
) -> str:
    normalized = scheduler_name.strip().lower()
    if normalized == "dpm":
        pipe.scheduler = diffusers.CogVideoXDPMScheduler.from_config(
            pipe.scheduler.config,
            timestep_spacing="trailing",
        )
        return "dpm"
    if normalized == "ddim":
        pipe.scheduler = diffusers.CogVideoXDDIMScheduler.from_config(
            pipe.scheduler.config
        )
        return "ddim"
    raise ValueError(f"Unsupported CogVideoX scheduler: {scheduler_name}")


def _configure_offload(pipe: Any, mode: str, device: Any) -> Tuple[str, List[Any]]:
    normalized = mode.strip().lower()
    if normalized == "sequential":
        pipe.enable_sequential_cpu_offload(device=device)
        return "sequential", []
    if normalized == "model":
        pipe.enable_model_cpu_offload(device=device)
        return "model", []
    if normalized == "hybrid":
        # T5-XXL (~8.87 GB in fp16) exceeds 8 GB VRAM, so offload it sequentially
        # while keeping the transformer (~3.23 GB) and VAE (~0.82 GB) whole on GPU.
        # Do not attach these to pipe._all_hooks so maybe_free_model_hooks()
        # does not overwrite text_encoder's sequential offload on reuse.
        accelerate: Any = importlib.import_module("accelerate")
        accelerate.cpu_offload(
            pipe.text_encoder,
            execution_device=device,
            offload_buffers=len(pipe.text_encoder._parameters) > 0,
        )
        _, transformer_hook = accelerate.cpu_offload_with_hook(
            pipe.transformer,
            execution_device=device,
        )
        _, vae_hook = accelerate.cpu_offload_with_hook(
            pipe.vae,
            execution_device=device,
            prev_module_hook=transformer_hook,
        )
        return "hybrid", [transformer_hook, vae_hook]
    raise ValueError(f"Unsupported CogVideoX CPU offload mode: {mode}")


def _resolve_device_and_dtype(payload: Dict[str, Any], torch: Any) -> Tuple[Any, str, Any, str]:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA is unavailable; CPU-only CogVideoX inference is disabled.")

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    if "nvidia" not in gpu_name.lower():
        raise RuntimeError(f"An NVIDIA CUDA GPU is required; found '{gpu_name}'.")

    dtype_name = str(payload.get("dtype", "float16")).strip().lower()
    if dtype_name == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype_name == "float16":
        torch_dtype = torch.float16
    else:
        raise ValueError(f"Unsupported CogVideoX dtype: {dtype_name}")

    return device, gpu_name, torch_dtype, dtype_name


def _pipeline_cache_key(payload: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        payload.get("model_path"),
        payload["model_id"],
        payload["model_revision"],
        payload.get("cache_dir"),
        str(payload.get("dtype", "float16")).strip().lower(),
        str(payload.get("cpu_offload", "sequential")).strip().lower(),
        bool(payload.get("vae_slicing", True)),
        bool(payload.get("vae_tiling", True)),
        bool(payload.get("attention_slicing", False)),
        bool(payload.get("model_path")) or bool(payload.get("local_files_only", False)),
    )


def _load_pipeline(
    payload: Dict[str, Any],
    torch: Any,
    diffusers: Any,
    device: Any,
    torch_dtype: Any,
) -> Tuple[Any, str, List[Any]]:
    model_path = payload.get("model_path")
    if model_path:
        model_source = Path(model_path)
        if not model_source.is_dir():
            raise FileNotFoundError(
                f"Configured CogVideoX model path does not exist: {model_path}"
            )
        model_source = str(model_source)
    else:
        model_source = payload["model_id"]

    local_files_only = bool(payload.get("model_path")) or bool(
        payload.get("local_files_only", False)
    )
    from_pretrained_kwargs: Dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "use_safetensors": True,
        "local_files_only": local_files_only,
    }
    if payload.get("model_path") is None:
        from_pretrained_kwargs["revision"] = payload["model_revision"]
    if payload.get("cache_dir"):
        from_pretrained_kwargs["cache_dir"] = payload["cache_dir"]

    pipe = diffusers.CogVideoXPipeline.from_pretrained(
        model_source, **from_pretrained_kwargs
    )
    offload_used, offload_hooks = _configure_offload(
        pipe,
        str(payload.get("cpu_offload", "sequential")),
        device,
    )
    if payload.get("vae_slicing", True):
        pipe.vae.enable_slicing()
    if payload.get("vae_tiling", True):
        pipe.vae.enable_tiling()
    if payload.get("attention_slicing", False):
        pipe.enable_attention_slicing("auto")
    return pipe, offload_used, offload_hooks


def _run_inference(
    pipe: Any,
    payload: Dict[str, Any],
    *,
    torch: Any,
    imageio: Any,
    export_to_video: Any,
    device: Any,
    gpu_name: str,
    dtype_name: str,
    offload_used: str,
    offload_hooks: List[Any],
    scheduler_used: str,
) -> Dict[str, Any]:
    transformer_inputs = []

    def capture_cuda_input(_module, args, kwargs):
        del _module
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None and args:
            hidden_states = args[0]
        transformer_inputs.append(
            str(hidden_states.device) if isinstance(hidden_states, torch.Tensor) else None
        )

    hook = pipe.transformer.register_forward_pre_hook(
        capture_cuda_input,
        with_kwargs=True,
    )
    try:
        torch.cuda.synchronize(device)
        free_bytes, _ = torch.cuda.mem_get_info(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        result = pipe(
            prompt=payload["prompt"],
            height=payload["height"],
            width=payload["width"],
            num_frames=payload["frames"],
            num_inference_steps=payload["inference_steps"],
            num_videos_per_prompt=1,
            guidance_scale=payload["guidance_scale"],
            use_dynamic_cfg=bool(payload.get("use_dynamic_cfg", False)),
            generator=torch.Generator(device=device).manual_seed(payload["seed"]),
        )
        torch.cuda.synchronize(device)
        inference_time = time.perf_counter() - started
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
    finally:
        hook.remove()
        for offload_hook in offload_hooks:
            offload_hook.offload()
        gc.collect()
        torch.cuda.empty_cache()

    cuda_used = any(device_name == "cuda:0" for device_name in transformer_inputs)
    if not cuda_used:
        raise RuntimeError(
            "CogVideoX transformer did not receive CUDA inputs; refusing CPU fallback."
        )

    frames = result.frames[0]
    if len(frames) != payload["frames"]:
        raise RuntimeError(
            f"CogVideoX produced {len(frames)} frames; expected {payload['frames']}."
        )

    output_path = Path(payload["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(output_path), fps=payload["fps"])
    reader = imageio.get_reader(str(output_path), format="ffmpeg")
    try:
        video_metadata = reader.get_meta_data()
        file_frame_count = reader.count_frames()
    finally:
        reader.close()

    actual_size = video_metadata.get("size")
    if (
        file_frame_count != payload["frames"]
        or actual_size != (payload["width"], payload["height"])
        or not output_path.is_file()
        or output_path.stat().st_size <= 0
    ):
        raise RuntimeError("CogVideoX output failed MP4 frame, resolution, or file validation.")

    return {
        "width": payload["width"],
        "height": payload["height"],
        "frames": file_frame_count,
        "fps": payload["fps"],
        "inference_time_seconds": inference_time,
        "device": str(device),
        "dtype": dtype_name,
        "cpu_offload": offload_used,
        "scheduler": scheduler_used,
        "gpu": gpu_name,
        "gpu_vram_free_before_inference_bytes": free_bytes,
        "gpu_peak_allocated_bytes": peak_allocated,
        "gpu_peak_reserved_bytes": peak_reserved,
        "model_revision": payload["model_revision"],
    }


def _generate(payload: Dict[str, Any]) -> Dict[str, Any]:
    imageio: Any = importlib.import_module("imageio.v2")
    torch: Any = importlib.import_module("torch")
    diffusers: Any = importlib.import_module("diffusers")
    export_to_video: Any = importlib.import_module("diffusers.utils").export_to_video

    device, gpu_name, torch_dtype, dtype_name = _resolve_device_and_dtype(payload, torch)
    pipe, offload_used, offload_hooks = _load_pipeline(
        payload, torch, diffusers, device, torch_dtype
    )
    scheduler_used = _configure_scheduler(
        pipe,
        diffusers,
        str(payload.get("scheduler", "ddim")),
    )
    return _run_inference(
        pipe,
        payload,
        torch=torch,
        imageio=imageio,
        export_to_video=export_to_video,
        device=device,
        gpu_name=gpu_name,
        dtype_name=dtype_name,
        offload_used=offload_used,
        offload_hooks=offload_hooks,
        scheduler_used=scheduler_used,
    )


def _run_batch_stdio() -> int:
    imageio: Any = importlib.import_module("imageio.v2")
    torch: Any = importlib.import_module("torch")
    diffusers: Any = importlib.import_module("diffusers")
    export_to_video: Any = importlib.import_module("diffusers.utils").export_to_video

    cached_key: Optional[Tuple[Any, ...]] = None
    cached_pipe: Any = None
    cached_offload_used: str = "sequential"
    cached_offload_hooks: List[Any] = []
    cached_scheduler: Optional[str] = None
    cached_device: Any = None
    cached_gpu_name: str = ""
    cached_dtype_name: str = "float16"

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            if payload.get("command") == "shutdown":
                json.dump({"ok": True, "shutdown": True}, sys.stdout)
                sys.stdout.write("\n")
                sys.stdout.flush()
                return 0

            with contextlib.redirect_stdout(sys.stderr):
                key = _pipeline_cache_key(payload)
                if cached_pipe is None or key != cached_key:
                    cached_pipe = None
                    cached_offload_hooks = []
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    (
                        cached_device,
                        cached_gpu_name,
                        torch_dtype,
                        cached_dtype_name,
                    ) = _resolve_device_and_dtype(payload, torch)
                    (
                        cached_pipe,
                        cached_offload_used,
                        cached_offload_hooks,
                    ) = _load_pipeline(
                        payload, torch, diffusers, cached_device, torch_dtype
                    )
                    cached_key = key
                    cached_scheduler = None

                requested_scheduler = str(payload.get("scheduler", "ddim")).strip().lower()
                if cached_scheduler != requested_scheduler:
                    cached_scheduler = _configure_scheduler(
                        cached_pipe, diffusers, requested_scheduler
                    )

                result = _run_inference(
                    cached_pipe,
                    payload,
                    torch=torch,
                    imageio=imageio,
                    export_to_video=export_to_video,
                    device=cached_device,
                    gpu_name=cached_gpu_name,
                    dtype_name=cached_dtype_name,
                    offload_used=cached_offload_used,
                    offload_hooks=cached_offload_hooks,
                    scheduler_used=cached_scheduler,
                )

            json.dump({"ok": True, "result": result}, sys.stdout)
            sys.stdout.write("\n")
            sys.stdout.flush()
        except Exception as exc:
            json.dump(
                {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                sys.stdout,
            )
            sys.stdout.write("\n")
            sys.stdout.flush()
            traceback.print_exc(file=sys.stderr)
    return 0


def main() -> int:
    if "--batch-stdio" in sys.argv[1:]:
        return _run_batch_stdio()
    try:
        payload = json.load(sys.stdin)
        with contextlib.redirect_stdout(sys.stderr):
            result = _generate(payload)
        json.dump({"ok": True, "result": result}, sys.stdout)
        sys.stdout.write("\n")
        return 0
    except Exception as exc:
        json.dump(
            {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
            sys.stdout,
        )
        sys.stdout.write("\n")
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
