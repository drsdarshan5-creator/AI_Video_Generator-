import asyncio
import gc
import html
import importlib
import importlib.util
import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from ...config import BASE_DIR, settings
from .base import BaseImageProvider, ImageGenerationResult, ImageProviderError

logger = logging.getLogger(__name__)


class _ClipTokenizer:
    def __init__(self, directory: Path) -> None:
        import json

        self.vocab = json.loads((directory / "vocab.json").read_text(encoding="utf-8"))
        merge_lines = (directory / "merges.txt").read_text(encoding="utf-8").splitlines()
        self.ranks = {
            tuple(line.split()): rank
            for rank, line in enumerate(merge_lines[1:])
            if line.strip()
        }
        self.byte_encoder = self._bytes_to_unicode()
        self.pattern = re.compile(
            r"'s|'t|'re|'ve|'m|'ll|'d|[A-Za-z]+|[0-9]|[^A-Za-z0-9\s]+",
            re.IGNORECASE,
        )

    @staticmethod
    def _bytes_to_unicode() -> Dict[int, str]:
        visible = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        codepoints = visible[:]
        extra = 0
        for value in range(256):
            if value not in visible:
                visible.append(value)
                codepoints.append(256 + extra)
                extra += 1
        return {value: chr(codepoint) for value, codepoint in zip(visible, codepoints)}

    def _bpe(self, token: str) -> List[str]:
        word = list(token[:-1]) + [token[-1] + "</w>"]
        while len(word) > 1:
            pairs = set(zip(word, word[1:]))
            pair = min(pairs, key=lambda item: self.ranks.get(item, float("inf")))
            if pair not in self.ranks:
                break
            merged = []
            index = 0
            while index < len(word):
                if (
                    index < len(word) - 1
                    and word[index] == pair[0]
                    and word[index + 1] == pair[1]
                ):
                    merged.append(word[index] + word[index + 1])
                    index += 2
                else:
                    merged.append(word[index])
                    index += 1
            word = merged
        return word

    def encode(self, text: str) -> Any:
        import numpy as np

        text = html.unescape(html.unescape(text)).strip().lower()
        text = re.sub(r"\s+", " ", text)
        token_ids = [self.vocab["<|startoftext|>"]]
        for token in self.pattern.findall(text):
            encoded = "".join(
                self.byte_encoder[value] for value in token.encode("utf-8")
            )
            token_ids.extend(self.vocab[piece] for piece in self._bpe(encoded))
        token_ids.append(self.vocab["<|endoftext|>"])
        token_ids = token_ids[:77]
        token_ids.extend([self.vocab["<|endoftext|>"]] * (77 - len(token_ids)))
        return np.asarray(token_ids, dtype=np.int32)


class _PNDMSampler:
    def __init__(self, config: Dict[str, Any], steps: int) -> None:
        import numpy as np

        self.train_steps = int(config["num_train_timesteps"])
        self.step_ratio = self.train_steps // steps
        roots = np.linspace(
            float(config["beta_start"]) ** 0.5,
            float(config["beta_end"]) ** 0.5,
            self.train_steps,
            dtype=np.float32,
        )
        self.alphas_cumprod = np.cumprod(1.0 - roots**2, dtype=np.float32)
        self.final_alpha = (
            1.0
            if bool(config["set_alpha_to_one"])
            else float(self.alphas_cumprod[0])
        )
        ascending = (
            np.arange(steps, dtype=np.int64) * self.step_ratio
            + int(config["steps_offset"])
        )
        self.timesteps = np.concatenate(
            (ascending[:-1], ascending[-2:-1], ascending[-1:])
        )[::-1].copy()
        self.history = []
        self.initial_sample = None

    def step(
        self,
        model_output: Any,
        sample: Any,
        timestep: int,
        counter: int,
    ) -> Any:
        import numpy as np

        current_timestep = timestep
        previous_timestep = timestep - self.step_ratio
        if counter != 1:
            self.history = self.history[-3:]
            self.history.append(model_output)
        else:
            previous_timestep = timestep
            current_timestep += self.step_ratio

        update_sample = sample
        if len(self.history) == 1 and counter == 0:
            adjusted_output = model_output
            self.initial_sample = sample
        elif len(self.history) == 1 and counter == 1:
            adjusted_output = (model_output + self.history[-1]) / 2.0
            if self.initial_sample is None:
                raise RuntimeError("PNDM initial sample was not recorded.")
            update_sample = self.initial_sample
            self.initial_sample = None
        elif len(self.history) == 2:
            adjusted_output = (3.0 * self.history[-1] - self.history[-2]) / 2.0
        elif len(self.history) == 3:
            adjusted_output = (
                23.0 * self.history[-1]
                - 16.0 * self.history[-2]
                + 5.0 * self.history[-3]
            ) / 12.0
        else:
            adjusted_output = (
                55.0 * self.history[-1]
                - 59.0 * self.history[-2]
                + 37.0 * self.history[-3]
                - 9.0 * self.history[-4]
            ) / 24.0

        alpha_t = float(self.alphas_cumprod[current_timestep])
        alpha_previous = (
            float(self.alphas_cumprod[previous_timestep])
            if previous_timestep >= 0
            else self.final_alpha
        )
        beta_t = 1.0 - alpha_t
        beta_previous = 1.0 - alpha_previous
        sample_coefficient = np.sqrt(alpha_previous / alpha_t)
        noise_denominator = (
            alpha_t * np.sqrt(beta_previous)
            + np.sqrt(alpha_t * beta_t * alpha_previous)
        )
        return (
            sample_coefficient * update_sample
            - (alpha_previous - alpha_t) * adjusted_output / noise_denominator
        ).astype(np.float32)


class OnnxDirectMLImageProvider(BaseImageProvider):
    """Runs the validated SD 1.5 ONNX pipeline with per-call DirectML sessions."""

    provider_name = "onnx_directml"
    width = 512
    height = 512
    inference_steps = 4
    guidance_scale = 7.5
    latent_scale = 0.18215
    required_model_files = (
        "model_index.json",
        "scheduler/scheduler_config.json",
        "tokenizer/vocab.json",
        "tokenizer/merges.txt",
        "tokenizer/tokenizer_config.json",
        "tokenizer/special_tokens_map.json",
        "text_encoder/model.onnx",
        "unet/model.onnx",
        "unet/weights.pb",
        "vae_decoder/model.onnx",
    )

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.model_dir = self._resolve_path(
            model_dir if model_dir is not None else settings.IMAGE_MODEL_DIR
        )
        self.output_dir = self._resolve_path(
            output_dir if output_dir is not None else settings.IMAGE_OUTPUT_DIR
        )
        self.width = int(width if width is not None else settings.IMAGE_WIDTH)
        self.height = int(height if height is not None else settings.IMAGE_HEIGHT)
        self.seed = int(seed if seed is not None else settings.IMAGE_SEED)

    @staticmethod
    def _resolve_path(path: Path) -> Path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = BASE_DIR / candidate
        return candidate.resolve()

    def _runtime_installed(self) -> bool:
        return importlib.util.find_spec("onnxruntime") is not None

    def _missing_model_files(self) -> List[str]:
        return [
            relative
            for relative in self.required_model_files
            if not (self.model_dir / relative).is_file()
        ]

    def is_available(self) -> bool:
        return (
            self.width == 512
            and self.height == 512
            and self._runtime_installed()
            and not self._missing_model_files()
        )

    def _load_runtime(self) -> Any:
        try:
            runtime = importlib.import_module("onnxruntime")
        except (ImportError, OSError) as exc:
            raise ImageProviderError(
                f"ONNX Runtime could not be loaded: {exc}"
            ) from exc
        if "DmlExecutionProvider" not in runtime.get_available_providers():
            raise ImageProviderError(
                "ONNX Runtime does not provide DmlExecutionProvider."
            )
        return runtime

    def _create_sessions(self, runtime: Any) -> List[Any]:
        sessions = []
        graph_paths = (
            self.model_dir / "text_encoder" / "model.onnx",
            self.model_dir / "unet" / "model.onnx",
            self.model_dir / "vae_decoder" / "model.onnx",
        )
        try:
            for graph_path in graph_paths:
                options = runtime.SessionOptions()
                options.execution_mode = runtime.ExecutionMode.ORT_SEQUENTIAL
                options.enable_mem_pattern = False
                session = runtime.InferenceSession(
                    str(graph_path),
                    sess_options=options,
                    providers=["DmlExecutionProvider"],
                )
                if "DmlExecutionProvider" not in session.get_providers():
                    raise ImageProviderError(
                        f"DmlExecutionProvider was not enabled for {graph_path.name}."
                    )
                sessions.append(session)
            return sessions
        except Exception:
            sessions.clear()
            del sessions
            gc.collect()
            raise

    def _run_core(
        self,
        prompt: str,
        seed: int,
        sessions: List[Any],
    ) -> tuple[Any, Dict[str, float]]:
        import numpy as np

        tokenizer = _ClipTokenizer(self.model_dir / "tokenizer")
        input_ids = np.stack(
            (tokenizer.encode(""), tokenizer.encode(prompt)),
            axis=0,
        )
        text_start = time.perf_counter()
        embeddings = sessions[0].run(
            ["last_hidden_state"],
            {"input_ids": input_ids},
        )[0].astype(np.float32)
        text_seconds = time.perf_counter() - text_start

        rng = np.random.default_rng(seed)
        latents = rng.standard_normal(
            (1, 4, self.height // 8, self.width // 8),
            dtype=np.float32,
        )
        scheduler_config = json.loads(
            (self.model_dir / "scheduler" / "scheduler_config.json").read_text(
                encoding="utf-8"
            )
        )
        scheduler = _PNDMSampler(scheduler_config, self.inference_steps)
        unet_start = time.perf_counter()
        for counter, timestep in enumerate(scheduler.timesteps):
            noise = sessions[1].run(
                ["out_sample"],
                {
                    "sample": np.concatenate((latents, latents), axis=0),
                    "timestep": np.asarray([timestep], dtype=np.float32),
                    "encoder_hidden_states": embeddings,
                },
            )[0]
            unconditioned, conditioned = np.split(noise, 2, axis=0)
            guided_noise = unconditioned + self.guidance_scale * (
                conditioned - unconditioned
            )
            latents = scheduler.step(
                guided_noise,
                latents,
                int(timestep),
                counter,
            )
        unet_seconds = time.perf_counter() - unet_start

        vae_start = time.perf_counter()
        decoded = sessions[2].run(
            ["sample"],
            {"latent_sample": latents / self.latent_scale},
        )[0]
        vae_seconds = time.perf_counter() - vae_start
        image = np.clip(decoded.transpose(0, 2, 3, 1) / 2.0 + 0.5, 0.0, 1.0)
        pixels = np.rint(image[0] * 255.0).astype(np.uint8)
        return pixels, {
            "text_encoder_seconds": text_seconds,
            "unet_seconds": unet_seconds,
            "vae_seconds": vae_seconds,
        }

    def _generate_sync(
        self,
        prompt: str,
        metadata: Dict[str, Any],
    ) -> ImageGenerationResult:
        sessions: List[Any] = []
        started = time.perf_counter()
        try:
            if (self.width, self.height) != (512, 512):
                raise ImageProviderError(
                    "The current ONNX DirectML model supports 512x512 generation only; "
                    "configure IMAGE_WIDTH and IMAGE_HEIGHT to 512."
                )
            missing = self._missing_model_files()
            if missing:
                raise ImageProviderError(
                    "Required ONNX model files are missing: " + ", ".join(missing)
                )
            seed = int(metadata.get("seed", self.seed))
            if seed < 0:
                raise ImageProviderError("IMAGE_SEED must be a non-negative integer.")

            runtime = self._load_runtime()
            sessions = self._create_sessions(runtime)
            pixels, timings = self._run_core(prompt, seed, sessions)

            scene_number = metadata.get("scene_number", "frame")
            file_name = (
                f"onnx_sd15_scene_{scene_number}_{uuid.uuid4().hex[:8]}.png"
            )
            file_path = self.output_dir / file_name
            self.output_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pixels).save(file_path, format="PNG")

            result_metadata = dict(metadata)
            result_metadata.update(
                {
                    "width": self.width,
                    "height": self.height,
                    "seed": seed,
                    "inference_steps": self.inference_steps,
                    **timings,
                    "generation_seconds": time.perf_counter() - started,
                }
            )
            return ImageGenerationResult(
                image_path=str(file_path),
                image_url=f"/generated/images/{file_name}",
                provider_name=self.provider_name,
                status="generated",
                prompt_used=prompt,
                metadata=result_metadata,
            )
        finally:
            sessions.clear()
            del sessions
            gc.collect()

    async def generate_image(
        self,
        prompt: str,
        metadata: Dict[str, Any],
    ) -> ImageGenerationResult:
        try:
            return await asyncio.to_thread(self._generate_sync, prompt, metadata)
        except Exception as exc:
            logger.exception(
                "ONNX DirectML image generation failed for scene %s",
                metadata.get("scene_number", "unknown"),
            )
            return ImageGenerationResult(
                provider_name=self.provider_name,
                status="failed",
                prompt_used=prompt,
                metadata=metadata,
                error=f"ONNX DirectML image generation failed: {exc}",
            )
