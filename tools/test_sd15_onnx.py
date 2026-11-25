import argparse
import gc
import hashlib
import html
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "image" / "sd15-onnx"
OUTPUT_PATH = MODEL_DIR / "test_output.png"
PROMPT = "a cinematic film still of a quiet street at night"
WIDTH = 512
HEIGHT = 512
NUM_INFERENCE_STEPS = 4
GUIDANCE_SCALE = 7.5
LATENT_SCALE = 0.18215


class ClipTokenizer:
    def __init__(self, directory: Path) -> None:
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
    def _bytes_to_unicode() -> dict[int, str]:
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

    def _bpe(self, token: str) -> list[str]:
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

    def encode(self, text: str) -> np.ndarray:
        text = html.unescape(html.unescape(text)).strip().lower()
        text = re.sub(r"\s+", " ", text)
        token_ids = [self.vocab["<|startoftext|>"]]
        for token in self.pattern.findall(text):
            encoded = "".join(self.byte_encoder[value] for value in token.encode("utf-8"))
            token_ids.extend(self.vocab[piece] for piece in self._bpe(encoded))
        token_ids.append(self.vocab["<|endoftext|>"])
        token_ids = token_ids[:77]
        token_ids.extend([self.vocab["<|endoftext|>"]] * (77 - len(token_ids)))
        return np.asarray(token_ids, dtype=np.int32)


class PNDMSampler:
    def __init__(self, config: dict[str, object], steps: int) -> None:
        self.train_steps = int(config["num_train_timesteps"])
        self.step_ratio = self.train_steps // steps
        start = float(config["beta_start"])
        end = float(config["beta_end"])
        roots = np.linspace(np.sqrt(start), np.sqrt(end), self.train_steps, dtype=np.float32)
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
        self.history: list[np.ndarray] = []
        self.initial_sample: np.ndarray | None = None

    def step(
        self,
        model_output: np.ndarray,
        sample: np.ndarray,
        timestep: int,
        counter: int,
    ) -> np.ndarray:
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


def create_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.enable_mem_pattern = False
    return ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=["DmlExecutionProvider"],
    )


def report_gpu_memory(label: str) -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        snapshot = f"unavailable: {type(exc).__name__}: {exc}"
    else:
        snapshot = result.stdout.strip()
    print(f"{label}: {snapshot}")
    return snapshot


def create_sessions() -> list[ort.InferenceSession]:
    sessions = [
        create_session(MODEL_DIR / "text_encoder" / "model.onnx"),
        create_session(MODEL_DIR / "unet" / "model.onnx"),
        create_session(MODEL_DIR / "vae_decoder" / "model.onnx"),
    ]
    for name, session in zip(("text encoder", "UNet", "VAE decoder"), sessions):
        print(f"{name} providers: {session.get_providers()}")
    return sessions


def report_generation(result: dict[str, float | str]) -> None:
    print(
        f"seed {result['seed']}: total={result['total']:.2f}s, "
        f"text={result['text_encoder']:.2f}s, "
        f"UNet={result['unet']:.2f}s, VAE={result['vae']:.2f}s, "
        f"bytes={result['bytes']}, sha256={result['sha256']}, "
        f"path={result['path']}"
    )


def run_lifecycle_test() -> None:
    print("creating first DirectML sessions", flush=True)
    first_sessions = create_sessions()
    print("starting generation 1 (seed 42)", flush=True)
    first = generate_image(
        first_sessions,
        42,
        MODEL_DIR / "lifecycle_test_42.png",
    )
    report_generation(first)
    report_gpu_memory("VRAM after generation 1")

    del first_sessions
    collected = gc.collect()
    print(f"released first session references; garbage collected: {collected}")
    report_gpu_memory("VRAM after session release")

    print("creating fresh DirectML sessions", flush=True)
    second_sessions = create_sessions()
    print("starting generation 2 (seed 43)", flush=True)
    second = generate_image(
        second_sessions,
        43,
        MODEL_DIR / "lifecycle_test_43.png",
    )
    report_generation(second)
    report_gpu_memory("VRAM after generation 2")
    print("session reuse: NO")
    print("session recreation: PASS")
    print(f"images different: {'yes' if first['sha256'] != second['sha256'] else 'no'}")


def run_fresh_session_stability() -> None:
    jobs = (
        (42, MODEL_DIR / "fresh_session_test_42.png"),
        (43, MODEL_DIR / "fresh_session_test_43.png"),
        (44, MODEL_DIR / "fresh_session_test_44.png"),
    )
    results = []
    print(
        "safety checker not executed; prior DirectML inference failure: "
        "887A0006 at DmlCommandRecorder.cpp(371)"
    )
    for generation_number, (seed, output_path) in enumerate(jobs, start=1):
        print(
            f"creating fresh sessions for generation {generation_number}",
            flush=True,
        )
        sessions = create_sessions()
        print(f"starting generation {generation_number} (seed {seed})", flush=True)
        result = generate_image(sessions, seed, output_path)
        report_generation(result)
        report_gpu_memory(f"VRAM after generation {generation_number}")

        del sessions
        collected = gc.collect()
        print(
            f"released sessions after generation {generation_number}; "
            f"garbage collected: {collected}"
        )
        report_gpu_memory(f"VRAM after release {generation_number}")
        results.append(result)

    hashes = [str(result["sha256"]) for result in results]
    average = sum(float(result["total"]) for result in results) / len(results)
    print(f"successful generations: {len(results)}")
    print("fresh sessions used for every generation: PASS")
    print(f"average generation time: {average:.2f}s")
    print(f"all image hashes different: {'yes' if len(set(hashes)) == 3 else 'no'}")


def generate_image(
    sessions: list[ort.InferenceSession],
    seed: int,
    output_path: Path,
) -> dict[str, float | str]:
    core_generation_start = time.perf_counter()
    tokenizer = ClipTokenizer(MODEL_DIR / "tokenizer")
    prompt_ids = tokenizer.encode(PROMPT)
    negative_ids = tokenizer.encode("")
    input_ids = np.stack((negative_ids, prompt_ids), axis=0)

    text_start = time.perf_counter()
    text_outputs = sessions[0].run(
        ["last_hidden_state"],
        {"input_ids": input_ids},
    )[0]
    text_encoder_seconds = time.perf_counter() - text_start
    prompt_embeddings = text_outputs.astype(np.float32)

    rng = np.random.default_rng(seed)
    latents = rng.standard_normal(
        (1, 4, HEIGHT // 8, WIDTH // 8),
        dtype=np.float32,
    )
    scheduler_config = json.loads(
        (MODEL_DIR / "scheduler" / "scheduler_config.json").read_text(
            encoding="utf-8"
        )
    )
    scheduler = PNDMSampler(scheduler_config, NUM_INFERENCE_STEPS)

    unet_start = time.perf_counter()
    for counter, timestep in enumerate(scheduler.timesteps):
        latent_input = np.concatenate((latents, latents), axis=0)
        timestep_input = np.asarray([timestep], dtype=np.float32)
        noise = sessions[1].run(
            ["out_sample"],
            {
                "sample": latent_input,
                "timestep": timestep_input,
                "encoder_hidden_states": prompt_embeddings,
            },
        )[0]
        unconditioned, conditioned = np.split(noise, 2, axis=0)
        guided_noise = unconditioned + GUIDANCE_SCALE * (
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
        {"latent_sample": latents / LATENT_SCALE},
    )[0]
    vae_seconds = time.perf_counter() - vae_start
    image = np.clip(decoded.transpose(0, 2, 3, 1) / 2.0 + 0.5, 0.0, 1.0)
    pixels = np.rint(image[0] * 255.0).astype(np.uint8)
    output_image = Image.fromarray(pixels).convert("RGB")
    output_image.save(output_path, format="PNG")

    if not output_path.is_file():
        raise RuntimeError(f"PNG was not created: {output_path}")
    with Image.open(output_path) as verified:
        if verified.format != "PNG":
            raise RuntimeError(f"Unexpected image format: {verified.format}")
        if verified.size != (WIDTH, HEIGHT):
            raise RuntimeError(f"Unexpected image dimensions: {verified.size}")
        if verified.mode != "RGB":
            raise RuntimeError(f"Unexpected image color mode: {verified.mode}")
        verified.load()
        print(
            f"validated PNG: {output_path.name}, format={verified.format}, "
            f"size={verified.size}, mode={verified.mode}, pixels=loaded"
        )
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    timings: dict[str, float | str] = {
        "seed": seed,
        "total": time.perf_counter() - core_generation_start,
        "text_encoder": text_encoder_seconds,
        "unet": unet_seconds,
        "vae": vae_seconds,
        "bytes": output_path.stat().st_size,
        "sha256": digest,
        "path": str(output_path),
    }
    return timings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SD 1.5 ONNX DirectML prototype.")
    parser.add_argument(
        "--repeatability",
        action="store_true",
        help="Generate seeds 42, 43, and 44 sequentially using reused sessions.",
    )
    parser.add_argument(
        "--lifecycle-test",
        action="store_true",
        help="Generate seeds 42 and 43 with session deletion and recreation between them.",
    )
    parser.add_argument(
        "--fresh-session-stability",
        action="store_true",
        help="Generate seeds 42, 43, and 44 with sessions recreated between images.",
    )
    args = parser.parse_args()

    if args.fresh_session_stability:
        run_fresh_session_stability()
        return
    if args.lifecycle_test:
        run_lifecycle_test()
        return

    session_start = time.perf_counter()
    sessions = create_sessions()
    session_load_seconds = time.perf_counter() - session_start
    print(f"session load seconds: {session_load_seconds:.2f}")
    print(
        "safety checker not executed; prior DirectML inference failure: "
        "887A0006 at DmlCommandRecorder.cpp(371)"
    )

    if args.repeatability:
        jobs = [
            (42, MODEL_DIR / "stability_test_42.png"),
            (43, MODEL_DIR / "stability_test_43.png"),
            (44, MODEL_DIR / "stability_test_44.png"),
        ]
    else:
        jobs = [(42, OUTPUT_PATH)]

    original_session_ids = tuple(id(session) for session in sessions)
    results = []
    for seed, output_path in jobs:
        print(f"starting generation for seed {seed}", flush=True)
        result = generate_image(sessions, seed, output_path)
        results.append(result)
        print(
            f"seed {seed}: total={result['total']:.2f}s, "
            f"text={result['text_encoder']:.2f}s, "
            f"UNet={result['unet']:.2f}s, VAE={result['vae']:.2f}s, "
            f"bytes={result['bytes']}, sha256={result['sha256']}"
        )
        report_gpu_memory(f"VRAM after generation {seed}")

    if args.repeatability:
        session_reused = original_session_ids == tuple(id(session) for session in sessions)
        hashes = [str(result["sha256"]) for result in results]
        average = sum(float(result["total"]) for result in results) / len(results)
        print(f"total successful generations: {len(results)}")
        print(f"session reuse: {'PASS' if session_reused else 'FAIL'}")
        print(f"average generation time: {average:.2f}s")
        print(f"images confirmed different: {'yes' if len(set(hashes)) == 3 else 'no'}")


if __name__ == "__main__":
    main()
