# CineAI

CineAI is a local story-to-film workspace: FastAPI analyzes a premise and
persists its story, characters, and scenes in PostgreSQL; React/Vite presents
those records and lets the user explicitly request media; provider-neutral
movie assembly combines local scene assets into an MP4.

## Architecture

- **Frontend:** React 18 + Vite; API and media requests are centralized in
  `frontend/src/services/api.js`.
- **Backend:** FastAPI routes, Pydantic schemas, services, and provider
  registries under `backend/app/`.
- **Persistence:** PostgreSQL stores stories, characters, scenes, and movie
  job status/result metadata. Generated media files are stored under
  `generated/`, not in PostgreSQL. Per-scene audio result details remain in
  process memory for the lifetime of the backend process.
- **Assembly:** FFmpeg via `imageio-ffmpeg` reads existing assets and creates
  H.264/AAC MP4 output. It does not invoke model runtimes.
- **Models:** Optional local model weights are kept under `models/`; the
  repository ignore rules exclude local weights, generated media, virtual
  environments, and environment files.

## Local development

Requirements: Python 3.11, Node.js/npm, Docker Compose, and a local PostgreSQL
instance. The default database connection is for the included local Compose
service and is not suitable for a shared or deployed environment.

1. Start PostgreSQL from the project root:

   ```powershell
   docker compose up -d postgres
   ```

2. Create and configure the backend environment:

   ```powershell
   Copy-Item backend\.env.example backend\.env
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install --upgrade pip
   .\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
   .\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
   ```

   For development auto-reload, add `--reload`. The API docs are available at
   `http://127.0.0.1:8000/docs`.

3. In a separate terminal:

   ```powershell
   Set-Location frontend
   npm ci
   npm run dev
   ```

   Open `http://127.0.0.1:5173`. Vite proxies `/api` and `/generated` to the
   local backend by default. `frontend/.env.example` documents optional
   `VITE_API_BASE_URL`, `VITE_MEDIA_BASE_URL`, proxy, and movie polling
   settings. Never put provider credentials in a `VITE_` variable.

4. The UI flow is **Story analysis → review characters/scenes/screenplay →
   generate selected assets → assemble movie → poll job status → play or
   download MP4**. Media/model generation is explicit; loading the UI does not
   invoke providers.

## Providers and defaults

| Asset | Default | Optional provider | Runtime |
|---|---|---|---|
| Story analysis | `rule_based` | `local_llm` | Isolated CPU llama.cpp environment and a separately downloaded local GGUF |
| Storyboard image | `placeholder` | `onnx_directml` | ONNX Runtime DirectML and local SD 1.5 ONNX files; verified dimensions are 512×512 |
| Dialogue voice | `placeholder` | `piper_onnx` | Piper + ONNX Runtime DirectML and a local Piper ONNX voice |
| Scene video | `placeholder` | `cogvideox` | Isolated `.venv-video-cogvideox`, NVIDIA CUDA, local/cached CogVideoX-2B |
| Scene audio | `placeholder` | `local_audio`, `ai_audio` | Deterministic procedural WAV, or isolated `.venv-audio` + local AudioLDM 2 + NVIDIA CUDA |

`placeholder` providers are safe defaults and do not claim to produce AI
media. `local_audio` writes procedural WAV files and is **not AI-generated**.
ONNX DirectML image generation and Piper are real local inference; CogVideoX
and AudioLDM 2 are GPU-backed and opt-in. Avoid running multiple GPU providers
at once on an 8 GB GPU. Optional model weights are not installed by the main
backend requirements.

The root `backend/.env.example` lists safe defaults and relevant environment
variables. Copy it to `backend/.env`; keep the latter private. Optional
credentials are needed only for a provider that explicitly uses them.
`DEBUG=false` and loopback binding are the defaults. Set `HOST` or `CORS_ORIGINS`
only when intentionally allowing access from another machine/origin.

Generated media locations:

- Storyboards: `generated/images/`
- Voice previews: `generated/voice/`
- Scene audio: `generated/audio/`
- Scene videos: `generated/videos/`
- Final movies: `generated/movies/`

### Optional ONNX DirectML Storyboard Provider

The storyboard image provider defaults to `placeholder`. To opt into the
experimental `onnx_directml` provider, make the Stable Diffusion 1.5 ONNX model
files available under `models/image/sd15-onnx/` (or set `IMAGE_MODEL_DIR`), and
run the backend in a Python environment with ONNX Runtime DirectML available.
Configure `IMAGE_PROVIDER=onnx_directml`, `IMAGE_WIDTH=512`, and
`IMAGE_HEIGHT=512`; `IMAGE_SEED` controls the default seed. The provider creates
fresh DirectML sessions for each image and does not load or run the optional
safety checker. The placeholder provider remains the default.

### Optional CogVideoX scene-video provider

Scene video generation defaults to `VIDEO_PROVIDER=placeholder`. To opt into
the experimental `cogvideox` provider, use the isolated
`.venv-video-cogvideox` environment and set `VIDEO_PROVIDER=cogvideox`.
`COGVIDEOX_PYTHON_PATH` selects that environment's Python executable;
`COGVIDEOX_MODEL_PATH` can point to a local model snapshot, while
`COGVIDEOX_CACHE_DIR` selects the Hugging Face cache when loading by model ID.
The default model ID and revision are pinned in the backend settings.

CogVideoX is not imported or loaded during FastAPI startup. The provider starts
the isolated runtime only when a scene-generation request is made, uses FP16
with sequential CPU offload, and requires an NVIDIA CUDA GPU. Its safe defaults
are 720x480, 8 frames, 8 FPS, and 2 inference steps; the 2B model is restricted
to its verified 720x480 output resolution. CogVideoX dependencies are not added
to the main backend requirements.

#### Resumable video batches

The batch runner stores an atomic `manifest.json` and one MP4 per attempt under
`COGVIDEOX_OUTPUT_DIR\batches\<batch-id>`. Completed, validated shots are reused
when a batch is resumed; failed shots can be retried individually without
overwriting prior outputs. Prompts and request metadata are kept in the
manifest. Warm-worker mode uses the CogVideoX provider's existing worker
session, so its pipeline stays loaded across sequential shots.

Create a `shots.json` file with this structure:

```json
{
  "shots": [
    {
      "shot_number": 1,
      "request": {
        "prompt": "A cinematic landscape at dawn.",
        "duration_seconds": 2
      }
    }
  ]
}
```

From the repository root, create and run a batch (or resume it by running the
same `run` command again):

```powershell
$env:PYTHONPATH='backend'
python -m app.services.video_batch create --batch-id landscape-test --shots-json .\shots.json
python -m app.services.video_batch run --batch-id landscape-test --warm-worker
```

Retry a single failed shot explicitly:

```powershell
python -m app.services.video_batch retry --batch-id landscape-test --shot-number 1 --warm-worker
```

The batch runner does not change CogVideoX settings or assemble a movie. It
requires the selected provider to produce a decodable MP4, so the existing
placeholder provider cannot complete a batch.

The real GPU smoke test is skipped unless explicitly enabled:

```powershell
$env:CINEAI_COGVIDEOX_SMOKE='1'
$env:PYTHONPATH='backend'
python -m pytest backend/tests/test_cogvideox_smoke.py -v
```

### Step 7.2 Local audio provider

`local_audio` is an optional, CPU-only provider that uses Python's standard
library to write deterministic WAV assets for ambience, simple sound effects,
and procedural music patterns. These functional test assets are not
AI-generated music. No model, external API, FFmpeg, or additional package is
required. The provider remains opt-in; `AUDIO_PROVIDER=placeholder` is still
the default. Generated files use `AUDIO_OUTPUT_DIR` unless a request specifies
another output directory.

The local audio smoke test is opt-in:

```powershell
$env:CINEAI_LOCAL_AUDIO_SMOKE='1'
$env:PYTHONPATH='backend'
python -m pytest backend/tests/test_local_audio_smoke.py -v
```

### Audio Provider Architecture

Audio generation is provider-based, with `AUDIO_PROVIDER=placeholder` as the
default. `local_audio` creates deterministic procedural test assets, while
`ai_audio` is an optional isolated model runtime. Piper remains responsible
for dialogue and narration.

### Step 7.3 — AI Audio Generation

The selected first AI model is **AudioLDM 2 base** (`cvssp/audioldm2`). Its
official Diffusers pipeline supports text-to-audio generation for sound
effects and music, making it a broader fit for scene ambience and effects than
MusicGen-small, which is music-only. AudioLDM 2 is 1.1B parameters. The pinned
repository revision's safetensors weigh approximately **4.47 GB** (4.17 GiB);
the repository also contains duplicate `.bin` weights, so a full unfiltered
checkout is larger. Its model license is **CC BY-NC-SA 4.0**; it is not cleared
for commercial use.

The target machine has an RTX 5050 Laptop GPU with about 8 GB VRAM and 24 GB
RAM. FP16 weights are approximately 2.2 GB; with activations and working
buffers, peak VRAM is estimated at 4–7 GB for short clips. CPU offload is
enabled. Host memory use is estimated at 8–12 GB for the offloaded pipeline.
These are estimates, not verified requirements. CPU-only generation is
disabled. The model accepts up to 10 seconds per request in this integration,
with a default maximum of 5 seconds and 20 inference steps. The model card and
[Diffusers AudioLDM 2 documentation](https://huggingface.co/docs/diffusers/api/pipelines/audioldm2)
describe the supported text-to-audio tasks and inference controls.

Create and install the isolated CUDA environment (this does not download the
model):

```powershell
python -m venv .venv-audio
.\.venv-audio\Scripts\python.exe -m pip install --upgrade pip
.\.venv-audio\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r backend/requirements-audio.txt
```

Download the pinned model explicitly to `models/audio/audioldm2/` (about
4.47 GB of safetensors plus small configuration/tokenizer files):

```powershell
.\.venv-audio\Scripts\python.exe tools/download_audioldm2.py
```

The download script selects safetensors and supporting config files only; it
does not download the duplicate `.bin` checkpoints. Runtime model loading is
local-only and never downloads weights during a request. Configure
`AUDIO_PROVIDER=ai_audio` to opt in; `placeholder` remains the default. The
worker runs in `.venv-audio`, validates CUDA/NVIDIA availability, uses FP16,
batch size 1, and Diffusers model CPU offload.

Example request:

```python
AudioGenerationRequest(
    prompt="Rain falling on a quiet city street at night",
    audio_type="ambience",
    duration_seconds=2,
    sample_rate=22050,
    channels=1,
    seed=42,
)
```

The AI audio smoke test is opt-in and requires both the isolated runtime and
the downloaded local model:

```powershell
$env:CINEAI_AI_AUDIO_SMOKE='1'
$env:PYTHONPATH='backend'
python -m pytest backend/tests/test_ai_audio_smoke.py -v
```

It generates a short ambience WAV in a temporary directory, validates PCM,
sample rate, channel count, duration and non-silence, and records the
generation time. It is not run during normal tests. Output quality varies
with prompts and seeds; generated clips are short and may need curation.
Movie assembly below combines available scene audio and dialogue assets into the movie.
Procedural `local_audio` remains explicitly non-AI
(`ai_generated=false`); successful AudioLDM 2 results are marked
`ai_generated=true`.

### Scene and Movie Audio

`AUDIO_PROVIDER=placeholder` remains the default and does not generate audio
for movie jobs. Set `AUDIO_PROVIDER=local_audio` to create deterministic
procedural WAV assets, or opt in to `AUDIO_PROVIDER=ai_audio` for GPU-backed
AudioLDM 2 generation through `.venv-audio`. AI generation is computationally
expensive and uses only the local model; request-time downloads are disabled.

The scene endpoint `POST /api/scenes/{scene_number}/audio` builds a
deterministic prompt from the stored scene and returns an
`AudioGenerationResult`. Its optional JSON body accepts `story_id`,
`audio_type`, `duration_seconds`, and `seed`. The audio type is selected from
scene cues when omitted; dialogue text is not sent to the audio model.

When a non-placeholder audio provider is configured, movie rendering attempts
to generate one scene-audio asset per scene in the background. Use
`GET /api/movie/audio/{movie_id}` to retrieve per-scene results and metadata
while the backend process is running. Audio files remain in the
generated-media directory; no binary audio data is stored in PostgreSQL.

### Movie Assembly and Final Rendering

`POST /api/movie/render` now reads the story's ordered scenes and assembles
their existing local media into a real H.264/AAC MP4. It prefers each scene's
video file and falls back to its storyboard image. Dialogue WAV references and
configured scene ambience, effects, and music are placed on the scene timeline;
audio is mixed and encoded as AAC. Scene transitions default to cuts, with
crossfades supported by the provider-neutral assembly request.

Movie assembly uses the CPU-side FFmpeg binary provided by
`imageio-ffmpeg`; it does not import or call CogVideoX, AudioLDM 2, Piper, ONNX
Runtime, PyTorch, or Diffusers. It combines already-generated files only, so
movie rendering does not start image, video, or voice generation. Configure
those existing providers and create assets separately. `AUDIO_PROVIDER` remains
`placeholder` by default; choosing a non-placeholder audio provider continues
to opt in to Step 7.4 scene-audio generation.

Each scene must have an existing local video or storyboard image under the
generated-media directory. Missing or invalid assets cause the movie job to
finish as `failed` rather than returning a dummy MP4. Poll
`GET /api/movie/status/{movie_id}` for completion; the existing `video_url`
points to the final file under `/generated/movies/`. Media stays on disk and no
binary assets are stored in PostgreSQL. Dialogue clips start at the beginning
of their scene because the current dialogue schema does not record per-line
timing; audio asset offsets can be supplied to the assembly service directly.

### Piper ONNX voice

Set `VOICE_PROVIDER=piper_onnx` only in a backend Python environment with the
`piper` and ONNX Runtime DirectML modules available. `PIPER_MODEL_PATH` must
name a local `.onnx` voice model with its companion `.onnx.json` configuration.
The provider validates that DirectML is enabled, writes validated PCM WAV
files to `generated/voice/`, and removes incomplete outputs. The frontend
requests one representative voice preview per character; if its text exactly
matches a stored dialogue line, that line's existing `audio_url` field is
updated so the movie assembler can include it. Other dialogue lines are not
automatically synthesized.

## Verification

Start PostgreSQL before running the full backend suite. From the project root:

```powershell
$env:PYTHONPATH='backend'
python -m pytest backend/tests -v
python -m pip check
Set-Location frontend
npm ci
npm audit
npm run build
```

The normal backend suite skips model/GPU smoke tests. Run them explicitly only
when the corresponding local environments, weights, and hardware are ready:

```powershell
$env:PYTHONPATH='backend'
$env:CINEAI_LOCAL_AUDIO_SMOKE='1'
python -m pytest backend/tests/test_local_audio_smoke.py -v
$env:CINEAI_AI_AUDIO_SMOKE='1'
python -m pytest backend/tests/test_ai_audio_smoke.py -v
$env:CINEAI_COGVIDEOX_SMOKE='1'
python -m pytest backend/tests/test_cogvideox_smoke.py -v
```

The AI audio test uses the local AudioLDM 2 model and isolated `.venv-audio`;
the CogVideoX test uses its isolated runtime and pinned local/cache revision.
ONNX DirectML image and Piper checks require the backend to run in the
environment that provides DirectML and their model files. Do not treat
placeholder output as AI-generated media.

## Operational limits and security

- The current app is intended for local development/demo use. It has no user
  authentication or authorization, and `/generated` is served as public
  static media. Do not expose the backend or Vite development server to an
  untrusted network.
- Story, character, scene, and movie job status/result records are stored in
  PostgreSQL. If the backend restarts during a render, the next status request
  marks that interrupted job as failed; generated media files already written
  to disk remain available. Per-scene audio result details are held in process
  memory and are not restored after a backend restart. Run one backend process:
  active render tasks and their audio results are process-local, and the small
  job/audio metadata cache is retained until restart. Multi-worker deployment
  and sustained high-volume workloads are not supported.
- The movie API requires an existing scene image or video file for every
  scene. Missing or invalid assets fail the job rather than creating a
  success-shaped placeholder MP4.
- CogVideoX and AudioLDM 2 are expensive GPU operations. AudioLDM 2 is
  licensed CC BY-NC-SA 4.0 and is not cleared for commercial use. Verify all
  model licenses and generated content before use.
- Main backend requirements intentionally omit model runtimes. `pip check`
  should pass for the active environment; optional GPU environments may have
  provider-specific dependency metadata differences and should be validated
  separately.
