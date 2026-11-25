import os
from pathlib import Path
from typing import List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent.parent
GENERATED_DIR = BASE_DIR / "generated"
MODELS_DIR = BASE_DIR / "models" / "llm"


class Settings(BaseSettings):
    """Application configuration settings."""

    # Server Settings
    APP_NAME: str = "AI Video Generator API"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

    # Database Configuration (PostgreSQL)
    # Uses host port 5433
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5433/ai_video_db"

    # AI Service Settings
    STORY_ANALYZER_PROVIDER: str = "rule_based"
    DEFAULT_GENRE: str = "Cinematic Drama"
    VOICE_PROVIDER: str = "placeholder"
    AUDIO_PROVIDER: str = "placeholder"
    AUDIO_OUTPUT_DIR: Path = GENERATED_DIR / "audio"
    AUDIO_DEFAULT_SAMPLE_RATE: int = Field(default=22050, gt=0)
    AUDIO_DEFAULT_CHANNELS: int = Field(default=1, gt=0)
    AUDIO_DEFAULT_FORMAT: str = "wav"
    AI_AUDIO_MODEL_ID: str = "cvssp/audioldm2"
    AI_AUDIO_MODEL_REVISION: str = "c8e7e189d324425c05c4c2f81214041ef4107983"
    AI_AUDIO_MODEL_PATH: Path = BASE_DIR / "models" / "audio" / "audioldm2"
    AI_AUDIO_PYTHON_PATH: Path = (
        BASE_DIR
        / ".venv-audio"
        / ("Scripts" if os.name == "nt" else "bin")
        / ("python.exe" if os.name == "nt" else "python")
    )
    AI_AUDIO_DEVICE: str = "cuda"
    AI_AUDIO_MAX_DURATION_SECONDS: float = Field(default=5.0, gt=0, le=10.0)
    AI_AUDIO_STEPS: int = Field(default=20, ge=1, le=200)
    AI_AUDIO_SEED: int = Field(default=42, ge=0)
    AI_AUDIO_TIMEOUT_SECONDS: int = Field(default=900, ge=30, le=3600)
    PIPER_MODEL_PATH: Path = Path(
        "models/voice/en_US-hifi_tts_92-medium/en_US-hifi_tts_92-medium.onnx"
    )
    VIDEO_PROVIDER: str = "placeholder"
    VIDEO_MODEL_PATH: Path = BASE_DIR / "models" / "video"
    VIDEO_OUTPUT_DIR: Path = GENERATED_DIR / "videos"
    VIDEO_DEFAULT_WIDTH: int = Field(default=1280, gt=0)
    VIDEO_DEFAULT_HEIGHT: int = Field(default=720, gt=0)
    VIDEO_DEFAULT_FPS: float = Field(default=24.0, gt=0)
    VIDEO_DEFAULT_DURATION_SECONDS: float = Field(default=5.0, gt=0)

    # Optional CogVideoX provider. Its runtime is isolated from the backend env.
    COGVIDEOX_MODEL_ID: str = "zai-org/CogVideoX-2b"
    COGVIDEOX_MODEL_REVISION: str = (
        "1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01"
    )
    COGVIDEOX_MODEL_PATH: Optional[Path] = None
    COGVIDEOX_CACHE_DIR: Optional[Path] = None
    COGVIDEOX_PYTHON_PATH: Path = (
        BASE_DIR
        / ".venv-video-cogvideox"
        / ("Scripts" if os.name == "nt" else "bin")
        / ("python.exe" if os.name == "nt" else "python")
    )
    COGVIDEOX_WIDTH: int = Field(default=720, ge=16, le=720)
    COGVIDEOX_HEIGHT: int = Field(default=480, ge=16, le=480)
    COGVIDEOX_FRAMES: int = Field(default=8, ge=4, le=49)
    COGVIDEOX_FPS: float = Field(default=8.0, ge=1.0, le=30.0)
    COGVIDEOX_STEPS: int = Field(default=2, ge=1, le=50)
    COGVIDEOX_GUIDANCE_SCALE: float = Field(default=1.0, ge=0.0, le=20.0)
    COGVIDEOX_SCHEDULER: str = "ddim"
    COGVIDEOX_CPU_OFFLOAD: str = "sequential"
    COGVIDEOX_USE_DYNAMIC_CFG: bool = False
    COGVIDEOX_ATTENTION_SLICING: bool = False
    COGVIDEOX_VAE_SLICING: bool = True
    COGVIDEOX_VAE_TILING: bool = True
    COGVIDEOX_DTYPE: str = "float16"
    COGVIDEOX_LOCAL_FILES_ONLY: bool = False
    COGVIDEOX_SEED: int = Field(default=42, ge=0)
    COGVIDEOX_OUTPUT_DIR: Path = GENERATED_DIR / "videos"
    COGVIDEOX_TIMEOUT_SECONDS: int = Field(default=1800, ge=30, le=7200)

    @field_validator("COGVIDEOX_FRAMES")
    @classmethod
    def validate_cogvideox_frames(cls, value: int) -> int:
        if value != 49 and value % 4 != 0:
            raise ValueError(
                "COGVIDEOX_FRAMES must be divisible by 4 or equal 49."
            )
        return value

    @field_validator("COGVIDEOX_SCHEDULER")
    @classmethod
    def validate_cogvideox_scheduler(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"ddim", "dpm"}:
            raise ValueError("COGVIDEOX_SCHEDULER must be 'ddim' or 'dpm'.")
        return normalized

    @field_validator("COGVIDEOX_CPU_OFFLOAD")
    @classmethod
    def validate_cogvideox_cpu_offload(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"sequential", "hybrid", "model"}:
            raise ValueError(
                "COGVIDEOX_CPU_OFFLOAD must be 'sequential', 'hybrid', or 'model'."
            )
        return normalized

    @field_validator("COGVIDEOX_DTYPE")
    @classmethod
    def validate_cogvideox_dtype(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"float16", "bfloat16"}:
            raise ValueError("COGVIDEOX_DTYPE must be 'float16' or 'bfloat16'.")
        return normalized

    # Local LLM Configuration (Step 2)
    LOCAL_LLM_MODEL_PATH: Path = MODELS_DIR / "qwen2.5-3b-instruct-q4_k_m.gguf"
    LOCAL_LLM_N_GPU_LAYERS: int = -1
    LOCAL_LLM_CONTEXT_SIZE: int = 4096
    LOCAL_LLM_FALLBACK_TO_RULE_BASED: bool = True

    # Storyboard image provider settings
    IMAGE_PROVIDER: str = "placeholder"
    IMAGE_MODEL_PATH: Path = BASE_DIR / "models" / "image"
    IMAGE_MODEL_DIR: Path = BASE_DIR / "models" / "image" / "sd15-onnx"
    IMAGE_SEED: int = Field(default=42, ge=0)
    IMAGE_OUTPUT_DIR: Path = GENERATED_DIR / "images"
    IMAGE_WIDTH: int = Field(default=1280, gt=0)
    IMAGE_HEIGHT: int = Field(default=720, gt=0)

    @property
    def resolved_local_llm_model_path(self) -> Path:
        """Resolve model path to absolute path relative to BASE_DIR if not absolute."""
        if self.LOCAL_LLM_MODEL_PATH.is_absolute():
            return self.LOCAL_LLM_MODEL_PATH
        return BASE_DIR / self.LOCAL_LLM_MODEL_PATH

    @property
    def resolved_piper_model_path(self) -> Path:
        """Resolve the configured Piper model path relative to BASE_DIR."""
        if self.PIPER_MODEL_PATH.is_absolute():
            return self.PIPER_MODEL_PATH
        return BASE_DIR / self.PIPER_MODEL_PATH

    # Directories
    MEDIA_ROOT: Path = GENERATED_DIR
    IMAGES_DIR: Path = GENERATED_DIR / "images"
    AUDIO_DIR: Path = GENERATED_DIR / "audio"
    VIDEOS_DIR: Path = GENERATED_DIR / "videos"
    MOVIES_DIR: Path = GENERATED_DIR / "movies"
    MOVIE_ASSEMBLY_TIMEOUT_SECONDS: int = Field(default=1800, ge=30, le=7200)
    MODELS_DIR: Path = MODELS_DIR

    # AI Service API Keys (Optional/Configurable via .env)
    GEMINI_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    ELEVENLABS_API_KEY: str = ""
    RUNWAY_API_KEY: str = ""
    LUMA_API_KEY: str = ""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / "backend" / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

# Ensure generated & model directories exist
for path in [
    settings.IMAGES_DIR,
    settings.IMAGE_OUTPUT_DIR,
    settings.AUDIO_DIR,
    settings.VIDEOS_DIR,
    settings.MOVIES_DIR,
    settings.MODELS_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)
