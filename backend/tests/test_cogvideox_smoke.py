import os

import pytest

from app.config import Settings, settings
from app.services.video_providers import CogVideoXVideoProvider, VideoGenerationRequest


pytestmark = pytest.mark.skipif(
    os.getenv("CINEAI_COGVIDEOX_SMOKE") != "1",
    reason="Set CINEAI_COGVIDEOX_SMOKE=1 to run the local CogVideoX CUDA smoke test.",
)


@pytest.mark.anyio
async def test_cogvideox_minimal_cuda_smoke(tmp_path):
    config = Settings(
        _env_file=None,
        COGVIDEOX_MODEL_ID="zai-org/CogVideoX-2b",
        COGVIDEOX_MODEL_REVISION="1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01",
        COGVIDEOX_MODEL_PATH=None,
        COGVIDEOX_PYTHON_PATH=settings.COGVIDEOX_PYTHON_PATH,
        COGVIDEOX_WIDTH=720,
        COGVIDEOX_HEIGHT=480,
        COGVIDEOX_FRAMES=8,
        COGVIDEOX_FPS=8,
        COGVIDEOX_STEPS=2,
        COGVIDEOX_GUIDANCE_SCALE=1.0,
        COGVIDEOX_SEED=42,
        COGVIDEOX_OUTPUT_DIR=tmp_path,
    )
    provider = CogVideoXVideoProvider(config=config)
    result = await provider.generate_video(
        VideoGenerationRequest(
            prompt=(
                "A cinematic shot of a young man walking through a rainy city street "
                "at night, realistic lighting, natural camera movement, photorealistic."
            ),
            width=720,
            height=480,
            fps=8,
            seed=42,
        )
    )

    assert result.status == "generated"
    assert result.output_path is not None and result.output_path.is_file()
    assert result.width == 720
    assert result.height == 480
    assert result.fps == 8
    assert result.metadata["frames"] == 8
    assert result.metadata["inference_steps"] == 2
    assert result.metadata["device"] == "cuda:0"
    assert result.metadata["dtype"] == "float16"
    assert result.metadata["cpu_offload"] == "sequential"
