from pathlib import Path

import numpy as np
import pytest

from app.config import BASE_DIR, settings
from app.models import Scene
from app.services.image_providers import (
    OnnxDirectMLImageProvider,
    PlaceholderImageProvider,
    get_image_provider,
    list_image_providers,
)
from app.services.storyboard import StoryboardService


MODEL_FILES = OnnxDirectMLImageProvider.required_model_files


def create_model_files(directory: Path) -> None:
    for relative in MODEL_FILES:
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test file")


def make_provider(model_dir: Path, output_dir: Path) -> OnnxDirectMLImageProvider:
    return OnnxDirectMLImageProvider(
        model_dir=model_dir,
        output_dir=output_dir,
        width=512,
        height=512,
        seed=42,
    )


def make_scene() -> Scene:
    return Scene(
        scene_number=1,
        heading="INT. ARCHIVE - NIGHT",
        location="Archive",
        mood="Quiet suspense",
        lighting="Low amber light",
        action_description="An archivist opens a sealed case.",
        visual_prompt="Stored prompt",
        duration_seconds=7,
    )


def fake_core_result(prompt, seed, sessions):
    assert prompt
    assert len(sessions) == 3
    return np.zeros((512, 512, 3), dtype=np.uint8), {
        "text_encoder_seconds": 0.1,
        "unet_seconds": 0.2,
        "vae_seconds": 0.3,
    }


def test_provider_registration_and_configuration_selection(monkeypatch):
    assert settings.IMAGE_PROVIDER == "placeholder"
    assert isinstance(get_image_provider(), PlaceholderImageProvider)
    assert isinstance(get_image_provider("onnx_directml"), OnnxDirectMLImageProvider)
    assert "placeholder" in list_image_providers()
    assert "onnx_directml" in list_image_providers()

    monkeypatch.setattr(settings, "IMAGE_PROVIDER", "onnx_directml")
    assert isinstance(get_image_provider(), OnnxDirectMLImageProvider)
    assert isinstance(
        StoryboardService().image_provider,
        OnnxDirectMLImageProvider,
    )


def test_provider_initialization_uses_image_configuration(monkeypatch, tmp_path):
    model_dir = tmp_path / "configured-model"
    output_dir = tmp_path / "configured-output"
    monkeypatch.setattr(settings, "IMAGE_MODEL_DIR", model_dir)
    monkeypatch.setattr(settings, "IMAGE_OUTPUT_DIR", output_dir)
    monkeypatch.setattr(settings, "IMAGE_WIDTH", 512)
    monkeypatch.setattr(settings, "IMAGE_HEIGHT", 512)
    monkeypatch.setattr(settings, "IMAGE_SEED", 73)

    provider = OnnxDirectMLImageProvider()

    assert provider.model_dir == model_dir.resolve()
    assert provider.output_dir == output_dir.resolve()
    assert provider.width == 512
    assert provider.height == 512
    assert provider.seed == 73


def test_relative_model_path_resolves_from_project_root():
    provider = OnnxDirectMLImageProvider(
        model_dir=Path("models/image/sd15-onnx"),
        width=512,
        height=512,
    )

    assert provider.model_dir == (BASE_DIR / "models" / "image" / "sd15-onnx").resolve()


@pytest.mark.anyio
async def test_missing_model_files_return_structured_failure(tmp_path):
    provider = make_provider(tmp_path / "missing", tmp_path / "output")

    assert provider.is_available() is False
    result = await provider.generate_image("A test prompt", {"scene_number": 1})

    assert result.provider_name == "onnx_directml"
    assert result.status == "failed"
    assert result.prompt_used == "A test prompt"
    assert "Required ONNX model files are missing" in result.error


@pytest.mark.anyio
async def test_success_returns_result_and_releases_sessions(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    create_model_files(model_dir)
    provider = make_provider(model_dir, tmp_path / "output")
    monkeypatch.setattr(provider, "_runtime_installed", lambda: True)
    monkeypatch.setattr(
        provider,
        "_load_runtime",
        lambda: type("Runtime", (), {"get_available_providers": lambda self: ["DmlExecutionProvider"]})(),
    )
    created_sessions = []

    def create_sessions(_runtime):
        sessions = [object(), object(), object()]
        created_sessions.append(sessions)
        return sessions

    monkeypatch.setattr(provider, "_create_sessions", create_sessions)
    monkeypatch.setattr(provider, "_run_core", fake_core_result)
    collected = []
    monkeypatch.setattr(
        "app.services.image_providers.onnx_directml.gc.collect",
        lambda: collected.append(True),
    )

    result = await provider.generate_image(
        "A test prompt",
        {"scene_number": 2},
    )

    assert result.provider_name == "onnx_directml"
    assert result.status == "generated"
    assert result.image_path
    assert result.image_url.startswith("/generated/images/")
    assert Path(result.image_path).is_file()
    assert result.metadata["width"] == 512
    assert result.metadata["height"] == 512
    assert result.metadata["seed"] == 42
    assert result.metadata["unet_seconds"] == 0.2
    assert created_sessions == [[]]
    assert collected


@pytest.mark.anyio
async def test_failure_returns_error_and_releases_sessions(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    create_model_files(model_dir)
    provider = make_provider(model_dir, tmp_path / "output")
    monkeypatch.setattr(provider, "_runtime_installed", lambda: True)
    monkeypatch.setattr(
        provider,
        "_load_runtime",
        lambda: type("Runtime", (), {"get_available_providers": lambda self: ["DmlExecutionProvider"]})(),
    )
    created_sessions = []

    def create_sessions(_runtime):
        sessions = [object(), object(), object()]
        created_sessions.append(sessions)
        return sessions

    def fail_core(_prompt, _seed, _sessions):
        raise RuntimeError("simulated DirectML failure")

    monkeypatch.setattr(provider, "_create_sessions", create_sessions)
    monkeypatch.setattr(provider, "_run_core", fail_core)
    collected = []
    monkeypatch.setattr(
        "app.services.image_providers.onnx_directml.gc.collect",
        lambda: collected.append(True),
    )

    result = await provider.generate_image("A test prompt", {"scene_number": 3})

    assert result.status == "failed"
    assert "simulated DirectML failure" in result.error
    assert created_sessions == [[]]
    assert collected


@pytest.mark.anyio
async def test_storyboard_keeps_provider_failure_per_frame(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    create_model_files(model_dir)
    provider = make_provider(model_dir, tmp_path / "output")
    monkeypatch.setattr(provider, "_runtime_installed", lambda: True)
    monkeypatch.setattr(
        provider,
        "_load_runtime",
        lambda: type("Runtime", (), {"get_available_providers": lambda self: ["DmlExecutionProvider"]})(),
    )
    monkeypatch.setattr(
        provider,
        "_create_sessions",
        lambda _runtime: [object(), object(), object()],
    )

    def fail_core(_prompt, _seed, _sessions):
        raise RuntimeError("simulated inference failure")

    monkeypatch.setattr(provider, "_run_core", fail_core)
    service = StoryboardService(image_provider=provider)

    rendered = await service.render_frames(
        service.build_storyboard([make_scene()])
    )

    assert rendered[0].status == "failed"
    assert rendered[0].provider_name == "onnx_directml"
    assert "simulated inference failure" in rendered[0].generation_error


@pytest.mark.anyio
async def test_unsupported_model_dimensions_fail_explicitly(tmp_path):
    model_dir = tmp_path / "model"
    create_model_files(model_dir)
    provider = OnnxDirectMLImageProvider(
        model_dir=model_dir,
        output_dir=tmp_path / "output",
        width=1280,
        height=720,
    )

    result = await provider.generate_image("Prompt", {})

    assert result.status == "failed"
    assert "supports 512x512 generation only" in result.error
