import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw

from ...config import settings
from .base import BaseImageProvider, ImageGenerationResult

logger = logging.getLogger(__name__)


class PlaceholderImageProvider(BaseImageProvider):
    """Local deterministic storyboard-card renderer used as the default provider."""

    provider_name = "placeholder"

    def __init__(self) -> None:
        self.output_dir: Path = settings.IMAGE_OUTPUT_DIR
        self.width = settings.IMAGE_WIDTH
        self.height = settings.IMAGE_HEIGHT

    def is_available(self) -> bool:
        return True

    @staticmethod
    def _wrap_text(text: str, max_length: int = 90) -> List[str]:
        lines = []
        current = ""
        for word in text.split():
            candidate = f"{current} {word}".strip()
            if len(candidate) > max_length and current:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
        return lines

    async def generate_image(
        self,
        prompt: str,
        metadata: Dict[str, Any],
    ) -> ImageGenerationResult:
        scene_number = metadata.get("scene_number")
        if not isinstance(scene_number, int) or scene_number < 1:
            return ImageGenerationResult(
                provider_name=self.provider_name,
                status="failed",
                prompt_used=prompt,
                metadata=metadata,
                error="A positive scene_number is required to render a placeholder.",
            )

        file_name = f"scene_{scene_number}_{uuid.uuid4().hex[:6]}.png"
        file_path = self.output_dir / file_name
        width = metadata.get("width", self.width)
        height = metadata.get("height", self.height)

        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            image = Image.new("RGB", (width, height), color=(18, 22, 31))
            draw = ImageDraw.Draw(image)
            draw.rectangle(
                [20, 20, width - 20, height - 20],
                outline=(56, 189, 248),
                width=3,
            )
            draw.rectangle(
                [28, 28, width - 28, height - 28],
                outline=(30, 41, 59),
                width=2,
            )

            draw.text(
                (60, 60),
                f"STORYBOARD FRAME - SCENE #{scene_number:02d}",
                fill=(248, 250, 252),
            )
            draw.text(
                (60, 110),
                str(metadata.get("scene_heading") or "SCENE HEADING UNAVAILABLE")[:100],
                fill=(148, 163, 184),
            )
            draw.text(
                (60, 155),
                f"SHOT: {metadata.get('shot_type') or 'UNSPECIFIED'}",
                fill=(148, 163, 184),
            )
            duration = metadata.get("duration_seconds")
            if duration is not None:
                draw.text((60, 195), f"DURATION: {duration}s", fill=(148, 163, 184))

            names = ", ".join(
                str(name) for name in metadata.get("character_names", []) if name
            )
            if names:
                draw.text((60, 230), f"CHARACTERS: {names[:100]}", fill=(148, 163, 184))

            action = str(metadata.get("action_description") or prompt)
            for index, line in enumerate(self._wrap_text(action)[:5]):
                draw.text(
                    (60, 275 + index * 36),
                    line,
                    fill=(203, 213, 225),
                )
            draw.text(
                (60, height - 80),
                "AI VIDEO GENERATOR - PREVISUALIZATION",
                fill=(100, 116, 139),
            )
            image.save(file_path, format="PNG")

            result_metadata = dict(metadata)
            result_metadata.update({"width": width, "height": height})
            return ImageGenerationResult(
                image_path=str(file_path),
                image_url=f"/generated/images/{file_name}",
                provider_name=self.provider_name,
                status="placeholder",
                prompt_used=prompt,
                metadata=result_metadata,
            )
        except Exception as exc:
            logger.exception("Placeholder image generation failed for scene %s", scene_number)
            return ImageGenerationResult(
                provider_name=self.provider_name,
                status="failed",
                prompt_used=prompt,
                metadata=metadata,
                error=f"Placeholder image generation failed: {exc}",
            )
