from typing import Dict, List, Any, Optional
from ..models import (
    Character,
    Scene,
    DialogueLine,
    ScreenplayDocument,
    ScreenplayScene,
)


class ScreenplayGenerationError(ValueError):
    """Raised when story or scene data cannot be converted into a screenplay."""

    pass


class ScreenplayService:
    """Convert structured story data into deterministic screenplay output."""

    @staticmethod
    def _required_text(value: Any, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ScreenplayGenerationError(
                f"Missing required story field: {field_name}"
            )
        return value.strip()

    @staticmethod
    def format_heading(scene: Scene) -> str:
        """Normalize a scene heading into a consistent slugline."""
        heading = scene.heading.strip() if scene.heading else ""
        if heading:
            upper = heading.upper()
            if upper.startswith(("INT.", "EXT.", "INT./EXT.", "I/E.")):
                return upper

        location = (scene.location or "UNSPECIFIED LOCATION").strip().upper()
        time = (scene.time_of_day or "DAY").strip().upper()
        return f"{location} - {time}" if not heading else f"{heading.upper()} - {time}"

    @staticmethod
    def format_dialogue(dialogue: DialogueLine) -> List[str]:
        """Format one structured dialogue line."""
        character_name = ScreenplayService._required_text(
            dialogue.character_name, "dialogue.character_name"
        )
        text = ScreenplayService._required_text(
            dialogue.text, "dialogue.text"
        )
        return [
            " " * 24 + character_name.upper(),
            " " * 30 + text,
            "",
        ]

    @staticmethod
    def format_scene_to_script(scene: Scene) -> str:
        """Convert a structured scene into screenplay text."""
        if not scene.heading and not scene.location:
            raise ScreenplayGenerationError(
                "Scene is missing both a heading and a location."
            )
        if not scene.action_description.strip():
            raise ScreenplayGenerationError(
                f"Scene {scene.scene_number} is missing an action description."
            )

        lines = [
            f"SCENE {scene.scene_number}",
            ScreenplayService.format_heading(scene),
            "",
            scene.action_description,
            "",
        ]

        if scene.camera_movement:
            lines.append(f"(Camera: {scene.camera_movement.strip()})")
            lines.append("")

        for dialogue in scene.dialogues:
            lines.extend(ScreenplayService.format_dialogue(dialogue))

        return "\n".join(lines)

    @classmethod
    def generate_from_story(
        cls,
        story: Any,
        scenes: Optional[List[Scene]] = None,
        characters: Optional[List[Character]] = None,
    ) -> ScreenplayDocument:
        """Generate a structured screenplay from a StoryAnalysisResponse-like object."""
        if isinstance(story, dict):
            title = story.get("title")
            genre = story.get("genre")
            logline = story.get("logline")
            theme = story.get("theme")
            tone = story.get("tone")
            provider = story.get("provider")
            raw_scenes = scenes if scenes is not None else story.get("scenes", [])
            raw_characters = characters if characters is not None else story.get("characters", [])
        else:
            title = getattr(story, "title", None)
            genre = getattr(story, "genre", None)
            logline = getattr(story, "logline", None)
            theme = getattr(story, "theme", None)
            tone = getattr(story, "tone", None)
            provider = getattr(story, "provider", None)
            raw_scenes = scenes if scenes is not None else getattr(story, "scenes", [])
            raw_characters = characters if characters is not None else getattr(story, "characters", [])

        title = cls._required_text(title, "title")
        genre = cls._required_text(genre or "Cinematic Drama", "genre")
        logline = cls._required_text(logline, "logline")

        if not isinstance(raw_scenes, list) or not raw_scenes:
            raise ScreenplayGenerationError("The story has no scenes.")

        structured_scenes: List[ScreenplayScene] = []
        ordered_scenes = sorted(raw_scenes, key=lambda s: getattr(s, "scene_number", 0) if not isinstance(s, dict) else s.get("scene_number", 0))
        for index, scene in enumerate(ordered_scenes):
            if isinstance(scene, dict):
                try:
                    scene = Scene.model_validate(scene)
                except Exception as exc:
                    raise ScreenplayGenerationError(
                        f"Malformed scene {scene.get('scene_number', index + 1)}."
                    ) from exc

            heading = cls.format_heading(scene)
            transition = "FADE OUT." if index == len(ordered_scenes) - 1 else "CUT TO:"
            structured_scenes.append(
                ScreenplayScene(
                    scene_number=scene.scene_number,
                    heading=heading,
                    location=scene.location,
                    time_of_day=scene.time_of_day,
                    mood=scene.mood,
                    lighting=scene.lighting,
                    weather=scene.weather,
                    shot_type=scene.shot_type,
                    action_description=cls._required_text(
                        scene.action_description,
                        f"scene {scene.scene_number} action_description",
                    ),
                    visual_prompt=scene.visual_prompt,
                    camera_movement=scene.camera_movement,
                    duration_seconds=scene.duration_seconds,
                    dialogues=scene.dialogues,
                    transition=transition,
                )
            )

        if isinstance(raw_characters, list):
            character_models = [
                item if isinstance(item, Character) else Character.model_validate(item)
                for item in raw_characters
            ]
        else:
            character_models = []

        formatted_text = cls.generate_full_screenplay(
            title=title,
            genre=genre,
            logline=logline,
            scenes=[Scene(
                scene_number=s.scene_number,
                heading=s.heading,
                location=s.location,
                time_of_day=s.time_of_day,
                mood=s.mood,
                lighting=s.lighting,
                weather=s.weather,
                shot_type=s.shot_type,
                action_description=s.action_description,
                visual_prompt=s.visual_prompt or "",
                camera_movement=s.camera_movement,
                duration_seconds=s.duration_seconds,
                dialogues=s.dialogues,
            ) for s in structured_scenes],
            theme=theme,
            tone=tone,
        )

        return ScreenplayDocument(
            title=title,
            genre=genre,
            logline=logline,
            theme=theme,
            tone=tone,
            provider=provider,
            characters=character_models,
            scenes=structured_scenes,
            formatted_text=formatted_text,
        )

    @classmethod
    def generate_full_screenplay(
        cls,
        title: str,
        scenes: List[Scene],
        genre: Optional[str] = None,
        logline: Optional[str] = None,
        theme: Optional[str] = None,
        tone: Optional[str] = None,
    ) -> str:
        """Generate the complete formatted screenplay document."""
        header = [
            "FADE IN:",
            "",
            title.upper(),
            f"Genre: {genre}" if genre else "",
            f"Logline: {logline}" if logline else "",
            f"Theme: {theme}" if theme else "",
            f"Tone: {tone}" if tone else "",
            "",
        ]

        formatted_scenes = [cls.format_scene_to_script(s) for s in scenes]
        for index, scene in enumerate(scenes):
            if index < len(scenes) - 1:
                formatted_scenes[index] = formatted_scenes[index].rstrip() + "\n\nCUT TO:"
            else:
                formatted_scenes[index] = formatted_scenes[index].rstrip() + "\n\nFADE OUT."

        return "\n".join(line for line in header if line != "") + "\n\n" + "\n\n".join(formatted_scenes)
