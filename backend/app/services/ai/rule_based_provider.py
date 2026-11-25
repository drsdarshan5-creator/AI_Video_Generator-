import re
import uuid
from typing import Dict, Any, List
from .base import BaseStoryLLMProvider
from ..cinematic_intelligence import (
    build_character_visual_prompt,
    build_scene_visual_prompt,
)


class RuleBasedStoryProvider(BaseStoryLLMProvider):
    """
    Intelligent rule-based narrative decomposition provider.
    Operates without external LLM dependencies, providing instant,
    reproducible cinematic story structures, consistent characters,
    and formatted screenplay scenes.
    """

    provider_name: str = "rule_based"

    def is_available(self) -> bool:
        """Rule-based provider is always available without external services."""
        return True

    async def analyze(
        self,
        prompt: str,
        genre: str = "Cinematic Drama",
        target_duration_minutes: int = 1,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Decompose prompt into cinematic story elements using structured heuristics."""
        clean_prompt = prompt.strip()
        title = self._derive_title(clean_prompt)
        logline = self._derive_logline(clean_prompt)
        theme = self._derive_theme(clean_prompt, genre)
        tone = self._derive_tone(genre)
        synopsis = self._derive_synopsis(clean_prompt, logline, theme)
        characters = self._extract_characters(clean_prompt, genre)
        scenes = self._generate_scenes(clean_prompt, characters, target_duration_minutes, genre)

        return {
            "title": title,
            "logline": logline,
            "synopsis": synopsis,
            "theme": theme,
            "tone": tone,
            "genre": genre,
            "characters": characters,
            "scenes": scenes,
            "provider": self.provider_name,
        }

    def extract_characters(self, prompt: str, genre: str) -> List[Dict[str, Any]]:
        return self._extract_characters(prompt, genre)

    def generate_scenes(
        self,
        prompt: str,
        characters: List[Dict[str, Any]],
        duration_minutes: int,
        genre: str,
    ) -> List[Dict[str, Any]]:
        return self._generate_scenes(prompt, characters, duration_minutes, genre)

    def get_structured_output_schema(self) -> Dict[str, Any]:
        return {
            "title": "string",
            "logline": "string",
            "synopsis": "string",
            "theme": "string",
            "tone": "string",
            "genre": "string",
            "characters": "list",
            "scenes": "list",
            "provider": "string",
        }

    def _derive_title(self, prompt: str) -> str:
        """Derive a cinematic title from the story premise."""
        words = prompt.split()
        if len(words) <= 5:
            return prompt.title()

        first_sentence = re.split(r"[.!?]", prompt)[0]
        stopwords = {
            "the", "and", "in", "a", "an", "of", "to", "at", "by", "for", "with",
            "about", "against", "between", "into", "through", "during", "before",
            "after", "above", "below", "from", "up", "down", "is", "are", "was", "were"
        }
        tokens = [w.strip(",.-;:\"'()") for w in first_sentence.split()]
        filtered = [t for t in tokens if t.lower() not in stopwords and len(t) > 2]

        if len(filtered) >= 2:
            return f"{filtered[0].title()} {filtered[1].title()}"
        elif filtered:
            return f"The {filtered[0].title()}"
        return "Echoes of Destiny"

    def _derive_logline(self, prompt: str) -> str:
        """Derive a one-sentence dramatic logline."""
        sentences = [s.strip() for s in re.split(r"[.!?]", prompt) if s.strip()]
        if sentences:
            logline = sentences[0]
            if not logline.endswith("."):
                logline += "."
            return logline
        return "A relentless protagonist embarks on an odyssey that challenges the boundaries of reality."

    def _derive_theme(self, prompt: str, genre: str) -> str:
        """Infer core thematic focus based on genre and keywords."""
        lower_prompt = prompt.lower()
        if "cyber" in lower_prompt or "tech" in lower_prompt or "neural" in lower_prompt or "sci-fi" in genre.lower():
            return "Human identity, synthetic consciousness, and the consequences of technological transcendence"
        elif "monolith" in lower_prompt or "ancient" in lower_prompt or "mystery" in lower_prompt:
            return "The pursuit of forgotten truth and human insignificance before the cosmos"
        elif "detective" in lower_prompt or "investigat" in lower_prompt or "crime" in lower_prompt:
            return "Obsession with justice and moral ambiguity in a fractured society"
        return "The courage to confront the unknown and reconcile with one's destiny"

    def _derive_tone(self, genre: str) -> str:
        """Derive cinematic tone from genre."""
        genre_lower = genre.lower()
        if "cyber" in genre_lower or "noir" in genre_lower:
            return "Atmospheric Neo-Noir, drenched in neon rain with palpable suspense"
        elif "sci-fi" in genre_lower:
            return "Cerebral, visionary, and visually breathtaking with philosophical gravitas"
        elif "horror" in genre_lower or "thriller" in genre_lower:
            return "Tense, psychological, claustrophobic dread with sudden revelations"
        elif "action" in genre_lower:
            return "Kinetic, high-velocity adrenaline with sharp character stakes"
        return "Cinematic, richly textured, emotionally grounded drama"

    def _derive_synopsis(self, prompt: str, logline: str, theme: str) -> str:
        """Build a coherent three-act synopsis."""
        return (
            f"ACT I (Setup): {logline} As ordinary routines dissolve, the inciting event triggers an irreversible "
            f"chain of discovery.\n\n"
            f"ACT II (Confrontation): Venturing deeper into uncharted territory, hidden adversaries and unexpected "
            f"allies emerge. Every choice forces a reckoning with the central truth: {theme}.\n\n"
            f"ACT III (Resolution): Standing at the precipice of revelation, the protagonist makes a decisive stand, "
            f"forever altering the landscape and leaving an indelible mark upon the world."
        )

    def _extract_characters(self, prompt: str, genre: str) -> List[Dict[str, Any]]:
        """Extract or infer protagonist, mentor/ally, and antagonist."""
        characters = [
            {
                "id": "char-1",
                "name": "Alex Vance",
                "role": "Protagonist",
                "description": "A seasoned, driven investigator driven by an unyielding search for truth.",
                "personality": "Observant, cautious, resolute under extreme pressure, possessing dry humor.",
                "appearance_description": "sharp weathered facial features",
                "age_range": "mid-30s",
                "clothing_style": "tactical trenchcoat with subtle tech accents",
                "voice_name": "Adam (Warm & Resolute)",
                "voice_id": "voice-adam-01",
            },
            {
                "id": "char-2",
                "name": "Dr. Elena Ramos",
                "role": "Mentor / Scientific Ally",
                "description": "An astute quantum researcher whose theories decipher anomalies others deem impossible.",
                "personality": "Analytical, empathetic, intensely focused, skeptical of dogma.",
                "appearance_description": "intelligent expressive eyes",
                "age_range": "late-40s",
                "clothing_style": "modern minimalist research coat",
                "voice_name": "Rachel (Calm & Intellectual)",
                "voice_id": "voice-rachel-02",
            },
        ]
        for character in characters:
            character["visual_prompt"] = build_character_visual_prompt(character)
        return characters

    def _generate_scenes(
        self,
        prompt: str,
        characters: List[Dict[str, Any]],
        duration_minutes: int,
        genre: str,
    ) -> List[Dict[str, Any]]:
        """Generate structured film scenes complete with sluglines, camera movements, and dialogues."""
        lead_char = characters[0]["name"] if characters else "Protagonist"
        second_char = characters[1]["name"] if len(characters) > 1 else "Advisor"

        scenes = [
            {
                "scene_number": 1,
                "heading": "EXT. CITY HORIZON - DAWN",
                "location": "City Skyline",
                "time_of_day": "DAWN",
                "mood": "Mysterious & Contemplative",
                "lighting": "Volumetric morning haze with sharp golden highlights catching reflective towers",
                "action_description": (
                    f"Thick fog drifts between towering metallic spires. {lead_char} stands upon an observation "
                    f"ledge, examining a handheld scanner that pulses with a rhythmic, encrypted frequency."
                ),
                "visual_prompt": (
                    "Cinematic drone shot of futuristic neo-city skyline at sunrise, volumetric morning haze, "
                    "raytracing reflections on glass spires, anamorphic lens, ultra-realistic 8k masterpiece"
                ),
                "camera_movement": "Slow forward tracking shot",
                "duration_seconds": 6,
                "dialogues": [
                    {
                        "character_name": lead_char,
                        "text": "Every signal has an origin. We just have to follow it to where it leads.",
                    }
                ],
            },
            {
                "scene_number": 2,
                "heading": "INT. RESEARCH FACILITY - CONTINUOUS",
                "location": "Subterranean Laboratory",
                "time_of_day": "CONTINUOUS",
                "mood": "Tense & Technical",
                "lighting": "Cool cyan and cobalt blue holographic displays illuminating dark brushed-steel workstations",
                "action_description": (
                    f"{second_char} isolates the incoming signal on a spherical holographic console. The waveform "
                    f"shifts, echoing harmonic patterns that defy conventional physics."
                ),
                "visual_prompt": (
                    "Atmospheric sci-fi laboratory interior with floating translucent cyan data streams, cinematic "
                    "depth of field, moody shadows, anamorphic horizontal lens flare, photorealistic render"
                ),
                "camera_movement": "Pan left to medium two-shot",
                "duration_seconds": 7,
                "dialogues": [
                    {
                        "character_name": second_char,
                        "text": f"Be careful, {lead_char.split()[0]}. Once this frequency opens, there is no shutting the gate.",
                    },
                    {
                        "character_name": lead_char,
                        "text": "Then let's find out what's waiting on the other side.",
                    },
                ],
            },
            {
                "scene_number": 3,
                "heading": "EXT. MONOLITH SITE - DUSK",
                "location": "Ancient Desert Valley",
                "time_of_day": "DUSK",
                "mood": "Awe-inspiring & Climactic",
                "lighting": "Deep amber sunset casting elongated shadows, accented by pulsating crystalline glow",
                "action_description": (
                    f"{lead_char} arrives before the colossal obsidian monolith. As their hand nears the surface, "
                    f"the structure resonates with deep infrasound, bathing the valley in golden luminescence."
                ),
                "visual_prompt": (
                    "Epic wide shot of a pulsing ancient obsidian monolith in an open desert canyon during golden hour, "
                    "majestic scale, shimmering energy rings, cinematic masterpiece, 70mm IMAX composition"
                ),
                "camera_movement": "High crane tilt down and slow push-in",
                "duration_seconds": 8,
                "dialogues": [
                    {
                        "character_name": lead_char,
                        "text": "It isn't a weapon after all... it's an invitation.",
                    }
                ],
            },
        ]
        weather_by_scene = ["Light mist", "No precipitation", "Dry wind"]
        shot_type_by_scene = ["Wide establishing shot", "Medium two-shot", "Slow push-in"]
        for index, scene in enumerate(scenes):
            scene["weather"] = weather_by_scene[index]
            scene["shot_type"] = shot_type_by_scene[index]
            scene["visual_prompt"] = build_scene_visual_prompt(scene, characters)
        return scenes
