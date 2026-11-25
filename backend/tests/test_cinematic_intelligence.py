import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from app.database import init_db
from app.main import app
from app.models import Character, DialogueLine, Scene
from app.services.cinematic_intelligence import (
    CinematicMetadataError,
    build_character_visual_prompt,
    build_scene_visual_prompt,
)


def make_character(name="Mara"):
    return Character(
        id=f"{name.lower()}-1",
        name=name,
        role="Protagonist",
        description="A determined investigator.",
        personality="Observant and resolute",
        appearance_description="sharp weathered features",
        age_range="mid-30s",
        clothing_style="dark tactical coat",
        visual_prompt="placeholder",
    )


def make_scene(dialogue_names=None, multiple_characters=False):
    dialogues = []
    action = "Mara enters the station."
    if dialogue_names:
        dialogues = [
            DialogueLine(character_name=name, text=f"{name} speaks.")
            for name in dialogue_names
        ]
    if multiple_characters:
        action = "Mara and Jon study the signal together."
    return Scene(
        scene_number=1,
        heading="INT. RESEARCH STATION - NIGHT",
        location="Research Station",
        time_of_day="NIGHT",
        mood="Tense",
        lighting="Cold blue practical lighting",
        weather="Light rain",
        shot_type="Medium two-shot",
        action_description=action,
        visual_prompt="placeholder",
        camera_movement="Slow push-in",
        duration_seconds=8,
        dialogues=dialogues,
    )


def test_character_visual_prompt_is_deterministic_and_reusable():
    character = make_character()
    first = build_character_visual_prompt(character)
    second = build_character_visual_prompt(character)

    assert first == second
    assert "cinematic character portrait" in first
    assert "sharp weathered features" in first
    assert "dark tactical coat" in first
    assert "consistent character design" in first


def test_character_visual_prompt_rejects_missing_name():
    with pytest.raises(ValidationError, match="name"):
        build_character_visual_prompt(Character(
            id="char-1",
            name=" ",
            description="Unknown",
            visual_prompt="placeholder",
        ))


def test_scene_visual_prompt_includes_location_metadata_and_character():
    mara = make_character("Mara")
    scene = make_scene(dialogue_names=["Mara"])
    prompt = build_scene_visual_prompt(scene, [mara])

    assert "Research Station" in prompt
    assert "NIGHT" in prompt
    assert "Light rain" in prompt
    assert "Mara: sharp weathered features" in prompt
    assert "Medium two-shot" in prompt
    assert "Slow push-in" in prompt


def test_character_consistency_across_multiple_scene_prompts():
    mara = make_character("Mara")
    mara.visual_prompt = build_character_visual_prompt(mara)

    scene_one = make_scene(dialogue_names=["Mara"])
    scene_two = make_scene(multiple_characters=True)

    prompt_one = build_scene_visual_prompt(scene_one, [mara])
    prompt_two = build_scene_visual_prompt(scene_two, [mara])

    assert "Mara: sharp weathered features" in prompt_one
    assert "Mara: sharp weathered features" in prompt_two


def test_scene_prompt_supports_multiple_referenced_characters():
    mara = make_character("Mara")
    jon = make_character("Jon")
    jon.appearance_description = "calm analytical eyes"
    jon.clothing_style = "lab jacket"
    scene = make_scene(multiple_characters=True)

    prompt = build_scene_visual_prompt(scene, [mara, jon])

    assert "Mara: sharp weathered features" in prompt
    assert "Jon: calm analytical eyes" in prompt


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("name", "", "at least"),
        ("visual_prompt", "", "at least"),
    ],
)
def test_invalid_character_data_is_rejected(field, value, error):
    data = {
        "id": "char-1",
        "name": "Mara",
        "description": "A character.",
        "visual_prompt": "Valid prompt",
    }
    data[field] = value
    with pytest.raises(ValidationError, match=error):
        Character.model_validate(data)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("heading", ""),
        ("duration_seconds", 0),
        ("duration_seconds", 61),
    ],
)
def test_invalid_scene_data_is_rejected(field, value):
    data = {
        "scene_number": 1,
        "heading": "INT. ROOM - NIGHT",
        "action_description": "The team enters.",
        "visual_prompt": "Cinematic room.",
        "duration_seconds": 5,
    }
    data[field] = value
    with pytest.raises(ValidationError):
        Scene.model_validate(data)


def test_malformed_dialogue_is_rejected():
    with pytest.raises(ValidationError):
        DialogueLine.model_validate({"character_name": "", "text": ""})


@pytest.mark.anyio
async def test_character_and_scene_retrieval_api():
    await init_db()
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        character_response = await client.post(
            "/api/characters/",
            json={
                "id": "char-api-1",
                "name": "Mara",
                "role": "Protagonist",
                "description": "A determined investigator.",
                "personality": "Resolute",
                "appearance_description": "sharp weathered features",
                "age_range": "mid-30s",
                "clothing_style": "dark tactical coat",
                "visual_prompt": "placeholder",
                "voice_name": "Test Voice",
            },
        )
        assert character_response.status_code == 200
        assert "cinematic character portrait" in character_response.json()["visual_prompt"]

        character_get = await client.get("/api/characters/char-api-1")
        assert character_get.status_code == 200
        assert character_get.json()["name"] == "Mara"
        assert character_get.json()["age_range"] == "mid-30s"

        scene_response = await client.post(
            "/api/scenes/",
            json={
                "story_id": "story-api-1",
                "scene_number": 1,
                "heading": "INT. LAB - NIGHT",
                "location": "Laboratory",
                "time_of_day": "NIGHT",
                "mood": "Tense",
                "lighting": "Cold blue",
                "weather": "Indoor",
                "shot_type": "Medium shot",
                "action_description": "Mara reviews the signal.",
                "visual_prompt": "placeholder",
                "camera_movement": "Slow push-in",
                "duration_seconds": 8,
                "dialogues": [
                    {"character_name": "Mara", "text": "There it is."}
                ],
            },
        )
        assert scene_response.status_code == 200
        assert "Mara" in scene_response.json()["visual_prompt"]

        scene_get = await client.get(
            "/api/scenes/1?story_id=story-api-1"
        )
        assert scene_get.status_code == 200
        assert scene_get.json()["weather"] == "Indoor"
        assert scene_get.json()["shot_type"] == "Medium shot"
