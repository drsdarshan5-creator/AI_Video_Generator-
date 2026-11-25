import pytest
from httpx import ASGITransport, AsyncClient

from app.database import init_db
from app.main import app
from app.models import (
    Character,
    DialogueLine,
    Scene,
)
from app.services.screenplay import (
    ScreenplayGenerationError,
    ScreenplayService,
)


def make_story():
    characters = [
        Character(
            id="char-1",
            name="Mara",
            role="Protagonist",
            description="A determined investigator.",
            visual_prompt="Cinematic portrait of Mara.",
        ),
        Character(
            id="char-2",
            name="Jon",
            role="Mentor",
            description="A cautious field scientist.",
            visual_prompt="Cinematic portrait of Jon.",
        ),
    ]
    scenes = [
        Scene(
            scene_number=1,
            heading="EXT. FLOODED STREET - NIGHT",
            location="Flooded Street",
            time_of_day="NIGHT",
            mood="Tense",
            lighting="Cold blue moonlight",
            action_description="Mara enters the abandoned district.",
            visual_prompt="Flooded cyberpunk street.",
            camera_movement="Slow tracking shot",
            duration_seconds=6,
            dialogues=[
                DialogueLine(character_name="Mara", text="The signal is coming from below."),
            ],
        ),
        Scene(
            scene_number=2,
            heading="INT. RESEARCH VAN - CONTINUOUS",
            location="Research Van",
            time_of_day="CONTINUOUS",
            mood="Urgent",
            lighting="Amber instrument glow",
            action_description="Jon analyzes the recovered signal.",
            visual_prompt="Research van interior.",
            camera_movement="Pan between the characters",
            duration_seconds=8,
            dialogues=[
                DialogueLine(character_name="Jon", text="We have less time than we thought."),
                DialogueLine(character_name="Mara", text="Then we move now."),
            ],
        ),
    ]
    return {
        "title": "Signal in the Flood",
        "genre": "Cyberpunk Mystery",
        "logline": "Mara follows a signal beneath a drowned city.",
        "theme": "Truth and survival",
        "tone": "Atmospheric",
        "provider": "rule_based",
        "characters": characters,
        "scenes": scenes,
    }


def test_valid_story_generates_structured_screenplay():
    document = ScreenplayService.generate_from_story(make_story())

    assert document.title == "Signal in the Flood"
    assert document.genre == "Cyberpunk Mystery"
    assert len(document.characters) == 2
    assert len(document.scenes) == 2
    assert document.scenes[0].heading == "EXT. FLOODED STREET - NIGHT"
    assert document.scenes[0].dialogues[0].character_name == "Mara"


def test_screenplay_formatted_text_contains_dialogue_and_transitions():
    document = ScreenplayService.generate_from_story(make_story())

    assert "FADE IN:" in document.formatted_text
    assert "EXT. FLOODED STREET - NIGHT" in document.formatted_text
    assert "MARA" in document.formatted_text
    assert "The signal is coming from below." in document.formatted_text
    assert "CUT TO:" in document.formatted_text
    assert "FADE OUT." in document.formatted_text


def test_missing_required_story_data_is_rejected():
    with pytest.raises(ScreenplayGenerationError, match="title"):
        ScreenplayService.generate_from_story({
            "genre": "Drama",
            "logline": "A valid logline.",
            "characters": [],
            "scenes": [],
        })


def test_missing_scenes_is_rejected():
    with pytest.raises(ScreenplayGenerationError, match="no scenes"):
        ScreenplayService.generate_from_story({
            "title": "No Scenes",
            "genre": "Drama",
            "logline": "A valid logline.",
            "characters": [],
            "scenes": [],
        })


def test_malformed_scene_is_rejected():
    with pytest.raises(ScreenplayGenerationError, match="Malformed scene"):
        ScreenplayService.generate_from_story({
            "title": "Malformed",
            "genre": "Drama",
            "logline": "A valid logline.",
            "characters": [],
            "scenes": [{"scene_number": 1}],
        })


@pytest.mark.anyio
async def test_api_generates_screenplay_for_analyzed_story():
    await init_db()
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        analyze_response = await client.post(
            "/api/story/analyze",
            json={
                "prompt": "A marine architect searches a flooded city for a lost research team.",
                "genre": "Cyberpunk Mystery",
                "provider": "rule_based",
            },
        )
        assert analyze_response.status_code == 201
        story_id = analyze_response.json()["story_id"]

        screenplay_response = await client.get(
            f"/api/scenes/screenplay/export?story_id={story_id}"
        )
        assert screenplay_response.status_code == 200
        data = screenplay_response.json()
        assert data["title"]
        assert data["logline"]
        assert len(data["scenes"]) >= 1
        assert data["formatted_text"]
        assert data["screenplay"] == data["formatted_text"]


@pytest.mark.anyio
async def test_api_screenplay_rejects_unknown_story():
    await init_db()
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/api/scenes/screenplay/export?story_id=does-not-exist"
        )
        assert response.status_code == 404
        assert "was not found" in response.json()["detail"]
