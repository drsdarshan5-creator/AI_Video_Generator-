import pytest
from httpx import ASGITransport, AsyncClient

from app.database import init_db
from app.main import app
from app.models import Character, DialogueLine, Scene
from app.services.storyboard import (
    StoryboardGenerationError,
    StoryboardService,
)


def make_character(name, visual_prompt):
    return Character(
        id=f"{name.lower()}-id",
        name=name,
        role="Supporting",
        description=f"{name} has a distinctive, consistent appearance.",
        appearance_description=f"recognizable features of {name}",
        clothing_style=f"practical clothing worn by {name}",
        visual_prompt=visual_prompt,
    )


def make_scene(scene_number=1, action=None, speaker="Mara"):
    return Scene(
        scene_number=scene_number,
        story_id="story-board-test",
        heading=f"INT. CONTROL ROOM - NIGHT",
        location="Orbital control room",
        time_of_day="NIGHT",
        mood="Urgent",
        lighting="Cool instrument light",
        weather="Meteor shower outside",
        shot_type="Wide shot",
        camera_movement="Slow push-in",
        duration_seconds=9,
        action_description=action or "Mara studies the display while Jon waits.",
        visual_prompt="Original prompt is replaced by the deterministic scene prompt.",
        dialogues=[
            DialogueLine(character_name=speaker, text="The signal is changing.")
        ],
    )


def test_single_scene_builds_structured_storyboard_frame():
    service = StoryboardService()
    mara = make_character("Mara", "Mara's stable visual design")

    frames = service.build_storyboard([make_scene()], [mara])

    assert len(frames) == 1
    frame = frames[0]
    assert frame.scene_number == 1
    assert frame.scene_heading == "INT. CONTROL ROOM - NIGHT"
    assert frame.location == "Orbital control room"
    assert frame.weather == "Meteor shower outside"
    assert frame.shot_type == "Wide shot"
    assert frame.duration_seconds == 9
    assert frame.character_references[0].visual_description == "Mara's stable visual design"
    assert frame.dialogue_summary[0].text == "The signal is changing."
    assert frame.status == "pending"


def test_multi_scene_storyboard_is_sorted_and_deterministic():
    service = StoryboardService()
    scenes = [
        make_scene(2, "Jon leaves the control room."),
        make_scene(1, "Mara studies the display."),
    ]

    first = service.build_storyboard(scenes)
    second = service.build_storyboard(scenes)

    assert [frame.scene_number for frame in first] == [1, 2]
    assert [frame.visual_prompt for frame in first] == [
        frame.visual_prompt for frame in second
    ]


def test_storyboard_rejects_duplicate_scene_numbers():
    with pytest.raises(StoryboardGenerationError, match="Duplicate scene numbers"):
        StoryboardService().build_storyboard(
            [make_scene(1), make_scene(1, "Mara exits the room.")]
        )


def test_storyboard_rejects_missing_scene_numbers():
    with pytest.raises(StoryboardGenerationError, match="Missing scene numbers: 2"):
        StoryboardService().build_storyboard(
            [make_scene(1), make_scene(3, "Mara exits the room.")]
        )


def test_multiple_character_references_keep_reusable_visuals():
    service = StoryboardService()
    mara = make_character("Mara", "stable Mara design")
    jon = make_character("Jon", "stable Jon design")

    frame = service.build_frame(
        make_scene(action="Mara and Jon study the display."),
        [mara, jon],
    )

    assert [item.name for item in frame.character_references] == ["Mara", "Jon"]
    assert [item.visual_description for item in frame.character_references] == [
        "stable Mara design",
        "stable Jon design",
    ]
    assert "recognizable features of Mara" in frame.visual_prompt
    assert "recognizable features of Jon" in frame.visual_prompt


def test_recurring_characters_have_explicit_scene_continuity_metadata():
    mara = make_character("Mara", "stable Mara design")
    frames = StoryboardService().build_storyboard(
        [
            make_scene(1, "Mara studies the display."),
            make_scene(2, "Mara follows the signal outside."),
        ],
        [mara],
    )

    assert len(frames) == 2
    assert frames[1].previous_scene_reference == 1
    assert frames[1].recurring_character_references[0].character_id == mara.id
    assert any("Mara" in note for note in frames[1].continuity_notes)


def test_frame_metadata_is_complete():
    frame = StoryboardService().build_frame(make_scene())
    frame_data = frame.model_dump()

    for field in (
        "scene_number",
        "scene_heading",
        "shot_type",
        "camera_movement",
        "lighting",
        "weather",
        "mood",
        "action_description",
        "duration_seconds",
        "visual_prompt",
        "character_references",
        "status",
    ):
        assert field in frame_data


def test_character_reference_is_not_invented_when_scene_does_not_reference_them():
    scene = make_scene(action="An empty control room hums quietly.")
    scene.dialogues = []
    frame = StoryboardService().build_frame(
        scene,
        [make_character("Mara", "stable Mara design")],
    )

    assert frame.character_references == []


@pytest.mark.anyio
async def test_placeholder_frame_contains_useful_scene_details(tmp_path, monkeypatch):
    service = StoryboardService()
    service.output_dir = tmp_path
    mara = make_character("Mara", "stable Mara design")
    frame = service.build_frame(make_scene(), [mara])
    drawn_text = []
    original_draw = __import__(
        "app.services.image_providers.placeholder",
        fromlist=["ImageDraw"],
    ).ImageDraw.Draw

    class DrawSpy:
        def __init__(self, draw):
            self.draw = draw

        def rectangle(self, *args, **kwargs):
            return self.draw.rectangle(*args, **kwargs)

        def text(self, position, text, **kwargs):
            drawn_text.append(text)
            return self.draw.text(position, text, **kwargs)

    monkeypatch.setattr(
        "app.services.image_providers.placeholder.ImageDraw.Draw",
        lambda image: DrawSpy(original_draw(image)),
    )

    rendered = await service.render_placeholders([frame])

    assert rendered[0].status == "placeholder"
    assert rendered[0].image_url.startswith("/generated/images/scene_1_")
    output_file = tmp_path / rendered[0].image_url.rsplit("/", 1)[1]
    assert output_file.exists()
    assert output_file.stat().st_size > 0
    visible_text = "\n".join(drawn_text)
    assert "SCENE #01" in visible_text
    assert "INT. CONTROL ROOM - NIGHT" in visible_text
    assert "SHOT: Wide shot" in visible_text
    assert "DURATION: 9s" in visible_text
    assert "CHARACTERS: Mara" in visible_text
    assert "Mara studies the display" in visible_text


@pytest.mark.parametrize(
    "scene_data",
    [
        {"scene_number": 1, "heading": "", "action_description": "Action", "visual_prompt": "Prompt"},
        {"scene_number": 1, "heading": "INT. ROOM - DAY", "action_description": "Action", "visual_prompt": "Prompt", "duration_seconds": 0},
        {"scene_number": 1, "heading": "INT. ROOM - DAY", "action_description": "", "visual_prompt": "Prompt"},
        {"scene_number": 1, "heading": "INT. ROOM - DAY", "action_description": "Action", "visual_prompt": "Prompt", "dialogues": [{"character_name": "", "text": "Hi"}]},
    ],
)
def test_invalid_scene_data_returns_storyboard_validation_error(scene_data):
    with pytest.raises(StoryboardGenerationError):
        StoryboardService().build_frame(scene_data)


@pytest.mark.anyio
async def test_storyboard_api_returns_structured_frames_for_story():
    await init_db()
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        analysis = await client.post(
            "/api/story/analyze",
            json={
                "prompt": "A station engineer follows a distress signal through an orbital habitat.",
                "genre": "Science Fiction",
                "provider": "rule_based",
            },
        )
        assert analysis.status_code == 201
        story_id = analysis.json()["story_id"]

        response = await client.get(f"/api/scenes/storyboard/{story_id}")

        assert response.status_code == 200
        payload = response.json()
        assert payload["story_id"] == story_id
        assert len(payload["frames"]) >= 1
        assert payload["frames"][0]["scene_heading"]
        assert payload["frames"][0]["visual_prompt"]
        assert payload["frames"][0]["status"] == "placeholder"


@pytest.mark.anyio
async def test_single_scene_storyboard_endpoint_preserves_legacy_fields():
    await init_db()
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/scenes/storyboard",
            json={"scene_number": 1, "prompt_override": "A custom storyboard prompt."},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["visual_prompt"] == "A custom storyboard prompt."
        assert payload["image_url"]
        assert "frame" in payload
