import pytest
import asyncio
from httpx import AsyncClient, ASGITransport
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app
from app.database import AsyncSessionLocal, init_db
from app.services.ai import (
    get_provider,
    list_available_providers,
    RuleBasedStoryProvider,
    LocalLLMStoryProvider,
    ProviderNotAvailableError,
    InvalidProviderError,
)
from app.models.db_models import Story, Character, Movie, Scene
from sqlalchemy import select


@pytest.mark.anyio
async def test_provider_registry():
    """Verify provider interface and registry behavior."""
    providers = list_available_providers()
    assert "rule_based" in providers
    assert "local_llm" in providers

    rule_provider = get_provider("rule_based")
    assert isinstance(rule_provider, RuleBasedStoryProvider)
    assert rule_provider.is_available() is True

    local_provider = get_provider("local_llm")
    assert isinstance(local_provider, LocalLLMStoryProvider)
    assert local_provider.is_available() is False

    # Attempting analyze on inactive local_llm should raise ProviderNotAvailableError
    try:
        await local_provider.analyze(prompt="Test prompt that is long enough")
        assert False, "Expected ProviderNotAvailableError"
    except ProviderNotAvailableError:
        pass

    # Requesting unknown provider raises InvalidProviderError
    try:
        get_provider("non_existent_provider")
        assert False, "Expected InvalidProviderError"
    except InvalidProviderError:
        pass


@pytest.mark.anyio
async def test_analyze_story_endpoint_and_postgres_persistence():
    """Test /api/story/analyze endpoint and verify PostgreSQL persistence."""
    await init_db()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "prompt": "In the subterranean depths of Old Berlin, investigator Mara Thorne tracks a rogue synthetic entity holding memory keys to an ancient reactor.",
            "genre": "Cyberpunk Mystery",
            "target_duration_minutes": 1,
            "provider": "rule_based",
        }
        response = await client.post("/api/story/analyze", json=payload)
        assert response.status_code == 201, f"Response: {response.text}"
        data = response.json()

        # Check response schema
        assert "story_id" in data
        assert data["title"]
        assert data["logline"]
        assert data["synopsis"]
        assert data["theme"]
        assert data["tone"]
        assert data["genre"] == "Cyberpunk Mystery"
        assert data["provider"] == "rule_based"
        assert len(data["characters"]) >= 2
        assert len(data["scenes"]) >= 3

        story_id = data["story_id"]

        # Check character fields
        first_char = data["characters"][0]
        assert first_char["id"]
        assert first_char["name"]
        assert first_char["role"]
        assert first_char["description"]
        assert first_char["visual_prompt"]
        assert first_char["voice_name"]

        # Check scene fields
        first_scene = data["scenes"][0]
        assert first_scene["scene_number"] == 1
        assert first_scene["heading"]
        assert first_scene["location"]
        assert first_scene["time_of_day"]
        assert first_scene["mood"]
        assert first_scene["lighting"]
        assert first_scene["action_description"]
        assert first_scene["visual_prompt"]
        assert first_scene["camera_movement"]
        assert len(first_scene["dialogues"]) > 0

        # Verify PostgreSQL database persistence directly
        async with AsyncSessionLocal() as session:
            stmt = select(Story).where(Story.id == story_id)
            res = await session.execute(stmt)
            db_story = res.scalar_one_or_none()
            assert db_story is not None
            assert db_story.title == data["title"]
            assert db_story.synopsis == data["synopsis"]
            assert db_story.genre == data["genre"]

            # Verify characters saved in DB
            stmt_chars = select(Character).where(Character.story_id == story_id)
            res_chars = await session.execute(stmt_chars)
            db_chars = res_chars.scalars().all()
            assert len(db_chars) == len(data["characters"])
            assert db_chars[0].name == data["characters"][0]["name"]

            # Verify scenes saved in DB
            stmt_scenes = select(Scene).where(Scene.story_id == story_id)
            res_scenes = await session.execute(stmt_scenes)
            db_scenes = res_scenes.scalars().all()
            assert len(db_scenes) == len(data["scenes"])
            assert db_scenes[0].heading == data["scenes"][0]["heading"]
            assert db_scenes[0].dialogues is not None


@pytest.mark.anyio
async def test_existing_endpoints_preserved():
    """Verify all existing endpoints remain fully functional."""
    await init_db()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Health check
        res_root = await client.get("/")
        assert res_root.status_code == 200
        assert res_root.json()["status"] == "healthy"

        # 2. Sample story
        res_sample = await client.get("/api/story/sample")
        assert res_sample.status_code == 200
        assert len(res_sample.json()["sample_prompts"]) >= 2

        # 3. List characters (with and without trailing slash)
        res_chars = await client.get("/api/characters/")
        assert res_chars.status_code == 200
        assert isinstance(res_chars.json(), list)

        res_chars_no_slash = await client.get("/api/characters")
        assert res_chars_no_slash.status_code == 200

        # 4. Voice test preview
        res_voice = await client.post("/api/characters/char-1/voice?text=Testing+character+speech")
        assert res_voice.status_code == 200
        assert "audio_url" in res_voice.json()
        assert res_voice.json()["audio_url"] is None

        # 5. List scenes (with and without trailing slash)
        res_scenes = await client.get("/api/scenes/")
        assert res_scenes.status_code == 200
        assert isinstance(res_scenes.json(), list)

        res_scenes_no_slash = await client.get("/api/scenes")
        assert res_scenes_no_slash.status_code == 200

        # 6. Generate storyboard frame
        res_frame = await client.post("/api/scenes/storyboard", json={"scene_number": 1})
        assert res_frame.status_code == 200
        assert "image_url" in res_frame.json()

        # 7. Screenplay export
        res_export = await client.get("/api/scenes/screenplay/export?title=Cyberpunk+Test")
        assert res_export.status_code == 200
        assert "screenplay" in res_export.json()

        # 8. Render a persisted story and verify its movie job is stored.
        story_response = await client.post(
            "/api/story/analyze",
            json={
                "prompt": (
                    "A cartographer uncovers a hidden route beneath a coastal "
                    "city while a storm closes in."
                ),
                "genre": "Mystery",
                "target_duration_minutes": 1,
            },
        )
        assert story_response.status_code == 201
        story_id = story_response.json()["story_id"]
        res_render = await client.post(
            "/api/movie/render",
            json={"story_id": story_id, "title": "Test Movie"},
        )
        assert res_render.status_code == 200
        movie_id = res_render.json()["movie_id"]
        async with AsyncSessionLocal() as session:
            db_movie = await session.get(Movie, movie_id)
            assert db_movie is not None
            assert db_movie.story_id == story_id
            assert db_movie.status in {"processing", "failed"}

        res_status = await client.get(f"/api/movie/status/{movie_id}")
        assert res_status.status_code == 200
        assert res_status.json()["movie_id"] == movie_id
        for _ in range(20):
            if res_status.json()["status"] in {"completed", "failed"}:
                break
            await asyncio.sleep(0.05)
            res_status = await client.get(f"/api/movie/status/{movie_id}")
        assert res_status.json()["status"] == "failed"
        async with AsyncSessionLocal() as session:
            db_movie = await session.get(Movie, movie_id)
            assert db_movie is not None
            assert db_movie.status == "failed"
        deleted = await client.delete(f"/api/story/{story_id}")
        assert deleted.status_code == 200


@pytest.mark.anyio
async def test_validation_and_error_handling():
    """Test validation errors and provider stub behavior."""
    await init_db()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Short prompt (< 10 chars)
        res_short = await client.post("/api/story/analyze", json={"prompt": "Short"})
        assert res_short.status_code == 422

        # Inactive local_llm provider returns 400
        res_local = await client.post("/api/story/analyze", json={
            "prompt": "A valid narrative premise that meets the length threshold.",
            "provider": "local_llm",
        })
        assert res_local.status_code == 400
        assert "Local LLM provider" in res_local.json()["detail"]

        # Unknown provider returns 400
        res_unknown = await client.post("/api/story/analyze", json={
            "prompt": "A valid narrative premise that meets the length threshold.",
            "provider": "unknown_ai",
        })
        assert res_unknown.status_code == 400
        assert "Unknown story analyzer provider" in res_unknown.json()["detail"]


if __name__ == "__main__":
    import asyncio
    async def run_all():
        await init_db()
        print("Running test_provider_registry...")
        await test_provider_registry()
        print("Running test_analyze_story_endpoint_and_postgres_persistence...")
        await test_analyze_story_endpoint_and_postgres_persistence()
        print("Running test_existing_endpoints_preserved...")
        await test_existing_endpoints_preserved()
        print("Running test_validation_and_error_handling...")
        await test_validation_and_error_handling()
        print("\nALL STEP 1 TESTS PASSED SUCCESSFULLY!")

    asyncio.run(run_all())
