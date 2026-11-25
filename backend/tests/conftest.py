import pytest

from app.database import engine


@pytest.fixture(autouse=True)
async def dispose_database_connections(anyio_backend):
    yield
    await engine.dispose()
