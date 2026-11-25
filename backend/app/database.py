import logging
from typing import AsyncGenerator
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from .config import settings

logger = logging.getLogger("uvicorn")

Base = declarative_base()

# Configure SQLAlchemy async engine
# Supports SQLite fallback if PostgreSQL is not yet configured/reachable
db_url = settings.DATABASE_URL
if db_url.startswith("postgresql://"):
    # Ensure async driver is used with SQLAlchemy async engine
    db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)

connect_args = {}
if "sqlite" in db_url:
    connect_args = {"check_same_thread": False}

engine = create_async_engine(
    db_url,
    echo=settings.DEBUG,
    connect_args=connect_args,
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency that yields an asynchronous database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    """Create database tables and perform column migrations on startup."""
    try:
        from .models import db_models  # noqa: F401
        async with engine.begin() as conn:
            # Create any tables not yet existing (e.g. projects)
            await conn.run_sync(Base.metadata.create_all)

            # Apply idempotent column additions for existing tables if needed
            migration_statements = [
                # Stories table updates
                "ALTER TABLE stories ADD COLUMN IF NOT EXISTS project_id VARCHAR(64);",
                "ALTER TABLE stories ADD COLUMN IF NOT EXISTS synopsis TEXT;",
                "ALTER TABLE stories ADD COLUMN IF NOT EXISTS theme VARCHAR(255);",
                "ALTER TABLE stories ADD COLUMN IF NOT EXISTS tone VARCHAR(100);",
                "ALTER TABLE stories ADD COLUMN IF NOT EXISTS status VARCHAR(50) DEFAULT 'analyzed';",
                "ALTER TABLE stories ADD COLUMN IF NOT EXISTS provider VARCHAR(50) DEFAULT 'rule_based';",
                "ALTER TABLE stories ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP;",

                # Characters table updates
                "ALTER TABLE characters ADD COLUMN IF NOT EXISTS role VARCHAR(100) DEFAULT 'Protagonist';",
                "ALTER TABLE characters ADD COLUMN IF NOT EXISTS personality TEXT;",

                # Scenes table updates
                "ALTER TABLE scenes ADD COLUMN IF NOT EXISTS location VARCHAR(255);",
                "ALTER TABLE scenes ADD COLUMN IF NOT EXISTS time_of_day VARCHAR(50);",
                "ALTER TABLE scenes ADD COLUMN IF NOT EXISTS mood VARCHAR(100);",
                "ALTER TABLE scenes ADD COLUMN IF NOT EXISTS lighting VARCHAR(255);",
                "ALTER TABLE scenes ADD COLUMN IF NOT EXISTS dialogues JSON;",
            ]

            for stmt in migration_statements:
                try:
                    await conn.execute(text(stmt))
                except Exception as m_err:
                    logger.debug(f"Migration note for statement [{stmt}]: {m_err}")

        logger.info("Database tables and schema migrations verified successfully.")
    except Exception as e:
        logger.warning(
            "Could not initialize the configured database (%s). "
            "Verify PostgreSQL is running and DATABASE_URL is valid.",
            type(e).__name__,
        )
