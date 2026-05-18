"""
Unified database configuration supporting both Async (FastAPI) and Sync (Celery) operations.
Uses connection pooling optimized for each use case.
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from src.constants import settings

# ============================================================================
# ASYNC ENGINE (FastAPI)
# ============================================================================
async_engine = create_async_engine(
    settings.database_url_async,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=3600,
    pool_timeout=30,
    echo=False,
    # Async-specific optimizations
    future=True,
)

AsyncSessionLocal = sessionmaker(
    async_engine, 
    class_=AsyncSession, 
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)

# ============================================================================
# SYNC ENGINE (Celery Workers)
# ============================================================================
sync_engine = create_engine(
    settings.database_url_sync,
    pool_size=25,
    max_overflow=15,
    pool_pre_ping=True,
    pool_recycle=3600,
    pool_timeout=30,
    echo=False,
    # Sync-specific optimizations
    future=True,
)

SyncSessionLocal = sessionmaker(
    bind=sync_engine,
    autoflush=False,
    autocommit=False,
)

# ============================================================================
# BASE MODEL
# ============================================================================
Base = declarative_base()

# ============================================================================
# DEPENDENCIES
# ============================================================================
async def get_db():
    """FastAPI dependency for async database sessions."""
    db = AsyncSessionLocal()
    try:
        yield db
    finally:
        await db.close()


def get_sync_db():
    """Context manager for sync database sessions (Celery tasks)."""
    db = SyncSessionLocal()
    try:
        yield db
    finally:
        db.close()