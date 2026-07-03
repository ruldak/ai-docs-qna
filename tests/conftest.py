import pytest
import pytest_asyncio
import os
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.main import app
from src.database import get_db, Base
from src.app import models, utils

# Gunakan SQLite in-memory untuk testing (super cepat & terisolasi)
SQLALCHEMY_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

engine = create_async_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool, # Penting untuk SQLite in-memory agar sharing connection
)
TestingSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

@pytest_asyncio.fixture
async def db_session():
    """Membuat tabel sebelum test dan menghapusnya setelah test selesai."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async with TestingSessionLocal() as session:
        yield session
        
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

@pytest_asyncio.fixture
async def client(db_session: AsyncSession):
    """HTTP Client yang terhubung ke database testing."""
    async def override_get_db():
        try:
            yield db_session
        finally:
            pass # Biarkan fixture db_session yang handle closing

    app.dependency_overrides[get_db] = override_get_db
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
        
    app.dependency_overrides.clear()

@pytest_asyncio.fixture
async def test_user(db_session: AsyncSession):
    """Membuat user dummy langsung di database testing."""
    user = models.User(
        email="testuser@example.com",
        password=utils.get_password_hash("password123"), # Hash asli agar login valid
        full_name="Test User",
        role="user",
        is_active=True
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user

@pytest_asyncio.fixture
async def auth_headers(test_user: models.User):
    """Generate JWT Token valid untuk test_user."""
    access_token = utils.access_security.create_access_token(
        subject={"user_id": test_user.id}
    )
    return {"Authorization": f"Bearer {access_token}"}