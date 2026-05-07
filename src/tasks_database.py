from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.constants import TASKS_DATABASE_URL

# Konfigurasi pool untuk Celery worker
# Sesuaikan pool_size dengan --concurrency worker
engine = create_engine(
    TASKS_DATABASE_URL,
    pool_size=25,
    max_overflow=15,
    pool_pre_ping=True,
    pool_recycle=3600,
    pool_timeout=30,
    echo=False
)

SessionLocal = sessionmaker(bind=engine)