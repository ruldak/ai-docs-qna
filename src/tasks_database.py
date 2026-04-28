from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.constants import TASKS_DATABASE_URL

# Celery worker runs in a separate process, use the SYNC driver
engine = create_engine(TASKS_DATABASE_URL, pool_size=5, max_overflow=10)
SessionLocal = sessionmaker(bind=engine)