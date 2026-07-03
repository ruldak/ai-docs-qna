import os
from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    db_user: Optional[str] = Field(None, env="DB_USER")
    db_password: Optional[str] = Field(None, env="DB_PASSWORD")
    db_host: Optional[str] = Field(None, env="DB_HOST")
    db_name: Optional[str] = Field(None, env="DB_NAME")
    db_port: str = Field("5432", env="DB_PORT")

    database_url: Optional[str] = Field(None, env="DATABASE_URL")

    groq_api_key: Optional[str] = Field(None, env="GROQ_API_KEY")
    hugging_face_api_key: Optional[str] = Field(None, env="HUGGING_FACE_API_KEY")
    cohere_api_key: Optional[str] = Field(None, env="COHERE_API_KEY")
    secret_key: Optional[str] = Field(None, env="SECRET_KEY")

    supabase_url: Optional[str] = Field(None, env="SUPABASE_URL")
    supabase_service_role_key: Optional[str] = Field(None, env="SUPABASE_SERVICE_ROLE_KEY")
    bucket_name: Optional[str] = Field(None, env="BUCKET_NAME")

    lancedb_uri: str = Field("./lancedb_data", env="LANCEDB_URI")
    lancedb_table: str = Field("documents", env="LANCEDB_TABLE")

    celery_broker_url: Optional[str] = Field(None, env="CELERY_BROKER_URL")
    celery_backend_url: Optional[str] = Field(None, env="CELERY_BACKEND_URL")

    llm_model: str = Field("llama-3.3-70b-versatile", env="LLM_MODEL")
    llm_temperature: float = Field(0.0, env="LLM_TEMPERATURE")

    max_file_size_mb: int = Field(5, env="MAX_FILE_SIZE_MB")
    access_token_expire_minutes: int = Field(60, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    refresh_token_expire_days: int = Field(7, env="REFRESH_TOKEN_EXPIRE_DAYS")

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore"
    )

    @property
    def database_url_async(self) -> str:
        """Async database URL for FastAPI (asyncpg)."""
        if self.database_url:
            return self.database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        if all([self.db_user, self.db_password, self.db_host, self.db_name]):
            return f"postgresql+asyncpg://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
        raise ValueError("Database configuration incomplete. Provide either DATABASE_URL or DB_USER/DB_PASSWORD/DB_HOST/DB_NAME")

    @property
    def database_url_sync(self) -> str:
        """Sync database URL for Celery workers (psycopg2)."""
        if self.database_url:
            return self.database_url
        if all([self.db_user, self.db_password, self.db_host, self.db_name]):
            return f"postgresql+psycopg2://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
        raise ValueError("Database configuration incomplete. Provide either DATABASE_URL or DB_USER/DB_PASSWORD/DB_HOST/DB_NAME")

    @property
    def max_file_size_bytes(self) -> int:
        """Max file size in bytes."""
        return self.max_file_size_mb * 1024 * 1024


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
DATABASE_URL = settings.database_url_async
TASKS_DATABASE_URL = settings.database_url_sync