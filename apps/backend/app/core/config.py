from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Nexus MCP"
    app_version: str = "0.1.0"
    app_env: str = "development"
    debug: bool = True

    api_v1_prefix: str = "/api/v1"
    cors_origins: list[str] = ["http://localhost:5173"]

    log_level: str = "INFO"

    # SQLite keeps the initial bootstrap runnable without Docker.
    # We'll switch the development environment to PostgreSQL later.
    database_url: str = "sqlite:///./nexus.db"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
