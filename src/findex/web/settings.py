"""Configuration from the environment (12-factor rule III), validated by pydantic-settings.

    INDEX_PATH=/data/index.bin LOG_LEVEL=info LOG_JSON=true uvicorn findex.web.app:app

A ``.env`` file in the working directory is read too (development only).
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    index_path: Path = Path("index.bin")
    log_level: Literal["debug", "info", "warning", "error"] = "info"
    log_json: bool = False  # one JSON object per log line, for the platform's log collector
    cors_origins: list[str] = Field(default_factory=list[str])  # empty: same-origin only
    search_cache_size: int = Field(default=512, ge=0)


@functools.cache
def get_settings() -> Settings:
    """One Settings per process; tests override the dependency instead of the environment."""
    return Settings()
