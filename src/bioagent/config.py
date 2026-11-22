"""All settings, read once from the environment / `.env` in the project root."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_SRC_ROOT = Path(__file__).resolve().parents[2]
# Source checkout (editable install): the repo root. Installed wheel (e.g. Docker): the working directory.
PROJECT_ROOT = _SRC_ROOT if (_SRC_ROOT / "pyproject.toml").exists() else Path.cwd()


class ConfigError(RuntimeError):
    """A required setting is missing or invalid."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        populate_by_name=True,
    )

    llm_provider: Literal["gemini", "anthropic"] = "gemini"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.5-flash-lite"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-haiku-5-5"

    # NCBI E-utilities (email is required by NCBI policy; key raises the rate limit)
    ncbi_email: str | None = None
    ncbi_api_key: str | None = None

    http_timeout_s: float = Field(20.0, validation_alias=AliasChoices("BIOAGENT_HTTP_TIMEOUT_S", "http_timeout_s"))
    cache_enabled: bool = Field(True, validation_alias=AliasChoices("BIOAGENT_CACHE_ENABLED", "cache_enabled"))
    cache_path: Path = Field(
        PROJECT_ROOT / ".cache" / "bioagent.sqlite",
        validation_alias=AliasChoices("BIOAGENT_CACHE_PATH", "cache_path"),
    )

    max_steps: int = Field(6, ge=2, le=12, validation_alias=AliasChoices("BIOAGENT_MAX_STEPS", "max_steps"))
    max_tokens_per_run: int = Field(
        60_000, ge=1_000, validation_alias=AliasChoices("BIOAGENT_MAX_TOKENS_PER_RUN", "max_tokens_per_run")
    )
    citation_mode: Literal["strip", "flag"] = Field(
        "strip", validation_alias=AliasChoices("BIOAGENT_CITATION_MODE", "citation_mode")
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
