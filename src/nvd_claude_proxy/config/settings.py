from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Search for .env in these locations, in priority order (first found wins).
# This lets `ncp` work from any working directory when installed globally.
_ENV_FILE_CANDIDATES: list[str] = [
    ".env",  # cwd (local dev / docker)
    str(Path.home() / ".config" / "nvd-claude-proxy" / ".env"),  # XDG
    str(Path.home() / ".nvd-claude-proxy"),  # legacy dot-file
]


class Settings(BaseSettings):
    # `protected_namespaces=()` disables pydantic's guard on the `model_`
    # prefix so fields like `model_config_path` don't collide with pydantic's
    # own `model_config` attribute.
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE_CANDIDATES,
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    nvidia_api_key: str = Field(..., alias="NVIDIA_API_KEY")
    nvidia_base_url: str = Field("https://integrate.api.nvidia.com/v1", alias="NVIDIA_BASE_URL")
    proxy_host: str = Field("127.0.0.1", alias="PROXY_HOST")
    proxy_port: int = Field(8787, alias="PROXY_PORT")
    proxy_api_key: str | None = Field(default=None, alias="PROXY_API_KEY")
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    model_config_path: str | None = Field(default=None, alias="MODEL_CONFIG_PATH")
    request_timeout_seconds: float = Field(600.0, alias="REQUEST_TIMEOUT_SECONDS")
    max_retries: int = Field(2, alias="MAX_RETRIES")
    # Per-client rate limiting (disabled by default; set > 0 to enable).
    # Keyed on metadata.user_id if present, otherwise on client IP.
    rate_limit_rpm: int = Field(0, alias="RATE_LIMIT_RPM")
    # Max request body size in megabytes (0 = unlimited).
    max_request_body_mb: float = Field(0.0, alias="MAX_REQUEST_BODY_MB")


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
