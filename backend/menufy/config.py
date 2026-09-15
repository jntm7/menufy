"""Application settings loaded from environment variables and `.env`."""

from functools import lru_cache
from pathlib import Path
from typing import Literal
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["ollama", "gemini"]

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

# App settings
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    # LLM provider
    llm_provider: LLMProvider = "ollama"

    # Ollama (local development)
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen3-vl:8b"

    # Gemini (production)
    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-3.8-flash"

    # Runtime
    request_timeout_seconds: float = 60.0
    debug: bool = False

    @property
    def llm_model(self) -> str:
        return self.gemini_model if self.llm_provider == "gemini" else self.ollama_model

@lru_cache
def get_settings() -> Settings:
    return Settings()
