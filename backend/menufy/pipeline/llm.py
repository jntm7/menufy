"""Provider-agnostic LLM interface with Ollama (local) and Gemini (production) implementations."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol
from google import genai
from google.genai import types
from ollama import Client as OllamaClient
from menufy.config import Settings

# Image input
@dataclass(frozen=True, slots=True)
class ImageInput:
    data: bytes
    mime_type: str

# LLM interface
class LLM(Protocol):
    def generate(
        self,
        prompt: str,
        *,
        images: Sequence[ImageInput] = (),
        json_schema: dict[str, Any] | None = None,
    ) -> str: ...

# Ollama provider (local development)
class OllamaLLM:
    def __init__(self, client: OllamaClient, model: str) -> None:
        self._client = client
        self._model = model

    @classmethod
    def from_settings(cls, settings: Settings) -> "OllamaLLM":
        client = OllamaClient(host=settings.ollama_host, timeout=settings.request_timeout_seconds)
        return cls(client, settings.ollama_model)

    def generate(
        self,
        prompt: str,
        *,
        images: Sequence[ImageInput] = (),
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        message: dict[str, Any] = {"role": "user", "content": prompt}
        if images:
            message["images"] = [image.data for image in images]
        response = self._client.chat(model=self._model, messages=[message], format=json_schema)
        return response.message.content or ""

# Gemini provider (production)
class GeminiLLM:
    def __init__(self, client: genai.Client, model: str) -> None:
        self._client = client
        self._model = model

    @classmethod
    def from_settings(cls, settings: Settings) -> "GeminiLLM":
        if settings.gemini_api_key is None:
            raise ValueError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini")
        client = genai.Client(
            api_key=settings.gemini_api_key.get_secret_value(),
            http_options=types.HttpOptions(timeout=int(settings.request_timeout_seconds * 1000)),
        )
        return cls(client, settings.gemini_model)

    def generate(
        self,
        prompt: str,
        *,
        images: Sequence[ImageInput] = (),
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        contents: list[types.Part | str] = [
            types.Part.from_bytes(data=image.data, mime_type=image.mime_type) for image in images
        ]
        contents.append(prompt)
        config = types.GenerateContentConfig(
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        if json_schema is not None:
            config.response_mime_type = "application/json"
            config.response_json_schema = json_schema
        response = self._client.models.generate_content(
            model=self._model, contents=contents, config=config
        )
        return response.text or ""

# Provider factory
def build_llm(settings: Settings) -> LLM:
    match settings.llm_provider:
        case "ollama":
            return OllamaLLM.from_settings(settings)
        case "gemini":
            return GeminiLLM.from_settings(settings)
