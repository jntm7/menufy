"""Provider-agnostic LLM interface with Ollama (local) and Gemini (production) implementations."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol
import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from ollama import Client as OllamaClient
from ollama import ResponseError
from menufy.config import Settings
from menufy.pipeline.errors import (
    LLMEmptyResponseError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
)

# HTTP statuses both providers use
RATE_LIMIT_STATUS = 429
NOT_FOUND_STATUS = 404

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
        try:
            response = self._client.chat(model=self._model, messages=[message], format=json_schema)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"{self._model} timed out") from exc
        # The Ollama client turns a refused connection into a plain ConnectionError.
        except ConnectionError as exc:
            raise LLMUnavailableError("Cannot reach Ollama. Is it running?") from exc
        except ResponseError as exc:
            raise self._as_llm_error(exc) from exc
        content = response.message.content or ""
        if not content.strip():
            raise LLMEmptyResponseError(f"{self._model} returned an empty response")
        return content

    def _as_llm_error(self, exc: ResponseError) -> LLMError:
        if exc.status_code == RATE_LIMIT_STATUS:
            return LLMRateLimitError(f"Ollama rate limited the request: {exc}")
        if exc.status_code == NOT_FOUND_STATUS:
            return LLMUnavailableError(
                f"Model {self._model} is missing. Run: ollama pull {self._model}"
            )
        return LLMError(f"Ollama request failed ({exc.status_code}): {exc}")

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
        try:
            response = self._client.models.generate_content(
                model=self._model, contents=contents, config=config
            )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"{self._model} timed out") from exc
        except genai_errors.APIError as exc:
            raise self._as_llm_error(exc) from exc
        text = response.text or ""
        if not text.strip():
            reason = _stop_reason(response)
            raise LLMEmptyResponseError(
                f"{self._model} returned no text (reason: {reason or 'unknown'})", reason=reason
            )
        return text

    def _as_llm_error(self, exc: genai_errors.APIError) -> LLMError:
        if exc.code == RATE_LIMIT_STATUS:
            return LLMRateLimitError(f"Gemini quota or rate limit exceeded: {exc}")
        if isinstance(exc, genai_errors.ServerError):
            return LLMUnavailableError(f"Gemini is unavailable ({exc.code}): {exc}")
        return LLMError(f"Gemini request failed ({exc.code}): {exc}")

# Why Gemini returned nothing: a blocked prompt or a non-STOP finish reason
def _stop_reason(response: types.GenerateContentResponse) -> str | None:
    feedback = response.prompt_feedback
    if feedback is not None and feedback.block_reason is not None:
        return str(feedback.block_reason.name)
    if response.candidates:
        finish_reason = response.candidates[0].finish_reason
        if finish_reason is not None:
            return str(finish_reason.name)
    return None

# Provider factory
def build_llm(settings: Settings) -> LLM:
    match settings.llm_provider:
        case "ollama":
            return OllamaLLM.from_settings(settings)
        case "gemini":
            return GeminiLLM.from_settings(settings)
