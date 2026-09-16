from types import SimpleNamespace
from typing import Any
import httpx
import pytest
from google.genai import errors as genai_errors
from google.genai import types
from ollama import ResponseError
from pydantic import SecretStr
from menufy.config import Settings
from menufy.pipeline.errors import (
    LLMEmptyResponseError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from menufy.pipeline.llm import GeminiLLM, ImageInput, OllamaLLM, build_llm

# Shared test inputs
IMAGE = ImageInput(data=b"img", mime_type="image/jpeg")
SCHEMA = {"type": "object"}

# Fake Ollama client
class FakeOllamaClient:
    def __init__(self, reply: str = "ok", error: Exception | None = None) -> None:
        self.reply = reply
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def chat(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(message=SimpleNamespace(content=self.reply))

# Fake Gemini client
class FakeGeminiClient:
    def __init__(
        self,
        reply: str = "ok",
        error: Exception | None = None,
        block_reason: types.BlockedReason | None = None,
        finish_reason: types.FinishReason | None = None,
    ) -> None:
        self.reply = reply
        self.error = error
        self.block_reason = block_reason
        self.finish_reason = finish_reason
        self.calls: list[dict[str, Any]] = []
        self.models = self

    def generate_content(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        blocked = self.block_reason is not None
        feedback = SimpleNamespace(block_reason=self.block_reason) if blocked else None
        candidates = (
            [SimpleNamespace(finish_reason=self.finish_reason)]
            if self.finish_reason is not None
            else None
        )
        return SimpleNamespace(text=self.reply, prompt_feedback=feedback, candidates=candidates)

# Helpers
def ollama_llm(client: FakeOllamaClient, model: str = "qwen3-vl:8b") -> OllamaLLM:
    return OllamaLLM(client, model)  # type: ignore[arg-type]

def gemini_llm(client: FakeGeminiClient, model: str = "gemini-3.8-flash") -> GeminiLLM:
    return GeminiLLM(client, model)  # type: ignore[arg-type]

# Ollama provider tests
def test_ollama_passes_model_images_and_schema() -> None:
    fake = FakeOllamaClient(reply='{"dishes": []}')

    result = ollama_llm(fake).generate("read this menu", images=[IMAGE], json_schema=SCHEMA)

    assert result == '{"dishes": []}'
    (call,) = fake.calls
    assert call["model"] == "qwen3-vl:8b"
    assert call["format"] == SCHEMA
    assert call["messages"] == [{"role": "user", "content": "read this menu", "images": [b"img"]}]

def test_ollama_omits_images_for_text_only() -> None:
    fake = FakeOllamaClient()

    ollama_llm(fake).generate("hi")

    assert "images" not in fake.calls[0]["messages"][0]

# Ollama error mapping tests
def test_ollama_timeout_raises_llm_timeout() -> None:
    fake = FakeOllamaClient(error=httpx.ReadTimeout("timed out"))

    with pytest.raises(LLMTimeoutError):
        ollama_llm(fake).generate("hi")

def test_ollama_connection_refused_raises_unavailable() -> None:
    fake = FakeOllamaClient(error=ConnectionError("connection refused"))

    with pytest.raises(LLMUnavailableError, match="Is it running"):
        ollama_llm(fake).generate("hi")

def test_ollama_missing_model_suggests_pull() -> None:
    fake = FakeOllamaClient(error=ResponseError("model not found", 404))

    with pytest.raises(LLMUnavailableError, match="ollama pull qwen3-vl:8b"):
        ollama_llm(fake).generate("hi")

def test_ollama_rate_limit_raises_rate_limit() -> None:
    fake = FakeOllamaClient(error=ResponseError("slow down", 429))

    with pytest.raises(LLMRateLimitError):
        ollama_llm(fake).generate("hi")

def test_ollama_other_status_raises_generic_llm_error() -> None:
    fake = FakeOllamaClient(error=ResponseError("boom", 500))

    with pytest.raises(LLMError, match="500"):
        ollama_llm(fake).generate("hi")

def test_ollama_blank_reply_raises_empty_response() -> None:
    fake = FakeOllamaClient(reply="   ")

    with pytest.raises(LLMEmptyResponseError):
        ollama_llm(fake).generate("hi")

# Gemini provider tests
def test_gemini_sends_image_before_prompt_and_requests_json() -> None:
    fake = FakeGeminiClient(reply='{"dishes": []}')

    result = gemini_llm(fake).generate("read this menu", images=[IMAGE], json_schema=SCHEMA)

    assert result == '{"dishes": []}'
    (call,) = fake.calls
    assert call["model"] == "gemini-3.8-flash"
    image_part, prompt = call["contents"]
    assert isinstance(image_part, types.Part)
    assert image_part.inline_data is not None
    assert image_part.inline_data.mime_type == "image/jpeg"
    assert prompt == "read this menu"
    assert call["config"].response_mime_type == "application/json"
    assert call["config"].response_json_schema == SCHEMA

def test_gemini_text_only_disables_function_calling_without_json() -> None:
    fake = FakeGeminiClient()

    gemini_llm(fake).generate("hi")

    (call,) = fake.calls
    assert call["contents"] == ["hi"]
    assert call["config"].automatic_function_calling.disable is True
    assert call["config"].response_mime_type is None

# Gemini error mapping tests
def test_gemini_timeout_raises_llm_timeout() -> None:
    fake = FakeGeminiClient(error=httpx.ReadTimeout("timed out"))

    with pytest.raises(LLMTimeoutError):
        gemini_llm(fake).generate("hi")

def test_gemini_quota_raises_rate_limit() -> None:
    error = genai_errors.ClientError(429, {"error": {"message": "quota exceeded"}})
    fake = FakeGeminiClient(error=error)

    with pytest.raises(LLMRateLimitError):
        gemini_llm(fake).generate("hi")

def test_gemini_server_error_raises_unavailable() -> None:
    error = genai_errors.ServerError(503, {"error": {"message": "overloaded"}})
    fake = FakeGeminiClient(error=error)

    with pytest.raises(LLMUnavailableError):
        gemini_llm(fake).generate("hi")

def test_gemini_client_error_raises_generic_llm_error() -> None:
    error = genai_errors.ClientError(400, {"error": {"message": "bad request"}})
    fake = FakeGeminiClient(error=error)

    with pytest.raises(LLMError, match="400"):
        gemini_llm(fake).generate("hi")

def test_gemini_blocked_prompt_reports_reason() -> None:
    fake = FakeGeminiClient(reply="", block_reason=types.BlockedReason.SAFETY)

    with pytest.raises(LLMEmptyResponseError, match="SAFETY") as caught:
        gemini_llm(fake).generate("hi")
    assert caught.value.reason == "SAFETY"

def test_gemini_truncated_reply_reports_finish_reason() -> None:
    fake = FakeGeminiClient(reply="", finish_reason=types.FinishReason.MAX_TOKENS)

    with pytest.raises(LLMEmptyResponseError, match="MAX_TOKENS"):
        gemini_llm(fake).generate("hi")

# Provider factory tests
def test_build_llm_selects_provider() -> None:
    assert isinstance(build_llm(Settings()), OllamaLLM)
    gemini = build_llm(Settings(llm_provider="gemini", gemini_api_key=SecretStr("key")))
    assert isinstance(gemini, GeminiLLM)

def test_gemini_requires_api_key() -> None:
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        build_llm(Settings(llm_provider="gemini"))
