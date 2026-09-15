from types import SimpleNamespace
from typing import Any
import pytest
from google.genai import types
from pydantic import SecretStr
from menufy.config import Settings
from menufy.pipeline.llm import GeminiLLM, ImageInput, OllamaLLM, build_llm

# Shared test inputs
IMAGE = ImageInput(data=b"img", mime_type="image/jpeg")
SCHEMA = {"type": "object"}

# Fake Ollama client
class FakeOllamaClient:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    def chat(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(message=SimpleNamespace(content=self.reply))

# Fake Gemini client
class FakeGeminiClient:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []
        self.models = self

    def generate_content(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(text=self.reply)

# Ollama provider tests
def test_ollama_passes_model_images_and_schema() -> None:
    fake = FakeOllamaClient(reply='{"dishes": []}')

    result = OllamaLLM(fake, "qwen3-vl:8b").generate(  # type: ignore[arg-type]
        "read this menu", images=[IMAGE], json_schema=SCHEMA
    )

    assert result == '{"dishes": []}'
    (call,) = fake.calls
    assert call["model"] == "qwen3-vl:8b"
    assert call["format"] == SCHEMA
    assert call["messages"] == [{"role": "user", "content": "read this menu", "images": [b"img"]}]

def test_ollama_omits_images_for_text_only() -> None:
    fake = FakeOllamaClient(reply="ok")

    OllamaLLM(fake, "m").generate("hi")  # type: ignore[arg-type]

    assert "images" not in fake.calls[0]["messages"][0]

# Gemini provider tests
def test_gemini_sends_image_before_prompt_and_requests_json() -> None:
    fake = FakeGeminiClient(reply='{"dishes": []}')

    result = GeminiLLM(fake, "gemini-3.8-flash").generate(  # type: ignore[arg-type]
        "read this menu", images=[IMAGE], json_schema=SCHEMA
    )

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
    fake = FakeGeminiClient(reply="ok")

    GeminiLLM(fake, "m").generate("hi")  # type: ignore[arg-type]

    (call,) = fake.calls
    assert call["contents"] == ["hi"]
    assert call["config"].automatic_function_calling.disable is True
    assert call["config"].response_mime_type is None

# Provider factory tests
def test_build_llm_selects_provider() -> None:
    assert isinstance(build_llm(Settings()), OllamaLLM)
    gemini = build_llm(Settings(llm_provider="gemini", gemini_api_key=SecretStr("key")))
    assert isinstance(gemini, GeminiLLM)

def test_gemini_requires_api_key() -> None:
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        build_llm(Settings(llm_provider="gemini"))
