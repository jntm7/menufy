from fastapi.testclient import TestClient
from pydantic import SecretStr
from menufy.config import Settings, get_settings
from menufy.main import app

# Health request helper
def get_health(settings: Settings) -> dict[str, str]:
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        response = TestClient(app).get("/health")
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200
    body: dict[str, str] = response.json()
    return body

# Health endpoint tests
def test_health_defaults_to_local_ollama() -> None:
    assert get_health(Settings()) == {
        "status": "ok",
        "llm_provider": "ollama",
        "llm_model": "qwen3-vl:8b",
    }

def test_health_reports_gemini_in_production() -> None:
    settings = Settings(llm_provider="gemini", gemini_api_key=SecretStr("key"))

    assert get_health(settings)["llm_model"] == "gemini-3.8-flash"
