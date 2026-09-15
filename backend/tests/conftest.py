from collections.abc import Iterator
import pytest
from menufy.config import Settings, get_settings

# Settings isolation fixture
@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep tests independent of the developer's `.env` file and shell environment."""
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    for field in Settings.model_fields:
        monkeypatch.delenv(field.upper(), raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
