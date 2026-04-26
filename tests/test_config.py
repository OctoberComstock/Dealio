from app.config import Settings


def test_runtime_limits_overridable_via_env(monkeypatch):
    monkeypatch.setenv("MAX_SEARCHES", "10")
    monkeypatch.setenv("REQUEST_TIMEOUT_SECONDS", "60")
    settings = Settings()
    assert settings.max_searches == 10
    assert settings.request_timeout_seconds == 60


def test_tavily_config_overridable_via_env(monkeypatch):
    monkeypatch.setenv("TAVILY_SEARCH_DEPTH", "advanced")
    monkeypatch.setenv("TAVILY_MAX_RESULTS_PER_QUERY", "10")
    settings = Settings()
    assert settings.tavily_search_depth == "advanced"
    assert settings.tavily_max_results_per_query == 10


def test_anthropic_model_overridable_via_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-opus-4-7")
    settings = Settings()
    assert settings.anthropic_model == "claude-opus-4-7"
