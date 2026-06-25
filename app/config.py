from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    tavily_api_key: str = ""
    tavily_search_depth: str = "basic"
    tavily_max_results_per_query: int = 5

    anthropic_api_key: str = ""
    anthropic_model: str = ""
    agent_timeout_seconds: int = 140

    max_searches: int = 8
    max_fetched_pages: int = 8
    meaningful_savings_threshold: float = 0.10
    request_timeout_seconds: int = 60
    supporting_page_timeout_seconds: int = 15

    search_cache_ttl_seconds: int = 900
    page_cache_ttl_seconds: int = 900
    parsed_content_cache_ttl_seconds: int = 900

    database_path: str = "./dealio.db"
    render_page_timeout_seconds: int = 30
    log_level: str = "DEBUG"

    dealio_demo_password: str = ""
    session_secret_key: str = "dev-only-secret-key-change-in-production"


settings = Settings()
