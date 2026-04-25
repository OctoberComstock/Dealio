from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    tavily_api_key: str = ""
    anthropic_api_key: str = ""

    max_searches: int = 5
    max_fetched_pages: int = 8
    request_timeout_seconds: int = 45

    search_cache_ttl_seconds: int = 900
    page_cache_ttl_seconds: int = 900
    parsed_content_cache_ttl_seconds: int = 900

    database_path: str = "./dealio.db"


settings = Settings()
