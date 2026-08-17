from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)

    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")

    deepseek_api_key: str | None = Field(default=None, alias="DEEPSEEK_API_KEY")
    deepseek_model: str = Field(default="deepseek-chat", alias="DEEPSEEK_MODEL")
    deepseek_base_url: str = Field(default="https://api.deepseek.com/v1", alias="DEEPSEEK_BASE_URL")

    qwen_api_key: str | None = Field(default=None, alias="QWEN_API_KEY")
    qwen_model: str = Field(default="qwen-plus", alias="QWEN_MODEL")
    qwen_base_url: str = Field(
        default="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        alias="QWEN_BASE_URL",
    )

    default_chat_provider: Literal["groq", "deepseek", "qwen"] = Field(alias="DEFAULT_CHAT_PROVIDER")
    chat_fallback_order: str = Field(default="groq", alias="CHAT_FALLBACK_ORDER")

    embedding_provider: Literal["qwen_api", "local"] = Field(alias="EMBEDDING_PROVIDER")
    local_embedding_model: str = Field(default="intfloat/multilingual-e5-base", alias="LOCAL_EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=768, alias="EMBEDDING_DIMENSION")

    database_url: str = Field(alias="DATABASE_URL")
    qdrant_url: str = Field(alias="QDRANT_URL")
    qdrant_api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")
    qdrant_collection_name: str = Field(default="knowledge_base", alias="QDRANT_COLLECTION_NAME")
    redis_url: str = Field(alias="REDIS_URL")

    rate_limit_per_minute: int = Field(default=20, alias="RATE_LIMIT_PER_MINUTE")
    rate_limit_tokens_per_day: int = Field(default=50000, alias="RATE_LIMIT_TOKENS_PER_DAY")

    site_base_url: str = Field(alias="SITE_BASE_URL")
    env: Literal["development", "staging", "production"] = Field(default="production", alias="ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    sync_debounce_seconds: int = Field(default=2, alias="SYNC_DEBOUNCE_SECONDS")

    @property
    def fallback_providers(self) -> list[str]:
        return [provider.strip() for provider in self.chat_fallback_order.split(",") if provider.strip()]

    @field_validator("embedding_provider")
    @classmethod
    def validate_embedding_provider_for_available_keys(cls, value: str, info):
        data = info.data
        if value == "qwen_api" and not data.get("qwen_api_key"):
            raise ValueError("QWEN_API_KEY is required when EMBEDDING_PROVIDER=qwen_api")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
