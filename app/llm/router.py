from collections.abc import Iterable

from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from app.core.config import Settings


class LLMRouter:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _build_model(self, provider: str) -> BaseChatModel:
        if provider == "groq":
            if not self.settings.groq_api_key:
                raise ValueError("GROQ_API_KEY is not configured")
            return ChatGroq(api_key=self.settings.groq_api_key, model=self.settings.groq_model, temperature=0)
        if provider == "deepseek":
            if not self.settings.deepseek_api_key:
                raise ValueError("DEEPSEEK_API_KEY is not configured")
            return ChatOpenAI(
                api_key=self.settings.deepseek_api_key,
                model=self.settings.deepseek_model,
                base_url=self.settings.deepseek_base_url,
                temperature=0,
            )
        if provider == "qwen":
            if not self.settings.qwen_api_key:
                raise ValueError("QWEN_API_KEY is not configured")
            return ChatOpenAI(
                api_key=self.settings.qwen_api_key,
                model=self.settings.qwen_model,
                base_url=self.settings.qwen_base_url,
                temperature=0,
            )
        raise ValueError(f"Unsupported provider: {provider}")

    def resolve_chain(self, requested_model: str | None = None) -> list[tuple[BaseChatModel, str]]:
        fallback_order: Iterable[str] = [requested_model] if requested_model else [self.settings.default_chat_provider]
        fallback_order = list(dict.fromkeys([*fallback_order, *self.settings.fallback_providers]))

        models: list[tuple[BaseChatModel, str]] = []
        for provider in fallback_order:
            if not provider:
                continue
            try:
                models.append((self._build_model(provider), provider))
            except Exception:
                continue
        
        if not models:
            raise RuntimeError("No chat provider is available due to missing configuration.")
        
        return models
