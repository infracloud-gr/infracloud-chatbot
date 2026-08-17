from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str
    user_id: str
    session_id: int | None = None
    model: str | None = Field(default=None, pattern="^(groq|deepseek|qwen)$")


class SourceOut(BaseModel):
    article_id: int
    title: str
    heading_path: str | None = None
    url: str
    thumbnail_url: str | None = None
    description: str | None = None


class ChatResponse(BaseModel):
    session_id: int
    answer: str
    model_used: str
    sources: list[SourceOut]
