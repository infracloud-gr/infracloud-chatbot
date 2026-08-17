from datetime import datetime

from pydantic import BaseModel


class SourceSchema(BaseModel):
    article_id: int
    title: str
    heading_path: str | None = None
    url: str
    thumbnail_url: str | None = None
    description: str | None = None


class MessageRead(BaseModel):
    message_id: int
    role: str
    content: str
    sources: list[SourceSchema] | None = None
    model_used: str | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class SessionRead(BaseModel):
    session_id: int
    user_id: str
    title: str | None
    created_at: datetime

    class Config:
        from_attributes = True
