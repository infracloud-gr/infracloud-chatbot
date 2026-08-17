from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.history.models import Message, Session as ChatSession
from app.rag.chain import RAGService
from app.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="", tags=["chat"])


def get_rag_service() -> RAGService:
    from app.main import rag_service

    return rag_service


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, db: Session = Depends(get_db), rag: RAGService = Depends(get_rag_service)):
    session = None
    if payload.session_id:
        session = db.get(ChatSession, payload.session_id)
    if session is None:
        session = ChatSession(user_id=payload.user_id, title=payload.message[:80])
        db.add(session)
        db.flush()

    db.add(Message(session_id=session.session_id, role="user", content=payload.message))
    db.flush()

    history_rows = db.execute(
        select(Message).where(Message.session_id == session.session_id).order_by(Message.created_at.asc())
    ).scalars().all()
    history_payload = [{"role": row.role, "content": row.content} for row in history_rows]

    result = rag.ask(payload.message, history_payload, requested_model=payload.model)

    db.add(
        Message(
            session_id=session.session_id,
            role="assistant",
            content=result["answer"],
            sources=result["sources"],
            model_used=result["model_used"],
        )
    )
    db.commit()

    return ChatResponse(
        session_id=session.session_id,
        answer=result["answer"],
        model_used=result["model_used"],
        sources=result["sources"],
    )
