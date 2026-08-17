from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.history.models import Message, Session as ChatSession
from app.history.schemas import MessageRead, SessionRead

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("", response_model=list[SessionRead])
def list_sessions(user_id: str, db: Session = Depends(get_db)):
    result = db.execute(select(ChatSession).where(ChatSession.user_id == user_id).order_by(ChatSession.created_at.desc()))
    return result.scalars().all()


@router.get("/{session_id}/messages", response_model=list[MessageRead])
def get_messages(session_id: int, db: Session = Depends(get_db)):
    session = db.get(ChatSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    result = db.execute(select(Message).where(Message.session_id == session_id).order_by(Message.created_at.asc()))
    return result.scalars().all()


@router.delete("/{session_id}")
def delete_session(session_id: int, db: Session = Depends(get_db)):
    session = db.get(ChatSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    db.delete(session)
    db.commit()
    return {"deleted": True}
