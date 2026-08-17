from fastapi import FastAPI
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi import _rate_limit_exceeded_handler

from app.api import admin, chat, sessions
from app.core.config import get_settings
from app.core.rate_limiter import limiter
from app.db.database import Base, engine
from app.rag.chain import RAGService
from app.rag.sync import SyncManager
from app.rag.vectorstore import VectorStoreService

settings = get_settings()

app = FastAPI(title="InfraCloud RAG Chatbot", version="0.1.0")
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.include_router(chat.router)
app.include_router(sessions.router)
app.include_router(admin.router)

rag_service = RAGService(settings)
sync_manager = SyncManager(VectorStoreService(settings), debounce_seconds=settings.sync_debounce_seconds)


@app.on_event("startup")
def startup_event():
    Base.metadata.create_all(bind=engine)
    sync_manager.start()


@app.get("/healthz")
def healthz():
    return {"status": "ok"}
