from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.content.models import Article, Category
from app.db.database import get_db
from app.rag.sync import SyncManager

router = APIRouter(prefix="/admin", tags=["admin"])


def get_sync_manager() -> SyncManager:
    from app.main import sync_manager

    return sync_manager


@router.post("/reindex")
def reindex(db: Session = Depends(get_db), manager: SyncManager = Depends(get_sync_manager)):
    manager.full_reindex(db)
    return {"status": "ok"}


@router.get("/index-status")
def index_status(db: Session = Depends(get_db)):
    articles_count = db.execute(select(func.count(Article.article_id)).where(Article.status == "published")).scalar_one()
    categories_count = db.execute(select(func.count(Category.category_id))).scalar_one()
    return {
        "published_articles": articles_count,
        "categories": categories_count,
    }

def get_rag_service():
    from app.main import rag_service
    return rag_service

@router.get("/llm-health")
def llm_health(rag = Depends(get_rag_service)):
    from langchain_core.messages import HumanMessage
    models = rag.router.resolve_chain()
    results = {}
    for llm, provider in models:
        try:
            llm.invoke([HumanMessage(content="Hello, this is a health check. Please reply 'OK'.")])
            results[provider] = "ok"
        except Exception as e:
            results[provider] = f"error: {str(e)}"
    
    overall_status = "ok" if any(v == "ok" for v in results.values()) else "error"
    return {"status": overall_status, "providers": results}
