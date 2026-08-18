import logging
from collections import defaultdict

from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy.orm import Session

from app.rag.chunker import ParagraphChunker
from app.rag.loader import ArticleLoader
from app.rag.vectorstore import VectorStoreService

logger = logging.getLogger(__name__)


class SyncManager:
    def __init__(self, vector_service: VectorStoreService, debounce_seconds: int = 2):
        self.vector_service = vector_service
        self.debounce_seconds = debounce_seconds
        self.pending_updates: dict[int, set[str]] = defaultdict(set)
        self.scheduler = BackgroundScheduler()

    def start(self):
        if not self.scheduler.running:
            self.scheduler.add_job(self.flush, "interval", seconds=self.debounce_seconds)
            self.scheduler.start()

    def mark_article_changed(self, article_id: int, action: str):
        self.pending_updates[article_id].add(action)

    def flush(self):
        if not self.pending_updates:
            return
            
        updates_to_process = self.pending_updates.copy()
        self.pending_updates.clear()
        
        logger.info("Flushing incremental sync for %s articles", len(updates_to_process))
        
        from app.db.database import SessionLocal
        from app.content.models import Article
        
        with SessionLocal() as db:
            chunker = ParagraphChunker()
            loader = ArticleLoader(db)
            
            for article_id, actions in updates_to_process.items():
                try:
                    article = db.get(Article, article_id)
                    if article is None or article.status != "published":
                        self.vector_service.delete_article(article_id)
                        logger.info("Deleted article %s from vector store", article_id)
                    else:
                        article_docs = list(loader.load([article_id]))
                        if not article_docs:
                            self.vector_service.delete_article(article_id)
                            continue
                            
                        article_doc = article_docs[0]
                        chunks = chunker.split(article_doc.page_content, article_doc.metadata)
                        self.vector_service.reindex_article(chunks)
                        logger.info("Reindexed article %s with %s chunks", article_id, len(chunks))
                except Exception as e:
                    logger.error("Failed to sync article %s: %s", article_id, e)

    def full_reindex(self, db: Session):
        loader = ArticleLoader(db)
        chunker = ParagraphChunker()
        by_article: dict[int, list] = defaultdict(list)
        for article_doc in loader.load():
            chunks = chunker.split(article_doc.page_content, article_doc.metadata)
            by_article[article_doc.metadata["article_id"]].extend(chunks)

        for _, docs in by_article.items():
            self.vector_service.reindex_article(docs)
