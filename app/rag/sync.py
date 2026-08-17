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
        logger.info("Flushing incremental sync for %s articles", len(self.pending_updates))

    def full_reindex(self, db: Session):
        loader = ArticleLoader(db)
        chunker = ParagraphChunker()
        by_article: dict[int, list] = defaultdict(list)
        for article_doc in loader.load():
            chunks = chunker.split(article_doc.page_content, article_doc.metadata)
            by_article[article_doc.metadata["article_id"]].extend(chunks)

        for _, docs in by_article.items():
            self.vector_service.reindex_article(docs)
