from collections.abc import Iterator

from langchain_core.documents import Document
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.content.models import Article, Category


class ArticleLoader:
    def __init__(self, db: Session):
        self.db = db

    def load(self, article_ids: list[int] | None = None) -> Iterator[Document]:
        stmt = (
            select(Article, Category)
            .join(Category, Category.category_id == Article.category_id)
            .where(Article.status == "published")
        )
        if article_ids:
            stmt = stmt.where(Article.article_id.in_(article_ids))

        rows = self.db.execute(stmt).all()
        for article, category in rows:
            yield Document(
                page_content=article.content,
                metadata={
                    "article_id": article.article_id,
                    "category_id": category.category_id,
                    "category_name": category.name,
                    "category_slug": category.slug,
                    "article_slug": article.slug,
                    "title": article.title,
                    "thumbnail_url": article.thumbnail_url,
                    "meta_description": article.meta_description,
                    "content_hash": article.content_hash,
                    "updated_at": article.updated_at.isoformat() if article.updated_at else None,
                },
            )
