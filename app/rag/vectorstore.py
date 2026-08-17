import uuid

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.core.config import Settings


class VectorStoreService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        self.embeddings = self._build_embeddings()

    def _build_embeddings(self):
        if self.settings.embedding_provider == "qwen_api":
            return OpenAIEmbeddings(
                api_key=self.settings.qwen_api_key,
                model="text-embedding-v3",
                base_url=self.settings.qwen_base_url,
            )
        return HuggingFaceEmbeddings(model_name=self.settings.local_embedding_model)

    def ensure_collection(self):
        collections = [c.name for c in self.client.get_collections().collections]
        if self.settings.qdrant_collection_name in collections:
            return
        self.client.create_collection(
            collection_name=self.settings.qdrant_collection_name,
            vectors_config=models.VectorParams(size=self.settings.embedding_dimension, distance=models.Distance.COSINE),
        )

    def get_store(self) -> QdrantVectorStore:
        self.ensure_collection()
        return QdrantVectorStore(
            client=self.client,
            collection_name=self.settings.qdrant_collection_name,
            embedding=self.embeddings,
        )

    def reindex_article(self, documents):
        store = self.get_store()
        article_id = documents[0].metadata["article_id"] if documents else None
        if article_id is not None:
            self.client.delete(
                collection_name=self.settings.qdrant_collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[models.FieldCondition(key="metadata.article_id", match=models.MatchValue(value=article_id))]
                    )
                ),
            )
        if not documents:
            return
        ids = [str(uuid.uuid4()) for _ in documents]
        store.add_documents(documents=documents, ids=ids)

    def delete_article(self, article_id: int):
        self.client.delete(
            collection_name=self.settings.qdrant_collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="metadata.article_id", match=models.MatchValue(value=article_id))]
                )
            ),
        )
