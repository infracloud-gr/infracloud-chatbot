import logging
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from app.core.config import Settings
from app.llm.router import LLMRouter
from app.rag.source_links import build_source
from app.rag.vectorstore import VectorStoreService

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.router = LLMRouter(settings)
        self.vector_service = VectorStoreService(settings)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Bạn là trợ lý chỉ trả lời dựa trên ngữ cảnh cung cấp. Nếu thiếu thông tin, hãy nói rõ không đủ dữ liệu.",
                ),
                (
                    "human",
                    "Lịch sử hội thoại:\n{history}\n\nNgữ cảnh:\n{context}\n\nCâu hỏi: {question}",
                ),
            ]
        )

    def retrieve(self, question: str, k: int = 4) -> list[Document]:
        store = self.vector_service.get_store()
        return store.similarity_search(question, k=k)

    @staticmethod
    def _format_history(history: list[dict]) -> str:
        return "\n".join(f"{item['role']}: {item['content']}" for item in history[-10:])

    def ask(self, question: str, history: list[dict], requested_model: str | None = None) -> dict:
        docs = self.retrieve(question)
        context = "\n\n".join(doc.page_content for doc in docs)
        
        models = self.router.resolve_chain(requested_model)
        messages = self.prompt.format_messages(
            history=self._format_history(history),
            context=context,
            question=question,
        )
        
        answer = None
        used_provider = None
        last_error = None
        
        for llm, provider in models:
            try:
                answer = llm.invoke(messages).content
                used_provider = provider
                break
            except Exception as e:
                logger.warning("Provider %s failed with error: %s", provider, e)
                last_error = e
                continue
                
        if answer is None:
            raise RuntimeError(f"All chat providers failed. Last error: {last_error}")

        deduplicated: dict[int, dict] = {}
        for doc in docs:
            source = build_source(doc.metadata, self.settings.site_base_url)
            deduplicated[source["article_id"]] = source

        return {
            "answer": answer,
            "model_used": used_provider,
            "sources": list(deduplicated.values()),
        }
