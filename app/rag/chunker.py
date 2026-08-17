from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


class ParagraphChunker:
    def __init__(self, min_chars: int = 80, max_chars: int = 1200):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
            strip_headers=False,
        )
        self.paragraph_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chars,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    @staticmethod
    def _heading_path(chunk_metadata: dict) -> str | None:
        headings = [chunk_metadata.get("h1"), chunk_metadata.get("h2"), chunk_metadata.get("h3")]
        cleaned = [item.strip() for item in headings if isinstance(item, str) and item.strip()]
        if not cleaned:
            return None
        return " > ".join(cleaned)

    def split(self, markdown: str, metadata: dict) -> list[Document]:
        section_docs = self.header_splitter.split_text(markdown)
        docs: list[Document] = []
        chunk_index = 0

        for section_doc in section_docs:
            section_chunks = self.paragraph_splitter.split_text(section_doc.page_content)
            for text in section_chunks:
                content = text.strip()
                if not content:
                    continue
                if len(content) < self.min_chars and docs:
                    docs[-1].page_content = f"{docs[-1].page_content}\n\n{content}".strip()
                    continue

                section_metadata = {**metadata, **section_doc.metadata}
                section_metadata["heading_path"] = self._heading_path(section_doc.metadata)
                section_metadata["chunk_index"] = chunk_index
                docs.append(Document(page_content=content, metadata=section_metadata))
                chunk_index += 1

        return docs
