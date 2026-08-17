from langchain_core.documents import Document


class ParagraphChunker:
    def __init__(self, min_chars: int = 80, max_chars: int = 1200):
        self.min_chars = min_chars
        self.max_chars = max_chars

    def _split_long_paragraph(self, paragraph: str) -> list[str]:
        if len(paragraph) <= self.max_chars:
            return [paragraph]
        sentences = [item.strip() for item in paragraph.split(". ") if item.strip()]
        chunks: list[str] = []
        current = ""
        for sentence in sentences:
            candidate = f"{current}. {sentence}" if current else sentence
            if len(candidate) > self.max_chars and current:
                chunks.append(current.strip())
                current = sentence
            else:
                current = candidate
        if current:
            chunks.append(current.strip())
        return chunks

    def split(self, markdown: str, metadata: dict) -> list[Document]:
        paragraphs = [p.strip() for p in markdown.split("\n\n") if p.strip()]
        merged: list[str] = []
        for paragraph in paragraphs:
            if merged and len(paragraph) < self.min_chars:
                merged[-1] = f"{merged[-1]}\n\n{paragraph}".strip()
            else:
                merged.append(paragraph)

        docs: list[Document] = []
        chunk_index = 0
        for paragraph in merged:
            for text in self._split_long_paragraph(paragraph):
                chunk_metadata = {**metadata, "chunk_index": chunk_index}
                docs.append(Document(page_content=text, metadata=chunk_metadata))
                chunk_index += 1
        return docs
