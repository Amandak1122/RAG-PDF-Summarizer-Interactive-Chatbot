from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings

class ChunkingService:
    """
    Handles semantic text chunking using LangChain's RecursiveCharacterTextSplitter,
    preserving page-level metadata and eliminating empty or duplicate chunks.
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be strictly less than chunk_size ({self.chunk_size})."
            )

        # Configures prioritized separation boundaries:
        # Paragraphs (\n\n) -> Sentences (\n) -> Words (space) -> Characters ("")
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            keep_separator=True
        )

    def split_pages_into_chunks(self, pages: List[Dict[str, Any]]) -> List[Document]:
        """
        Splits a list of extracted page dictionaries into LangChain Document chunks.
        Preserves exact page numbers for accurate citations.
        Filters empty and duplicate chunks.
        """
        chunk_documents: List[Document] = []
        seen_contents = set()
        global_chunk_idx = 0

        for page in pages:
            raw_text = page.get("text", "").strip()
            if not raw_text:
                continue

            page_number = page.get("page_number", 1)
            total_pages = page.get("total_pages", 1)
            source_file = page.get("source", "document.pdf")

            # Split text within the boundary of this page
            page_chunks = self.splitter.split_text(raw_text)

            for chunk_text in page_chunks:
                clean_chunk = chunk_text.strip()
                # Skip empty chunks or very short noise (< 10 chars)
                if not clean_chunk or len(clean_chunk) < 10:
                    continue

                # Deduplicate identical chunks within document
                if clean_chunk in seen_contents:
                    continue

                seen_contents.add(clean_chunk)
                global_chunk_idx += 1

                doc = Document(
                    page_content=clean_chunk,
                    metadata={
                        "chunk_id": global_chunk_idx,
                        "source": source_file,
                        "page": page_number,
                        "total_pages": total_pages,
                        "character_length": len(clean_chunk)
                    }
                )
                chunk_documents.append(doc)

        return chunk_documents

