import os
from pathlib import Path
from typing import List, Dict, Any
import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from backend.config import settings

class PDFProcessor:
    """Handles PDF text extraction and semantic chunking with metadata tracking."""

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def extract_text_with_metadata(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extracts text from each page of a PDF using PyMuPDF.
        Returns a list of dictionaries with page numbers and raw text.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found at: {file_path}")

        extracted_pages = []
        doc = fitz.open(file_path)

        try:
            total_pages = len(doc)
            for page_index in range(total_pages):
                page = doc[page_index]
                text = page.get_text("text").strip()
                if text:
                    extracted_pages.append({
                        "page_number": page_index + 1,
                        "total_pages": total_pages,
                        "source": path.name,
                        "text": text
                    })
        finally:
            doc.close()

        return extracted_pages

    def process_and_chunk(self, file_path: str) -> List[Document]:
        """
        Extracts text page by page and chunks it with LangChain's RecursiveCharacterTextSplitter.
        Retains source file name and page number metadata for every chunk (crucial for citations).
        """
        pages = self.extract_text_with_metadata(file_path)
        documents: List[Document] = []

        for page_data in pages:
            raw_text = page_data["text"]
            # Split text within the page
            chunks = self.text_splitter.split_text(raw_text)

            for chunk_idx, chunk_text in enumerate(chunks):
                doc = Document(
                    page_content=chunk_text,
                    metadata={
                        "source": page_data["source"],
                        "page": page_data["page_number"],
                        "total_pages": page_data["total_pages"],
                        "chunk_index": chunk_idx
                    }
                )
                documents.append(doc)

        return documents

    def get_full_text(self, file_path: str) -> str:
        """Returns the complete raw text of the document for whole-document summarization."""
        pages = self.extract_text_with_metadata(file_path)
        return "\n\n".join(p["text"] for p in pages)

