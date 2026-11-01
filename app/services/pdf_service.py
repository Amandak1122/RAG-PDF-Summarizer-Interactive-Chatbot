import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import pymupdf  # Official PyMuPDF API (replacing deprecated import fitz)

from app.config import settings

class PDFProcessingError(Exception):
    """Base exception for PDF parsing and extraction errors."""
    pass

class PasswordProtectedPDFError(PDFProcessingError):
    """Raised when an uploaded PDF requires a password."""
    pass

class ScannedPDFError(PDFProcessingError):
    """Raised when an uploaded PDF contains no machine-readable text."""
    pass

class PDFService:
    """Handles PDF validation, safe file handling, and page-by-page text extraction."""

    def __init__(self, max_file_size_bytes: Optional[int] = None):
        self.max_file_size_bytes = max_file_size_bytes or settings.MAX_FILE_SIZE_BYTES

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Removes dangerous characters, path traversal indicators (../), and whitespace
        to ensure safe file storage on Windows and Linux.
        """
        clean_name = Path(filename).name  # Strips directory prefixes
        # Replace non-alphanumeric characters (except dots, underscores, hyphens)
        clean_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', clean_name)
        if not clean_name.lower().endswith(".pdf"):
            clean_name += ".pdf"
        return clean_name

    def validate_file(self, file_path: Path) -> None:
        """Validates file existence, format, and size limit."""
        if not file_path.exists():
            raise PDFProcessingError(f"File not found: {file_path}")

        file_size = file_path.stat().st_size
        if file_size == 0:
            raise PDFProcessingError("The uploaded PDF file is empty (0 bytes).")

        if file_size > self.max_file_size_bytes:
            mb_limit = self.max_file_size_bytes // (1024 * 1024)
            raise PDFProcessingError(f"File size exceeds the {mb_limit} MB limit.")

    def extract_pages(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Extracts text and page metadata page-by-page using PyMuPDF.

        Raises:
            PasswordProtectedPDFError: If the document is encrypted.
            ScannedPDFError: If no extractable text is found (scanned/image-only PDF).
            PDFProcessingError: For corrupted or invalid PDF structures.
        """
        self.validate_file(file_path)

        doc = None
        try:
            try:
                doc = pymupdf.open(str(file_path))
            except Exception as e:
                raise PDFProcessingError(f"Could not open or parse PDF file: {str(e)}")

            # Check for encryption / password protection
            if doc.is_encrypted or doc.needs_pass:
                raise PasswordProtectedPDFError(
                    "This PDF document is encrypted or password-protected. "
                    "Please provide an unencrypted PDF."
                )

            total_pages = len(doc)
            if total_pages == 0:
                raise PDFProcessingError("The PDF contains 0 pages.")

            extracted_pages: List[Dict[str, Any]] = []
            total_chars = 0

            for page_idx in range(total_pages):
                page = doc[page_idx]
                page_text = page.get_text("text").strip()
                char_count = len(page_text)
                total_chars += char_count

                if page_text:
                    extracted_pages.append({
                        "page_number": page_idx + 1,
                        "total_pages": total_pages,
                        "source": file_path.name,
                        "char_count": char_count,
                        "text": page_text
                    })

            # Detect image-only or scanned PDFs
            if total_chars < 15 or len(extracted_pages) == 0:
                raise ScannedPDFError(
                    "No machine-readable text was found in this PDF. "
                    "It appears to be a scanned document or contains only raster images. "
                    "An Optical Character Recognition (OCR) engine is required to read image-only PDFs."
                )

            return extracted_pages

        finally:
            if doc is not None:
                try:
                    doc.close()
                except Exception:
                    pass

    def get_full_document_text(self, file_path: Path) -> str:
        """Returns concatenated text of all pages for whole-document processing."""
        pages = self.extract_pages(file_path)
        return "\n\n".join(page["text"] for page in pages)

