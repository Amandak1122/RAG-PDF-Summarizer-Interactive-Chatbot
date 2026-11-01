import pytest
from pathlib import Path
from app.services.pdf_service import PDFService, PDFProcessingError

@pytest.fixture
def pdf_service():
    return PDFService()

@pytest.fixture
def sample_dir():
    return Path(__file__).resolve().parent.parent / "sample_documents"

def test_sanitize_filename(pdf_service):
    assert pdf_service.sanitize_filename("../malicious/doc.pdf") == "doc.pdf"
    assert pdf_service.sanitize_filename("my file name!.pdf") == "my_file_name_.pdf"
    assert pdf_service.sanitize_filename("sample") == "sample.pdf"

def test_extract_pages_valid_pdf(pdf_service, sample_dir):
    valid_pdf = sample_dir / "ai_primer.pdf"
    pages = pdf_service.extract_pages(valid_pdf)

    assert len(pages) == 3
    assert pages[0]["page_number"] == 1
    assert pages[0]["total_pages"] == 3
    assert "Retrieval-Augmented Generation" in pages[0]["text"]
    assert pages[1]["page_number"] == 2
    assert "Semantic Chunking" in pages[1]["text"]

def test_extract_pages_empty_file(pdf_service, sample_dir):
    empty_pdf = sample_dir / "empty.pdf"
    with pytest.raises(PDFProcessingError, match="empty"):
        pdf_service.extract_pages(empty_pdf)

def test_extract_pages_corrupt_file(pdf_service, sample_dir):
    corrupt_pdf = sample_dir / "corrupt.pdf"
    with pytest.raises(PDFProcessingError):
        pdf_service.extract_pages(corrupt_pdf)

def test_nonexistent_file(pdf_service, sample_dir):
    missing_pdf = sample_dir / "does_not_exist.pdf"
    with pytest.raises(PDFProcessingError, match="not found"):
        pdf_service.extract_pages(missing_pdf)

