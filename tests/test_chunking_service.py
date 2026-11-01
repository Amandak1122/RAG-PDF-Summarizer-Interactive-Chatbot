import pytest
from app.services.chunking_service import ChunkingService

def test_chunking_service_valid_split():
    service = ChunkingService(chunk_size=200, chunk_overlap=40)
    sample_pages = [
        {
            "page_number": 1,
            "total_pages": 2,
            "source": "test.pdf",
            "text": "This is page one. " * 30
        },
        {
            "page_number": 2,
            "total_pages": 2,
            "source": "test.pdf",
            "text": "This is page two. " * 30
        }
    ]

    chunks = service.split_pages_into_chunks(sample_pages)

    assert len(chunks) > 0
    # Verify page metadata preserved
    page_numbers = {c.metadata["page"] for c in chunks}
    assert 1 in page_numbers
    assert 2 in page_numbers
    assert chunks[0].metadata["source"] == "test.pdf"

def test_chunking_invalid_overlap():
    with pytest.raises(ValueError, match="must be strictly less"):
        ChunkingService(chunk_size=100, chunk_overlap=120)

def test_chunking_filters_empty():
    service = ChunkingService(chunk_size=200, chunk_overlap=20)
    empty_pages = [
        {"page_number": 1, "total_pages": 1, "source": "test.pdf", "text": "    \n\n  "}
    ]
    chunks = service.split_pages_into_chunks(empty_pages)
    assert len(chunks) == 0

