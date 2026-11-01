import pytest
import tempfile
from pathlib import Path
from langchain_core.documents import Document

from app.services.embedding_service import EmbeddingService
from app.services.vector_store_service import VectorStoreService

@pytest.fixture(scope="module")
def embedding_service():
    # Use real embedding service to verify embedding model compatibility
    return EmbeddingService()

@pytest.fixture
def temp_vector_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

def test_vector_store_lifecycle_and_isolation(embedding_service, temp_vector_dir):
    store_service = VectorStoreService(embedding_service=embedding_service, base_dir=temp_vector_dir)

    # Prepare chunks for Session 1 (Machine Learning)
    docs_session_1 = [
        Document(page_content="Supervised learning involves training models on labeled datasets.", metadata={"page": 1, "source": "ml.pdf"}),
        Document(page_content="Neural networks are composed of layers of interconnected nodes.", metadata={"page": 2, "source": "ml.pdf"})
    ]

    # Prepare chunks for Session 2 (Cooking)
    docs_session_2 = [
        Document(page_content="To make pasta, boil water and add salt before adding pasta.", metadata={"page": 1, "source": "cooking.pdf"})
    ]

    # Index both sessions separately
    store_service.create_index_for_session("session_ml", docs_session_1)
    store_service.create_index_for_session("session_cook", docs_session_2)

    # 1. Verify Session 1 query
    results_ml = store_service.similarity_search_with_score("session_ml", "What is supervised learning?", top_k=1)
    assert len(results_ml) == 1
    assert "Supervised learning" in results_ml[0][0].page_content
    assert results_ml[0][0].metadata["page"] == 1

    # 2. Verify Session Isolation: session_cook should NOT match machine learning
    results_cook = store_service.similarity_search_with_score("session_cook", "What is supervised learning?", top_k=1)
    assert len(results_cook) == 1
    assert "pasta" in results_cook[0][0].page_content

    # 3. Test Clearing
    store_service.clear_session("session_ml")
    assert not store_service.is_session_indexed("session_ml")

