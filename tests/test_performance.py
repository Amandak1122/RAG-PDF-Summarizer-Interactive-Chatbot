import time
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from app.services.embedding_service import EmbeddingService
from app.services.vector_store_service import VectorStoreService
from app.services.retrieval_service import RetrievalService
from app.services.llm_service import LLMService

@pytest.fixture
def sample_documents():
    return [
        Document(page_content="Retrieval-Augmented Generation combines search with LLMs.", metadata={"page": 1, "source": "doc.pdf"}),
        Document(page_content="Vector databases like FAISS store dense embeddings.", metadata={"page": 2, "source": "doc.pdf"}),
        Document(page_content="LangChain provides modular components for RAG pipelines.", metadata={"page": 3, "source": "doc.pdf"})
    ]

def test_query_embedding_cache():
    """Verify that query embedding cache returns cached vectors without re-computing."""
    service = EmbeddingService()
    query = "What is Retrieval-Augmented Generation?"

    # Mock the internal model to count calls
    mock_model = MagicMock()
    mock_model.embed_query.return_value = [0.1, 0.2, 0.3]
    service._embeddings_model = mock_model

    # First call: computes embedding
    v1 = service.embed_query(query)
    assert v1 == [0.1, 0.2, 0.3]
    assert mock_model.embed_query.call_count == 1

    # Second call with identical query: served from cache (0 model calls)
    v2 = service.embed_query(query)
    assert v2 == [0.1, 0.2, 0.3]
    assert mock_model.embed_query.call_count == 1  # Not called again!

def test_chunk_embedding_cache():
    """Verify that chunk embeddings are cached and not re-computed for identical passages."""
    service = EmbeddingService()
    chunks = [
        "First chunk of document text.",
        "Second chunk of document text."
    ]

    mock_model = MagicMock()
    mock_model.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]
    service._embeddings_model = mock_model

    # First batch
    res1 = service.embed_documents(chunks)
    assert len(res1) == 2
    assert mock_model.embed_documents.call_count == 1

    # Re-embedding same chunks: served 100% from cache
    res2 = service.embed_documents(chunks)
    assert len(res2) == 2
    assert res1 == res2
    # embed_documents was NOT called again
    assert mock_model.embed_documents.call_count == 1

def test_retrieval_filters_irrelevant_chunks():
    """Verify that retrieval filters out distant chunks for specific questions."""
    embedding_service = EmbeddingService()
    vector_store = VectorStoreService(embedding_service)

    # Mock similarity search with scores:
    # Chunk 1 (highly relevant: 0.75), Chunk 2 (relevant: 0.85), Chunk 3 (distant: 1.55), Chunk 4 (irrelevant: 1.70)
    mock_matches = [
        (Document(page_content="RAG reduces hallucinations.", metadata={"page": 1, "source": "doc.pdf"}), 0.75),
        (Document(page_content="FAISS enables fast semantic search.", metadata={"page": 2, "source": "doc.pdf"}), 0.85),
        (Document(page_content="Photosynthesis in plants.", metadata={"page": 3, "source": "doc.pdf"}), 1.55),
        (Document(page_content="Quantum computing mechanics.", metadata={"page": 4, "source": "doc.pdf"}), 1.70),
    ]

    vector_store.similarity_search_with_score = MagicMock(return_value=mock_matches)
    retrieval_service = RetrievalService(vector_store)

    res = retrieval_service.retrieve_context(
        session_id="test_sess",
        query="How does RAG reduce hallucinations?",
        top_k=4
    )

    # Only relevant chunks should be retained in citations and context
    assert len(res["citations"]) == 2
    assert res["citations"][0].relevance_score == 0.75
    assert res["citations"][1].relevance_score == 0.85
    # Irrelevant chunks (1.55 and 1.70) were pruned
    assert "Photosynthesis" not in res["formatted_context"]
    assert "Quantum" not in res["formatted_context"]

def test_out_of_domain_fast_rejection():
    """Verify that completely out-of-domain queries are rejected without LLM invocation."""
    embedding_service = EmbeddingService()
    vector_store = VectorStoreService(embedding_service)

    # Out of domain chunks with distance > 1.50
    mock_matches = [
        (Document(page_content="Business analytics overview.", metadata={"page": 1, "source": "doc.pdf"}), 1.62),
        (Document(page_content="Office location details.", metadata={"page": 2, "source": "doc.pdf"}), 1.71),
    ]

    vector_store.similarity_search_with_score = MagicMock(return_value=mock_matches)
    retrieval_service = RetrievalService(vector_store)

    res = retrieval_service.retrieve_context(
        session_id="test_sess",
        query="What is deep learning and how do CNNs work?",
        top_k=4
    )

    assert res["has_sufficient_relevance"] is False
    assert res["best_distance_score"] == 1.62

def test_llm_streaming_generator():
    """Verify that LLM streaming generator yields token chunks correctly."""
    llm_service = LLMService()

    # Mock Groq client and streaming completion
    mock_chunk1 = MagicMock()
    mock_chunk1.choices = [MagicMock(delta=MagicMock(content="Retrieval "))]
    mock_chunk2 = MagicMock()
    mock_chunk2.choices = [MagicMock(delta=MagicMock(content="Augmented "))]
    mock_chunk3 = MagicMock()
    mock_chunk3.choices = [MagicMock(delta=MagicMock(content="Generation."))]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = [mock_chunk1, mock_chunk2, mock_chunk3]
    llm_service.get_client = MagicMock(return_value=mock_client)

    tokens = list(llm_service.generate_rag_answer_stream(
        question="What is RAG?",
        context="RAG context here."
    ))

    assert tokens == ["Retrieval ", "Augmented ", "Generation."]

