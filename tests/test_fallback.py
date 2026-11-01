import io
import pytest
from pathlib import Path
from starlette.testclient import TestClient

from app.main import app

client = TestClient(app)

@pytest.fixture(scope="module")
def setup_test_document():
    """Uploads the sample AI primer PDF to initialize the vector store for testing."""
    sample_pdf_path = Path(__file__).resolve().parent.parent / "sample_documents" / "ai_primer.pdf"
    with open(sample_pdf_path, "rb") as f:
        files = {"file": ("ai_primer.pdf", f, "application/pdf")}
        response = client.post("/api/upload?session_id=fallback_eval_suite", files=files)
    assert response.status_code == 201
    return "fallback_eval_suite"

def test_scenario_1_question_clearly_answered(setup_test_document):
    """Scenario 1: A question clearly answered by the uploaded PDF."""
    session_id = setup_test_document
    payload = {
        "question": "What is Retrieval-Augmented Generation and what are its advantages?",
        "top_k": 3
    }
    response = client.post(f"/api/ask?session_id={session_id}", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["has_sufficient_context"] is True
    assert data["requires_fallback_permission"] is False
    assert data["source_type"] == "pdf"
    assert len(data["citations"]) > 0
    assert "Page 1" in data["answer"] or "hallucination" in data["answer"].lower()

def test_scenario_2_wording_variation_keywords_extraction(setup_test_document):
    """Scenario 2: A question using different wording / meta-synthesis ('Generate top 5 keywords')."""
    session_id = setup_test_document
    payload = {
        "question": "Generate the top 5 relevant keywords from this document.",
        "top_k": 4
    }
    response = client.post(f"/api/ask?session_id={session_id}", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Must answer directly from the PDF without triggering fallback
    assert data["has_sufficient_context"] is True
    assert data["requires_fallback_permission"] is False
    assert data["source_type"] == "pdf"
    assert len(data["citations"]) > 0
    assert "Retrieval-Augmented Generation" in data["answer"] or "FAISS" in data["answer"]

def test_scenario_3_multi_chunk_synthesis(setup_test_document):
    """Scenario 3: A question requiring information synthesized from multiple PDF chunks/pages."""
    session_id = setup_test_document
    payload = {
        "question": "What components are used for chunking, vector storage, and system evaluation?",
        "top_k": 4
    }
    response = client.post(f"/api/ask?session_id={session_id}", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["has_sufficient_context"] is True
    assert data["requires_fallback_permission"] is False
    assert data["source_type"] == "pdf"
    # Citations should span across multiple pages (e.g. Page 2 and Page 3)
    cited_pages = {c["page_number"] for c in data["citations"]}
    assert len(cited_pages) >= 2

def test_scenario_4_completely_unrelated_question(setup_test_document):
    """Scenario 4: A question completely unrelated to the PDF triggers fallback permission."""
    session_id = setup_test_document
    payload = {
        "question": "What is quantum computing and how do qubits achieve superposition?",
        "top_k": 3
    }
    response = client.post(f"/api/ask?session_id={session_id}", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["has_sufficient_context"] is False
    assert data["requires_fallback_permission"] is True
    assert data["source_type"] == "none"
    assert "I couldn't find sufficient information" in data["answer"]
    assert len(data["citations"]) == 0

def test_scenario_5_partially_related_with_insufficient_context(setup_test_document):
    """Scenario 5: Topic mentions AI but asks for specific facts genuinely absent from document."""
    session_id = setup_test_document
    payload = {
        "question": "Who was the lead author and founder who originally published the RAG paper in 2020?",
        "top_k": 3
    }
    response = client.post(f"/api/ask?session_id={session_id}", json=payload)
    assert response.status_code == 200
    data = response.json()

    # The AI primer does not name the authors, so the system triggers fallback permission
    assert data["has_sufficient_context"] is False
    assert data["requires_fallback_permission"] is True
    assert data["source_type"] == "none"

def test_scenario_6_general_knowledge_fallback_after_approval(setup_test_document):
    """Scenario 6: Fallback generation after explicit user approval with disclaimer."""
    session_id = setup_test_document
    question = "Explain how photosynthesis works in green plants."

    # 1. Ask unrelated question -> triggers fallback
    ask_res = client.post(
        f"/api/ask?session_id={session_id}",
        json={"question": question, "top_k": 2}
    )
    assert ask_res.json()["requires_fallback_permission"] is True

    # 2. Approve fallback
    fb_res = client.post(
        f"/api/fallback?session_id={session_id}",
        json={"question": question, "permission_granted": True}
    )
    assert fb_res.status_code == 200
    fb_data = fb_res.json()

    assert fb_data["source_type"] == "general_knowledge"
    assert "Note: This answer is generated using general knowledge" in fb_data["disclaimer"]
    assert len(fb_data["answer"]) > 10

def test_scenario_7_cancellation_without_fallback_generation(setup_test_document):
    """Scenario 7: User cancels fallback; no general knowledge LLM answer is generated."""
    session_id = setup_test_document
    question = "What is the capital of Mars?"

    # 1. Ask unrelated question
    client.post(
        f"/api/ask?session_id={session_id}",
        json={"question": question, "top_k": 2}
    )

    # 2. Cancel fallback
    cancel_res = client.post(
        f"/api/fallback?session_id={session_id}",
        json={"question": question, "permission_granted": False}
    )
    assert cancel_res.status_code == 200
    cancel_data = cancel_res.json()

    assert cancel_data["source_type"] == "cancelled"
    assert "won't generate an answer outside the uploaded PDF" in cancel_data["answer"]
    assert cancel_data["disclaimer"] is None
