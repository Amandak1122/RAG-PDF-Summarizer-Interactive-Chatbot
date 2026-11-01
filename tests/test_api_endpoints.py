import io
import pytest
from pathlib import Path
from starlette.testclient import TestClient

from app.main import app

client = TestClient(app)

@pytest.fixture
def sample_pdf_path():
    path = Path(__file__).resolve().parent.parent / "sample_documents" / "ai_primer.pdf"
    return path

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "/docs" in data["swagger_docs"]

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "llm_model" in data

def test_upload_invalid_filetype():
    file_content = b"Not a PDF"
    files = {"file": ("test.txt", io.BytesIO(file_content), "text/plain")}
    response = client.post("/api/upload?session_id=test_session", files=files)
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]

def test_upload_valid_pdf(sample_pdf_path):
    with open(sample_pdf_path, "rb") as f:
        files = {"file": ("ai_primer.pdf", f, "application/pdf")}
        response = client.post("/api/upload?session_id=test_session", files=files)

    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "success"
    assert data["total_pages"] == 3
    assert data["total_chunks"] > 0

def test_ask_endpoint_retrieves_citations_safely():
    payload = {
        "question": "What are the advantages of RAG architecture?",
        "top_k": 2
    }
    response = client.post("/api/ask?session_id=test_session", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Citations must be populated from the uploaded sample PDF
    assert len(data["citations"]) > 0
    assert data["citations"][0]["page_number"] in [1, 2, 3]
    # Without API key, answer gracefully explains requirement rather than failing with 500
    assert "answer" in data

def test_summarize_endpoint_safety():
    payload = {"strategy": "direct"}
    response = client.post("/api/summarize?session_id=test_session", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["document_name"] == "ai_primer.pdf"
    assert "executive_summary" in data

def test_reset_endpoint():
    response = client.delete("/api/reset?session_id=test_session")
    assert response.status_code == 200
    assert response.json()["status"] == "success"

    # Subsequent ask on cleared session should return 400
    res = client.post("/api/ask?session_id=test_session", json={"question": "Where is it?"})
    assert res.status_code == 400

