import os
import shutil
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.config import settings
from backend.pdf_processor import PDFProcessor
from backend.vector_store import VectorStoreManager
from backend.rag_engine import RAGEngine

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="FastAPI Backend for RAG-based PDF Summarization and Interactive Q&A"
)

# Enable CORS for Streamlit frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize application singletons
pdf_processor = PDFProcessor()
vector_store = VectorStoreManager()
rag_engine = RAGEngine()

# In-memory document session state
current_document_state = {
    "filename": None,
    "file_path": None,
    "total_pages": 0,
    "total_chunks": 0
}

class ChatRequest(BaseModel):
    query: str
    top_k: Optional[int] = 4
    api_key: Optional[str] = None

class SummarizeRequest(BaseModel):
    api_key: Optional[str] = None

@app.get("/")
def root():
    return {
        "message": "Welcome to RAG PDF Chatbot API",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
def health():
    """Returns system status, index state, and Groq API readiness."""
    return {
        "status": "healthy",
        "has_active_document": current_document_state["filename"] is not None,
        "active_document": current_document_state["filename"],
        "total_pages": current_document_state["total_pages"],
        "total_chunks": current_document_state["total_chunks"],
        "is_vector_indexed": vector_store.is_indexed(),
        "api_key_configured": rag_engine.is_api_key_configured(),
        "model": settings.GROQ_MODEL
    }

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Accepts a PDF file, extracts text, chunks it with LangChain,
    and builds a persistent FAISS vector store.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = settings.UPLOAD_DIR / file.filename
    try:
        # Save uploaded PDF to disk
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1. Extract and chunk text
        chunks = pdf_processor.process_and_chunk(str(file_path))
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No readable text found in PDF. It might be scanned/image-only."
            )

        # 2. Build and persist FAISS index
        vector_store.create_and_save_index(chunks)

        # 3. Update session state
        total_pages = chunks[0].metadata.get("total_pages", 1) if chunks else 0
        current_document_state["filename"] = file.filename
        current_document_state["file_path"] = str(file_path)
        current_document_state["total_pages"] = total_pages
        current_document_state["total_chunks"] = len(chunks)

        return {
            "status": "success",
            "message": f"Successfully indexed '{file.filename}'",
            "filename": file.filename,
            "total_pages": total_pages,
            "total_chunks": len(chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

@app.post("/summarize")
def summarize_document(payload: SummarizeRequest = None):
    """Generates an executive summary of the currently loaded PDF."""
    if not current_document_state["file_path"]:
        raise HTTPException(
            status_code=400,
            detail="No document has been uploaded yet. Please upload a PDF first."
        )

    file_path = current_document_state["file_path"]
    full_text = pdf_processor.get_full_text(file_path)

    api_key_override = payload.api_key if payload else None
    result = rag_engine.generate_summary(
        full_text=full_text,
        override_api_key=api_key_override
    )
    return result

@app.post("/chat")
def chat_with_pdf(payload: ChatRequest):
    """
    Retrieves top_k context chunks from FAISS for the given question
    and queries Groq Llama 3 for a grounded response.
    """
    if not vector_store.is_indexed():
        raise HTTPException(
            status_code=400,
            detail="No PDF has been uploaded and indexed. Please upload a PDF first."
        )

    # 1. Semantic search in FAISS
    top_k = payload.top_k or settings.TOP_K_RESULTS
    retrieved_docs_with_scores = vector_store.similarity_search_with_score(
        query=payload.query,
        top_k=top_k
    )

    if not retrieved_docs_with_scores:
        return {
            "answer": "No relevant context found in the document to answer your question.",
            "citations": [],
            "api_key_present": rag_engine.is_api_key_configured(payload.api_key)
        }

    # 2. Synthesize answer with RAG Engine
    result = rag_engine.answer_question(
        query=payload.query,
        retrieved_docs_with_scores=retrieved_docs_with_scores,
        override_api_key=payload.api_key
    )
    return result

@app.delete("/reset")
def reset_session():
    """Resets the vector index, uploads, and in-memory document state."""
    vector_store.clear_index()
    current_document_state["filename"] = None
    current_document_state["file_path"] = None
    current_document_state["total_pages"] = 0
    current_document_state["total_chunks"] = 0
    return {"status": "success", "message": "Session reset successfully."}

