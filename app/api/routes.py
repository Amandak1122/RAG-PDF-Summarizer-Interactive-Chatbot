import os
import time
import json
import shutil
import uuid
import hashlib
import logging
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, status, Query
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

from app.config import settings
from app.models import (
    HealthResponse,
    UploadResponse,
    QuestionRequest,
    QuestionResponse,
    FallbackRequest,
    FallbackResponse,
    SummarizeRequest,
    SummarizeResponse,
    ResetResponse,
)
from app.services.pdf_service import (
    PDFService,
    PDFProcessingError,
    PasswordProtectedPDFError,
    ScannedPDFError,
)
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService
from app.services.vector_store_service import VectorStoreService
from app.services.retrieval_service import RetrievalService
from app.services.llm_service import LLMService
from app.services.summarization_service import SummarizationService

router = APIRouter(tags=["RAG Core Endpoints"])

# Initialize Services
pdf_service = PDFService()
chunking_service = ChunkingService()
embedding_service = EmbeddingService()
vector_store_service = VectorStoreService(embedding_service=embedding_service)
retrieval_service = RetrievalService(vector_store_service=vector_store_service)
llm_service = LLMService()
summarization_service = SummarizationService(llm_service=llm_service)

# In-memory document session registry (maps session_id -> state)
# For local use, default session_id is "default_session"
DEFAULT_SESSION_ID = "default_session"

session_state = {
    DEFAULT_SESSION_ID: {
        "filename": None,
        "file_path": None,
        "total_pages": 0,
        "total_chunks": 0,
        "chunks": [],
        "full_text": "",
        "pending_fallback": None
    }
}

def get_current_session(session_id: str = DEFAULT_SESSION_ID) -> dict:
    if session_id not in session_state:
        session_state[session_id] = {
            "filename": None,
            "file_path": None,
            "total_pages": 0,
            "total_chunks": 0,
            "chunks": [],
            "full_text": "",
            "pending_fallback": None
        }
    return session_state[session_id]

@router.get("/health", response_model=HealthResponse)
def get_health(session_id: str = Query(DEFAULT_SESSION_ID)):
    """
    Health check and session diagnostic endpoint.
    Reports API readiness, vector store status, and loaded document metadata.
    """
    session = get_current_session(session_id)
    is_indexed = vector_store_service.is_session_indexed(session_id)
    api_ready = llm_service.is_configured()

    return HealthResponse(
        status="healthy",
        app_name=settings.APP_NAME,
        version=settings.VERSION,
        active_document=session["filename"],
        total_pages=session["total_pages"],
        total_chunks=session["total_chunks"],
        is_indexed=is_indexed,
        api_key_configured=api_ready,
        llm_model=settings.GROQ_MODEL
    )

def _safe_unlink(path: Path) -> None:
    """Safely unlinks a temporary file, suppressing Windows file-lock errors."""
    try:
        if path.exists():
            path.unlink(missing_ok=True)
    except (PermissionError, OSError):
        pass

@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_pdf(
    file: UploadFile = File(..., description="PDF document to process"),
    session_id: str = Query(DEFAULT_SESSION_ID)
):
    """
    Validates, extracts, chunks, and vector-indexes an uploaded PDF document.
    """
    # 1. Validate file extension
    filename = file.filename or "uploaded.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file format. Please upload a PDF file."
        )

    # Fast hash check: prevent re-processing identical uploads in the same session
    file_bytes = await file.read()
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    session = get_current_session(session_id)

    if (
        session.get("file_hash") == file_hash
        and session.get("filename") == filename
        and vector_store_service.is_session_indexed(session_id)
    ):
        logger.info(f"Reusing existing indexed document for session '{session_id}' (file_hash: {file_hash[:8]}).")
        return UploadResponse(
            status="success",
            message=f"Reusing existing indexed chunks for '{filename}' (0ms re-indexing).",
            filename=filename,
            total_pages=session["total_pages"],
            total_chunks=session["total_chunks"],
            file_size_bytes=len(file_bytes)
        )

    # 2. Sanitize filename and save locally
    safe_name = pdf_service.sanitize_filename(filename)
    saved_path = settings.UPLOAD_DIR / f"{session_id}_{safe_name}"

    try:
        with open(saved_path, "wb") as buffer:
            buffer.write(file_bytes)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded file: {str(e)}"
        )

    # 3. Extract text and validate contents
    try:
        pages = pdf_service.extract_pages(saved_path)
        full_text = pdf_service.get_full_document_text(saved_path)
    except PasswordProtectedPDFError as e:
        _safe_unlink(saved_path)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except ScannedPDFError as e:
        _safe_unlink(saved_path)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except PDFProcessingError as e:
        _safe_unlink(saved_path)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    # 4. Chunk text semantically
    try:
        chunks = chunking_service.split_pages_into_chunks(pages)
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Could not split PDF text into valid chunks."
            )
    except Exception as e:
        _safe_unlink(saved_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chunking process failed: {str(e)}"
        )

    # 5. Build FAISS vector store
    try:
        vector_store_service.create_index_for_session(session_id, chunks)
    except Exception as e:
        _safe_unlink(saved_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vector store indexing failed: {str(e)}"
        )

    # 6. Update session state
    session = get_current_session(session_id)
    session["filename"] = safe_name
    session["file_path"] = str(saved_path)
    session["file_hash"] = file_hash
    session["total_pages"] = pages[0]["total_pages"] if pages else 0
    session["total_chunks"] = len(chunks)
    session["chunks"] = chunks
    session["full_text"] = full_text

    file_size = saved_path.stat().st_size

    return UploadResponse(
        status="success",
        message=f"Successfully extracted and indexed {len(chunks)} chunks from '{safe_name}'.",
        filename=safe_name,
        total_pages=session["total_pages"],
        total_chunks=len(chunks),
        file_size_bytes=file_size
    )

@router.post("/ask", response_model=QuestionResponse)
def ask_question(
    payload: QuestionRequest,
    session_id: str = Query(DEFAULT_SESSION_ID)
):
    """
    RAG Question-Answering endpoint.
    Retrieves top_k context chunks from FAISS, evaluates relevance score,
    and returns either a grounded PDF answer or prompts for fallback permission.
    """
    session = get_current_session(session_id)
    if not vector_store_service.is_session_indexed(session_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No document has been uploaded and indexed for this session. Please upload a PDF first."
        )

    t_start = time.perf_counter()

    # Reset any previous pending fallback for this session
    session["pending_fallback"] = None

    # 1. Retrieve context & citations from FAISS
    retrieval_result = retrieval_service.retrieve_context(
        session_id=session_id,
        query=payload.question,
        top_k=payload.top_k
    )

    t_retrieval_end = time.perf_counter()
    retrieval_ms = round((t_retrieval_end - t_start) * 1000, 2)

    citations = retrieval_result["citations"]
    formatted_context = retrieval_result["formatted_context"]
    best_score = retrieval_result.get("best_distance_score")
    has_sufficient_relevance = retrieval_result.get("has_sufficient_relevance", False)

    fallback_prompt_message = (
        "I couldn't find sufficient information related to your question in the uploaded PDF. "
        "Would you like me to generate an answer using the LLM's general knowledge instead?"
    )

    # Check 1: If vector similarity search found no chunks or all scores are above the threshold
    if not has_sufficient_relevance or not retrieval_result["has_context"]:
        total_ms = round((time.perf_counter() - t_start) * 1000, 2)
        logger.info(f"[Ask Route Decision] Question: '{payload.question}' -> Fallback Triggered (Check 1 - Vector Metric in {total_ms}ms). Reason: {retrieval_result.get('relevance_reason')}")
        session["pending_fallback"] = {
            "question": payload.question.strip(),
            "context": formatted_context,
            "is_meta": retrieval_service.is_document_meta_query(payload.question)
        }
        return QuestionResponse(
            question=payload.question,
            answer=fallback_prompt_message,
            citations=[],
            llm_model=settings.GROQ_MODEL,
            has_sufficient_context=False,
            requires_fallback_permission=True,
            source_type="none",
            best_distance_score=best_score,
            latency_ms={"retrieval_ms": retrieval_ms, "llm_ms": 0.0, "total_ms": total_ms}
        )

    # 2. Call LLM Service with retrieved PDF context
    t_llm_start = time.perf_counter()
    llm_res = llm_service.generate_rag_answer(
        question=payload.question,
        context=formatted_context,
        api_key_override=payload.api_key
    )
    t_llm_end = time.perf_counter()
    llm_ms = round((t_llm_end - t_llm_start) * 1000, 2)
    total_ms = round((t_llm_end - t_start) * 1000, 2)

    # Check 2: Even if distance score was within range, check if LLM declared lack of context
    grounded_answer = llm_res["answer"]
    negative_indicators = [
        "cannot find this information",
        "cannot find sufficient information",
        "not mentioned in the provided",
        "not provided in the document",
        "not available in the context",
        "does not contain information",
        "not contain enough information",
        "based on the provided document, i cannot",
        "cannot find any information",
        "unable to find this information"
    ]
    is_negative = any(ind in grounded_answer.lower() for ind in negative_indicators)

    if is_negative:
        logger.info(f"[Ask Route Decision] Question: '{payload.question}' -> Fallback Triggered (Check 2 - LLM Grounding Refusal). Model Output: '{grounded_answer[:120]}'")
        session["pending_fallback"] = {
            "question": payload.question.strip(),
            "context": formatted_context,
            "is_meta": retrieval_service.is_document_meta_query(payload.question)
        }
        return QuestionResponse(
            question=payload.question,
            answer=fallback_prompt_message,
            citations=[],
            llm_model=llm_res["model_used"],
            has_sufficient_context=False,
            requires_fallback_permission=True,
            source_type="none",
            best_distance_score=best_score,
            latency_ms={"retrieval_ms": retrieval_ms, "llm_ms": llm_ms, "total_ms": total_ms}
        )

    logger.info(f"[Ask Route Decision] Question: '{payload.question}' -> Grounded PDF Answer formulated in {total_ms}ms (Retrieval: {retrieval_ms}ms | LLM: {llm_ms}ms) with {len(citations)} citations.")

    # Valid grounded answer found in PDF
    return QuestionResponse(
        question=payload.question,
        answer=grounded_answer,
        citations=citations,
        llm_model=llm_res["model_used"],
        has_sufficient_context=True,
        requires_fallback_permission=False,
        source_type="pdf",
        best_distance_score=best_score,
        latency_ms={"retrieval_ms": retrieval_ms, "llm_ms": llm_ms, "total_ms": total_ms}
    )

@router.post("/ask/stream")
def ask_question_stream(
    payload: QuestionRequest,
    session_id: str = Query(DEFAULT_SESSION_ID)
):
    """
    High-Performance Streaming RAG endpoint using Server-Sent Events (SSE).
    Returns real-time token stream with metadata and citations.
    """
    session = get_current_session(session_id)
    if not vector_store_service.is_session_indexed(session_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No document has been uploaded and indexed for this session."
        )

    session["pending_fallback"] = None
    t0 = time.perf_counter()

    retrieval_result = retrieval_service.retrieve_context(
        session_id=session_id,
        query=payload.question,
        top_k=payload.top_k
    )
    t_retrieval_end = time.perf_counter()
    retrieval_ms = round((t_retrieval_end - t0) * 1000, 2)

    citations = retrieval_result["citations"]
    formatted_context = retrieval_result["formatted_context"]
    best_score = retrieval_result.get("best_distance_score")
    has_sufficient_relevance = retrieval_result.get("has_sufficient_relevance", False)

    fallback_prompt_message = (
        "I couldn't find sufficient information related to your question in the uploaded PDF. "
        "Would you like me to generate an answer using the LLM's general knowledge instead?"
    )

    def event_stream():
        # Check 1: Vector relevance
        if not has_sufficient_relevance or not retrieval_result["has_context"]:
            session["pending_fallback"] = {
                "question": payload.question.strip(),
                "context": formatted_context,
                "is_meta": retrieval_service.is_document_meta_query(payload.question)
            }
            meta_event = {
                "type": "metadata",
                "requires_fallback_permission": True,
                "has_sufficient_context": False,
                "citations": [],
                "source_type": "none",
                "best_distance_score": best_score,
                "latency_ms": {"retrieval_ms": retrieval_ms, "llm_ms": 0.0, "total_ms": retrieval_ms}
            }
            yield f"data: {json.dumps(meta_event)}\n\n"
            yield f"data: {json.dumps({'type': 'token', 'content': fallback_prompt_message})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        # Context is sufficient -> stream tokens from Groq
        meta_event = {
            "type": "metadata",
            "requires_fallback_permission": False,
            "has_sufficient_context": True,
            "citations": [c.model_dump() for c in citations],
            "source_type": "pdf",
            "best_distance_score": best_score,
            "latency_ms": {"retrieval_ms": retrieval_ms}
        }
        yield f"data: {json.dumps(meta_event)}\n\n"

        t_stream_start = time.perf_counter()
        full_tokens = []
        for token in llm_service.generate_rag_answer_stream(
            question=payload.question,
            context=formatted_context,
            api_key_override=payload.api_key
        ):
            full_tokens.append(token)
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        t_stream_end = time.perf_counter()
        llm_ms = round((t_stream_end - t_stream_start) * 1000, 2)
        total_ms = round((t_stream_end - t0) * 1000, 2)

        yield f"data: {json.dumps({'type': 'done', 'latency_ms': {'retrieval_ms': retrieval_ms, 'llm_ms': llm_ms, 'total_ms': total_ms}})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@router.post("/fallback", response_model=FallbackResponse)
def handle_fallback(
    payload: FallbackRequest,
    session_id: str = Query(DEFAULT_SESSION_ID)
):
    """
    Permission-Based LLM Fallback endpoint.
    Requires an active pending fallback permission request for this exact question and session.
    Direct unauthorized calls without a prior pending request are blocked with HTTP 400.
    """
    session = get_current_session(session_id)
    pending = session.get("pending_fallback")

    # Enforce security: must match currently pending question in this session
    if not pending or pending.get("question") != payload.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No pending fallback permission request matches this question or session. Permission must be requested first."
        )

    # Context to pass if question references the document
    pending_context = pending.get("context", "")
    is_meta_ref = pending.get("is_meta", False) or retrieval_service.is_document_meta_query(payload.question)

    # Invalidate pending request immediately (single-use permission)
    session["pending_fallback"] = None

    # Scenario A: User Cancelled
    if not payload.permission_granted:
        return FallbackResponse(
            question=payload.question,
            answer="Okay, I won't generate an answer outside the uploaded PDF.",
            source_type="cancelled",
            llm_model=settings.GROQ_MODEL,
            disclaimer=None
        )

    # Scenario B: User Approved
    doc_context = pending_context if (pending_context and is_meta_ref) else (session.get("full_text", "")[:4000] if is_meta_ref else None)

    t_fb_start = time.perf_counter()
    llm_res = llm_service.generate_general_knowledge_answer(
        question=payload.question,
        document_context=doc_context,
        api_key_override=payload.api_key
    )
    t_fb_end = time.perf_counter()
    fb_llm_ms = round((t_fb_end - t_fb_start) * 1000, 2)

    disclaimer = "Note: This answer is generated using general knowledge and is not supported by the uploaded PDF."

    return FallbackResponse(
        question=payload.question,
        answer=llm_res["answer"],
        source_type="general_knowledge",
        llm_model=llm_res["model_used"],
        disclaimer=disclaimer,
        latency_ms={"retrieval_ms": 0.0, "llm_ms": fb_llm_ms, "total_ms": fb_llm_ms}
    )

@router.post("/summarize", response_model=SummarizeResponse)
def summarize_document(
    payload: Optional[SummarizeRequest] = None,
    session_id: str = Query(DEFAULT_SESSION_ID)
):
    """
    Document Summarization endpoint.
    Executes direct or hierarchical map-reduce summarization based on document length.
    """
    session = get_current_session(session_id)
    if not session["file_path"] or not session["full_text"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No document has been uploaded for this session. Please upload a PDF first."
        )

    strategy = payload.strategy if payload else "auto"
    api_key_override = payload.api_key if payload else None

    summary_result = summarization_service.summarize(
        full_text=session["full_text"],
        chunks=session["chunks"],
        strategy=strategy,
        api_key_override=api_key_override
    )

    return SummarizeResponse(
        document_name=session["filename"] or "document.pdf",
        total_pages=session["total_pages"],
        strategy_used=summary_result["strategy_used"],
        executive_summary=summary_result["executive_summary"],
        key_points=summary_result.get("key_points", []),
        action_items_or_conclusions=summary_result.get("action_items", []),
        raw_summary=summary_result.get("raw_summary", "")
    )

@router.delete("/reset", response_model=ResetResponse)
def reset_session(session_id: str = Query(DEFAULT_SESSION_ID)):
    """Clears uploaded document files, session memory, and FAISS vector index."""
    vector_store_service.clear_session(session_id)

    if session_id in session_state:
        file_path_str = session_state[session_id].get("file_path")
        if file_path_str:
            Path(file_path_str).unlink(missing_ok=True)
        del session_state[session_id]

    return ResetResponse(
        status="success",
        message=f"Session '{session_id}' has been reset successfully."
    )

