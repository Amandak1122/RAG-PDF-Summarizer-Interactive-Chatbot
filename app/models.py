from typing import List, Optional
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    """Schema for GET /health endpoint."""
    status: str = Field(..., json_schema_extra={"example": "healthy"})
    app_name: str
    version: str
    active_document: Optional[str] = None
    total_pages: int = 0
    total_chunks: int = 0
    is_indexed: bool = False
    api_key_configured: bool = False
    llm_model: str

class CitationItem(BaseModel):
    """Schema representing a retrieved context passage citation."""
    citation_id: int
    source_file: str
    page_number: int
    excerpt: str
    relevance_score: float

class QuestionRequest(BaseModel):
    """Schema for POST /ask (or /chat) endpoint."""
    question: str = Field(..., min_length=2, max_length=1000, json_schema_extra={"example": "What are the main findings in the document?"})
    top_k: Optional[int] = Field(default=4, ge=1, le=10, description="Number of context chunks to retrieve")
    api_key: Optional[str] = Field(default=None, description="Optional Groq API key override")

class QuestionResponse(BaseModel):
    """Schema for question answer output."""
    question: str
    answer: str
    citations: List[CitationItem] = []
    llm_model: str
    has_sufficient_context: bool = True
    requires_fallback_permission: bool = False
    source_type: str = "pdf"  # "pdf", "none", "general_knowledge", "cancelled"
    best_distance_score: Optional[float] = None
    latency_ms: Optional[dict] = Field(default=None, description="Diagnostic performance timings in milliseconds")

class FallbackRequest(BaseModel):
    """Schema for POST /fallback endpoint."""
    question: str = Field(..., min_length=2, max_length=1000)
    permission_granted: bool = Field(..., description="True if user approved fallback, False if cancelled")
    api_key: Optional[str] = Field(default=None, description="Optional Groq API key override")

class FallbackResponse(BaseModel):
    """Schema for fallback answer output."""
    question: str
    answer: str
    source_type: str  # "general_knowledge" or "cancelled"
    llm_model: str
    disclaimer: Optional[str] = None
    latency_ms: Optional[dict] = Field(default=None, description="Diagnostic performance timings in milliseconds")

class SummarizeRequest(BaseModel):
    """Schema for POST /summarize endpoint."""
    strategy: Optional[str] = Field(default="auto", description="Summarization strategy: 'direct', 'hierarchical', or 'auto'")
    api_key: Optional[str] = Field(default=None, description="Optional Groq API key override")

class SummarizeResponse(BaseModel):
    """Schema for document summarization output."""
    document_name: str
    total_pages: int
    strategy_used: str
    executive_summary: str
    key_points: List[str] = []
    action_items_or_conclusions: List[str] = []
    raw_summary: str

class UploadResponse(BaseModel):
    """Schema for POST /upload endpoint."""
    status: str
    message: str
    filename: str
    total_pages: int
    total_chunks: int
    file_size_bytes: int

class ResetResponse(BaseModel):
    """Schema for DELETE /reset endpoint."""
    status: str
    message: str

