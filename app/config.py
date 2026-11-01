import os
from pathlib import Path
from dotenv import load_dotenv

# Locate project root directory (parent of 'app')
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load environment variables from .env
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)

class Settings:
    """Application-wide configuration settings."""

    # Project Information
    APP_NAME: str = "RAG PDF Summarizer & Chatbot"
    VERSION: str = "1.0.0"

    # LLM Settings (Groq)
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "").strip()
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b").strip()

    # Embedding Settings (Local Sentence Transformers)
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2").strip()

    # Chunking Parameters
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))

    # Retrieval Parameters
    DEFAULT_TOP_K: int = int(os.getenv("TOP_K_RESULTS", "4"))
    SIMILARITY_DISTANCE_THRESHOLD: float = float(os.getenv("SIMILARITY_DISTANCE_THRESHOLD", "1.45"))
    SIMILARITY_COSINE_MIN: float = float(os.getenv("SIMILARITY_COSINE_MIN", "-0.05"))

    # File and Vector Store Paths
    STORAGE_DIR: Path = PROJECT_ROOT / "data"
    UPLOAD_DIR: Path = PROJECT_ROOT / "data" / "uploads"
    VECTOR_STORE_DIR: Path = PROJECT_ROOT / "data" / "faiss_index"
    SAMPLE_DOCS_DIR: Path = PROJECT_ROOT / "sample_documents"

    # Upload Constraints (50 MB limit, PDF only)
    MAX_FILE_SIZE_BYTES: int = 50 * 1024 * 1024  # 50 MB

    def ensure_directories(self) -> None:
        """Creates necessary runtime directories on startup."""
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
        self.SAMPLE_DOCS_DIR.mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.ensure_directories()

