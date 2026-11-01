import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

class Settings:
    PROJECT_NAME: str = "RAG PDF Summarizer & Chatbot"
    VERSION: str = "1.0.0"

    # Groq API Configuration
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "").strip()
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()

    # RAG & Chunking Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "4"))

    # File and Vector Storage Paths
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    FAISS_INDEX_DIR: Path = BASE_DIR / "faiss_index"

    @classmethod
    def ensure_directories(cls):
        """Creates necessary folders if they don't exist."""
        cls.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        cls.FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.ensure_directories()

