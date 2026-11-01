import logging
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from app.config import settings
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

class VectorStoreService:
    """
    Manages FAISS vector indexing, session-scoped index persistence,
    and semantic similarity search.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        base_dir: Optional[Path] = None
    ):
        self.embedding_service = embedding_service
        self.base_dir = base_dir or settings.VECTOR_STORE_DIR
        self._active_stores: dict[str, FAISS] = {}

    def get_session_dir(self, session_id: str) -> Path:
        """Returns the specific directory path for a session's index."""
        # Sanitize session_id to avoid path traversal
        safe_session_id = "".join(c for c in session_id if c.isalnum() or c in ("-", "_"))
        return self.base_dir / safe_session_id

    def create_index_for_session(
        self,
        session_id: str,
        documents: List[Document]
    ) -> FAISS:
        """
        Builds a new FAISS index from chunks and persists it in the session-scoped directory.
        Prevents mixing content between separate documents or users.
        """
        if not documents:
            raise ValueError("Cannot create vector index: document chunk list is empty.")

        vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embedding_service
        )

        session_path = self.get_session_dir(session_id)
        session_path.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(str(session_path))

        self._active_stores[session_id] = vector_store
        logger.info(f"Successfully indexed {len(documents)} chunks for session '{session_id}'.")
        return vector_store

    def load_index_for_session(self, session_id: str) -> Optional[FAISS]:
        """Loads a persisted FAISS index for the given session."""
        if session_id in self._active_stores:
            return self._active_stores[session_id]

        session_path = self.get_session_dir(session_id)
        index_file = session_path / "index.faiss"

        if not index_file.exists():
            return None

        vector_store = FAISS.load_local(
            folder_path=str(session_path),
            embeddings=self.embedding_service,
            allow_dangerous_deserialization=True  # Scoped strictly to our own local storage dir
        )
        self._active_stores[session_id] = vector_store
        return vector_store

    def similarity_search_with_score(
        self,
        session_id: str,
        query: str,
        top_k: int = 4
    ) -> List[Tuple[Document, float]]:
        """
        Searches the session's index for chunks most similar to the query.
        Returns tuples of (Document, distance_score).
        """
        store = self.load_index_for_session(session_id)
        if store is None:
            return []

        return store.similarity_search_with_score(query=query, k=top_k)

    def is_session_indexed(self, session_id: str) -> bool:
        """Checks if a vector index exists for the specified session."""
        if session_id in self._active_stores:
            return True
        session_path = self.get_session_dir(session_id)
        return (session_path / "index.faiss").exists()

    def clear_session(self, session_id: str) -> None:
        """Removes the session's vector store from memory and deletes its files on disk."""
        if session_id in self._active_stores:
            del self._active_stores[session_id]

        session_path = self.get_session_dir(session_id)
        if session_path.exists():
            shutil.rmtree(session_path)
            logger.info(f"Cleared vector store files for session '{session_id}'.")

