import os
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from backend.config import settings

class VectorStoreManager:
    """Manages local Sentence Transformers embeddings and FAISS vector indexing/retrieval."""

    def __init__(self, model_name: str = None, index_dir: Path = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.index_dir = index_dir or settings.FAISS_INDEX_DIR
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._vector_store: Optional[FAISS] = None

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Lazy loader for Sentence Transformers model (loads into memory only when needed)."""
        if self._embeddings is None:
            # Runs locally on CPU with normalized embeddings for cosine similarity
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
        return self._embeddings

    def create_and_save_index(self, documents: List[Document]) -> FAISS:
        """Creates a FAISS vector store from chunks and persists it to disk."""
        if not documents:
            raise ValueError("No documents provided for indexing.")

        self._vector_store = FAISS.from_documents(documents, self.embeddings)
        self.save_index()
        return self._vector_store

    def save_index(self) -> None:
        """Saves current FAISS index to the configured directory."""
        if self._vector_store is not None:
            self.index_dir.mkdir(parents=True, exist_ok=True)
            self._vector_store.save_local(str(self.index_dir))

    def load_index(self) -> Optional[FAISS]:
        """Loads FAISS index from disk if present."""
        index_file = self.index_dir / "index.faiss"
        if not index_file.exists():
            return None

        # allow_dangerous_deserialization is required for local pickle files in FAISS
        self._vector_store = FAISS.load_local(
            str(self.index_dir),
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        return self._vector_store

    def similarity_search_with_score(
        self, query: str, top_k: int = None
    ) -> List[Tuple[Document, float]]:
        """
        Retrieves top_k most similar document chunks along with similarity distance score.
        Lower score in L2 distance or higher in cosine indicates closer similarity.
        """
        k = top_k or settings.TOP_K_RESULTS
        if self._vector_store is None:
            loaded = self.load_index()
            if loaded is None:
                return []

        return self._vector_store.similarity_search_with_score(query, k=k)

    def is_indexed(self) -> bool:
        """Checks if a vector index exists in memory or on disk."""
        if self._vector_store is not None:
            return True
        return (self.index_dir / "index.faiss").exists()

    def clear_index(self) -> None:
        """Clears in-memory store and removes saved index files."""
        self._vector_store = None
        if self.index_dir.exists():
            shutil.rmtree(self.index_dir)
            self.index_dir.mkdir(parents=True, exist_ok=True)

