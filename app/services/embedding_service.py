import os
import gc
import hashlib
import logging
import threading
from typing import List, Optional
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from app.config import settings

logger = logging.getLogger(__name__)

class EmbeddingService(Embeddings):
    """
    Manages local text embeddings using Sentence Transformers (e.g. all-MiniLM-L6-v2).
    Generates normalized dense vector embeddings for both chunks and queries.
    Features:
    - In-memory LRU query embedding cache (0ms for repeated/similar queries).
    - Chunk text content hash cache (avoids re-embedding identical document passages).
    - Memory-optimized batching (batch_size=16) and single-thread execution for cloud hosting limits (Render Free 512MB).
    - Thread-safe lazy initialization to prevent startup OpenMP deadlocks and OOM.
    - Inherits from langchain_core.embeddings.Embeddings for direct vectorstore integration.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self._embeddings_model: Optional[HuggingFaceEmbeddings] = None
        self._lock = threading.Lock()
        self._query_cache: dict[str, List[float]] = {}
        self._chunk_cache: dict[str, List[float]] = {}
        self._max_query_cache_size: int = 1024
        self._max_chunk_cache_size: int = 4096

    @staticmethod
    def _hash_text(text: str) -> str:
        """Computes fast MD5 hash of text for caching."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    @property
    def model(self) -> HuggingFaceEmbeddings:
        """
        Loads the embedding model lazily on first inference request.
        Restricts PyTorch to single-thread to conserve RAM and prevent thread arena bloat.
        """
        if self._embeddings_model is None:
            with self._lock:
                if self._embeddings_model is None:
                    try:
                        import torch
                        torch.set_num_threads(1)
                        torch.set_num_interop_threads(1)

                        logger.info(f"Loading SentenceTransformers embedding model: {self.model_name}")
                        self._embeddings_model = HuggingFaceEmbeddings(
                            model_name=self.model_name,
                            model_kwargs={"device": "cpu"},
                            encode_kwargs={"normalize_embeddings": True, "batch_size": 16}
                        )
                    except Exception as e:
                        logger.error(f"Failed to load embedding model '{self.model_name}': {e}")
                        raise RuntimeError(
                            f"Could not load SentenceTransformers model '{self.model_name}'. "
                            f"Ensure you have internet access on first run to download the weights: {e}"
                        )
        return self._embeddings_model

    def warmup(self) -> None:
        """Pre-warms the embedding model into RAM if needed."""
        try:
            _ = self.embed_query("warmup system readiness")
            logger.info("EmbeddingService pre-warmed successfully.")
        except Exception as e:
            logger.warning(f"EmbeddingService warmup warning: {e}")

    def embed_query(self, text: str) -> List[float]:
        """
        Embeds a single query string into a vector.
        Uses in-memory cache to achieve sub-millisecond response for identical/repeated queries.
        """
        text_clean = text.strip()
        if not text_clean:
            raise ValueError("Query string cannot be empty.")

        # Check in-memory query cache
        if text_clean in self._query_cache:
            return self._query_cache[text_clean]

        # Generate embedding vector
        vec = self.model.embed_query(text_clean)

        # Cache with LRU eviction
        if len(self._query_cache) >= self._max_query_cache_size:
            self._query_cache.pop(next(iter(self._query_cache)))
        self._query_cache[text_clean] = vec

        return vec

    def __call__(self, text: str) -> List[float]:
        """Allows EmbeddingService to be called directly as a callable embedding function."""
        return self.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a batch of document strings into vectors.
        Reuses cached chunk embeddings and batches uncached chunks to stay within strict RAM limits.
        """
        if not texts:
            return []

        results: List[Optional[List[float]]] = [None] * len(texts)
        missing_indices: List[int] = []
        missing_texts: List[str] = []

        # 1. Lookup cached chunks
        for idx, text in enumerate(texts):
            thash = self._hash_text(text)
            if thash in self._chunk_cache:
                results[idx] = self._chunk_cache[thash]
            else:
                missing_indices.append(idx)
                missing_texts.append(text)

        # 2. Compute embeddings in small memory-conscious batches
        if missing_texts:
            logger.info(f"Computing embeddings for {len(missing_texts)} new chunks ({len(texts) - len(missing_texts)} cached).")
            batch_size = 16
            new_vectors = []
            for i in range(0, len(missing_texts), batch_size):
                batch = missing_texts[i:i + batch_size]
                new_vectors.extend(self.model.embed_documents(batch))

            for orig_idx, vec in zip(missing_indices, new_vectors):
                results[orig_idx] = vec
                thash = self._hash_text(texts[orig_idx])
                if len(self._chunk_cache) < self._max_chunk_cache_size:
                    self._chunk_cache[thash] = vec

            gc.collect()
        else:
            logger.info(f"All {len(texts)} chunks retrieved from chunk embedding cache (0ms inference).")

        return results  # type: ignore
