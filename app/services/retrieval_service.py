import logging
from typing import List, Tuple, Dict, Any, Optional
from langchain_core.documents import Document

from app.config import settings
from app.models import CitationItem
from app.services.vector_store_service import VectorStoreService

logger = logging.getLogger(__name__)

class RetrievalService:
    """
    Coordinates semantic context retrieval:
    Retrieves top_k chunks from FAISS, formats citations, interprets vector scores,
    and constructs a bounded context window for the LLM.
    """

    def __init__(self, vector_store_service: VectorStoreService):
        self.vector_store_service = vector_store_service

    @staticmethod
    def l2_to_cosine(l2_distance: float) -> float:
        """
        Converts Euclidean L2 distance of unit-normalized vectors (||u||=1, ||v||=1) to Cosine Similarity.
        Formula: L2^2 = 2 - 2 * cos(theta) => cos(theta) = 1.0 - (L2^2 / 2.0)
        - d = 0.0 -> cos = 1.0 (identical)
        - d = 1.0 -> cos = 0.5 (high similarity)
        - d = 1.414 -> cos = 0.0 (orthogonal / uncorrelated)
        - d = 1.65 -> cos = -0.36 (divergent / out-of-domain)
        """
        return round(1.0 - (float(l2_distance) ** 2) / 2.0, 4)

    @staticmethod
    def is_document_meta_query(query: str) -> bool:
        """
        Detects questions that target document-wide synthesis, meta-analysis,
        extraction, or document-derived tasks (e.g. 'Generate top 5 keywords from this JD',
        'Based on the job description, create 10 interview questions for this position').
        These queries do not quote specific sentences, so their literal vector distance
        to individual chunks is naturally broader (L2 ~1.20-1.38).
        """
        q_lower = query.lower()
        meta_signals = [
            "this document", "this jd", "this pdf", "the document", "the jd", "the pdf",
            "uploaded document", "uploaded pdf", "uploaded file", "from this", "in this document",
            "in this jd", "in this pdf", "keywords", "key words", "key concepts", "summary",
            "summarize", "main findings", "takeaways", "skills", "responsibilities", "overview",
            "main points", "bullet points", "what does this document", "what does this jd",
            "job description", "the position", "this position", "the role", "this role",
            "interview question", "interview questions", "interview", "create questions",
            "generate questions", "suggest questions", "qualifications", "requirements",
            "candidate", "responsibilities", "for this position", "for this role",
            "based on the job description", "based on the jd", "based on the document",
            "based on the pdf", "based on the uploaded", "according to the document",
            "from the document", "from the jd", "from the job description"
        ]
        return any(signal in q_lower for signal in meta_signals)

    def retrieve_context(
        self,
        session_id: str,
        query: str,
        top_k: int = 4,
        max_context_chars: int = 6000
    ) -> Dict[str, Any]:
        """
        Retrieves relevant passages, compiles citations with page numbers,
        interprets FAISS distance and cosine similarity, and logs diagnostic details.
        """
        is_meta = self.is_document_meta_query(query)
        # For document-wide synthesis, retrieve broader context if available
        effective_top_k = max(top_k, 6) if is_meta else top_k

        matches: List[Tuple[Document, float]] = (
            self.vector_store_service.similarity_search_with_score(
                session_id=session_id,
                query=query,
                top_k=effective_top_k
            )
        )

        if not matches:
            logger.warning(f"[Retrieval Debug] No vector chunks found in session '{session_id}' for query: '{query}'")
            return {
                "has_context": False,
                "has_sufficient_relevance": False,
                "best_distance_score": None,
                "best_cosine_score": None,
                "citations": [],
                "formatted_context": "",
                "relevance_reason": "No vector chunks returned by FAISS index."
            }

        raw_scores = [round(float(score), 4) for _, score in matches]
        cosine_scores = [self.l2_to_cosine(s) for s in raw_scores]
        best_distance_score = min(raw_scores)
        best_cosine_score = max(cosine_scores)

        is_meta = self.is_document_meta_query(query)

        # Multi-factor relevance evaluation:
        # 1. Meta-document questions targeting the uploaded PDF pass vector filtering to allow LLM synthesis.
        # 2. General questions require L2 <= SIMILARITY_DISTANCE_THRESHOLD or Cosine >= SIMILARITY_COSINE_MIN.
        if is_meta:
            has_sufficient_relevance = best_distance_score <= (settings.SIMILARITY_DISTANCE_THRESHOLD + 0.15)
            reason = f"Document-referencing query detected. L2 distance={best_distance_score} is within meta-synthesis range."
        else:
            has_sufficient_relevance = best_distance_score <= settings.SIMILARITY_DISTANCE_THRESHOLD
            reason = (
                f"Vector metric evaluated: Best L2={best_distance_score} (Cosine={best_cosine_score}) vs "
                f"threshold L2<={settings.SIMILARITY_DISTANCE_THRESHOLD}."
            )

        citations: List[CitationItem] = []
        context_blocks: List[str] = []
        accumulated_chars = 0
        pages_retrieved = []

        # Filter out irrelevant chunks for specific queries to save prompt tokens & reduce latency
        retained_matches: List[Tuple[Document, float]] = []
        if is_meta:
            retained_matches = matches
        else:
            # Dynamic relevance cutoff: keep chunks within relative delta of best match and <= threshold
            for doc, score in matches:
                if len(retained_matches) == 0:
                    retained_matches.append((doc, score))
                elif score <= min(settings.SIMILARITY_DISTANCE_THRESHOLD, best_distance_score + 0.25):
                    retained_matches.append((doc, score))

        for idx, (doc, score) in enumerate(retained_matches):
            page_num = doc.metadata.get("page", 1)
            pages_retrieved.append(page_num)
            source_file = doc.metadata.get("source", "document.pdf")
            content = doc.page_content.strip()

            excerpt = content[:280] + ("..." if len(content) > 280 else "")

            citation = CitationItem(
                citation_id=idx + 1,
                source_file=source_file,
                page_number=page_num,
                excerpt=excerpt,
                relevance_score=round(float(score), 4)
            )
            citations.append(citation)

            block = f"[Source: {source_file} | Page {page_num}]\n{content}"
            if accumulated_chars + len(block) <= max_context_chars:
                context_blocks.append(block)
                accumulated_chars += len(block)

        formatted_context = "\n\n---\n\n".join(context_blocks)

        logger.info(
            f"[Retrieval Debug]\n"
            f"  Question: '{query}'\n"
            f"  Retrieved Chunks: {len(matches)} on Pages: {sorted(list(set(pages_retrieved)))}\n"
            f"  Raw FAISS Scores (L2 Distance, lower=better): {raw_scores}\n"
            f"  Interpreted Cosine Similarities (higher=better): {cosine_scores}\n"
            f"  Best L2: {best_distance_score} | Best Cosine: {best_cosine_score}\n"
            f"  Meta-Query Detected: {is_meta}\n"
            f"  Sufficient Relevance: {has_sufficient_relevance} | Reason: {reason}"
        )

        return {
            "has_context": True,
            "has_sufficient_relevance": has_sufficient_relevance,
            "best_distance_score": best_distance_score,
            "best_cosine_score": best_cosine_score,
            "citations": citations,
            "formatted_context": formatted_context,
            "relevance_reason": reason
        }

