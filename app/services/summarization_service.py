import re
import logging
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

from app.config import settings
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)

class SummarizationService:
    """
    Implements document summarization with two distinct strategies:
    1. Direct Summarization: For short documents within context limits.
    2. Hierarchical / Map-Reduce: For longer documents to avoid token truncation and fact loss.
    """

    DIRECT_CHAR_THRESHOLD = 12000  # ~3,000 tokens

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def summarize(
        self,
        full_text: str,
        chunks: List[Document],
        strategy: str = "auto",
        api_key_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Coordinates document summarization based on chosen strategy ('direct', 'hierarchical', or 'auto').
        """
        client = self.llm_service.get_client(api_key_override)
        if not client:
            return {
                "strategy_used": strategy,
                "executive_summary": (
                    "⚠️ Groq API key is not configured. "
                    "Please set `GROQ_API_KEY` in .env or provide it in the web interface to generate summaries."
                ),
                "key_points": [],
                "action_items": [],
                "raw_summary": "API Key Required",
                "success": False
            }

        text_length = len(full_text)
        chosen_strategy = strategy.lower()

        if chosen_strategy == "auto":
            chosen_strategy = "direct" if text_length <= self.DIRECT_CHAR_THRESHOLD else "hierarchical"

        if chosen_strategy == "direct":
            return self._direct_summarize(full_text, client)
        else:
            return self._hierarchical_summarize(chunks, client)

    def _direct_summarize(self, full_text: str, client) -> Dict[str, Any]:
        """Strategy A: Single-pass direct summarization for short documents."""
        # Truncate slightly if it exceeds safe bounds
        safe_text = full_text[:16000]

        prompt = f"""You are a professional document analyst. Provide a structured, thorough summary of the following document.

Format your output using these exact section headers:
### Executive Overview
(Provide 2-3 concise sentences summarizing the primary purpose and conclusions.)

### Key Points
- (Bullet point 1)
- (Bullet point 2)
- (Bullet point 3)

### Action Items & Conclusions
(Outline the core takeaways, recommendations, or final outcomes.)

Document Text:
\"\"\"
{safe_text}
\"\"\"

Structured Summary:"""

        try:
            response = client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "You produce clear, executive-grade document summaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1024
            )
            raw_summary = response.choices[0].message.content.strip()
            parsed = self._parse_summary_sections(raw_summary)
            parsed["strategy_used"] = "direct"
            parsed["success"] = True
            return parsed
        except Exception as e:
            logger.error(f"Direct summarization error: {e}")
            return {
                "strategy_used": "direct",
                "executive_summary": f"Failed to generate direct summary: {str(e)}",
                "key_points": [],
                "action_items": [],
                "raw_summary": str(e),
                "success": False
            }

    def _hierarchical_summarize(self, chunks: List[Document], client) -> Dict[str, Any]:
        """
        Strategy B: Hierarchical (Map-Reduce) summarization for long documents.
        Map Step: Summarizes groups of chunks.
        Reduce Step: Synthesizes intermediate summaries into a coherent final summary.
        """
        if not chunks:
            return {
                "strategy_used": "hierarchical",
                "executive_summary": "No content available to summarize.",
                "key_points": [],
                "action_items": [],
                "raw_summary": "",
                "success": False
            }

        # Step 1: Batch chunks into groups of 4-6 chunks
        batch_size = 5
        batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
        intermediate_summaries: List[str] = []

        logger.info(f"Running Map phase: summarizing {len(batches)} batches of chunks.")

        for idx, batch in enumerate(batches):
            batch_text = "\n\n".join(doc.page_content for doc in batch)
            map_prompt = f"""Summarize the key facts, findings, and arguments from this section of the document in 3-4 bullet points:

Section Content:
\"\"\"
{batch_text[:4000]}
\"\"\"

Section Bullet Points:"""

            try:
                res = client.chat.completions.create(
                    model=settings.GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": "Extract critical factual points from document sections."},
                        {"role": "user", "content": map_prompt}
                    ],
                    temperature=0.2,
                    max_tokens=300
                )
                intermediate_summaries.append(res.choices[0].message.content.strip())
            except Exception as e:
                logger.warning(f"Map phase batch {idx} failed: {e}")

        # Step 2: Reduce phase - combine intermediate summaries into cohesive final summary
        combined_intermediate = "\n\n".join(intermediate_summaries)

        reduce_prompt = f"""You are a master technical editor. The following are intermediate section summaries extracted from a multi-page document.
Synthesize them into one coherent, non-redundant, and structured executive summary.

Format your output using these exact section headers:
### Executive Overview
(2-3 sentences synthesizing the entire document.)

### Key Points
- (Consolidated point 1)
- (Consolidated point 2)
- (Consolidated point 3)

### Action Items & Conclusions
(Overall takeaways and outcomes.)

Intermediate Summaries:
\"\"\"
{combined_intermediate}
\"\"\"

Final Synthesized Summary:"""

        try:
            final_res = client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "You synthesize multi-section summaries into unified executive reports."},
                    {"role": "user", "content": reduce_prompt}
                ],
                temperature=0.2,
                max_tokens=1024
            )
            raw_summary = final_res.choices[0].message.content.strip()
            parsed = self._parse_summary_sections(raw_summary)
            parsed["strategy_used"] = "hierarchical"
            parsed["success"] = True
            return parsed
        except Exception as e:
            logger.error(f"Hierarchical reduce phase error: {e}")
            return {
                "strategy_used": "hierarchical",
                "executive_summary": f"Failed in reduction phase: {str(e)}",
                "key_points": [],
                "action_items": [],
                "raw_summary": str(e),
                "success": False
            }

    @staticmethod
    def _parse_summary_sections(text: str) -> Dict[str, Any]:
        """Parses markdown output into clean structured dictionary fields."""
        executive = ""
        key_points = []
        action_items = []

        exec_match = re.search(r"### Executive Overview\s*\n(.*?)(?=\n###|\Z)", text, re.DOTALL)
        if exec_match:
            executive = exec_match.group(1).strip()

        points_match = re.search(r"### Key Points\s*\n(.*?)(?=\n###|\Z)", text, re.DOTALL)
        if points_match:
            lines = points_match.group(1).strip().split("\n")
            key_points = [line.lstrip("-*• ").strip() for line in lines if line.strip()]

        actions_match = re.search(r"### Action Items & Conclusions\s*\n(.*?)(?=\n###|\Z)", text, re.DOTALL)
        if actions_match:
            lines = actions_match.group(1).strip().split("\n")
            action_items = [line.lstrip("-*• ").strip() for line in lines if line.strip()]

        # Fallback if markdown headers were not preserved by the LLM
        if not executive and not key_points:
            executive = text[:400] + ("..." if len(text) > 400 else "")

        return {
            "executive_summary": executive or "Summary generated.",
            "key_points": key_points,
            "action_items": action_items,
            "raw_summary": text
        }

