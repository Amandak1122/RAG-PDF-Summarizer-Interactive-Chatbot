import os
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from groq import Groq
from backend.config import settings

class RAGEngine:
    """Orchestrates Groq LLM queries for document summarization and question answering."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.GROQ_API_KEY
        self.model = model or settings.GROQ_MODEL
        self._client: Optional[Groq] = None

    def get_client(self, override_api_key: Optional[str] = None) -> Optional[Groq]:
        """Initializes or retrieves the Groq client. Returns None if no key is available."""
        active_key = override_api_key or self.api_key or os.getenv("GROQ_API_KEY", "").strip()
        if not active_key:
            return None
        return Groq(api_key=active_key)

    def is_api_key_configured(self, override_api_key: Optional[str] = None) -> bool:
        """Checks whether a valid non-empty Groq API key is present."""
        key = override_api_key or self.api_key or os.getenv("GROQ_API_KEY", "").strip()
        return bool(key and len(key) > 5)

    def generate_summary(
        self, full_text: str, override_api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generates a structured, professional multi-section summary of the PDF text.
        If the API key is not yet provided, returns a placeholder explanation.
        """
        client = self.get_client(override_api_key)
        if not client:
            return {
                "summary": (
                    "### ⚠️ Groq API Key Not Configured Yet\n\n"
                    "The backend has successfully parsed your document! "
                    "To generate a live AI summary using Llama 3 on Groq, "
                    "please paste your `GROQ_API_KEY` into the `.env` file or in the Streamlit sidebar."
                ),
                "model_used": self.model,
                "api_key_present": False
            }

        # Truncate text if excessively long to fit within context window safely
        max_chars = 15000
        truncated_text = full_text[:max_chars]
        if len(full_text) > max_chars:
            truncated_text += "\n\n...[Content truncated for summary processing]..."

        prompt = f"""You are an expert document analyst. Provide a well-structured, comprehensive summary of the following document.

Format your response with the following markdown headers:
1. **Executive Overview**: High-level summary of the document's main objective and theme (2-3 sentences).
2. **Key Concepts & Findings**: Bullet points of critical facts, arguments, or data points.
3. **Action Items / Takeaways**: Core lessons, recommendations, or conclusion.

Document Content:
\"\"\"
{truncated_text}
\"\"\"

Summary:"""

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise, analytical document summarization specialist."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1024
            )
            return {
                "summary": response.choices[0].message.content,
                "model_used": self.model,
                "api_key_present": True
            }
        except Exception as e:
            return {
                "summary": f"Error calling Groq API: {str(e)}",
                "model_used": self.model,
                "api_key_present": True,
                "error": str(e)
            }

    def answer_question(
        self,
        query: str,
        retrieved_docs_with_scores: List[tuple],
        override_api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generates an answer to the user query using only retrieved document chunks.
        Formats citations with page numbers and relevance scores.
        """
        # Format citations metadata
        citations = []
        context_blocks = []

        for idx, (doc, score) in enumerate(retrieved_docs_with_scores):
            page_num = doc.metadata.get("page", "Unknown")
            source_file = doc.metadata.get("source", "Document")
            content = doc.page_content.strip()

            citations.append({
                "citation_id": idx + 1,
                "source": source_file,
                "page": page_num,
                "snippet": content[:250] + ("..." if len(content) > 250 else ""),
                "score": round(float(score), 4)
            })

            context_blocks.append(
                f"[Source: {source_file}, Page: {page_num}]\n{content}"
            )

        combined_context = "\n\n---\n\n".join(context_blocks)

        client = self.get_client(override_api_key)
        if not client:
            return {
                "answer": (
                    "### ⚠️ Groq API Key Not Configured Yet\n\n"
                    "Retrieved relevant chunks from your document (see citations below), "
                    "but cannot query Groq without an API key.\n\n"
                    "Once you add your `GROQ_API_KEY` to `.env` or in the sidebar, "
                    "Groq will generate answers synthesized directly from these excerpts."
                ),
                "citations": citations,
                "model_used": self.model,
                "api_key_present": False
            }

        system_instruction = (
            "You are a helpful, strictly grounded AI assistant answering questions about an uploaded PDF.\n"
            "Rules:\n"
            "1. Answer ONLY using the factual context provided below.\n"
            "2. If the answer cannot be found in the context, explicitly say: 'Based on the provided document, I cannot find this information.' Do not fabricate answers.\n"
            "3. Cite the page numbers when stating facts (e.g., 'According to Page 3...')."
        )

        user_content = f"""Context from PDF:
{combined_context}

Question: {query}

Answer:"""

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.2,
                max_tokens=800
            )
            return {
                "answer": response.choices[0].message.content,
                "citations": citations,
                "model_used": self.model,
                "api_key_present": True
            }
        except Exception as e:
            return {
                "answer": f"Error communicating with Groq API: {str(e)}",
                "citations": citations,
                "model_used": self.model,
                "api_key_present": True,
                "error": str(e)
            }

