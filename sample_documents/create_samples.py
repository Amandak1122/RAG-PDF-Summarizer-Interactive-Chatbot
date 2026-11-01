"""Utility script to create sample test PDFs for verification and unit tests."""
from pathlib import Path
import pymupdf

def generate_sample_pdfs():
    sample_dir = Path(__file__).parent
    sample_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create a clean, text-based multi-page sample PDF (AI Primer)
    doc = pymupdf.open()

    # Page 1: Overview of RAG
    page1 = doc.new_page()
    text_p1 = (
        "Retrieval-Augmented Generation (RAG) Architecture Overview\n\n"
        "Page 1: Introduction to RAG\n"
        "Retrieval-Augmented Generation (RAG) is a technique that combines an external knowledge base "
        "with large language models (LLMs). Instead of relying solely on the static weights learned during training, "
        "RAG fetches relevant document excerpts at query time and supplies them as ground truth in the prompt.\n\n"
        "Key Advantages:\n"
        "1. Mitigates hallucination by anchoring generation in verifiable facts.\n"
        "2. Eliminates the high computational cost of continuously fine-tuning models.\n"
        "3. Enables private enterprise data to be queried securely without sharing internal data for model retraining.\n"
    )
    page1.insert_text((50, 72), text_p1, fontsize=11)

    # Page 2: Chunking & Vector Search
    page2 = doc.new_page()
    text_p2 = (
        "Page 2: Semantic Chunking and Vector Similarity\n\n"
        "Text Chunking Strategy:\n"
        "Raw text must be divided into smaller passages. LangChain's RecursiveCharacterTextSplitter is often used "
        "with a chunk size of 800 characters and an overlap of 100 characters. Overlap prevents critical context "
        "from being severed across chunk boundaries.\n\n"
        "Vector Embeddings:\n"
        "Chunks are converted into 384-dimensional dense vectors using models like all-MiniLM-L6-v2 from "
        "Sentence Transformers. These embeddings represent semantic meaning mathematically in high-dimensional vector space.\n\n"
        "FAISS Indexing:\n"
        "Facebook AI Similarity Search (FAISS) enables millisecond nearest-neighbor lookups using cosine or Euclidean distance."
    )
    page2.insert_text((50, 72), text_p2, fontsize=11)

    # Page 3: Evaluation and Limitations
    page3 = doc.new_page()
    text_p3 = (
        "Page 3: Evaluation Metrics and Future Work\n\n"
        "Evaluating RAG Systems:\n"
        "Two core dimensions are evaluated:\n"
        "1. Retrieval Relevance: Did FAISS return passages containing the true answer?\n"
        "2. Faithfulness: Did the LLM answer strictly using retrieved context without hallucinating external claims?\n\n"
        "System Limitations:\n"
        "Standard text extractors cannot read scanned or photographed documents without an OCR engine. "
        "In addition, documents containing complex tables require layout-aware parsing."
    )
    page3.insert_text((50, 72), text_p3, fontsize=11)

    valid_pdf_path = sample_dir / "ai_primer.pdf"
    doc.save(str(valid_pdf_path))
    doc.close()
    print(f"Created: {valid_pdf_path}")

    # 2. Create an empty PDF file (0 bytes)
    empty_pdf = sample_dir / "empty.pdf"
    empty_pdf.write_bytes(b"")
    print(f"Created: {empty_pdf}")

    # 3. Create a corrupt PDF file (invalid bytes)
    corrupt_pdf = sample_dir / "corrupt.pdf"
    corrupt_pdf.write_text("This is not a real PDF file header.")
    print(f"Created: {corrupt_pdf}")

if __name__ == "__main__":
    generate_sample_pdfs()

