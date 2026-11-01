# 📚 RAG PDF Summarizer & Interactive Chatbot

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B.svg)](https://streamlit.io/)
[![FAISS](https://img.shields.io/badge/Vector%20Store-FAISS%20CPU-orange.svg)](https://github.com/facebookresearch/faiss)
[![Groq](https://img.shields.io/badge/LLM%20Inference-Groq%20LPU-black.svg)](https://groq.com/)

A production-grade **Retrieval-Augmented Generation (RAG)** platform designed for strict, hallucination-resistant document intelligence. Built with **FastAPI**, **Streamlit**, **PyMuPDF**, **Sentence Transformers**, **FAISS**, and **Groq LPU LLMs**.

---

## 🌟 Key Highlights & Engineering Features

1. **Permission-Based LLM Fallback (Hallucination Defense):**
   - When a question lacks sufficient support in the uploaded PDF, the system **does not hallucinate** or silently answer using external knowledge.
   - Evaluates vector metric bounds ($L2 \le 1.45$, $\text{Cosine} \ge -0.05$) in **<25 ms** with **0 LLM calls**.
   - Prompts the user: *"I couldn't find sufficient information related to your question in the uploaded PDF. Would you like me to generate an answer using the LLM's general knowledge instead?"*
   - Gives users two clear choices: **Generate Answer** (answers using general knowledge with an explicit disclaimer, suppressing fake PDF citations) or **Cancel** (refrains from calling the LLM).
   - Single-use permission token prevents direct unauthorized API calls.

2. **Real-Time Token Streaming (Server-Sent Events):**
   - High-throughput SSE endpoint (`POST /api/ask/stream`) yields delta tokens directly from Groq, delivering a Time-to-First-Token (TTFT) of **~700 ms**.

3. **Sub-Millisecond Caching & Latency Optimization:**
   - **In-Memory Query Vector Cache:** LRU cache returns embeddings for repeated questions in **0.43 ms** (bypassing CPU forward passes).
   - **Chunk Hash Caching:** MD5 content hashes prevent re-embedding identical text passages.
   - **Duplicate File SHA-256 Bypass:** Re-uploading an identical PDF reuses indexed chunks in **~4.5 ms** (a **300x+ speedup** over raw extraction).

4. **Multi-Page & Meta-Query Synthesis:**
   - Broad synthesis queries (e.g. *"Generate top 5 keywords"*, *"Based on this JD, create 10 interview questions"*) are recognized dynamically, retrieving broader context without triggering false out-of-domain rejections.

5. **Hierarchical Map-Reduce Summarization:**
   - Summarizes short documents in a single pass (`direct`) and long documents via a two-stage `hierarchical` map-reduce pipeline.

6. **Session & Document Isolation:**
   - Every user receives a unique session ID. Vector stores and temporary files are isolated under `data/faiss_index/<session_id>` and `data/uploads/<session_id>_*`, preventing cross-session data leakage.

---

## 🏗️ System Architecture & Data Flow

```mermaid
flowchart TD
    subgraph Client ["Client Presentation Layer"]
        UI["Streamlit Web UI (:8501)\n(Real-Time SSE Stream & Interactive Fallback Dialog)"]
    end

    subgraph API ["REST API Layer (FastAPI :8000)"]
        Routes["APIRouter (/upload, /ask, /ask/stream, /fallback, /summarize, /health, /reset)"]
        Models["Pydantic v2 Validation Schemas"]
    end

    subgraph Pipeline ["Core RAG Pipeline Services"]
        PDFSvc["PDFService\n(PyMuPDF / Page Extraction / Scanned OCR Detection)"]
        ChunkSvc["ChunkingService\n(RecursiveCharacterTextSplitter / 800c Size / 100c Overlap)"]
        EmbedSvc["EmbeddingService\n(Sentence Transformers all-MiniLM-L6-v2 + In-Memory LRU Cache)"]
        VectorSvc["VectorStoreService\n(FAISS Vector Index / Session-Scoped Persistence)"]
        RetrSvc["RetrievalService\n(L2-to-Cosine Conversion / Dynamic Pruning / Citations)"]
        LLMSvc["LLMService\n(Groq LPU / Anti-Injection Guards / Token Streaming)"]
        SummSvc["SummarizationService\n(Direct & Map-Reduce Hierarchical Synthesizer)"]
    end

    subgraph Storage ["Storage & External Providers"]
        DataDir[("Local Session Data (/data/uploads & /data/faiss_index)")]
        GroqCloud[("Groq Cloud LPU Engine (qwen/qwen3.8-27b)")]
    end

    UI -->|Upload PDF / HTTP POST| Routes
    Routes --> PDFSvc
    PDFSvc --> ChunkSvc
    ChunkSvc --> VectorSvc
    VectorSvc --> EmbedSvc
    VectorSvc --> DataDir

    UI -->|Ask Question (SSE Stream)| Routes
    Routes --> RetrSvc
    RetrSvc --> VectorSvc
    RetrSvc -->|Context & Page Citations| LLMSvc
    LLMSvc -->|Token Stream| GroqCloud
    GroqCloud -->|Real-Time Tokens| UI

    UI -->|Permission Approval / Cancel| Routes
    Routes -->|Controlled Fallback| LLMSvc
```

---

## 📂 Project Directory Structure

```text
RAG_PDF_Chatbot/
├── .env.example                     # Environment template with documented settings
├── .gitignore                       # Production gitignore (excludes secrets & user uploads)
├── README.md                        # Documentation & interview guide
├── requirements.txt                 # Compatible pinned dependencies
│
├── app/                             # Active application package
│   ├── __init__.py
│   ├── main.py                      # FastAPI server entry point & CORS configuration
│   ├── config.py                    # Central configuration & settings validation
│   ├── models.py                    # Pydantic v2 request/response contracts
│   │
│   ├── api/                         # REST API layer
│   │   ├── __init__.py
│   │   └── routes.py                # Endpoints: /upload, /ask, /ask/stream, /fallback, /summarize, /health, /reset
│   │
│   ├── services/                    # Core business logic & RAG engine
│   │   ├── __init__.py
│   │   ├── pdf_service.py           # Text parsing, filename sanitization, scanned PDF detection
│   │   ├── chunking_service.py      # Recursive semantic chunking & deduplication
│   │   ├── embedding_service.py     # SentenceTransformer embeddings + LRU query cache
│   │   ├── vector_store_service.py  # Session-scoped FAISS vector indexing & persistence
│   │   ├── retrieval_service.py     # L2/Cosine score interpretation & dynamic relevance pruning
│   │   ├── llm_service.py           # Groq LLM integration, prompt injection guards & streaming
│   │   └── summarization_service.py # Single-pass & Map-Reduce summarization engine
│   │
│   └── ui/                          # Frontend presentation layer
│       └── streamlit_app.py         # Streamlit chat interface, streaming & fallback UI
│
├── data/                            # Runtime storage (gitignored except .gitkeep)
│   ├── uploads/                     # Temporary uploaded PDFs (session-isolated)
│   └── faiss_index/                 # Session-scoped FAISS vector stores
│
├── sample_documents/                # Verified test documents & demo fixtures
│   ├── ai_primer.pdf                # 3-page synthetic AI Primer (for demos and tests)
│   ├── create_samples.py            # Generator script for reproducible test fixtures
│   ├── corrupt.pdf                  # Invalid PDF fixture (tests error resilience)
│   ├── empty.pdf                    # 0-byte fixture (tests validation)
│   ├── scanned_image_only.pdf       # Image-only fixture (tests OCR rejection notice)
│   └── test_document.txt            # Non-PDF fixture (tests extension enforcement)
│
├── tests/                           # Complete automated test suite (38+ scenarios)
│   ├── __init__.py
│   ├── test_pdf_service.py          # PDF parsing & boundary tests
│   ├── test_chunking_service.py     # Chunk size, overlap, and deduplication tests
│   ├── test_vector_store.py         # FAISS indexing, persistence & isolation tests
│   ├── test_performance.py          # Embedding cache, chunk cache, and pruning unit tests
│   ├── test_api_endpoints.py        # REST API contracts & status codes
│   ├── test_fallback.py             # Permission-Based Fallback comprehensive scenarios
│   ├── test_live_performance.py     # Millisecond latency benchmark suite
│   └── test_quality_assessment.py   # Full 38-scenario quality & reliability test runner
│
└── archive/                         # Archived legacy code (retained safely, isolated from app)
    ├── legacy_backend/              # Initial monolithic backend
    └── legacy_frontend/             # Initial frontend
```

---

## ⚡ Quickstart Guide

### 1. Environment Setup

```powershell
# Clone the repository
git clone https://github.com/your-username/RAG_PDF_Chatbot.git
cd RAG_PDF_Chatbot

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # On Windows
# source .venv/bin/activate    # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy the template and add your Groq API key:

```powershell
cp .env.example .env
```

Open `.env` and set:
```env
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
GROQ_MODEL=qwen/qwen3.8-27b
```
*(Get a free Groq API key at [console.groq.com/keys](https://console.groq.com/keys)).*

### 3. Run Automated Tests

Execute the complete test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

Run the end-to-end 38-scenario quality audit:

```powershell
.\.venv\Scripts\python.exe tests/test_quality_assessment.py
```

### 4. Start the Application

Open two terminal windows:

**Terminal 1 — FastAPI Backend Server:**
```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000 --reload
```
- Interactive Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health Check Diagnostic: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

**Terminal 2 — Streamlit Frontend Web App:**
```powershell
.\.venv\Scripts\python.exe -m streamlit run app/ui/streamlit_app.py --server.port 8501
```
- Open Web UI: [http://localhost:8501](http://localhost:8501)

### 5. Cloud Deployment (Streamlit Cloud + Render)
For production deployment with separate hosting (Streamlit Community Cloud for the frontend and Render for the FastAPI backend), see the complete step-by-step guide in [DEPLOYMENT.md](DEPLOYMENT.md).

---

## 📊 Live Benchmark Measurements

*Measured using `tests/test_live_performance.py` against live running services:*

| Operation / Step | Cold Execution | Cached Execution | Improvement / Speedup |
| :--- | :---: | :---: | :---: |
| **PDF Upload & Indexing** | ~1,350 ms | **4.5 ms** | **300x+ faster** (SHA-256 hash bypass) |
| **FAISS Vector Retrieval** | 19.6 ms | **0.43 ms** | **45x faster** (In-memory LRU cache) |
| **Streaming First-Token (TTFT)** | ~2,800 ms (blocking) | **747 ms** | **3.7x faster perceived latency** |
| **Out-of-Domain Detection** | ~2,400 ms (if LLM checked) | **23.5 ms** | **100x faster** (**0 ms** LLM inference) |

---

## 🎯 Technical Interview Guide: RAG Deep Dive

Use these concrete talking points when explaining this architecture during technical interviews:

### Q1: Why decouple FastAPI and Streamlit into separate services?
> **Answer:** Decoupling follows production microservice best practices. Streamlit excels as a rapid dashboarding tool, but embedding heavy vector stores and business logic directly in Streamlit causes state invalidation on UI reruns. By placing document parsing, vector search, and LLM orchestration behind a clean FastAPI REST API, the core engine remains stateless, scalable, independently testable, and capable of serving multiple clients (Streamlit, mobile apps, Slack bots, or React web frontends).

### Q2: How does chunking work, and why is chunk overlap necessary?
> **Answer:** Using LangChain's `RecursiveCharacterTextSplitter`, text is partitioned hierarchically (`\n\n` paragraphs $\rightarrow$ `\n` sentences $\rightarrow$ words). With `chunk_size=800` and `chunk_overlap=100`, the 100-character overlap prevents critical semantic context from being sliced across chunk boundaries (e.g. a condition appearing in one sentence and its conclusion in the next).

### Q3: Why run embeddings and FAISS locally instead of using cloud vector databases?
> **Answer:** 
> 1. **Zero Cost & Strict Privacy:** `all-MiniLM-L6-v2` runs locally on CPU with zero per-token cost, ensuring that private enterprise documents never leave the local environment during embedding.
> 2. **Low Latency:** In-memory FAISS similarity search executes in **<1 ms** without network roundtrip latency or external vector DB rate limits.

### Q4: How are FAISS similarity scores interpreted?
> **Answer:** FAISS returns squared Euclidean $L2$ distance ($\|\mathbf{u} - \mathbf{v}\|^2$). Since all embeddings are unit-normalized ($\|\mathbf{u}\|=1$), $L2$ relates to Cosine Similarity via:
> $$\text{Cosine}(\theta) = 1.0 - \frac{L2^2}{2.0}$$
> - $L2 = 0.0 \implies \text{Cosine} = 1.0$ (identical match)
> - $L2 = 1.0 \implies \text{Cosine} = 0.5$ (strong match)
> - $L2 = 1.414 \implies \text{Cosine} = 0.0$ (orthogonal / uncorrelated)
> - $L2 \ge 1.65 \implies \text{Cosine} \le -0.36$ (divergent / out-of-domain)

### Q5: How does the Permission-Based LLM Fallback prevent hallucinations?
> **Answer:** Standard RAG pipelines suffer from hallucinations when the retrieved context lacks the answer: the LLM either invents facts or provides a disclaimer inconsistently. This project implements a **two-tier gatekeeper**:
> 1. **Vector Relevance Gate:** If the best retrieved chunk has $L2 > 1.45$, the query is rejected at the vector search layer in **~20 ms** without calling the LLM.
> 2. **Explicit Permission Protocol:** The UI displays a warning and requests user approval. If approved, the answer is labeled as general knowledge with an explicit disclaimer, and fake document citations are suppressed. Unauthorized direct API calls without prior pending permission are blocked with HTTP 400.

### Q6: How is long-document summarization handled?
> **Answer:** Single-prompt summarization fails on long PDFs due to context window limits and attention dilution. We implement **Hierarchical (Map-Reduce) Summarization**:
> - **Map Step:** Document chunks are grouped into batches and summarized into intermediate takeaways.
> - **Reduce Step:** The intermediate summaries are synthesized into a coherent executive report with key findings and action items.

---

## 🔒 Security & Safe Handling

- **Never Commit Credentials:** `.env` is ignored by `.gitignore`.
- **API Key Secrecy:** All API key references are masked; responses never leak credential strings.
- **Path Traversal Defense:** Uploaded filenames and session IDs are sanitized using regex (`re.sub(r'[^a-zA-Z0-9_.-]', '_', name)`), stripping `../` traversal indicators.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
