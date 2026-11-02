# Cloud Deployment Guide: RAG PDF Summarizer & Chatbot

This guide provides step-by-step instructions for deploying the **RAG PDF Summarizer and Interactive Chatbot** across separate cloud hosting platforms:

- **Frontend:** [Streamlit Community Cloud](https://share.streamlit.io) (hosts `app/ui/streamlit_app.py`)
- **Backend:** [Render](https://render.com) Web Service (hosts `app/main.py` FastAPI backend)
- **LLM Inference:** [Groq Cloud](https://console.groq.com) (Qwen / Llama 3 models via API)

```
┌─────────────────────────────────┐        Server-to-Server HTTPS       ┌─────────────────────────────────┐
│    Streamlit Community Cloud    │  ─────────────────────────────────> │       Render Web Service        │
│   (app/ui/streamlit_app.py)     │  <───────────────────────────────── │         (app/main.py)           │
└─────────────────────────────────┘         SSE Streaming / REST        └──────────────┬──────────────────┘
                 │                                                                     │
          Browser Client                                                        HTTPS API Calls
                 │                                                                     │
                 ▼                                                                     ▼
     [End User Web Interface]                                                ┌───────────────────┐
                                                                             │  Groq Cloud LLM   │
                                                                             │(qwen/qwen3.8-27b) │
                                                                             └───────────────────┘
```

---

## Part 1: Deploy FastAPI Backend on Render

### Method A: Automated Deployment via `render.yaml` (Recommended)

1. Push your repository to GitHub (ensure `.env` is NOT committed; `.gitignore` is pre-configured).
2. Log in to [Render Dashboard](https://dashboard.render.com).
3. Click **New +** and select **Blueprint**.
4. Connect your GitHub repository.
5. Render will automatically detect [`render.yaml`](./render.yaml).
6. Set the required secret in the dashboard:
   - **`GROQ_API_KEY`**: Paste your Groq API key (`gsk_...`).
7. Click **Apply**. Render will build and deploy the web service.

---

### Method B: Manual Web Service Setup on Render

1. Log in to [Render Dashboard](https://dashboard.render.com).
2. Click **New +** > **Web Service**.
3. Connect your GitHub repository.
4. Configure the service settings:
   - **Name:** `rag-pdf-fastapi-backend`
   - **Region:** Any preferred region (e.g., Oregon, Frankfurt)
   - **Branch:** `main`
   - **Runtime:** `Python 3`
   - **Build Command:**
     ```bash
     pip install -r requirements.txt && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
     ```
   - **Start Command:**
     ```bash
     uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1
     ```
   - **Health Check Path:** `/api/health`

5. Add **Environment Variables** in the Render settings:
   | Key | Value | Description |
   | :--- | :--- | :--- |
   | `PYTHON_VERSION` | `3.11.9` | Ensures Python 3.11 runtime |
   | `GROQ_API_KEY` | `gsk_...` | **Secret:** Your private Groq API key |
   | `GROQ_MODEL` | `qwen/qwen3.8-27b` | Preserves the active reasoning model |
   | `ALLOWED_ORIGINS` | `*` | Or specify `https://<your-app>.streamlit.app` |
   | `MALLOC_ARENA_MAX` | `2` | **Essential for Render 512MB RAM:** Prevents glibc memory fragmentation |
   | `OMP_NUM_THREADS` | `1` | Restricts OpenMP thread memory bloat |
   | `MKL_NUM_THREADS` | `1` | Restricts MKL thread memory bloat |
   | `WEB_CONCURRENCY` | `1` | Restricts Uvicorn to 1 process worker |
   | `CHUNK_SIZE` | `800` | Semantic chunk token size |
   | `CHUNK_OVERLAP` | `100` | Chunk overlap tokens |
   | `TOP_K_RESULTS` | `4` | Chunks retrieved per query |
   | `SIMILARITY_DISTANCE_THRESHOLD` | `1.45` | L2 vector distance cutoff |
   | `SIMILARITY_COSINE_MIN` | `-0.05` | Cosine similarity cutoff |

6. Click **Create Web Service**.
7. Once deployed, copy your Render service URL (e.g., `https://rag-pdf-fastapi-backend.onrender.com`).

---

## Part 2: Deploy Frontend on Streamlit Community Cloud

1. Log in to [Streamlit Community Cloud](https://share.streamlit.io).
2. Click **New app**.
3. Select your GitHub repository and branch (`main`).
4. Set the **Main file path**:
   ```
   app/ui/streamlit_app.py
   ```
5. Click **Advanced settings**:
   - **Python Version:** Select `3.11`.
   - **Secrets:** Configure your Render backend URL:
     ```toml
     API_BASE_URL = "https://rag-pdf-fastapi-backend.onrender.com"
     ```
     > [!IMPORTANT]
     > **Security Rule:** Never put `GROQ_API_KEY` into Streamlit secrets or client code. The Groq API key is managed exclusively by the Render backend environment.

6. Click **Deploy!**
7. Streamlit Cloud will read [`.streamlit/config.toml`](./.streamlit/config.toml) and launch the web interface.

---

## Part 3: Free-Tier Hosting Limitations & Important Considerations

### 1. Memory Constraints (Render Free Tier: 512 MB RAM)
- The FastAPI backend loads PyTorch and the Sentence Transformers model (`all-MiniLM-L6-v2`) into memory to compute 384-dimensional dense vectors.
- Baseline resident memory with PyTorch and FAISS is approximately **300 MB - 420 MB RSS**.
- On Render's Free Tier (512 MB RAM limit), running multiple concurrent requests or processing very large PDFs (>50 pages) can push peak memory near the limit.
- **Optimization Measures Implemented:**
  - Uvicorn runs with `--workers 1` to prevent multiple process clones.
  - MD5 query and chunk embedding caches reduce unnecessary model re-computations.
- *Production Recommendation:* If you experience `Exit Code 137 (OOMKilled)` on heavy documents, upgrade the Render instance to the **Starter plan** (512MB - 1GB RAM) or host the backend container on a VM/service with $\ge 1\text{ GB}$ RAM.

### 2. Ephemeral Storage Behavior
- Render Web Services use **ephemeral disks**.
- Any uploaded PDFs in `data/uploads/` and FAISS index files in `data/faiss_index/` are stored in the active container filesystem.
- When Render restarts, deploys a new commit, or spins down due to inactivity, these files are reset.
- This is completely safe for interactive user sessions, but users should know that documents are indexed per active session and not persisted across container reboots.

### 3. Free-Tier Cold Starts (Inactivity Sleep)
- On Render Free Tier, services spin down after 15 minutes of inactivity.
- The first request after sleep may take **30 to 50 seconds** while the container boots, loads Python, and caches the MiniLM embedding model.
- The Streamlit frontend includes a 5-second health check cache (`get_backend_health`) and will display:
  `🔴 Backend API Offline`
  while Render is waking up. Once awake, it automatically switches to `🟢 Backend API Connected`.

---

## Part 4: Local Development (Unchanged & Preserved)

Local development requires zero cloud configuration. `API_BASE_URL` automatically falls back to `http://127.0.0.1:8000/api`.

### Running Locally:
1. **Start Backend:**
   ```bash
   .\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
   ```
2. **Start Frontend:**
   ```bash
   .\.venv\Scripts\python.exe -m streamlit run app/ui/streamlit_app.py
   ```
3. Open `http://localhost:8501` in your browser.

