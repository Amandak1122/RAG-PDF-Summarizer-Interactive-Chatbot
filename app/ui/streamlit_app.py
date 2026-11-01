import json
import uuid
import streamlit as st
import requests

# Set page layout and metadata
st.set_page_config(
    page_title="RAG PDF Summarizer & Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend REST API configuration
BACKEND_BASE_URL = "http://127.0.0.1:8000/api"

# Initialize Session State
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "summary_data" not in st.session_state:
    st.session_state.summary_data = None

if "document_info" not in st.session_state:
    st.session_state.document_info = None

if "pending_fallback" not in st.session_state:
    st.session_state.pending_fallback = None

@st.cache_data(ttl=5)
def get_backend_health(session_id: str):
    """Cached GET /api/health check (5s TTL) to prevent blocking HTTP overhead on UI reruns."""
    try:
        url = f"{BACKEND_BASE_URL}/health?session_id={session_id}"
        resp = requests.get(url, timeout=3)
        if resp.status_code == 200:
            return resp.json()
    except requests.RequestException:
        return None
    return None

# ==================== SIDEBAR CONTROLS ====================
with st.sidebar:
    st.title("⚙️ RAG Control Panel")
    st.caption(f"Session ID: `{st.session_state.session_id}`")

    # Backend Connectivity Badge
    health = get_backend_health(st.session_state.session_id)
    if health:
        st.success("🟢 Backend API Connected", icon="✅")
        if health.get("api_key_configured"):
            st.info("🔑 Groq API Key: Configured in `.env`")
        else:
            st.warning("⚠️ Groq API Key: Not set in `.env`")
    else:
        st.error("🔴 Backend API Offline\nEnsure FastAPI is running on port 8000.", icon="⚠️")

    st.markdown("---")

    # Optional Groq API Key Override
    st.subheader("🔑 Groq API Key")
    api_key_override = st.text_input(
        "API Key (Optional Override):",
        type="password",
        placeholder="gsk_... or use .env",
        help="Paste a temporary Groq API key here if not already saved in your .env file."
    )

    st.markdown("---")

    # PDF File Uploader
    st.subheader("📄 Upload Document")
    uploaded_pdf = st.file_uploader(
        label="Choose a PDF file",
        type=["pdf"],
        help="Upload a native text-based PDF document (up to 50 MB)."
    )

    if uploaded_pdf is not None:
        if st.button("🚀 Process & Index PDF", use_container_width=True):
            with st.spinner("Extracting text with PyMuPDF & building FAISS index..."):
                try:
                    files = {"file": (uploaded_pdf.name, uploaded_pdf.getvalue(), "application/pdf")}
                    upload_url = f"{BACKEND_BASE_URL}/upload?session_id={st.session_state.session_id}"
                    res = requests.post(upload_url, files=files, timeout=180)

                    if res.status_code == 201:
                        data = res.json()
                        st.session_state.document_info = {
                            "name": data["filename"],
                            "pages": data["total_pages"],
                            "chunks": data["total_chunks"],
                            "size_bytes": data["file_size_bytes"]
                        }
                        st.session_state.chat_history = []
                        st.session_state.summary_data = None
                        st.success(f"Indexed **{data['filename']}** successfully!")
                    else:
                        error_detail = res.json().get("detail", "Failed to process PDF.")
                        st.error(f"Error ({res.status_code}): {error_detail}")
                except Exception as e:
                    st.error(f"Upload connection error: {str(e)}")

    if st.session_state.document_info:
        doc = st.session_state.document_info
        st.success(
            f"**Active Document:**\n"
            f"- File: `{doc['name']}`\n"
            f"- Total Pages: `{doc['pages']}`\n"
            f"- Semantic Chunks: `{doc['chunks']}`"
        )

    st.markdown("---")

    # Retrieval & Summarization Parameters
    st.subheader("🎛️ Settings")
    top_k_chunks = st.slider(
        "Top-k Context Passages",
        min_value=1,
        max_value=8,
        value=4,
        help="Number of most similar chunks retrieved from FAISS."
    )
    summary_strategy = st.selectbox(
        "Summarization Strategy",
        options=["auto", "direct", "hierarchical"],
        help="'direct' is for short docs; 'hierarchical' (Map-Reduce) combines chunks for long docs."
    )

    st.markdown("---")

    # Reset Session Button
    if st.button("🗑️ Reset Session & Index", use_container_width=True):
        try:
            requests.delete(
                f"{BACKEND_BASE_URL}/reset?session_id={st.session_state.session_id}",
                timeout=5
            )
        except Exception:
            pass
        st.session_state.chat_history = []
        st.session_state.summary_data = None
        st.session_state.document_info = None
        st.session_state.pending_fallback = None
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.rerun()

# ==================== MAIN PANEL ====================
st.title("📚 RAG PDF Summarizer & Interactive Chatbot")
st.caption("Grounded PDF Intelligence with FastAPI, PyMuPDF, Sentence Transformers, FAISS, and Groq Llama 3")

# Document Overview & Summary Section
if st.session_state.document_info:
    col_hdr, col_btn = st.columns([3, 1])
    with col_hdr:
        st.subheader("📝 Document Summarization")
    with col_btn:
        if st.button("✨ Generate Summary", use_container_width=True):
            with st.spinner("Analyzing document and synthesizing executive summary..."):
                payload = {
                    "strategy": summary_strategy,
                    "api_key": api_key_override.strip() if api_key_override else None
                }
                try:
                    res = requests.post(
                        f"{BACKEND_BASE_URL}/summarize?session_id={st.session_state.session_id}",
                        json=payload,
                        timeout=90
                    )
                    if res.status_code == 200:
                        st.session_state.summary_data = res.json()
                    else:
                        st.error(res.json().get("detail", "Summarization failed."))
                except Exception as e:
                    st.error(f"Summarization request failed: {str(e)}")

    if st.session_state.summary_data:
        summ = st.session_state.summary_data
        with st.expander(f"📌 Executive Summary ({summ['strategy_used'].capitalize()} Mode)", expanded=True):
            st.markdown(f"**Document:** `{summ['document_name']}` ({summ['total_pages']} pages)")
            st.markdown("#### Executive Overview")
            st.write(summ["executive_summary"])

            if summ.get("key_points"):
                st.markdown("#### Key Points & Findings")
                for pt in summ["key_points"]:
                    st.markdown(f"- {pt}")

            if summ.get("action_items_or_conclusions"):
                st.markdown("#### Action Items & Conclusions")
                for act in summ["action_items_or_conclusions"]:
                    st.markdown(f"- {act}")

st.markdown("---")

# ==================== CHAT INTERFACE ====================
st.subheader("💬 Ask Questions Grounded in Your PDF")

# Display past messages
for idx, msg in enumerate(st.session_state.chat_history):
    with st.chat_message(msg["role"]):
        source_type = msg.get("source_type", "pdf")

        if msg["role"] == "user":
            st.markdown(msg["content"])
        elif source_type == "permission_prompt":
            st.warning(msg["content"])
            # Display interactive buttons only for the active pending fallback question
            if st.session_state.pending_fallback and st.session_state.pending_fallback.get("question") == msg.get("question"):
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("Generate Answer", key=f"btn_gen_{idx}", type="primary", use_container_width=True):
                        with st.spinner("Generating answer using general LLM knowledge..."):
                            fb_payload = {
                                "question": msg.get("question"),
                                "permission_granted": True,
                                "api_key": api_key_override.strip() if api_key_override else None
                            }
                            try:
                                fb_res = requests.post(
                                    f"{BACKEND_BASE_URL}/fallback?session_id={st.session_state.session_id}",
                                    json=fb_payload,
                                    timeout=60
                                )
                                if fb_res.status_code == 200:
                                    fb_data = fb_res.json()
                                    st.session_state.chat_history.append({
                                        "role": "assistant",
                                        "content": fb_data["answer"],
                                        "source_type": "general_knowledge",
                                        "disclaimer": fb_data.get("disclaimer"),
                                        "citations": [],
                                        "latency_ms": fb_data.get("latency_ms")
                                    })
                                    st.session_state.pending_fallback = None
                                    st.rerun()
                                else:
                                    st.error(fb_res.json().get("detail", "Fallback request failed."))
                            except Exception as e:
                                st.error(f"Fallback connection error: {str(e)}")
                with col2:
                    if st.button("Cancel", key=f"btn_cancel_{idx}", use_container_width=True):
                        fb_payload = {
                            "question": msg.get("question"),
                            "permission_granted": False,
                            "api_key": api_key_override.strip() if api_key_override else None
                        }
                        try:
                            fb_res = requests.post(
                                f"{BACKEND_BASE_URL}/fallback?session_id={st.session_state.session_id}",
                                json=fb_payload,
                                timeout=30
                            )
                            if fb_res.status_code == 200:
                                fb_data = fb_res.json()
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": fb_data["answer"],
                                    "source_type": "cancelled",
                                    "citations": []
                                })
                                st.session_state.pending_fallback = None
                                st.rerun()
                        except Exception as e:
                            st.error(f"Connection error: {str(e)}")
        elif source_type == "general_knowledge":
            st.markdown("### 🤖 AI-Generated Answer (General Knowledge)")
            st.markdown(msg["content"])
            disclaimer_text = msg.get("disclaimer") or "Note: This answer is generated using general knowledge and is not supported by the uploaded PDF."
            st.caption(f"⚠️ *{disclaimer_text}*")
            if msg.get("latency_ms"):
                lat = msg["latency_ms"]
                st.caption(f"⚡ *Response Latency: {lat.get('total_ms', 0)}ms (LLM: {lat.get('llm_ms', 0)}ms)*")
        elif source_type == "cancelled":
            st.info(msg["content"])
        else:
            # Standard PDF-grounded answer
            st.caption("🏷️ **Source:** Based on Uploaded PDF")
            st.markdown(msg["content"])
            if msg.get("latency_ms"):
                lat = msg["latency_ms"]
                st.caption(f"⚡ *Response Latency: {lat.get('total_ms', 0)}ms (Retrieval: {lat.get('retrieval_ms', 0)}ms | LLM: {lat.get('llm_ms', 0)}ms)*")
            if msg.get("citations"):
                with st.expander(f"📖 View Citations ({len(msg['citations'])} sources supporting this answer)"):
                    for cite in msg["citations"]:
                        st.markdown(
                            f"**Source:** `{cite['source_file']}` | **Page:** `{cite['page_number']}` | **Distance:** `{cite['relevance_score']}`\n\n"
                            f"> {cite['excerpt']}"
                        )

# Prompt input
user_question = st.chat_input("Ask a question about your uploaded document...")

if user_question:
    if not st.session_state.document_info:
        st.warning("⚠️ Please upload and index a PDF in the sidebar before asking questions.")
    else:
        # Display user question
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        # Query Backend RAG API with real-time streaming
        payload = {
            "question": user_question,
            "top_k": top_k_chunks,
            "api_key": api_key_override.strip() if api_key_override else None
        }

        stream_url = f"{BACKEND_BASE_URL}/ask/stream?session_id={st.session_state.session_id}"
        meta = {}
        tokens = []

        try:
            with requests.post(stream_url, json=payload, stream=True, timeout=60) as resp:
                if resp.status_code == 200:
                    for line in resp.iter_lines(decode_unicode=True):
                        if line and line.startswith("data: "):
                            event_data = json.loads(line[6:])
                            event_type = event_data.get("type")
                            if event_type == "metadata":
                                meta = event_data
                            elif event_type == "token":
                                tokens.append(event_data.get("content", ""))
                            elif event_type == "done":
                                if "latency_ms" in event_data:
                                    meta["latency_ms"] = event_data["latency_ms"]
                                break

                    full_answer = "".join(tokens)
                    requires_permission = meta.get("requires_fallback_permission", False)
                    citations = meta.get("citations", [])
                    source_type = meta.get("source_type", "pdf")
                    latency_stats = meta.get("latency_ms")

                    if requires_permission:
                        st.session_state.pending_fallback = {"question": user_question}
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": full_answer,
                            "source_type": "permission_prompt",
                            "question": user_question,
                            "citations": [],
                            "latency_ms": latency_stats
                        })
                    else:
                        st.session_state.pending_fallback = None
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": full_answer,
                            "source_type": source_type,
                            "citations": citations,
                            "latency_ms": latency_stats
                        })
                    st.rerun()
                else:
                    # Fallback to standard synchronous endpoint
                    res = requests.post(
                        f"{BACKEND_BASE_URL}/ask?session_id={st.session_state.session_id}",
                        json=payload,
                        timeout=60
                    )
                    if res.status_code == 200:
                        data = res.json()
                        answer = data.get("answer", "No answer received.")
                        citations = data.get("citations", [])
                        requires_permission = data.get("requires_fallback_permission", False)
                        source_type = data.get("source_type", "pdf")
                        lat = data.get("latency_ms")

                        if requires_permission:
                            st.session_state.pending_fallback = {"question": user_question}
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": answer,
                                "source_type": "permission_prompt",
                                "question": user_question,
                                "citations": [],
                                "latency_ms": lat
                            })
                        else:
                            st.session_state.pending_fallback = None
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": answer,
                                "source_type": source_type,
                                "citations": citations,
                                "latency_ms": lat
                            })
                        st.rerun()
                    else:
                        error_detail = res.json().get("detail", "Error generating answer.")
                        st.error(f"Error ({res.status_code}): {error_detail}")
        except Exception as e:
            st.error(f"Failed to communicate with RAG backend: {str(e)}")
