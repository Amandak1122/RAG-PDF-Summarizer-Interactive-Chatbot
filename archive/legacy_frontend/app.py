import streamlit as plt
import streamlit as st
import requests
import json

# Page Configuration
st.set_page_config(
    page_title="RAG PDF Summarizer & Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend API URL
BACKEND_URL = "http://127.0.0.1:8000"

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "document_summary" not in st.session_state:
    st.session_state.document_summary = None
if "active_document" not in st.session_state:
    st.session_state.active_document = None
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = {}

def check_backend_health():
    """Checks if FastAPI backend is running and returns health status."""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=2)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException:
        return None
    return None

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ RAG Control Panel")

    # Backend Connection Status
    health_info = check_backend_health()
    if health_info:
        st.success("🟢 Backend Connected", icon="✅")
    else:
        st.error("🔴 Backend Disconnected\nFastAPI is not running on port 8000.", icon="⚠️")

    st.markdown("---")
    st.subheader("🔑 Groq API Settings")
    groq_api_key_input = st.text_input(
        "Groq API Key (Optional Override):",
        type="password",
        placeholder="gsk_... or leave blank to use .env",
        help="Paste your Groq API key here or keep it saved in your .env file."
    )

    st.markdown("---")
    st.subheader("📄 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
        if st.button("🚀 Process & Index PDF", use_container_width=True):
            with st.spinner("Extracting text and generating vector embeddings..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    response = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=60)
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.active_document = data["filename"]
                        st.session_state.doc_stats = {
                            "pages": data["total_pages"],
                            "chunks": data["total_chunks"]
                        }
                        st.session_state.chat_history = []
                        st.session_state.document_summary = None
                        st.success(f"Indexed {data['filename']} successfully!")
                    else:
                        st.error(f"Error: {response.json().get('detail', 'Upload failed')}")
                except Exception as e:
                    st.error(f"Failed to connect to backend: {str(e)}")

    if st.session_state.active_document:
        st.info(
            f"**Active Document:** {st.session_state.active_document}\n\n"
            f"📄 Pages: {st.session_state.doc_stats.get('pages', 0)} | "
            f"🧩 Chunks: {st.session_state.doc_stats.get('chunks', 0)}"
        )

    st.markdown("---")
    st.subheader("🔍 Retrieval Settings")
    top_k = st.slider("Top Relevant Chunks (k)", min_value=1, max_value=8, value=4)

    st.markdown("---")
    if st.button("🗑️ Reset Session", use_container_width=True):
        try:
            requests.delete(f"{BACKEND_URL}/reset", timeout=5)
        except Exception:
            pass
        st.session_state.chat_history = []
        st.session_state.document_summary = None
        st.session_state.active_document = None
        st.session_state.doc_stats = {}
        st.rerun()

# --- MAIN PANEL ---
st.title("📚 RAG PDF Summarizer & Interactive Chatbot")
st.caption("Powered by FastAPI • PyMuPDF • Sentence Transformers • FAISS • Groq Llama 3")

# Document Summary Section
if st.session_state.active_document:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("📝 Document Overview")
    with col2:
        if st.button("✨ Generate AI Summary", use_container_width=True):
            with st.spinner("Generating executive summary with Groq Llama 3..."):
                payload = {"api_key": groq_api_key_input if groq_api_key_input.strip() else None}
                try:
                    res = requests.post(f"{BACKEND_URL}/summarize", json=payload, timeout=60)
                    if res.status_code == 200:
                        st.session_state.document_summary = res.json().get("summary")
                    else:
                        st.error(res.json().get("detail", "Failed to summarize"))
                except Exception as e:
                    st.error(f"Request failed: {str(e)}")

    if st.session_state.document_summary:
        with st.expander("📌 View Executive Summary", expanded=True):
            st.markdown(st.session_state.document_summary)

st.markdown("---")

# Chat Interface
st.subheader("💬 Chat with your Document")

# Display previous messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "citations" in msg and msg["citations"]:
            with st.expander(f"📖 Citations & Context ({len(msg['citations'])} sources)"):
                for cite in msg["citations"]:
                    st.markdown(
                        f"**Source:** `{cite['source']}` | **Page:** `{cite['page']}` | **Distance Score:** `{cite['score']}`\n\n"
                        f"> {cite['snippet']}"
                    )

# Prompt input
user_query = st.chat_input("Ask a question about your uploaded PDF...")

if user_query:
    if not st.session_state.active_document:
        st.warning("⚠️ Please upload and index a PDF in the sidebar first!")
    else:
        # Append and display user message
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Query Backend
        with st.chat_message("assistant"):
            with st.spinner("Searching FAISS vector index & querying Groq..."):
                payload = {
                    "query": user_query,
                    "top_k": top_k,
                    "api_key": groq_api_key_input if groq_api_key_input.strip() else None
                }
                try:
                    res = requests.post(f"{BACKEND_URL}/chat", json=payload, timeout=60)
                    if res.status_code == 200:
                        data = res.json()
                        answer = data.get("answer", "No response received.")
                        citations = data.get("citations", [])

                        st.markdown(answer)
                        if citations:
                            with st.expander(f"📖 Citations & Context ({len(citations)} sources)"):
                                for cite in citations:
                                    st.markdown(
                                        f"**Source:** `{cite['source']}` | **Page:** `{cite['page']}` | **Score:** `{cite['score']}`\n\n"
                                        f"> {cite['snippet']}"
                                    )

                        # Save to session history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer,
                            "citations": citations
                        })
                    else:
                        err_msg = res.json().get("detail", "Error processing request.")
                        st.error(err_msg)
                except Exception as e:
                    st.error(f"Failed to communicate with backend: {str(e)}")

