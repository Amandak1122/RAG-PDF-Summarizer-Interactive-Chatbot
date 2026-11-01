import os
import time
import requests
from pathlib import Path

BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000/api").rstrip("/")
if not BASE_URL.endswith("/api"):
    BASE_URL = f"{BASE_URL}/api"
SESSION_ID = "live_perf_session"
SAMPLE_PDF = Path("sample_documents/ai_primer.pdf")

def test_live_performance():
    print("=" * 65)
    print("      LIVE PRODUCTION RAG BENCHMARK & PERFORMANCE AUDIT       ")
    print("=" * 65)

    # 1. Health check latency (cached vs direct)
    t0 = time.perf_counter()
    res_health = requests.get(f"{BASE_URL}/health?session_id={SESSION_ID}", timeout=5)
    t_health = (time.perf_counter() - t0) * 1000
    assert res_health.status_code == 200
    print(f"[1] GET /api/health Response Time:           {t_health:8.2f} ms")

    # 2. PDF Upload & Initial Indexing (Cold chunks)
    with open(SAMPLE_PDF, "rb") as f:
        t0 = time.perf_counter()
        res_upload1 = requests.post(
            f"{BASE_URL}/upload?session_id={SESSION_ID}",
            files={"file": ("ai_primer.pdf", f, "application/pdf")},
            timeout=60
        )
        t_upload_initial = (time.perf_counter() - t0) * 1000
    assert res_upload1.status_code == 201
    print(f"[2] Initial PDF Upload & Indexing:           {t_upload_initial:8.2f} ms")

    # 3. Repeated PDF Upload (Duplicate Hash Cache Check)
    with open(SAMPLE_PDF, "rb") as f:
        t0 = time.perf_counter()
        res_upload2 = requests.post(
            f"{BASE_URL}/upload?session_id={SESSION_ID}",
            files={"file": ("ai_primer.pdf", f, "application/pdf")},
            timeout=10
        )
        t_upload_cached = (time.perf_counter() - t0) * 1000
    assert res_upload2.status_code == 200 or res_upload2.status_code == 201
    print(f"[3] Repeat PDF Upload (Hash Caching):        {t_upload_cached:8.2f} ms (Speedup: {t_upload_initial / max(t_upload_cached, 0.1):.1f}x)")

    # 4. Standard Q&A Latency (Cold Query)
    query = "What is Retrieval-Augmented Generation and its key benefits?"
    t0 = time.perf_counter()
    res_ask1 = requests.post(
        f"{BASE_URL}/ask?session_id={SESSION_ID}",
        json={"question": query, "top_k": 4},
        timeout=30
    )
    t_ask_cold = (time.perf_counter() - t0) * 1000
    data1 = res_ask1.json()
    assert res_ask1.status_code == 200
    assert data1["has_sufficient_context"] is True
    lat1 = data1.get("latency_ms", {})
    print(f"[4] Cold Q&A Response (Retrieval + LLM):     {t_ask_cold:8.2f} ms")
    print(f"    └─ Backend Retrieval Latency:            {lat1.get('retrieval_ms', 0):8.2f} ms")
    print(f"    └─ Backend LLM Inference Latency:        {lat1.get('llm_ms', 0):8.2f} ms")
    print(f"    └─ Chunks Sent to LLM (Filtered):        {len(data1['citations'])} citations")

    # 5. Repeated Q&A (Query Embedding Cache)
    t0 = time.perf_counter()
    res_ask2 = requests.post(
        f"{BASE_URL}/ask?session_id={SESSION_ID}",
        json={"question": query, "top_k": 4},
        timeout=30
    )
    t_ask_cached = (time.perf_counter() - t0) * 1000
    data2 = res_ask2.json()
    lat2 = data2.get("latency_ms", {})
    print(f"[5] Repeat Q&A (Query Vector Cached):        {t_ask_cached:8.2f} ms")
    print(f"    └─ Backend Retrieval (Cache Hit):        {lat2.get('retrieval_ms', 0):8.2f} ms")

    # 6. Streaming Time-to-First-Token (TTFT)
    t0 = time.perf_counter()
    ttft_ms = None
    stream_res = requests.post(
        f"{BASE_URL}/ask/stream?session_id={SESSION_ID}",
        json={"question": "Briefly describe FAISS vector index.", "top_k": 3},
        stream=True,
        timeout=30
    )
    tokens_received = 0
    for line in stream_res.iter_lines(decode_unicode=True):
        if line and line.startswith("data: "):
            if ttft_ms is None and "token" in line:
                ttft_ms = (time.perf_counter() - t0) * 1000
            tokens_received += 1
    t_stream_total = (time.perf_counter() - t0) * 1000
    print(f"[6] Streaming Time to First Token (TTFT):    {ttft_ms:8.2f} ms (Total: {t_stream_total:.2f} ms)")

    # 7. Out-of-Domain Detection & Fallback Permission Trigger (0ms LLM)
    ood_query = "What is quantum computing and superstring theory?"
    t0 = time.perf_counter()
    res_ood = requests.post(
        f"{BASE_URL}/ask?session_id={SESSION_ID}",
        json={"question": ood_query, "top_k": 4},
        timeout=10
    )
    t_ood = (time.perf_counter() - t0) * 1000
    data_ood = res_ood.json()
    assert data_ood["requires_fallback_permission"] is True
    lat_ood = data_ood.get("latency_ms", {})
    print(f"[7] Out-of-Domain Fallback Decision:         {t_ood:8.2f} ms (0ms LLM used)")
    print(f"    └─ Backend Vector Relevance Check:       {lat_ood.get('retrieval_ms', 0):8.2f} ms")
    print("=" * 65)

    # Clean up session
    requests.delete(f"{BASE_URL}/reset?session_id={SESSION_ID}", timeout=5)

if __name__ == "__main__":
    test_live_performance()

