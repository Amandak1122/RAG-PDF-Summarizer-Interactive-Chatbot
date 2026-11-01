"""
Comprehensive End-to-End Quality Assessment & Production-Readiness Test Suite.
Covers Sections 2 through 8 against the live running FastAPI backend and Groq LLM.
"""

import sys
import os
import time
import json
import uuid
import requests
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE_URL = "http://127.0.0.1:8000/api"
SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_documents"
AI_PRIMER_PDF = SAMPLE_DIR / "ai_primer.pdf"
EMPTY_PDF = SAMPLE_DIR / "empty.pdf"
CORRUPT_PDF = SAMPLE_DIR / "corrupt.pdf"
SCANNED_PDF = SAMPLE_DIR / "scanned_image_only.pdf"
TXT_FILE = SAMPLE_DIR / "test_document.txt"

class QualityAssessmentRunner:
    def __init__(self):
        self.results = {}
        self.timings = {}
        self.issues = []

    def log_result(self, category: str, test_name: str, passed: bool, details: str = "", timing_ms: float = 0.0):
        if category not in self.results:
            self.results[category] = []
        self.results[category].append({
            "test_name": test_name,
            "passed": passed,
            "details": details,
            "timing_ms": round(timing_ms, 2)
        })
        status_icon = "✅ PASS" if passed else "❌ FAIL"
        time_str = f" ({timing_ms:.1f}ms)" if timing_ms > 0 else ""
        print(f"[{status_icon}] {category} -> {test_name}{time_str}")
        if not passed:
            print(f"       Issue: {details}")

    def run_all(self):
        print("=" * 80)
        print("    STARTING RIGOROUS PRODUCTION QUALITY & RAG PIPELINE ASSESSMENT     ")
        print("=" * 80)
        
        self.test_section1_setup()
        self.test_section2_pdf_processing()
        self.test_section3_rag_pipeline()
        self.test_section4_question_answering()
        self.test_section5_permission_fallback()
        self.test_section6_llm_and_api()
        self.test_section7_performance()
        self.test_section8_security()

        print("\n" + "=" * 80)
        print("                   ASSESSMENT EXECUTION COMPLETE                      ")
        print("=" * 80)

    # -------------------------------------------------------------
    # SECTION 1: Setup & Health Check
    # -------------------------------------------------------------
    def test_section1_setup(self):
        category = "1. Setup & Connectivity"
        t0 = time.perf_counter()
        try:
            res = requests.get(f"{BASE_URL}/health", timeout=5)
            dt = (time.perf_counter() - t0) * 1000
            data = res.json()
            passed = res.status_code == 200 and data.get("status") == "healthy" and data.get("api_key_configured") is True
            self.log_result(category, "Health Check & Groq Key Configured", passed, f"Status: {res.status_code}, data: {data}", dt)
        except Exception as e:
            self.log_result(category, "Health Check & Groq Key Configured", False, str(e))

    # -------------------------------------------------------------
    # SECTION 2: PDF Processing Tests
    # -------------------------------------------------------------
    def test_section2_pdf_processing(self):
        category = "2. PDF Processing"
        session_id = f"test_pdf_{uuid.uuid4().hex[:6]}"

        # 2.1 Normal text PDF upload
        t0 = time.perf_counter()
        with open(AI_PRIMER_PDF, "rb") as f:
            res = requests.post(f"{BASE_URL}/upload?session_id={session_id}", files={"file": ("ai_primer.pdf", f, "application/pdf")})
        dt = (time.perf_counter() - t0) * 1000
        data = res.json()
        p1 = res.status_code == 201 and data.get("total_pages") == 3 and data.get("total_chunks", 0) > 0
        self.log_result(category, "Normal Text PDF Upload", p1, f"Pages: {data.get('total_pages')}, Chunks: {data.get('total_chunks')}", dt)

        # 2.2 Multi-page processing
        p2 = data.get("total_pages") == 3 and data.get("total_chunks") >= 3
        self.log_result(category, "Multi-page PDF Processing", p2, f"Total pages processed: {data.get('total_pages')}")

        # 2.3 Empty PDF (0 bytes)
        with open(EMPTY_PDF, "rb") as f:
            res_empty = requests.post(f"{BASE_URL}/upload?session_id=s_empty", files={"file": ("empty.pdf", f, "application/pdf")})
        p3 = res_empty.status_code == 400 and "empty" in res_empty.json().get("detail", "").lower()
        self.log_result(category, "Empty PDF Rejection", p3, f"HTTP {res_empty.status_code}: {res_empty.json().get('detail')}")

        # 2.4 Corrupted PDF
        with open(CORRUPT_PDF, "rb") as f:
            res_corrupt = requests.post(f"{BASE_URL}/upload?session_id=s_corrupt", files={"file": ("corrupt.pdf", f, "application/pdf")})
        try:
            d_corrupt = res_corrupt.json()
            detail_corrupt = d_corrupt.get("detail", "")
        except Exception:
            detail_corrupt = res_corrupt.text
        p4 = res_corrupt.status_code == 400 and ("could not open" in detail_corrupt.lower() or "error" in detail_corrupt.lower())
        self.log_result(category, "Corrupted PDF Rejection", p4, f"HTTP {res_corrupt.status_code}: {detail_corrupt[:100]}")

        # 2.5 Invalid file format (.txt)
        with open(TXT_FILE, "rb") as f:
            res_txt = requests.post(f"{BASE_URL}/upload?session_id=s_txt", files={"file": ("test_document.txt", f, "text/plain")})
        p5 = res_txt.status_code == 400 and "unsupported file format" in res_txt.json().get("detail", "").lower()
        self.log_result(category, "Invalid File Extension Rejection", p5, f"HTTP {res_txt.status_code}: {res_txt.json().get('detail')}")

        # 2.6 Scanned / Image-Only PDF handling
        with open(SCANNED_PDF, "rb") as f:
            res_scanned = requests.post(f"{BASE_URL}/upload?session_id=s_scanned", files={"file": ("scanned_image_only.pdf", f, "application/pdf")})
        p6 = res_scanned.status_code == 422 and "ocr" in res_scanned.json().get("detail", "").lower()
        self.log_result(category, "Scanned Image-Only PDF Rejection (OCR Required)", p6, f"HTTP {res_scanned.status_code}: {res_scanned.json().get('detail')}")

        # 2.7 Duplicate PDF upload in same session (Hash caching)
        t0 = time.perf_counter()
        with open(AI_PRIMER_PDF, "rb") as f:
            res_dup = requests.post(f"{BASE_URL}/upload?session_id={session_id}", files={"file": ("ai_primer.pdf", f, "application/pdf")})
        dt_dup = (time.perf_counter() - t0) * 1000
        p7 = res_dup.status_code in (200, 201) and dt_dup < 50.0  # Must be instant cache hit (<50ms)
        self.log_result(category, "Duplicate PDF Upload (SHA-256 Hash Caching)", p7, f"Re-index response time: {dt_dup:.2f}ms", dt_dup)

        # Cleanup
        requests.delete(f"{BASE_URL}/reset?session_id={session_id}")
        requests.delete(f"{BASE_URL}/reset?session_id=s_empty")
        requests.delete(f"{BASE_URL}/reset?session_id=s_corrupt")
        requests.delete(f"{BASE_URL}/reset?session_id=s_txt")
        requests.delete(f"{BASE_URL}/reset?session_id=s_scanned")

    # -------------------------------------------------------------
    # SECTION 3: RAG Pipeline Tests
    # -------------------------------------------------------------
    def test_section3_rag_pipeline(self):
        category = "3. RAG Pipeline"
        session_id = f"test_rag_{uuid.uuid4().hex[:6]}"
        with open(AI_PRIMER_PDF, "rb") as f:
            requests.post(f"{BASE_URL}/upload?session_id={session_id}", files={"file": ("ai_primer.pdf", f, "application/pdf")})

        # Test retrieval & score conversion
        from app.services.retrieval_service import RetrievalService
        cos_zero = RetrievalService.l2_to_cosine(0.0)
        cos_one = RetrievalService.l2_to_cosine(1.0)
        cos_ortho = RetrievalService.l2_to_cosine(1.41421356)
        p_math = (cos_zero == 1.0) and (cos_one == 0.5) and (round(cos_ortho, 2) == 0.0)
        self.log_result(category, "FAISS L2 to Cosine Conversion Logic", p_math, f"L2=0->Cos {cos_zero}, L2=1->Cos {cos_one}, L2=1.414->Cos {cos_ortho}")

        # Test context retrieval & citations
        q = "What are the advantages of Retrieval-Augmented Generation?"
        res = requests.post(f"{BASE_URL}/ask?session_id={session_id}", json={"question": q, "top_k": 4})
        data = res.json()
        cites = data.get("citations", [])
        p_cites = len(cites) > 0 and any(c.get("page_number") == 1 for c in cites)
        self.log_result(category, "Citation Source & Page-Number Accuracy", p_cites, f"Returned {len(cites)} citations, pages: {[c.get('page_number') for c in cites]}")

        requests.delete(f"{BASE_URL}/reset?session_id={session_id}")

    # -------------------------------------------------------------
    # SECTION 4: Question Answering Tests
    # -------------------------------------------------------------
    def test_section4_question_answering(self):
        category = "4. Question Answering"
        session_id = f"test_qa_{uuid.uuid4().hex[:6]}"
        with open(AI_PRIMER_PDF, "rb") as f:
            requests.post(f"{BASE_URL}/upload?session_id={session_id}", files={"file": ("ai_primer.pdf", f, "application/pdf")})

        # 4.1 Direct factual question
        q1 = "What is Retrieval-Augmented Generation and what are its key advantages?"
        res1 = requests.post(f"{BASE_URL}/ask?session_id={session_id}", json={"question": q1, "top_k": 3})
        d1 = res1.json()
        p1 = d1.get("has_sufficient_context") is True and d1.get("source_type") == "pdf" and ("hallucination" in d1.get("answer", "").lower() or "mitigate" in d1.get("answer", "").lower())
        self.log_result(category, "Direct Factual Question", p1, f"Context found: {d1.get('has_sufficient_context')}, Source: {d1.get('source_type')}")

        # 4.2 Different wording question
        q2 = "How does RAG solve hallucination issues in language models?"
        res2 = requests.post(f"{BASE_URL}/ask?session_id={session_id}", json={"question": q2, "top_k": 3})
        d2 = res2.json()
        p2 = d2.get("has_sufficient_context") is True and d2.get("source_type") == "pdf"
        self.log_result(category, "Paraphrased / Different Wording Question", p2, f"Context found: {d2.get('has_sufficient_context')}")

        # 4.3 Multi-chunk synthesis question
        q3 = "Explain how semantic chunking and FAISS vector indexing are used together."
        res3 = requests.post(f"{BASE_URL}/ask?session_id={session_id}", json={"question": q3, "top_k": 4})
        d3 = res3.json()
        p3 = d3.get("has_sufficient_context") is True and ("faiss" in d3.get("answer", "").lower() or "chunk" in d3.get("answer", "").lower())
        self.log_result(category, "Multi-Chunk Synthesis Question", p3, f"Context found: {d3.get('has_sufficient_context')}")

        # 4.4 Summarization endpoint
        res_sum = requests.post(f"{BASE_URL}/summarize?session_id={session_id}", json={"strategy": "auto"})
        d_sum = res_sum.json()
        p_sum = res_sum.status_code == 200 and len(d_sum.get("executive_summary", "")) > 50 and len(d_sum.get("key_points", [])) > 0
        self.log_result(category, "Document Summarization Endpoint", p_sum, f"Strategy: {d_sum.get('strategy_used')}, Key points: {len(d_sum.get('key_points', []))}")

        # 4.5 Keyword extraction
        q4 = "Generate the top 5 relevant keywords from this document."
        res4 = requests.post(f"{BASE_URL}/ask?session_id={session_id}", json={"question": q4, "top_k": 4})
        d4 = res4.json()
        p4 = d4.get("has_sufficient_context") is True and d4.get("requires_fallback_permission") is False
        self.log_result(category, "Keyword Extraction from PDF", p4, f"Permission requested: {d4.get('requires_fallback_permission')}")

        # 4.6 Interview question generation based on PDF
        q5 = "Based on the uploaded document, create 5 technical interview questions for a RAG engineer."
        res5 = requests.post(f"{BASE_URL}/ask?session_id={session_id}", json={"question": q5, "top_k": 4})
        d5 = res5.json()
        ans5 = d5.get("answer", "").lower()
        p5 = d5.get("has_sufficient_context") is True and d5.get("requires_fallback_permission") is False and "please paste" not in ans5
        self.log_result(category, "Interview Question Generation (PDF Contextual)", p5, f"Permission requested: {d5.get('requires_fallback_permission')}, Refusal: {'please paste' in ans5}")

        # 4.7 Questions unrelated to PDF
        q6 = "What is quantum computing and superstring theory in theoretical physics?"
        res6 = requests.post(f"{BASE_URL}/ask?session_id={session_id}", json={"question": q6, "top_k": 4})
        d6 = res6.json()
        p6 = d6.get("requires_fallback_permission") is True and d6.get("has_sufficient_context") is False and d6.get("source_type") == "none"
        self.log_result(category, "Unrelated Question (Permission Prompt Triggered)", p6, f"Permission requested: {d6.get('requires_fallback_permission')}")

        # 4.8 Question with insufficient context
        q7 = "What is the stock price of Google and when was DeepMind acquired?"
        res7 = requests.post(f"{BASE_URL}/ask?session_id={session_id}", json={"question": q7, "top_k": 4})
        d7 = res7.json()
        p7 = d7.get("requires_fallback_permission") is True and d7.get("has_sufficient_context") is False
        self.log_result(category, "Insufficient Context Question (Permission Prompt Triggered)", p7, f"Permission requested: {d7.get('requires_fallback_permission')}")

        requests.delete(f"{BASE_URL}/reset?session_id={session_id}")

    # -------------------------------------------------------------
    # SECTION 5: Permission-Based LLM Fallback Tests
    # -------------------------------------------------------------
    def test_section5_permission_fallback(self):
        category = "5. Permission-Based Fallback"
        session_id = f"test_fb_{uuid.uuid4().hex[:6]}"
        with open(AI_PRIMER_PDF, "rb") as f:
            requests.post(f"{BASE_URL}/upload?session_id={session_id}", files={"file": ("ai_primer.pdf", f, "application/pdf")})

        ood_question = "What are the core axioms of quantum mechanics and Hilbert spaces?"

        # 5.1 Trigger fallback
        res_trigger = requests.post(f"{BASE_URL}/ask?session_id={session_id}", json={"question": ood_question, "top_k": 4})
        d_trig = res_trigger.json()
        p1 = d_trig.get("requires_fallback_permission") is True and "couldn't find sufficient information" in d_trig.get("answer", "")
        self.log_result(category, "Fallback Permission Prompt Display", p1, d_trig.get("answer"))

        # 5.2 User Clicks 'Cancel'
        res_cancel = requests.post(f"{BASE_URL}/fallback?session_id={session_id}", json={"question": ood_question, "permission_granted": False})
        d_cancel = res_cancel.json()
        p2 = res_cancel.status_code == 200 and d_cancel.get("source_type") == "cancelled" and "won't generate" in d_cancel.get("answer", "")
        self.log_result(category, "Fallback User Cancellation", p2, f"Source: {d_cancel.get('source_type')}, Ans: {d_cancel.get('answer')}")

        # 5.3 Attempt fallback again after cancellation without re-triggering (Must be blocked)
        res_re_cancel = requests.post(f"{BASE_URL}/fallback?session_id={session_id}", json={"question": ood_question, "permission_granted": True})
        p3 = res_re_cancel.status_code == 400 and "no pending fallback" in res_re_cancel.json().get("detail", "").lower()
        self.log_result(category, "Single-Use Permission Token Invalidation", p3, f"HTTP {res_re_cancel.status_code}: {res_re_cancel.json().get('detail')}")

        # 5.4 Re-trigger and User Clicks 'Generate Answer'
        requests.post(f"{BASE_URL}/ask?session_id={session_id}", json={"question": ood_question, "top_k": 4})
        res_approve = requests.post(f"{BASE_URL}/fallback?session_id={session_id}", json={"question": ood_question, "permission_granted": True})
        d_appr = res_approve.json()
        p4 = (
            res_approve.status_code == 200
            and d_appr.get("source_type") == "general_knowledge"
            and d_appr.get("disclaimer") is not None
            and len(d_appr.get("answer", "")) > 50
        )
        self.log_result(category, "Fallback Approval & General Knowledge Generation", p4, f"Source: {d_appr.get('source_type')}, Disclaimer: {d_appr.get('disclaimer')}")

        # 5.5 Session Isolation Test (Session A vs Session B)
        session_b = f"session_b_{uuid.uuid4().hex[:6]}"
        res_b_ask = requests.post(f"{BASE_URL}/ask?session_id={session_b}", json={"question": "What is RAG?", "top_k": 4})
        p5 = res_b_ask.status_code == 400 and "no document has been uploaded" in res_b_ask.json().get("detail", "").lower()
        self.log_result(category, "Session Isolation (Cross-Session Access Prevention)", p5, f"Session B rejected unindexed query: HTTP {res_b_ask.status_code}")

        # 5.6 Direct unauthorized fallback request without any ask call
        res_unauth = requests.post(f"{BASE_URL}/fallback?session_id={session_id}", json={"question": "Arbitrary question without permission", "permission_granted": True})
        p6 = res_unauth.status_code == 400
        self.log_result(category, "Direct Unauthorized Fallback Request Blocked", p6, f"HTTP {res_unauth.status_code}: {res_unauth.json().get('detail')}")

        requests.delete(f"{BASE_URL}/reset?session_id={session_id}")

    # -------------------------------------------------------------
    # SECTION 6: LLM and API Tests
    # -------------------------------------------------------------
    def test_section6_llm_and_api(self):
        category = "6. LLM & API Tests"
        session_id = f"test_llm_{uuid.uuid4().hex[:6]}"
        with open(AI_PRIMER_PDF, "rb") as f:
            requests.post(f"{BASE_URL}/upload?session_id={session_id}", files={"file": ("ai_primer.pdf", f, "application/pdf")})

        # 6.1 Groq API connection test
        q = "What is RAG?"
        res = requests.post(f"{BASE_URL}/ask?session_id={session_id}", json={"question": q, "top_k": 2})
        p1 = res.status_code == 200 and len(res.json().get("answer", "")) > 10
        self.log_result(category, "Groq Live Model Invocation", p1, f"Model: {res.json().get('llm_model')}")

        # 6.2 Streaming endpoint test (/api/ask/stream)
        time.sleep(2.0)
        try:
            stream_res = requests.post(
                f"{BASE_URL}/ask/stream?session_id={session_id}",
                json={"question": "Briefly state FAISS purpose.", "top_k": 2},
                stream=True,
                timeout=60
            )
            tokens = []
            has_meta = False
            has_done = False
            for line in stream_res.iter_lines(decode_unicode=True):
                if line and line.startswith("data: "):
                    payload = json.loads(line[6:])
                    if payload.get("type") == "metadata":
                        has_meta = True
                    elif payload.get("type") == "token":
                        tokens.append(payload.get("content", ""))
                    elif payload.get("type") == "done":
                        has_done = True
            full_stream_text = "".join(tokens)
            p2 = stream_res.status_code == 200 and has_meta and has_done and len(full_stream_text) > 5
            self.log_result(category, "Server-Sent Events (SSE) Streaming", p2, f"Tokens: {len(tokens)}, Text: {full_stream_text[:60]}...")
        except Exception as e:
            self.log_result(category, "Server-Sent Events (SSE) Streaming", False, f"Stream error: {str(e)}")

        # 6.3 Missing / Invalid API Key Handling
        res_bad_key = requests.post(
            f"{BASE_URL}/ask?session_id={session_id}",
            json={"question": "What is RAG?", "top_k": 2, "api_key": "gsk_invalid_bogus_key_1234567890"}
        )
        data_bad = res_bad_key.json()
        p3 = "error calling groq api" in data_bad.get("answer", "").lower() or res_bad_key.status_code != 500
        self.log_result(category, "Invalid API Key Error Handling", p3, f"Graceful response: {data_bad.get('answer')[:70]}")

        # 6.4 API Key Secrecy (Never exposed in responses)
        resp_str = json.dumps(res.json())
        api_key_in_env = os.getenv("GROQ_API_KEY", "")
        p4 = (api_key_in_env not in resp_str) if len(api_key_in_env) > 10 else True
        self.log_result(category, "Secret Key Masking & Privacy", p4, "No raw API key found in API response payloads")

        requests.delete(f"{BASE_URL}/reset?session_id={session_id}")

    # -------------------------------------------------------------
    # SECTION 7: Performance Testing
    # -------------------------------------------------------------
    def test_section7_performance(self):
        category = "7. Performance Benchmarking"
        session_id = f"test_perf_{uuid.uuid4().hex[:6]}"

        # PDF Upload Time
        t0 = time.perf_counter()
        with open(AI_PRIMER_PDF, "rb") as f:
            res_up = requests.post(f"{BASE_URL}/upload?session_id={session_id}", files={"file": ("ai_primer.pdf", f, "application/pdf")})
        t_upload = (time.perf_counter() - t0) * 1000
        self.log_result(category, "PDF Upload & Vector Indexing Latency", res_up.status_code == 201, f"{t_upload:.2f} ms", t_upload)

        # Retrieval Latency
        q = "What is Retrieval-Augmented Generation?"
        res_ask = requests.post(f"{BASE_URL}/ask?session_id={session_id}", json={"question": q, "top_k": 4})
        lat = res_ask.json().get("latency_ms", {})
        retrieval_ms = lat.get("retrieval_ms", 0.0)
        llm_ms = lat.get("llm_ms", 0.0)
        total_ms = lat.get("total_ms", 0.0)
        self.log_result(category, "FAISS Vector Retrieval Latency", retrieval_ms < 200.0, f"{retrieval_ms:.2f} ms", retrieval_ms)
        self.log_result(category, "Groq LLM Generation Latency", llm_ms < 5000.0, f"{llm_ms:.2f} ms", llm_ms)
        self.log_result(category, "Total End-to-End Latency", total_ms < 6000.0, f"{total_ms:.2f} ms", total_ms)

        # Vector check on out-of-domain (Zero-LLM cost)
        res_ood = requests.post(f"{BASE_URL}/ask?session_id={session_id}", json={"question": "Superstring theory in physics", "top_k": 4})
        lat_ood = res_ood.json().get("latency_ms", {})
        self.log_result(category, "Out-of-Domain Zero-LLM Fallback Check Latency", lat_ood.get("llm_ms", 0.0) == 0.0, f"Retrieval: {lat_ood.get('retrieval_ms')}ms | LLM: {lat_ood.get('llm_ms')}ms", lat_ood.get("total_ms", 0.0))

        requests.delete(f"{BASE_URL}/reset?session_id={session_id}")

    # -------------------------------------------------------------
    # SECTION 8: Security & Reliability
    # -------------------------------------------------------------
    def test_section8_security(self):
        category = "8. Security & Reliability"
        from app.services.pdf_service import PDFService
        
        # 8.1 Path Traversal in Filename
        dirty_filename = "../../../etc/passwd.pdf"
        sanitized = PDFService.sanitize_filename(dirty_filename)
        p1 = ("/" not in sanitized) and ("\\" not in sanitized) and (".." not in sanitized)
        self.log_result(category, "Filename Sanitization (Path Traversal Prevention)", p1, f"'{dirty_filename}' -> '{sanitized}'")

        # 8.2 Session ID Path Traversal Sanitization
        from app.services.vector_store_service import VectorStoreService
        from app.services.embedding_service import EmbeddingService
        vss = VectorStoreService(embedding_service=EmbeddingService())
        malicious_session = "../../root_dir"
        safe_path = vss.get_session_dir(malicious_session)
        p2 = ".." not in safe_path.name
        self.log_result(category, "Session ID Storage Path Traversal Prevention", p2, f"'{malicious_session}' -> '{safe_path.name}'")

        # 8.3 Reset session cleanup
        session_id = f"test_cleanup_{uuid.uuid4().hex[:6]}"
        with open(AI_PRIMER_PDF, "rb") as f:
            requests.post(f"{BASE_URL}/upload?session_id={session_id}", files={"file": ("ai_primer.pdf", f, "application/pdf")})
        res_del = requests.delete(f"{BASE_URL}/reset?session_id={session_id}")
        p3 = res_del.status_code == 200
        # Verify index dir removed
        p3_cleaned = not vss.is_session_indexed(session_id)
        self.log_result(category, "Session & Disk Cleanup on /reset", p3 and p3_cleaned, f"Index cleaned: {p3_cleaned}")

if __name__ == "__main__":
    runner = QualityAssessmentRunner()
    runner.run_all()
