import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

from app.config import settings
from app.api.routes import router

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Production-Grade FastAPI Backend for RAG PDF Summarization and Grounded Q&A",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for Streamlit and external frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Router
app.include_router(router, prefix="/api")

@app.get("/")
def root():
    return {
        "app_name": settings.APP_NAME,
        "version": settings.VERSION,
        "status": "online",
        "swagger_docs": "/docs",
        "redoc": "/redoc",
        "health_check": "/api/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)

