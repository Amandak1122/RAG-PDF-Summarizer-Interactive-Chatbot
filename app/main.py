import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

from app.config import settings
from app.api.routes import router, embedding_service

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Production-Grade FastAPI Backend for RAG PDF Summarization and Grounded Q&A",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS for Streamlit and external frontend integration
raw_origins = [o.strip() for o in settings.ALLOWED_ORIGINS.split(",") if o.strip()]
origins = raw_origins if raw_origins else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Router
app.include_router(router, prefix="/api")

@app.on_event("startup")
def startup_event():
    """Startup hook: keep memory footprint minimal so container easily passes Render 512MB limit."""
    logger.info("Server started successfully. Embedding model will load lazily on first request to conserve memory.")

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
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=False)
