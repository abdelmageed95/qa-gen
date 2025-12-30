"""Main FastAPI application"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import ingest, questions
from app.models.schemas import HealthResponse
from app.services.vector_store import VectorStore
from app.core.config import settings
from app import __version__
import logging

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Multi-Agent MCQ Generation System",
    description="Generate Multiple Choice Questions from PDF files using RAG and multi-agent workflow",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(ingest.router, tags=["Ingestion"])
app.include_router(questions.router, tags=["Question Generation"])


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Multi-Agent MCQ Generation System")
    logger.info(f"Version: {__version__}")
    logger.info(f"ChromaDB persist directory: {settings.CHROMA_PERSIST_DIR}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Multi-Agent MCQ Generation System")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Multi-Agent MCQ Generation System",
        "version": __version__,
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    try:
        vector_store = VectorStore()
        chroma_healthy = vector_store.health_check()

        return HealthResponse(
            status="healthy" if chroma_healthy else "degraded",
            version=__version__,
            chroma_status="healthy" if chroma_healthy else "unhealthy"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            version=__version__,
            chroma_status="error"
        )

# uvicorn  for local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
