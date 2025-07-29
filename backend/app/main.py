import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import time
import uvicorn

from .core.config import settings
from .core.logging import logger, setup_logging
from .core.database import create_tables, close_redis
from .api import workspaces, query, knowledge, analytics, reasoning, auth
from .models.schemas import ErrorResponse
from .dependencies import get_document_processing_service
from .services.knowledge_extraction import get_knowledge_extraction_service, get_topic_modeling_service
from .dependencies import get_vector_store_service

# Setup logging
setup_logging("INFO" if not settings.debug else "DEBUG")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles application startup and shutdown events."""
    logger.info(f"Starting {settings.app_name} v{settings.version}")
    
    # Initialize database tables
    create_tables()
    logger.info("Database initialized successfully")

    # Connect to vector store and load all heavy ML models concurrently at startup
    logger.info("Connecting to services and loading ML models...")
    doc_service = get_document_processing_service()
    knowledge_service = get_knowledge_extraction_service()
    topic_service = get_topic_modeling_service()
    vector_service = get_vector_store_service()
    await asyncio.gather(
        vector_service.connect(),
        doc_service.load_models(),
        knowledge_service.load_models(),
        topic_service.load_models()
    )
    logger.info("✅ All services connected and ML models loaded successfully.")
    
    yield
    
    # Code to run on shutdown
    logger.info("Shutting down application")
    await close_redis()


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="An advanced agentic AI platform for intelligent question-answering based on user documents",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8501", "http://127.0.0.1:8501"],  # React dev server and Streamlit
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", settings.host]
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.version,
        "timestamp": time.time()
    }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP_ERROR",
            "message": exc.detail,
            "details": {"status_code": exc.status_code}
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_ERROR",
            "message": "An unexpected error occurred",
            "details": {"type": type(exc).__name__}
        }
    )


# API Routes
app.include_router(auth.router, prefix=settings.api_v1_prefix)
app.include_router(workspaces.router, prefix=settings.api_v1_prefix)
app.include_router(query.router, prefix=settings.api_v1_prefix)
app.include_router(knowledge.router, prefix=settings.api_v1_prefix)
app.include_router(analytics.router, prefix=settings.api_v1_prefix)
app.include_router(reasoning.router, prefix=settings.api_v1_prefix)


# Root endpoint
@app.get("/")
async def root():
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.version,
        "docs": "/docs" if settings.debug else "Documentation not available",
        "status": "running"
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )
