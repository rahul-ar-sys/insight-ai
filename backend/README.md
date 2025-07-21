# Insight AI Backend

This is the backend service for the Agentic AI Q&A Platform.

## Features

- **Multi-Agent Architecture**: LangGraph-based orchestration with specialized agents
- **Document Processing**: Intelligent ingestion with OCR support
- **Vector Search**: Semantic document retrieval using embeddings
- **Knowledge Graphs**: Entity extraction and relationship mapping  
- **Contradiction Detection**: Automatic detection of conflicting information
- **Web Search Integration**: Hybrid knowledge from documents and web
- **Authentication**: Firebase Auth with JWT tokens
- **Real-time Processing**: Async document processing pipeline

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. Initialize database:
```bash
alembic upgrade head
```

4. Run the server:
```bash
uvicorn app.main:app --reload
```

## API Documentation

When running in debug mode, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Architecture

```
app/
├── main.py              # FastAPI application entry point
├── core/                # Core configuration and utilities
│   ├── config.py        # Settings and configuration
│   ├── database.py      # Database connections
│   ├── auth.py          # Authentication and security
│   └── logging.py       # Logging configuration
├── models/              # Data models and schemas
│   ├── database.py      # SQLAlchemy models
│   └── schemas.py       # Pydantic schemas
├── api/                 # API route handlers
│   ├── workspaces.py    # Workspace management
│   ├── query.py         # Query processing
│   ├── knowledge.py     # Knowledge graph
│   └── analytics.py     # Analytics and insights
├── agents/              # Multi-agent system
│   └── orchestrator.py  # LangGraph orchestration
├── tools/               # Agent tools
│   └── __init__.py      # Tool registry and implementations
└── services/            # Business logic services
    ├── document_processing.py
    ├── storage.py       # File storage abstraction
    └── vector_store.py  # Vector database operations
```
