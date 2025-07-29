from .core.database import SessionLocal

def get_db_session_for_background():
    """
    Dependency to create a new database session for background tasks.
    Ensures that each background task has its own isolated session.
    """
    db = None
    try:
        db = SessionLocal()
        yield db
    finally:
        if db:
            db.close()
# In app/dependencies.py

from .services.vector_store import VectorStoreService


# --- Singleton instances ---
_vector_store_instance = None
_processing_service_instance = None

def get_vector_store_service():
    """Dependency function to get the singleton instance of the VectorStoreService."""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStoreService()
    return _vector_store_instance

def get_document_processing_service():
    """Dependency function to get the singleton instance of the DocumentProcessingService."""
    from .services.document_processing import DocumentProcessingService
    global _processing_service_instance
    if _processing_service_instance is None:
        vector_service = get_vector_store_service()
        _processing_service_instance = DocumentProcessingService(vector_service=vector_service)
    return _processing_service_instance