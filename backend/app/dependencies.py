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
