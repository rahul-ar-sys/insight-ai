import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # App Configuration
    app_name: str = "Agentic AI Q&A Platform"
    version: str = "1.1.0"
    debug: bool = False
    
    # API Configuration
    api_v1_prefix: str = "/api/v1"
    host: str = "localhost"
    port: int = 8000
    
    # Database Configuration
    postgres_user: str = "postgres"
    postgres_password: str = "password"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "insight_ai"
    
    @property
    def database_url(self) -> str:
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    # Redis Configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    @property
    def redis_url(self) -> str:
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    # Vector Database Configuration
    vector_db_type: str = "chroma"  # or "pinecone", "weaviate" 
    
    
    # LLM Configuration
    
    google_api_key: Optional[str] = None
    default_llm: str = "google" 
    
    # Cloud Storage Configuration
    storage_type: str = "gcs"  # or "s3"
    gcs_bucket_name: Optional[str] = None
    gcs_credentials_path: Optional[str] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_bucket_name: Optional[str] = None
    aws_region: str = "us-west-2"
    
    # Authentication Configuration
    firebase_credentials_path: Optional[str] = None
    jwt_secret_key: str = "your-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    
    # OCR Configuration
    # In app/core/config.py
    tesseract_path: Optional[str] = "C:/Program Files/Tesseract-OCR/tesseract.exe"


    # External APIs
    tavily_api_key: Optional[str] = None
    
    # Processing Configuration
    max_file_size_mb: int = 100
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_model: str = "E:/insight_ai/backend/all-MiniLM-L6-v2"
    
    # Session Configuration
    session_timeout_minutes: int = 60
    max_conversation_turns: int = 50
    
    # Additional Configuration
    secret_key: str = "your-secret-key-change-in-production"
    
    # ChromaDB Configuration
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_persist_directory: str = "./data/chroma"
    chroma_use_persistent_client: bool = True
    
   
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "allow"  # Allow extra fields from environment


settings = Settings()
