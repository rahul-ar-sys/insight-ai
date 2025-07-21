from pydantic import BaseModel, Field
from typing import List, Optional
from uuid import UUID
from enum import Enum

class ProcessingStepStatus(str, Enum):
    """Enum for the status of a processing step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class ProcessingStep(BaseModel):
    """Model for a single step in the document processing pipeline."""
    name: str
    status: ProcessingStepStatus = ProcessingStepStatus.PENDING
    description: Optional[str] = None

class DocumentStatus(BaseModel):
    """Model for tracking the real-time status of a document being processed."""
    document_id: UUID
    file_name: str
    status: str = "processing"  # Overall status: processing, ready, error
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    steps: List[ProcessingStep]
    error_message: Optional[str] = None
