from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
import uuid
from datetime import datetime
import json

from ..core.database import get_db, get_redis
from ..core.auth import get_current_active_user, verify_workspace_access
from ..core.config import settings
from ..models.database import Workspace as WorkspaceModel, Document as DocumentModel
from ..models.schemas import (
    Workspace, WorkspaceCreate, WorkspaceUpdate, WorkspaceWithStats,
    Document, FileUploadResponse, User
)
from ..models.document_status import DocumentStatus
from ..services.document_processing import DocumentProcessingService
from ..services.storage import StorageService
from ..core.logging import logger
from ..services.vector_store import VectorStoreService
from ..dependencies import get_db_session_for_background

router = APIRouter(prefix="/workspaces", tags=["workspaces"])


@router.get("/", response_model=List[WorkspaceWithStats])
async def list_workspaces(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """List all workspaces for the current user"""
    workspaces = db.query(WorkspaceModel).filter(
        WorkspaceModel.owner_id == current_user.user_id,
        WorkspaceModel.is_active == True
    ).all()
    
    workspace_stats = []
    for workspace in workspaces:
        doc_count = db.query(DocumentModel).filter(DocumentModel.workspace_id == workspace.workspace_id).count()
        total_chunks = db.query(DocumentModel).filter(DocumentModel.workspace_id == workspace.workspace_id).with_entities(DocumentModel.total_chunks).all()
        chunk_count = sum(chunks[0] or 0 for chunks in total_chunks)
        
        workspace_stat = WorkspaceWithStats(
            **workspace.__dict__,
            document_count=doc_count,
            session_count=0,  # Placeholder
            total_chunks=chunk_count
        )
        workspace_stats.append(workspace_stat)
    
    return workspace_stats


@router.post("/", response_model=Workspace, status_code=status.HTTP_201_CREATED)
async def create_workspace(
    workspace_data: WorkspaceCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Create a new workspace"""
    new_workspace = WorkspaceModel(
        name=workspace_data.name,
        description=workspace_data.description,
        owner_id=current_user.user_id,
        settings=workspace_data.settings or {}
    )
    db.add(new_workspace)
    db.commit()
    db.refresh(new_workspace)
    logger.info(f"Created workspace {new_workspace.workspace_id} for user {current_user.user_id}")
    return new_workspace


@router.get("/{workspace_id}", response_model=WorkspaceWithStats)
async def get_workspace(
    workspace_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get a specific workspace with statistics"""
    await verify_workspace_access(workspace_id, current_user, db)
    workspace = db.query(WorkspaceModel).filter(WorkspaceModel.workspace_id == workspace_id).first()
    if not workspace:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found")
    
    doc_count = db.query(DocumentModel).filter(DocumentModel.workspace_id == workspace_id).count()
    total_chunks = db.query(DocumentModel).filter(DocumentModel.workspace_id == workspace_id).with_entities(DocumentModel.total_chunks).all()
    chunk_count = sum(chunks[0] or 0 for chunks in total_chunks)
    
    return WorkspaceWithStats(
        **workspace.__dict__,
        document_count=doc_count,
        session_count=0, # Placeholder
        total_chunks=chunk_count
    )


@router.put("/{workspace_id}", response_model=Workspace)
async def update_workspace(
    workspace_id: str,
    workspace_data: WorkspaceUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update a workspace"""
    await verify_workspace_access(workspace_id, current_user, db, permission="write")
    workspace = db.query(WorkspaceModel).filter(WorkspaceModel.workspace_id == workspace_id).first()
    if not workspace:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found")
    
    update_data = workspace_data.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(workspace, field, value)
    
    workspace.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(workspace)
    logger.info(f"Updated workspace {workspace_id}")
    return workspace


@router.delete("/{workspace_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_workspace(
    workspace_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete a workspace (soft delete)"""
    await verify_workspace_access(workspace_id, current_user, db, permission="write")
    workspace = db.query(WorkspaceModel).filter(WorkspaceModel.workspace_id == workspace_id).first()
    if not workspace:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found")
    
    workspace.is_active = False
    workspace.updated_at = datetime.utcnow()
    db.commit()
    logger.info(f"Deleted workspace {workspace_id}")


@router.get("/{workspace_id}/documents", response_model=List[Document])
async def list_documents(
    workspace_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """List all documents in a workspace"""
    await verify_workspace_access(workspace_id, current_user, db)
    documents = db.query(DocumentModel).filter(DocumentModel.workspace_id == workspace_id).order_by(DocumentModel.created_at.desc()).all()
    return documents


@router.post("/{workspace_id}/documents/upload", response_model=FileUploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    workspace_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Upload a document and start background processing."""
    await verify_workspace_access(workspace_id, current_user, db, permission="write")
    
    storage_service = StorageService()
    file_content = await file.read()
    
    # Store the file first to get a persistent path
    storage_path = await storage_service.store_file(
        file_content,
        f"{workspace_id}/{uuid.uuid4()}_{file.filename}"
    )

    document = DocumentModel(
        workspace_id=uuid.UUID(workspace_id),
        file_name=file.filename,
        original_name=file.filename,
        file_type=file.content_type,
        file_size=len(file_content),
        storage_path=storage_path,
        status="processing"
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    # Use background task for processing
    # This ensures the background task gets its own DB session
    processing_service = DocumentProcessingService()
    background_tasks.add_task(
        processing_service.process_document, 
        document, 
        file_content, 
        get_db_session_for_background
    )

    logger.info(f"Document {document.document_id} upload accepted, processing started in background.")
    
    return FileUploadResponse(
        document_id=document.document_id,
        message="Document upload accepted and processing has started.",
        processing_started=True
    )


@router.get("/{workspace_id}/documents/{document_id}/status", response_model=DocumentStatus)
async def get_document_processing_status(
    workspace_id: str,
    document_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get detailed real-time processing status for a document from Redis."""
    await verify_workspace_access(workspace_id, current_user, db)
    
    redis = await get_redis()
    status_key = f"document_status:{document_id}"
    
    try:
        status_data = await redis.get(status_key)
        if status_data:
            return DocumentStatus.parse_raw(status_data)
    except Exception as e:
        logger.error(f"Could not retrieve status from Redis for {document_id}: {e}")

    # Fallback to database if Redis has no data
    logger.warning(f"No status in Redis for {document_id}. Falling back to DB.")
    document = db.query(DocumentModel).filter(DocumentModel.document_id == document_id).first()
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    # Construct a status object from the DB state
    return DocumentStatus(
        document_id=document.document_id,
        file_name=document.original_name,
        status=document.status,
        progress=100.0 if document.status in ["ready", "error"] else 0.0,
        steps=[], # Cannot reconstruct steps from DB
        error_message=document.error_message
    )


@router.get("/{workspace_id}/documents/{document_id}", response_model=Document)
async def get_document(
    workspace_id: str,
    document_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get a specific document's metadata from the database."""
    await verify_workspace_access(workspace_id, current_user, db)
    document = db.query(DocumentModel).filter(
        DocumentModel.document_id == document_id,
        DocumentModel.workspace_id == workspace_id
    ).first()
    
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    
    return document


@router.delete("/{workspace_id}/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    workspace_id: str,
    document_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete a document and its associated data."""
    await verify_workspace_access(workspace_id, current_user, db, permission="write")
    document = db.query(DocumentModel).filter(
        DocumentModel.document_id == document_id,
        DocumentModel.workspace_id == workspace_id
    ).first()

    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    # Use the VectorStoreService to delete embeddings
    vector_service = VectorStoreService()
    await vector_service.delete_document_embeddings(document_id=str(document.document_id), workspace_id=str(workspace_id))

    # Use the StorageService to delete the file
    storage_service = StorageService()
    if document.storage_path:
        await storage_service.delete_file(document.storage_path)

    # Delete the document record from the database
    db.delete(document)
    db.commit()
    
    logger.info(f"Deleted document {document_id} from workspace {workspace_id}")