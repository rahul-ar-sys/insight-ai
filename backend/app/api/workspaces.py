from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, BackgroundTasks, Response
from sqlalchemy.orm import Session
from typing import List, Optional
import uuid
from datetime import datetime
from sqlalchemy import func
import json
import tempfile
import os

from ..core.database import get_db, get_redis
from ..core.auth import get_current_active_user, verify_workspace_access, User
from ..core.config import settings
from ..models.database import Workspace as WorkspaceModel, Document as DocumentModel
from ..models.schemas import (
    Workspace, WorkspaceCreate, WorkspaceUpdate, WorkspaceWithStats,
    Document, FileUploadResponse, User
)
from ..models.document_status import DocumentStatus
from ..services.document_processing import DocumentProcessingService
from ..dependencies import get_document_processing_service, get_vector_store_service
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

    if not workspaces:
        return []

    workspace_ids = [w.workspace_id for w in workspaces]

    # Perform a single, efficient query to get stats for all workspaces
    stats_query = db.query(
        WorkspaceModel.workspace_id,
        func.count(DocumentModel.document_id).label("doc_count"),
        func.sum(func.coalesce(DocumentModel.total_chunks, 0)).label("chunk_count")
    ).outerjoin(DocumentModel, WorkspaceModel.workspace_id == DocumentModel.workspace_id) \
     .filter(WorkspaceModel.workspace_id.in_(workspace_ids)) \
     .group_by(WorkspaceModel.workspace_id)

    stats_map = {stat.workspace_id: stat for stat in stats_query.all()}

    workspace_stats = []
    for workspace in workspaces:
        workspace_stat_data = stats_map.get(workspace.workspace_id)
        doc_count = workspace_stat_data.doc_count if workspace_stat_data else 0
        chunk_count = workspace_stat_data.chunk_count if workspace_stat_data else 0

        workspace_stat = WorkspaceWithStats(
            **workspace.__dict__,
            document_count=doc_count,
            session_count=0,  # Placeholder
            total_chunks=int(chunk_count or 0)
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

    workspace_with_stats = db.query(
        WorkspaceModel,
        func.count(DocumentModel.document_id).label("doc_count"),
        func.sum(func.coalesce(DocumentModel.total_chunks, 0)).label("chunk_count")
    ).outerjoin(DocumentModel, WorkspaceModel.workspace_id == DocumentModel.workspace_id) \
     .filter(WorkspaceModel.workspace_id == workspace_id) \
     .group_by(WorkspaceModel.workspace_id).first()

    if not workspace_with_stats:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found")

    workspace, doc_count, chunk_count = workspace_with_stats

    return WorkspaceWithStats(
        **workspace.__dict__,
        document_count=doc_count,
        session_count=0,  # Placeholder
        total_chunks=int(chunk_count or 0)
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
    processing_service: DocumentProcessingService = Depends(get_document_processing_service),
    db: Session = Depends(get_db)
):
    """
    Accepts a document, saves it to a temporary location, and schedules
    background processing. Returns immediately with a 202 Accepted response.
    """
    await verify_workspace_access(workspace_id, current_user, db, permission="write")
    
    # Stream the upload to a temporary file to avoid holding large files in memory.
    try:
        # Suffix helps identify the file if it leaks, but it's not for security.
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp_file:
            tmp_file_path = tmp_file.name
            file_size = 0
            # Read the file in chunks to handle large files efficiently
            while content_chunk := await file.read(1024 * 1024):  # 1MB chunks
                tmp_file.write(content_chunk)
                file_size += len(content_chunk)
    except Exception as e:
        logger.error(f"Failed to write uploaded file to temporary storage: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to save uploaded file.")

    # Create the document record immediately with a 'pending' status.
    # The storage_path will be updated by the background task.
    document = DocumentModel(
        workspace_id=uuid.UUID(workspace_id),
        file_name=file.filename,
        original_name=file.filename,
        file_type=file.content_type,
        file_size=file_size,
        storage_path=None,  # Will be set by the background task
        status="pending"
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    # Pass the ID and the temporary file path to the background task.
    background_tasks.add_task(
        processing_service.process_document_from_temp, # This now uses the injected singleton
        document.document_id,
        tmp_file_path,
        get_db_session_for_background
    )

    logger.info(f"Document {document.document_id} upload accepted, processing scheduled from {tmp_file_path}.")
    
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


async def _delete_document_background(document_id: str, workspace_id: str, storage_path: Optional[str], db_session_factory):
    """Background task to delete document data from all services."""
    logger.info(f"Starting background deletion for document {document_id}")
    db = next(db_session_factory())
    try:
        # 1. Delete embeddings from vector store
        vector_service = get_vector_store_service()
        await vector_service.delete_document_embeddings(document_id=document_id, workspace_id=workspace_id)

        # 2. Delete file from storage
        if storage_path:
            storage_service = StorageService()
            await storage_service.delete_file(storage_path)

        # 3. Delete the document record from the database
        document = db.query(DocumentModel).filter(DocumentModel.document_id == document_id).first()
        if document:
            db.delete(document)
            db.commit()
            logger.info(f"✅ Successfully deleted document {document_id} and all associated data.")
        else:
            logger.warning(f"Document {document_id} was already deleted from DB during background task.")

    except Exception as e:
        logger.error(f"❌ Background deletion for document {document_id} failed: {e}", exc_info=True)
        db.rollback()
    finally:
        db.close()


@router.delete("/{workspace_id}/documents/{document_id}", status_code=status.HTTP_202_ACCEPTED)
async def delete_document(
    workspace_id: str,
    document_id: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Schedules a document and its associated data for deletion."""
    await verify_workspace_access(workspace_id, current_user, db, permission="write")
    document = db.query(DocumentModel).filter(
        DocumentModel.document_id == document_id,
        DocumentModel.workspace_id == workspace_id
    ).first()

    if not document:
        # Return 204 No Content if it's already gone to make client-side logic simpler.
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    # Schedule the background deletion
    background_tasks.add_task(
        _delete_document_background,
        str(document.document_id),
        str(document.workspace_id),
        document.storage_path,
        get_db_session_for_background
    )

    logger.info(f"Deletion scheduled for document {document_id} from workspace {workspace_id}")
    return Response(status_code=status.HTTP_202_ACCEPTED)