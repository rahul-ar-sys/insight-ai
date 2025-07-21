from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List, Optional
import uuid
from datetime import datetime

from ..core.database import get_db
from ..core.auth import get_current_active_user, verify_workspace_access, upload_rate_limiter, check_rate_limit
from ..core.config import settings
from ..models.database import Workspace as WorkspaceModel, Document as DocumentModel
from ..models.schemas import (
    Workspace, WorkspaceCreate, WorkspaceUpdate, WorkspaceWithStats,
    Document, FileUploadResponse, User, ErrorResponse
)
from ..services.document_processing import DocumentProcessingService
from ..core.logging import logger

router = APIRouter(prefix="/workspaces", tags=["workspaces"])


@router.get("/", response_model=List[WorkspaceWithStats])
async def list_workspaces(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """List all workspaces for the current user"""
    
    try:
        workspaces = db.query(WorkspaceModel).filter(
            WorkspaceModel.owner_id == current_user.user_id,
            WorkspaceModel.is_active == True
        ).all()
        
        # Add statistics for each workspace
        workspace_stats = []
        for workspace in workspaces:
            # Count documents
            doc_count = db.query(DocumentModel).filter(
                DocumentModel.workspace_id == workspace.workspace_id
            ).count()
            
            # Count sessions (would need Session model)
            session_count = 0  # TODO: Implement when Session model is available
            
            # Count total chunks
            total_chunks = db.query(DocumentModel).filter(
                DocumentModel.workspace_id == workspace.workspace_id
            ).with_entities(DocumentModel.total_chunks).all()
            chunk_count = sum(chunks[0] or 0 for chunks in total_chunks)
            
            workspace_stat = WorkspaceWithStats(
                workspace_id=workspace.workspace_id,
                name=workspace.name,
                description=workspace.description,
                settings=workspace.settings,
                owner_id=workspace.owner_id,
                created_at=workspace.created_at,
                updated_at=workspace.updated_at,
                is_active=workspace.is_active,
                document_count=doc_count,
                session_count=session_count,
                total_chunks=chunk_count
            )
            workspace_stats.append(workspace_stat)
        
        return workspace_stats
        
    except Exception as e:
        logger.error(f"Error listing workspaces: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve workspaces"
        )


@router.post("/", response_model=Workspace, status_code=status.HTTP_201_CREATED)
async def create_workspace(
    workspace_data: WorkspaceCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Create a new workspace"""
    
    try:
        # Create new workspace
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
        
        return Workspace(
            workspace_id=new_workspace.workspace_id,
            name=new_workspace.name,
            description=new_workspace.description,
            settings=new_workspace.settings,
            owner_id=new_workspace.owner_id,
            created_at=new_workspace.created_at,
            updated_at=new_workspace.updated_at,
            is_active=new_workspace.is_active
        )
        
    except Exception as e:
        logger.error(f"Error creating workspace: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create workspace"
        )


@router.get("/{workspace_id}", response_model=WorkspaceWithStats)
async def get_workspace(
    workspace_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get a specific workspace with statistics"""
    
    # Verify access
    await verify_workspace_access(workspace_id, current_user, db)
    
    try:
        workspace = db.query(WorkspaceModel).filter(
            WorkspaceModel.workspace_id == workspace_id
        ).first()
        
        if not workspace:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Workspace not found"
            )
        
        # Calculate statistics
        doc_count = db.query(DocumentModel).filter(
            DocumentModel.workspace_id == workspace_id
        ).count()
        
        session_count = 0  # TODO: Implement when Session model is available
        
        total_chunks = db.query(DocumentModel).filter(
            DocumentModel.workspace_id == workspace_id
        ).with_entities(DocumentModel.total_chunks).all()
        chunk_count = sum(chunks[0] or 0 for chunks in total_chunks)
        
        workspace_stats = WorkspaceWithStats(
            **workspace.__dict__,
            document_count=doc_count,
            session_count=session_count,
            total_chunks=chunk_count
        )
        
        return workspace_stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving workspace {workspace_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve workspace"
        )


@router.put("/{workspace_id}", response_model=Workspace)
async def update_workspace(
    workspace_id: str,
    workspace_data: WorkspaceUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update a workspace"""
    
    # Verify access
    await verify_workspace_access(workspace_id, current_user, db, permission="write")
    
    try:
        workspace = db.query(WorkspaceModel).filter(
            WorkspaceModel.workspace_id == workspace_id
        ).first()
        
        if not workspace:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Workspace not found"
            )
        
        # Update fields
        update_data = workspace_data.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(workspace, field, value)
        
        workspace.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(workspace)
        
        logger.info(f"Updated workspace {workspace_id}")
        
        return Workspace.from_orm(workspace)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating workspace {workspace_id}: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update workspace"
        )


@router.delete("/{workspace_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_workspace(
    workspace_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete a workspace (soft delete)"""
    
    # Verify access
    await verify_workspace_access(workspace_id, current_user, db, permission="write")
    
    try:
        workspace = db.query(WorkspaceModel).filter(
            WorkspaceModel.workspace_id == workspace_id
        ).first()
        
        if not workspace:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Workspace not found"
            )
        
        # Soft delete
        workspace.is_active = False
        workspace.updated_at = datetime.utcnow()
        db.commit()
        
        logger.info(f"Deleted workspace {workspace_id}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting workspace {workspace_id}: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete workspace"
        )


@router.get("/{workspace_id}/documents", response_model=List[Document])
async def list_documents(
    workspace_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """List all documents in a workspace"""
    
    # Verify access
    await verify_workspace_access(workspace_id, current_user, db)
    
    try:
        documents = db.query(DocumentModel).filter(
            DocumentModel.workspace_id == workspace_id
        ).order_by(DocumentModel.created_at.desc()).all()
        
        return [Document.from_orm(doc) for doc in documents]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing documents for workspace {workspace_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve documents"
        )


@router.post("/{workspace_id}/documents", response_model=FileUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    workspace_id: str,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Upload a document to a workspace"""
    
    # Verify access and rate limit
    await verify_workspace_access(workspace_id, current_user, db, permission="write")
    check_rate_limit(upload_rate_limiter, current_user)
    
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No file provided"
            )
        
        # Check file size
        file_content = await file.read()
        file_size = len(file_content)
        
        max_size = settings.max_file_size_mb * 1024 * 1024
        if file_size > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size exceeds maximum limit of {settings.max_file_size_mb}MB"
            )
        
        # Validate file type
        allowed_types = [
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
            "text/plain",
            "text/markdown",
            "image/png",
            "image/jpeg",
            "image/jpg"
        ]
        
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type {file.content_type} not supported"
            )
        
        # Create document record
        document = DocumentModel(
            workspace_id=uuid.UUID(workspace_id),
            file_name=file.filename,
            original_name=file.filename,
            file_type=file.content_type,
            file_size=file_size,
            storage_path="",  # Will be set by processing service
            status="processing",
            doc_metadata={
                "uploaded_by": str(current_user.user_id),
                "upload_timestamp": datetime.utcnow().isoformat()
            }
        )
        
        db.add(document)
        db.commit()
        db.refresh(document)
        
        # Start background processing
        processing_service = DocumentProcessingService()
        # Note: In production, this should be handled by a task queue like Celery
        import asyncio
        asyncio.create_task(processing_service.process_document(file_content, document, db))
        
        logger.info(f"Document {document.document_id} uploaded to workspace {workspace_id}")
        
        return FileUploadResponse(
            document_id=document.document_id,
            message="Document uploaded successfully and processing started",
            processing_started=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document to workspace {workspace_id}: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload document"
        )


@router.get("/{workspace_id}/documents/{document_id}", response_model=Document)
async def get_document(
    workspace_id: str,
    document_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get a specific document"""
    
    # Verify access
    await verify_workspace_access(workspace_id, current_user, db)
    
    try:
        document = db.query(DocumentModel).filter(
            DocumentModel.workspace_id == workspace_id,
            DocumentModel.document_id == document_id
        ).first()
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        return Document.from_orm(document)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving document {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document"
        )


@router.get("/{workspace_id}/documents/{document_id}/status")
async def get_document_processing_status(
    workspace_id: str,
    document_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get detailed processing status for a document"""
    
    # Verify access
    await verify_workspace_access(workspace_id, current_user, db)
    
    try:
        document = db.query(DocumentModel).filter(
            DocumentModel.workspace_id == workspace_id,
            DocumentModel.document_id == document_id
        ).first()
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        # Get processing steps and current status
        processing_steps = [
            {
                "step": "File Upload",
                "status": "completed" if document.storage_path else "pending",
                "description": "Document uploaded and stored securely"
            },
            {
                "step": "Text Extraction", 
                "status": "completed" if document.storage_path and document.status != "processing" else "in_progress" if document.status == "processing" else "pending",
                "description": "Extracting text content from document"
            },
            {
                "step": "OCR Processing",
                "status": "completed" if document.ocr_applied else "skipped" if document.status != "processing" else "pending",
                "description": "Optical Character Recognition for scanned content"
            },
            {
                "step": "Content Chunking",
                "status": "completed" if document.total_chunks > 0 else "in_progress" if document.status == "processing" else "pending", 
                "description": "Breaking document into semantic chunks"
            },
            {
                "step": "Embedding Generation",
                "status": "completed" if document.embeddings_generated else "in_progress" if document.status == "processing" else "pending",
                "description": "Creating AI embeddings for semantic search"
            },
            {
                "step": "Indexing",
                "status": "completed" if document.status == "ready" else "in_progress" if document.status == "processing" else "failed" if document.status == "error" else "pending",
                "description": "Storing in vector database for AI retrieval"
            }
        ]
        
        # Calculate overall progress
        completed_steps = sum(1 for step in processing_steps if step["status"] == "completed")
        total_steps = len([step for step in processing_steps if step["status"] != "skipped"])
        progress_percentage = int((completed_steps / total_steps) * 100) if total_steps > 0 else 0
        
        return {
            "document_id": document_id,
            "status": document.status,
            "progress_percentage": progress_percentage,
            "steps": processing_steps,
            "total_chunks": document.total_chunks,
            "ocr_applied": document.ocr_applied,
            "embeddings_generated": document.embeddings_generated,
            "error_message": document.error_message,
            "created_at": document.created_at,
            "updated_at": document.updated_at
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting processing status for document {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get processing status"
        )


@router.get("/{workspace_id}/documents/{document_id}", response_model=Document)
async def get_document(
    workspace_id: str,
    document_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get a specific document"""
    
    # Verify access
    await verify_workspace_access(workspace_id, current_user, db)
    
    try:
        document = db.query(DocumentModel).filter(
            DocumentModel.document_id == document_id,
            DocumentModel.workspace_id == workspace_id
        ).first()
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        return Document.from_orm(document)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving document {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document"
        )


@router.delete("/{workspace_id}/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    workspace_id: str,
    document_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete a document"""
    
    # Verify access
    await verify_workspace_access(workspace_id, current_user, db, permission="write")
    
    try:
        document = db.query(DocumentModel).filter(
            DocumentModel.document_id == document_id,
            DocumentModel.workspace_id == workspace_id
        ).first()
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        # Delete document using processing service
        processing_service = DocumentProcessingService()
        success = await processing_service.delete_document(uuid.UUID(document_id), db)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete document"
            )
        
        logger.info(f"Deleted document {document_id} from workspace {workspace_id}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete document"
        )
