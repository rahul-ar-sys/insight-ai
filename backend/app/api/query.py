from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Optional
import uuid
from datetime import datetime

from ..core.database import get_db
from ..core.auth import get_current_active_user, verify_workspace_access, query_rate_limiter, check_rate_limit
from ..models.database import Session as SessionModel, ConversationTurn as ConversationTurnModel
from ..models.schemas import (
    QueryRequest, QueryResponse, Session as SessionSchema, 
    ConversationTurn, User, ErrorResponse
)
from ..agents.orchestrator import AgentOrchestrator
from ..core.logging import logger

router = APIRouter(prefix="/query", tags=["query"])

# Initialize the agent orchestrator
orchestrator = AgentOrchestrator()


@router.post("/", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Process a user query through the agentic AI system"""
    
    # Verify workspace access and rate limit
    await verify_workspace_access(str(request.workspace_id), current_user, db)
    check_rate_limit(query_rate_limiter, current_user)
    
    try:
        # Execute query through agent orchestrator
        response = await orchestrator.execute_query(request, str(current_user.user_id))
        
        # Create or get session
        session = await _get_or_create_session(
            request.workspace_id, 
            current_user.user_id, 
            request.session_id,
            db
        )
        
        # Store conversation turn
        conversation_turn = ConversationTurnModel(
            session_id=session.session_id,
            user_id=current_user.user_id,
            query_text=request.query,
            response_text=response.response,
            agent_trace={},  # TODO: Extract from orchestrator
            sources=response.sources,
            reasoning_steps=response.reasoning_steps,
            processing_time_ms=response.processing_time_ms,
            confidence_score=response.confidence_score
        )
        
        db.add(conversation_turn)
        
        # Update session
        session.updated_at = datetime.utcnow()
        if not session.title:
            # Generate title from first query
            session.title = request.query[:100] + "..." if len(request.query) > 100 else request.query
        
        db.commit()
        db.refresh(conversation_turn)
        
        # Update response with actual IDs
        response.session_id = session.session_id
        response.turn_id = conversation_turn.turn_id
        
        logger.info(f"Processed query for user {current_user.user_id} in workspace {request.workspace_id}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process query"
        )


async def _get_or_create_session(
    workspace_id: uuid.UUID,
    user_id: uuid.UUID,
    session_id: Optional[uuid.UUID],
    db: Session
) -> SessionModel:
    """Get existing session or create new one"""
    
    if session_id:
        # Try to get existing session
        session = db.query(SessionModel).filter(
            SessionModel.session_id == session_id,
            SessionModel.workspace_id == workspace_id,
            SessionModel.user_id == user_id,
            SessionModel.is_active == True
        ).first()
        
        if session:
            return session
    
    # Create new session
    new_session = SessionModel(
        workspace_id=workspace_id,
        user_id=user_id,
        title=None  # Will be set from first query
    )
    
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    
    return new_session


@router.get("/sessions/{workspace_id}", response_model=list[SessionSchema])
async def list_sessions(
    workspace_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """List all sessions in a workspace for the current user"""
    
    # Verify access
    await verify_workspace_access(workspace_id, current_user, db)
    
    try:
        sessions = db.query(SessionModel).filter(
            SessionModel.workspace_id == workspace_id,
            SessionModel.user_id == current_user.user_id,
            SessionModel.is_active == True
        ).order_by(SessionModel.updated_at.desc()).all()
        
        return [SessionSchema.from_orm(session) for session in sessions]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve sessions"
        )


@router.get("/sessions/{workspace_id}/{session_id}/history", response_model=list[ConversationTurn])
async def get_conversation_history(
    workspace_id: str,
    session_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get conversation history for a session"""
    
    # Verify access
    await verify_workspace_access(workspace_id, current_user, db)
    
    try:
        # Verify session belongs to user and workspace
        session = db.query(SessionModel).filter(
            SessionModel.session_id == session_id,
            SessionModel.workspace_id == workspace_id,
            SessionModel.user_id == current_user.user_id,
            SessionModel.is_active == True
        ).first()
        
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        # Get conversation turns
        turns = db.query(ConversationTurnModel).filter(
            ConversationTurnModel.session_id == session_id
        ).order_by(ConversationTurnModel.timestamp.asc()).all()
        
        return [ConversationTurn.from_orm(turn) for turn in turns]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve conversation history"
        )


@router.delete("/sessions/{workspace_id}/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    workspace_id: str,
    session_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete a session (soft delete)"""
    
    # Verify access
    await verify_workspace_access(workspace_id, current_user, db, permission="write")
    
    try:
        # Verify session belongs to user and workspace
        session = db.query(SessionModel).filter(
            SessionModel.session_id == session_id,
            SessionModel.workspace_id == workspace_id,
            SessionModel.user_id == current_user.user_id
        ).first()
        
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        # Soft delete
        session.is_active = False
        session.updated_at = datetime.utcnow()
        db.commit()
        
        logger.info(f"Deleted session {session_id}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete session"
        )
