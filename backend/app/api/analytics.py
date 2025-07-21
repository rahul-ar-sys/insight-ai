from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from datetime import datetime

from ..core.database import get_db
from ..core.auth import get_current_active_user, verify_workspace_access
from ..models.database import (
    Contradiction as ContradictionModel,
    Workspace as WorkspaceModel,
    Document as DocumentModel,
    ConversationTurn as ConversationTurnModel
)
from ..models.schemas import (
    Contradiction, WorkspaceAnalytics, InsightData, User, ErrorResponse
)
from ..core.logging import logger

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/{workspace_id}/overview", response_model=WorkspaceAnalytics)
async def get_workspace_analytics(
    workspace_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get comprehensive analytics for a workspace"""
    
    # Verify access
    await verify_workspace_access(workspace_id, current_user, db)
    
    try:
        # Total documents
        total_documents = db.query(DocumentModel).filter(
            DocumentModel.workspace_id == workspace_id
        ).count()
        
        # Total chunks
        total_chunks_result = db.query(
            db.func.sum(DocumentModel.total_chunks)
        ).filter(
            DocumentModel.workspace_id == workspace_id
        ).scalar()
        total_chunks = total_chunks_result or 0
        
        # Total conversations
        total_conversations = db.query(ConversationTurnModel).join(
            # Assuming Session model exists and has workspace_id
            # For now, we'll count conversation turns
        ).count()
        
        # Total entities (assuming KnowledgeEntity model)
        total_entities = 0  # TODO: Implement when knowledge entities are available
        
        # Total relationships
        total_relationships = 0  # TODO: Implement when knowledge relationships are available
        
        # Total contradictions
        total_contradictions = db.query(ContradictionModel).filter(
            ContradictionModel.workspace_id == workspace_id
        ).count()
        
        # Recent activity (last 10 activities)
        recent_activity = []
        
        # Recent document uploads
        recent_docs = db.query(DocumentModel).filter(
            DocumentModel.workspace_id == workspace_id
        ).order_by(DocumentModel.created_at.desc()).limit(5).all()
        
        for doc in recent_docs:
            recent_activity.append({
                "type": "document_upload",
                "description": f"Uploaded document: {doc.file_name}",
                "timestamp": doc.created_at.isoformat() if doc.created_at else "",
                "metadata": {"document_id": str(doc.document_id), "status": doc.status}
            })
        
        # Recent conversations
        recent_turns = db.query(ConversationTurnModel).order_by(
            ConversationTurnModel.timestamp.desc()
        ).limit(5).all()
        
        for turn in recent_turns:
            recent_activity.append({
                "type": "conversation",
                "description": f"Q&A: {turn.query_text[:50]}...",
                "timestamp": turn.timestamp.isoformat() if turn.timestamp else "",
                "metadata": {"turn_id": str(turn.turn_id), "confidence": turn.confidence_score}
            })
        
        # Sort recent activity by timestamp
        recent_activity.sort(key=lambda x: x["timestamp"], reverse=True)
        recent_activity = recent_activity[:10]
        
        # Topic distribution (simplified - would use NLP in production)
        topic_distribution = {
            "General": total_documents,
            # TODO: Implement proper topic modeling
        }
        
        # Entity types distribution
        entity_types = {
            # TODO: Implement when knowledge entities are available
        }
        
        analytics = WorkspaceAnalytics(
            total_documents=total_documents,
            total_chunks=total_chunks,
            total_conversations=total_conversations,
            total_entities=total_entities,
            total_relationships=total_relationships,
            total_contradictions=total_contradictions,
            recent_activity=recent_activity,
            topic_distribution=topic_distribution,
            entity_types=entity_types
        )
        
        return analytics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving workspace analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve analytics"
        )


@router.get("/{workspace_id}/insights", response_model=InsightData)
async def get_proactive_insights(
    workspace_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get proactive insights for a workspace"""
    
    # Verify access
    await verify_workspace_access(workspace_id, current_user, db)
    
    try:
        # Get documents for analysis
        documents = db.query(DocumentModel).filter(
            DocumentModel.workspace_id == workspace_id,
            DocumentModel.status == "ready"
        ).all()
        
        if not documents:
            return InsightData(
                key_themes=[],
                frequent_entities=[],
                trend_summaries=[],
                contradiction_summary={},
                recommendations=["Upload documents to get insights"]
            )
        
        # Key themes (simplified - would use topic modeling in production)
        key_themes = [
            "General Information",
            # TODO: Implement proper topic extraction using BERTopic or similar
        ]
        
        # Frequent entities (placeholder)
        frequent_entities = [
            # TODO: Implement when knowledge entities are available
        ]
        
        # Trend summaries
        trend_summaries = [
            f"Total of {len(documents)} documents uploaded",
            f"Recent upload activity shows consistent document addition",
            # TODO: Implement temporal analysis
        ]
        
        # Contradiction summary
        contradictions = db.query(ContradictionModel).filter(
            ContradictionModel.workspace_id == workspace_id
        ).all()
        
        contradiction_summary = {
            "total_contradictions": len(contradictions),
            "unresolved_contradictions": len([c for c in contradictions if c.resolution_status == "unresolved"]),
            "contradiction_types": {},
            "recent_contradictions": []
        }
        
        if contradictions:
            # Group by type
            type_counts = {}
            for contradiction in contradictions:
                type_counts[contradiction.contradiction_type] = type_counts.get(contradiction.contradiction_type, 0) + 1
            contradiction_summary["contradiction_types"] = type_counts
            
            # Recent contradictions
            recent_contradictions = sorted(contradictions, key=lambda x: x.created_at, reverse=True)[:3]
            contradiction_summary["recent_contradictions"] = [
                {
                    "type": c.contradiction_type,
                    "description": c.description,
                    "confidence": c.confidence_score,
                    "status": c.resolution_status
                }
                for c in recent_contradictions
            ]
        
        # Proactive recommendations
        recommendations = []
        
        if len(documents) < 5:
            recommendations.append("Consider uploading more documents to improve answer quality")
        
        if contradiction_summary["unresolved_contradictions"] > 0:
            recommendations.append(f"Review {contradiction_summary['unresolved_contradictions']} unresolved contradictions")
        
        # Check for documents with processing errors
        error_docs = [d for d in documents if d.status == "error"]
        if error_docs:
            recommendations.append(f"Fix processing errors in {len(error_docs)} documents")
        
        # Check for low OCR confidence
        ocr_docs = [d for d in documents if d.ocr_applied]
        if ocr_docs:
            recommendations.append("Review OCR-processed documents for accuracy")
        
        if not recommendations:
            recommendations.append("Your workspace is well-organized and up to date!")
        
        insights = InsightData(
            key_themes=key_themes,
            frequent_entities=frequent_entities,
            trend_summaries=trend_summaries,
            contradiction_summary=contradiction_summary,
            recommendations=recommendations
        )
        
        return insights
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate insights"
        )


@router.get("/{workspace_id}/contradictions", response_model=List[Contradiction])
async def get_contradictions(
    workspace_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get all contradictions detected in a workspace"""
    
    # Verify access
    await verify_workspace_access(workspace_id, current_user, db)
    
    try:
        contradictions = db.query(ContradictionModel).filter(
            ContradictionModel.workspace_id == workspace_id
        ).order_by(ContradictionModel.created_at.desc()).all()
        
        return [Contradiction.from_orm(c) for c in contradictions]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving contradictions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve contradictions"
        )


@router.put("/{workspace_id}/contradictions/{contradiction_id}/resolve", response_model=Contradiction)
async def resolve_contradiction(
    workspace_id: str,
    contradiction_id: str,
    resolution_data: Dict[str, Any],
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Mark a contradiction as resolved"""
    
    # Verify access
    await verify_workspace_access(workspace_id, current_user, db, permission="write")
    
    try:
        contradiction = db.query(ContradictionModel).filter(
            ContradictionModel.contradiction_id == contradiction_id,
            ContradictionModel.workspace_id == workspace_id
        ).first()
        
        if not contradiction:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Contradiction not found"
            )
        
        # Update resolution status
        status_value = resolution_data.get("status", "resolved")
        if status_value not in ["resolved", "dismissed"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid resolution status"
            )
        
        contradiction.resolution_status = status_value
        contradiction.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(contradiction)
        
        logger.info(f"Resolved contradiction {contradiction_id} in workspace {workspace_id}")
        
        return Contradiction.from_orm(contradiction)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resolving contradiction: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to resolve contradiction"
        )
