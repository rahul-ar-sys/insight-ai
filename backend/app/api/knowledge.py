from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Dict, Any

from ..core.database import get_db
from ..core.auth import get_current_active_user, verify_workspace_access
from ..models.database import (
    KnowledgeEntity as KnowledgeEntityModel,
    KnowledgeRelationship as KnowledgeRelationshipModel,
    Workspace as WorkspaceModel
)
from ..models.schemas import (
    KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge,
    User, ErrorResponse
)
from ..core.logging import logger

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


@router.get("/{workspace_id}/graph", response_model=KnowledgeGraph)
async def get_knowledge_graph(
    workspace_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get the knowledge graph for a workspace"""
    
    # Verify access
    await verify_workspace_access(workspace_id, current_user, db)
    
    try:
        # Get all entities in the workspace
        entities = db.query(KnowledgeEntityModel).filter(
            KnowledgeEntityModel.workspace_id == workspace_id
        ).all()
        
        # Get all relationships in the workspace
        relationships = db.query(KnowledgeRelationshipModel).join(
            KnowledgeEntityModel,
            KnowledgeRelationshipModel.subject_id == KnowledgeEntityModel.entity_id
        ).filter(
            KnowledgeEntityModel.workspace_id == workspace_id
        ).all()
        
        # Create nodes
        nodes = []
        for entity in entities:
            node = KnowledgeGraphNode(
                id=str(entity.entity_id),
                label=entity.name,
                type=entity.entity_type,
                properties={
                    "description": entity.description or "",
                    "confidence": entity.confidence_score or 0,
                    "mention_count": entity.mention_count,
                    "created_at": entity.created_at.isoformat() if entity.created_at else "",
                    "metadata": entity.entity_metadata or {}
                }
            )
            nodes.append(node)
        
        # Create edges
        edges = []
        for relationship in relationships:
            edge = KnowledgeGraphEdge(
                source=str(relationship.subject_id),
                target=str(relationship.object_id),
                relationship=relationship.predicate,
                properties={
                    "confidence": relationship.confidence_score or 0,
                    "evidence_count": relationship.evidence_count,
                    "created_at": relationship.created_at.isoformat() if relationship.created_at else "",
                    "metadata": relationship.relationship_metadata or {}
                }
            )
            edges.append(edge)
        
        knowledge_graph = KnowledgeGraph(nodes=nodes, edges=edges)
        
        logger.info(f"Retrieved knowledge graph for workspace {workspace_id}: {len(nodes)} nodes, {len(edges)} edges")
        
        return knowledge_graph
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving knowledge graph: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve knowledge graph"
        )


@router.get("/{workspace_id}/entities", response_model=Dict[str, Any])
async def get_entity_summary(
    workspace_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get entity summary statistics for a workspace"""
    
    # Verify access
    await verify_workspace_access(workspace_id, current_user, db)
    
    try:
        # Get entity type distribution
        entity_types = db.query(
            KnowledgeEntityModel.entity_type,
            db.func.count(KnowledgeEntityModel.entity_id).label('count')
        ).filter(
            KnowledgeEntityModel.workspace_id == workspace_id
        ).group_by(KnowledgeEntityModel.entity_type).all()
        
        entity_type_distribution = {entity_type: count for entity_type, count in entity_types}
        
        # Get top entities by mention count
        top_entities = db.query(KnowledgeEntityModel).filter(
            KnowledgeEntityModel.workspace_id == workspace_id
        ).order_by(KnowledgeEntityModel.mention_count.desc()).limit(10).all()
        
        top_entities_list = [
            {
                "name": entity.name,
                "type": entity.entity_type,
                "mention_count": entity.mention_count,
                "confidence": entity.confidence_score or 0
            }
            for entity in top_entities
        ]
        
        # Get relationship type distribution
        relationship_types = db.query(
            KnowledgeRelationshipModel.predicate,
            db.func.count(KnowledgeRelationshipModel.relationship_id).label('count')
        ).join(
            KnowledgeEntityModel,
            KnowledgeRelationshipModel.subject_id == KnowledgeEntityModel.entity_id
        ).filter(
            KnowledgeEntityModel.workspace_id == workspace_id
        ).group_by(KnowledgeRelationshipModel.predicate).all()
        
        relationship_type_distribution = {predicate: count for predicate, count in relationship_types}
        
        summary = {
            "total_entities": sum(entity_type_distribution.values()),
            "total_relationships": sum(relationship_type_distribution.values()),
            "entity_type_distribution": entity_type_distribution,
            "relationship_type_distribution": relationship_type_distribution,
            "top_entities": top_entities_list
        }
        
        return summary
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving entity summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve entity summary"
        )
