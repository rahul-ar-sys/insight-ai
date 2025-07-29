from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from ..core.database import get_db
from ..core.auth import get_current_user
from ..core.logging import logger
from ..models.database import (
    User, Workspace, KnowledgeEntity, KnowledgeRelationship, Document
)
from ..models.schemas import (
    KnowledgeGraphResponse, KnowledgeEntityResponse, KnowledgeRelationshipResponse,
    TopicModelingResponse, EntitySearchResponse
)
from ..services.knowledge_extraction import KnowledgeExtractionService, TopicModelingService, get_knowledge_extraction_service, get_topic_modeling_service
from ..services.analytics import AnalyticsService

router = APIRouter(prefix="/knowledge", tags=["Knowledge Graph"])


@router.get("/{workspace_id}/graph", response_model=KnowledgeGraphResponse)
async def get_knowledge_graph(
    workspace_id: str,
    limit: int = Query(100, ge=1, le=1000),
    entity_types: Optional[List[str]] = Query(None),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    include_relationships: bool = Query(True),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get the knowledge graph for a workspace with filtering options"""
    
    try:
        # Verify workspace access
        workspace = db.query(Workspace).filter(
            Workspace.workspace_id == workspace_id,
            Workspace.owner_id == current_user.user_id
        ).first()
        
        if not workspace:
            raise HTTPException(status_code=404, detail="Workspace not found")
        
        # Build entity query with filters
        entity_query = db.query(KnowledgeEntity).filter(
            KnowledgeEntity.workspace_id == workspace_id,
            KnowledgeEntity.confidence_score >= int(min_confidence * 100)
        )
        
        if entity_types:
            entity_query = entity_query.filter(
                KnowledgeEntity.entity_type.in_(entity_types)
            )
        
        entities = entity_query.limit(limit).all()
        
        # Get relationships if requested
        relationships = []
        if include_relationships and entities:
            entity_ids = [e.entity_id for e in entities]
            
            relationships = db.query(KnowledgeRelationship).filter(
                KnowledgeRelationship.subject_id.in_(entity_ids),
                KnowledgeRelationship.object_id.in_(entity_ids),
                KnowledgeRelationship.confidence_score >= int(min_confidence * 100)
            ).all()
        
        # Calculate graph statistics
        total_entities = db.query(KnowledgeEntity).filter(
            KnowledgeEntity.workspace_id == workspace_id
        ).count()
        
        total_relationships = db.query(KnowledgeRelationship).join(
            KnowledgeEntity,
            KnowledgeRelationship.subject_id == KnowledgeEntity.entity_id
        ).filter(
            KnowledgeEntity.workspace_id == workspace_id
        ).count()
        
        # Convert to response format
        entity_responses = [
            KnowledgeEntityResponse(
                entity_id=str(e.entity_id),
                name=e.name,
                entity_type=e.entity_type,
                description=e.description,
                confidence_score=e.confidence_score / 100.0,
                mention_count=e.mention_count,
                metadata=e.entity_metadata or {},
                created_at=e.created_at,
                updated_at=e.updated_at
            ) for e in entities
        ]
        
        relationship_responses = [
            KnowledgeRelationshipResponse(
                relationship_id=str(r.relationship_id),
                subject_id=str(r.subject_id),
                object_id=str(r.object_id),
                predicate=r.predicate,
                confidence_score=r.confidence_score / 100.0,
                evidence_count=r.evidence_count,
                metadata=r.relationship_metadata or {},
                created_at=r.created_at,
                updated_at=r.updated_at
            ) for r in relationships
        ]
        
        return KnowledgeGraphResponse(
            entities=entity_responses,
            relationships=relationship_responses,
            statistics={
                "total_entities": total_entities,
                "total_relationships": total_relationships,
                "filtered_entities": len(entities),
                "filtered_relationships": len(relationships),
                "entity_types": list(set(e.entity_type for e in entities)),
                "relationship_types": list(set(r.predicate for r in relationships))
            },
            metadata={
                "workspace_id": workspace_id,
                "generated_at": datetime.utcnow().isoformat(),
                "filters_applied": {
                    "entity_types": entity_types,
                    "min_confidence": min_confidence,
                    "limit": limit
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Error retrieving knowledge graph: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve knowledge graph")


@router.post("/{workspace_id}/extract")
async def extract_knowledge(
    workspace_id: str,
    background_tasks: BackgroundTasks,
    knowledge_service: KnowledgeExtractionService = Depends(get_knowledge_extraction_service),
    force_reprocessing: bool = Query(False),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Extract knowledge from all documents in a workspace"""
    
    try:
        # Verify workspace access
        workspace = db.query(Workspace).filter(
            Workspace.workspace_id == workspace_id,
            Workspace.owner_id == current_user.user_id
        ).first()
        
        if not workspace:
            raise HTTPException(status_code=404, detail="Workspace not found")
        
        # Check if extraction is needed
        if not force_reprocessing:
            # Check if we have recent knowledge extraction
            entities_count = db.query(KnowledgeEntity).filter(
                KnowledgeEntity.workspace_id == workspace_id,
                KnowledgeEntity.created_at > datetime.utcnow() - timedelta(days=1)
            ).count()
            
            if entities_count > 0:
                return {
                    "message": "Knowledge extraction already performed recently",
                    "existing_entities": entities_count,
                    "use_force_reprocessing": "Set force_reprocessing=true to re-extract"
                }
        
        # Start background extraction
        background_tasks.add_task(
            knowledge_service.process_workspace_knowledge,
            workspace_id,
            db
        )
        
        return {
            "message": "Knowledge extraction started",
            "status": "processing",
            "workspace_id": workspace_id,
            "started_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error starting knowledge extraction: {e}")
        raise HTTPException(status_code=500, detail="Failed to start knowledge extraction")


@router.get("/{workspace_id}/entities/search", response_model=EntitySearchResponse)
async def search_entities(
    workspace_id: str,
    query: str = Query(..., min_length=1),
    entity_types: Optional[List[str]] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Search entities in a workspace"""
    
    try:
        # Verify workspace access
        workspace = db.query(Workspace).filter(
            Workspace.workspace_id == workspace_id,
            Workspace.owner_id == current_user.user_id
        ).first()
        
        if not workspace:
            raise HTTPException(status_code=404, detail="Workspace not found")
        
        # Build search query
        search_query = db.query(KnowledgeEntity).filter(
            KnowledgeEntity.workspace_id == workspace_id,
            KnowledgeEntity.name.ilike(f"%{query}%")
        )
        
        if entity_types:
            search_query = search_query.filter(
                KnowledgeEntity.entity_type.in_(entity_types)
            )
        
        entities = search_query.order_by(
            KnowledgeEntity.mention_count.desc(),
            KnowledgeEntity.confidence_score.desc()
        ).limit(limit).all()
        
        # Get related entities (entities connected via relationships)
        if entities:
            entity_ids = [e.entity_id for e in entities]
            
            related_entity_ids = db.query(KnowledgeRelationship.object_id).filter(
                KnowledgeRelationship.subject_id.in_(entity_ids)
            ).union(
                db.query(KnowledgeRelationship.subject_id).filter(
                    KnowledgeRelationship.object_id.in_(entity_ids)
                )
            ).distinct().all()
            
            related_ids = [r[0] for r in related_entity_ids if r[0] not in entity_ids]
            
            related_entities = db.query(KnowledgeEntity).filter(
                KnowledgeEntity.entity_id.in_(related_ids[:10])  # Limit related entities
            ).all()
        else:
            related_entities = []
        
        return EntitySearchResponse(
            query=query,
            entities=[
                KnowledgeEntityResponse(
                    entity_id=str(e.entity_id),
                    name=e.name,
                    entity_type=e.entity_type,
                    description=e.description,
                    confidence_score=e.confidence_score / 100.0,
                    mention_count=e.mention_count,
                    metadata=e.entity_metadata or {},
                    created_at=e.created_at,
                    updated_at=e.updated_at
                ) for e in entities
            ],
            related_entities=[
                KnowledgeEntityResponse(
                    entity_id=str(e.entity_id),
                    name=e.name,
                    entity_type=e.entity_type,
                    description=e.description,
                    confidence_score=e.confidence_score / 100.0,
                    mention_count=e.mention_count,
                    metadata=e.entity_metadata or {},
                    created_at=e.created_at,
                    updated_at=e.updated_at
                ) for e in related_entities
            ],
            total_results=len(entities),
            search_metadata={
                "workspace_id": workspace_id,
                "search_time": datetime.utcnow().isoformat(),
                "filters": {
                    "entity_types": entity_types,
                    "limit": limit
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Error searching entities: {e}")
        raise HTTPException(status_code=500, detail="Failed to search entities")


@router.get("/{workspace_id}/topics", response_model=TopicModelingResponse)
async def get_workspace_topics(
    workspace_id: str,
    background_tasks: BackgroundTasks,
    topic_service: TopicModelingService = Depends(get_topic_modeling_service),
    refresh: bool = Query(False),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get topic modeling results for a workspace"""
    
    try:
        # Verify workspace access
        workspace = db.query(Workspace).filter(
            Workspace.workspace_id == workspace_id,
            Workspace.owner_id == current_user.user_id
        ).first()
        
        if not workspace:
            raise HTTPException(status_code=404, detail="Workspace not found")
        
        # Check if we need to refresh topics
        if refresh:
            background_tasks.add_task(
                topic_service.extract_topics_from_workspace,
                workspace_id,
                db
            )
            
            return TopicModelingResponse(
                topics=[],
                statistics={
                    "total_documents": 0,
                    "num_topics": 0,
                    "processing": True
                },
                metadata={
                    "workspace_id": workspace_id,
                    "status": "processing",
                    "message": "Topic modeling started in background"
                }
            )
        
        # Get existing topic modeling results from cache/database
        # For now, perform live topic modeling (in production, cache results)
        results = await topic_service.extract_topics_from_workspace(workspace_id, db)
        
        return TopicModelingResponse(
            topics=results.get("topics", []),
            statistics={
                "total_documents": results.get("total_documents", 0),
                "num_topics": results.get("num_topics", 0),
                "outliers": results.get("outliers", 0),
                "processing": False
            },
            metadata={
                "workspace_id": workspace_id,
                "generated_at": datetime.utcnow().isoformat(),
                "error": results.get("error"),
                "message": results.get("message")
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting workspace topics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get workspace topics")


@router.get("/{workspace_id}/entities/{entity_id}/details")
async def get_entity_details(
    workspace_id: str,
    entity_id: str,
    include_relationships: bool = Query(True),
    include_documents: bool = Query(True),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get detailed information about a specific entity"""
    
    try:
        # Verify workspace access
        workspace = db.query(Workspace).filter(
            Workspace.workspace_id == workspace_id,
            Workspace.owner_id == current_user.user_id
        ).first()
        
        if not workspace:
            raise HTTPException(status_code=404, detail="Workspace not found")
        
        # Get entity
        entity = db.query(KnowledgeEntity).filter(
            KnowledgeEntity.entity_id == entity_id,
            KnowledgeEntity.workspace_id == workspace_id
        ).first()
        
        if not entity:
            raise HTTPException(status_code=404, detail="Entity not found")
        
        result = {
            "entity": KnowledgeEntityResponse(
                entity_id=str(entity.entity_id),
                name=entity.name,
                entity_type=entity.entity_type,
                description=entity.description,
                confidence_score=entity.confidence_score / 100.0,
                mention_count=entity.mention_count,
                metadata=entity.entity_metadata or {},
                created_at=entity.created_at,
                updated_at=entity.updated_at
            )
        }
        
        # Get relationships
        if include_relationships:
            # Outgoing relationships (entity as subject)
            outgoing = db.query(KnowledgeRelationship).filter(
                KnowledgeRelationship.subject_id == entity_id
            ).all()
            
            # Incoming relationships (entity as object)
            incoming = db.query(KnowledgeRelationship).filter(
                KnowledgeRelationship.object_id == entity_id
            ).all()
            
            result["relationships"] = {
                "outgoing": [
                    KnowledgeRelationshipResponse(
                        relationship_id=str(r.relationship_id),
                        subject_id=str(r.subject_id),
                        object_id=str(r.object_id),
                        predicate=r.predicate,
                        confidence_score=r.confidence_score / 100.0,
                        evidence_count=r.evidence_count,
                        metadata=r.relationship_metadata or {},
                        created_at=r.created_at,
                        updated_at=r.updated_at
                    ) for r in outgoing
                ],
                "incoming": [
                    KnowledgeRelationshipResponse(
                        relationship_id=str(r.relationship_id),
                        subject_id=str(r.subject_id),
                        object_id=str(r.object_id),
                        predicate=r.predicate,
                        confidence_score=r.confidence_score / 100.0,
                        evidence_count=r.evidence_count,
                        metadata=r.relationship_metadata or {},
                        created_at=r.created_at,
                        updated_at=r.updated_at
                    ) for r in incoming
                ]
            }
        
        # Get related documents (where entity is mentioned)
        if include_documents:
            # This would require tracking entity mentions in documents
            # For now, return document count from metadata
            document_mentions = entity.entity_metadata.get("document_mentions", []) if entity.entity_metadata else []
            result["document_mentions"] = document_mentions
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting entity details: {e}")
        raise HTTPException(status_code=500, detail="Failed to get entity details")


@router.get("/{workspace_id}/analytics/knowledge")
async def get_knowledge_analytics(
    workspace_id: str,
    time_range: str = Query("7d", regex="^(1d|7d|30d|90d)$"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get analytics about the knowledge graph"""
    
    try:
        # Verify workspace access
        workspace = db.query(Workspace).filter(
            Workspace.workspace_id == workspace_id,
            Workspace.owner_id == current_user.user_id
        ).first()
        
        if not workspace:
            raise HTTPException(status_code=404, detail="Workspace not found")
        
        # Calculate time range
        time_delta_map = {
            "1d": timedelta(days=1),
            "7d": timedelta(days=7),
            "30d": timedelta(days=30),
            "90d": timedelta(days=90)
        }
        
        since_date = datetime.utcnow() - time_delta_map[time_range]
        
        # Entity statistics
        total_entities = db.query(KnowledgeEntity).filter(
            KnowledgeEntity.workspace_id == workspace_id
        ).count()
        
        recent_entities = db.query(KnowledgeEntity).filter(
            KnowledgeEntity.workspace_id == workspace_id,
            KnowledgeEntity.created_at >= since_date
        ).count()
        
        # Entity type distribution
        entity_types = db.query(
            KnowledgeEntity.entity_type,
            db.func.count(KnowledgeEntity.entity_id).label("count")
        ).filter(
            KnowledgeEntity.workspace_id == workspace_id
        ).group_by(KnowledgeEntity.entity_type).all()
        
        # Relationship statistics
        total_relationships = db.query(KnowledgeRelationship).join(
            KnowledgeEntity,
            KnowledgeRelationship.subject_id == KnowledgeEntity.entity_id
        ).filter(
            KnowledgeEntity.workspace_id == workspace_id
        ).count()
        
        # Top entities by mention count
        top_entities = db.query(KnowledgeEntity).filter(
            KnowledgeEntity.workspace_id == workspace_id
        ).order_by(KnowledgeEntity.mention_count.desc()).limit(10).all()
        
        return {
            "summary": {
                "total_entities": total_entities,
                "total_relationships": total_relationships,
                "recent_entities": recent_entities,
                "time_range": time_range
            },
            "entity_types": [
                {"type": et[0], "count": et[1]} for et in entity_types
            ],
            "top_entities": [
                {
                    "name": e.name,
                    "type": e.entity_type,
                    "mentions": e.mention_count,
                    "confidence": e.confidence_score / 100.0
                } for e in top_entities
            ],
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting knowledge analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get knowledge analytics")
