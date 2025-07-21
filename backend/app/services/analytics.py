from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import json

from ..core.logging import logger
from ..models.database import (
    User, Workspace, Document, DocumentChunk, Query as QueryModel,
    KnowledgeEntity, KnowledgeRelationship, UserSession
)


class AnalyticsService:
    """Service for generating analytics and insights"""
    
    def __init__(self):
        self.metrics_cache = {}
        self.cache_duration = timedelta(minutes=30)
    
    async def get_workspace_analytics(
        self,
        workspace_id: str,
        time_range: str,
        db: Session
    ) -> Dict[str, Any]:
        """Generate comprehensive workspace analytics"""
        
        try:
            # Calculate time range
            time_delta_map = {
                "1d": timedelta(days=1),
                "7d": timedelta(days=7),
                "30d": timedelta(days=30),
                "90d": timedelta(days=90)
            }
            
            since_date = datetime.utcnow() - time_delta_map.get(time_range, timedelta(days=7))
            
            # Document analytics
            doc_analytics = await self._get_document_analytics(workspace_id, since_date, db)
            
            # Query analytics
            query_analytics = await self._get_query_analytics(workspace_id, since_date, db)
            
            # Knowledge graph analytics
            knowledge_analytics = await self._get_knowledge_analytics(workspace_id, since_date, db)
            
            # User engagement analytics
            engagement_analytics = await self._get_engagement_analytics(workspace_id, since_date, db)
            
            return {
                "workspace_id": workspace_id,
                "time_range": time_range,
                "generated_at": datetime.utcnow().isoformat(),
                "documents": doc_analytics,
                "queries": query_analytics,
                "knowledge_graph": knowledge_analytics,
                "engagement": engagement_analytics
            }
            
        except Exception as e:
            logger.error(f"Error generating workspace analytics: {e}")
            return {"error": str(e)}
    
    async def _get_document_analytics(
        self,
        workspace_id: str,
        since_date: datetime,
        db: Session
    ) -> Dict[str, Any]:
        """Generate document-related analytics"""
        
        try:
            # Total documents
            total_docs = db.query(Document).filter(
                Document.workspace_id == workspace_id
            ).count()
            
            # Recent documents
            recent_docs = db.query(Document).filter(
                Document.workspace_id == workspace_id,
                Document.created_at >= since_date
            ).count()
            
            # Document status distribution
            status_dist = db.query(
                Document.status,
                db.func.count(Document.document_id).label("count")
            ).filter(
                Document.workspace_id == workspace_id
            ).group_by(Document.status).all()
            
            # Document type distribution
            type_dist = db.query(
                Document.mime_type,
                db.func.count(Document.document_id).label("count")
            ).filter(
                Document.workspace_id == workspace_id
            ).group_by(Document.mime_type).all()
            
            # Average processing time
            processed_docs = db.query(Document).filter(
                Document.workspace_id == workspace_id,
                Document.status == "ready",
                Document.processed_at.is_not(None)
            ).all()
            
            avg_processing_time = 0
            if processed_docs:
                processing_times = [
                    (doc.processed_at - doc.created_at).total_seconds()
                    for doc in processed_docs
                    if doc.processed_at and doc.created_at
                ]
                if processing_times:
                    avg_processing_time = sum(processing_times) / len(processing_times)
            
            # Total content size
            total_chunks = db.query(DocumentChunk).join(Document).filter(
                Document.workspace_id == workspace_id
            ).count()
            
            return {
                "total_documents": total_docs,
                "recent_documents": recent_docs,
                "total_chunks": total_chunks,
                "average_processing_time_seconds": round(avg_processing_time, 2),
                "status_distribution": [
                    {"status": s[0], "count": s[1]} for s in status_dist
                ],
                "type_distribution": [
                    {"type": t[0], "count": t[1]} for t in type_dist
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting document analytics: {e}")
            return {"error": str(e)}
    
    async def _get_query_analytics(
        self,
        workspace_id: str,
        since_date: datetime,
        db: Session
    ) -> Dict[str, Any]:
        """Generate query-related analytics"""
        
        try:
            # Total queries
            total_queries = db.query(QueryModel).filter(
                QueryModel.workspace_id == workspace_id
            ).count()
            
            # Recent queries
            recent_queries = db.query(QueryModel).filter(
                QueryModel.workspace_id == workspace_id,
                QueryModel.created_at >= since_date
            ).count()
            
            # Query status distribution
            status_dist = db.query(
                QueryModel.status,
                db.func.count(QueryModel.query_id).label("count")
            ).filter(
                QueryModel.workspace_id == workspace_id
            ).group_by(QueryModel.status).all()
            
            # Average response time
            completed_queries = db.query(QueryModel).filter(
                QueryModel.workspace_id == workspace_id,
                QueryModel.status == "completed",
                QueryModel.completed_at.is_not(None)
            ).all()
            
            avg_response_time = 0
            if completed_queries:
                response_times = [
                    (query.completed_at - query.created_at).total_seconds()
                    for query in completed_queries
                    if query.completed_at and query.created_at
                ]
                if response_times:
                    avg_response_time = sum(response_times) / len(response_times)
            
            # Popular query patterns (simplified)
            recent_query_texts = db.query(QueryModel.query_text).filter(
                QueryModel.workspace_id == workspace_id,
                QueryModel.created_at >= since_date
            ).limit(100).all()
            
            # Extract common words from queries
            query_words = []
            for query_text in recent_query_texts:
                if query_text[0]:
                    words = query_text[0].lower().split()
                    query_words.extend([w for w in words if len(w) > 3])
            
            word_freq = Counter(query_words).most_common(10)
            
            return {
                "total_queries": total_queries,
                "recent_queries": recent_queries,
                "average_response_time_seconds": round(avg_response_time, 2),
                "status_distribution": [
                    {"status": s[0], "count": s[1]} for s in status_dist
                ],
                "popular_terms": [
                    {"term": term, "frequency": freq} for term, freq in word_freq
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting query analytics: {e}")
            return {"error": str(e)}
    
    async def _get_knowledge_analytics(
        self,
        workspace_id: str,
        since_date: datetime,
        db: Session
    ) -> Dict[str, Any]:
        """Generate knowledge graph analytics"""
        
        try:
            # Entity counts
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
            
            # Relationship counts
            total_relationships = db.query(KnowledgeRelationship).join(
                KnowledgeEntity,
                KnowledgeRelationship.subject_id == KnowledgeEntity.entity_id
            ).filter(
                KnowledgeEntity.workspace_id == workspace_id
            ).count()
            
            # Relationship type distribution
            relationship_types = db.query(KnowledgeRelationship.predicate).join(
                KnowledgeEntity,
                KnowledgeRelationship.subject_id == KnowledgeEntity.entity_id
            ).filter(
                KnowledgeEntity.workspace_id == workspace_id
            ).all()
            
            rel_type_counter = Counter([r[0] for r in relationship_types])
            
            # Top entities by connections
            entity_connections = defaultdict(int)
            all_relationships = db.query(KnowledgeRelationship).join(
                KnowledgeEntity,
                KnowledgeRelationship.subject_id == KnowledgeEntity.entity_id
            ).filter(
                KnowledgeEntity.workspace_id == workspace_id
            ).all()
            
            for rel in all_relationships:
                entity_connections[rel.subject_id] += 1
                entity_connections[rel.object_id] += 1
            
            # Get top connected entities
            top_entity_ids = sorted(entity_connections.items(), key=lambda x: x[1], reverse=True)[:5]
            top_entities = []
            
            for entity_id, connection_count in top_entity_ids:
                entity = db.query(KnowledgeEntity).filter(
                    KnowledgeEntity.entity_id == entity_id
                ).first()
                
                if entity:
                    top_entities.append({
                        "name": entity.name,
                        "type": entity.entity_type,
                        "connections": connection_count,
                        "mentions": entity.mention_count
                    })
            
            return {
                "total_entities": total_entities,
                "recent_entities": recent_entities,
                "total_relationships": total_relationships,
                "entity_types": [
                    {"type": et[0], "count": et[1]} for et in entity_types
                ],
                "relationship_types": [
                    {"type": rel_type, "count": count} 
                    for rel_type, count in rel_type_counter.most_common(10)
                ],
                "top_connected_entities": top_entities
            }
            
        except Exception as e:
            logger.error(f"Error getting knowledge analytics: {e}")
            return {"error": str(e)}
    
    async def _get_engagement_analytics(
        self,
        workspace_id: str,
        since_date: datetime,
        db: Session
    ) -> Dict[str, Any]:
        """Generate user engagement analytics"""
        
        try:
            # Get workspace owner
            workspace = db.query(Workspace).filter(
                Workspace.workspace_id == workspace_id
            ).first()
            
            if not workspace:
                return {"error": "Workspace not found"}
            
            # Session analytics
            sessions = db.query(UserSession).filter(
                UserSession.user_id == workspace.owner_id,
                UserSession.created_at >= since_date
            ).all()
            
            total_sessions = len(sessions)
            total_time = sum([
                (s.last_activity - s.created_at).total_seconds() / 3600  # Convert to hours
                for s in sessions
                if s.last_activity and s.created_at
            ])
            
            avg_session_time = total_time / total_sessions if total_sessions > 0 else 0
            
            # Daily activity (queries per day)
            daily_queries = db.query(
                db.func.date(QueryModel.created_at).label("date"),
                db.func.count(QueryModel.query_id).label("count")
            ).filter(
                QueryModel.workspace_id == workspace_id,
                QueryModel.created_at >= since_date
            ).group_by(db.func.date(QueryModel.created_at)).all()
            
            # Most active hours
            hourly_queries = db.query(
                db.func.extract('hour', QueryModel.created_at).label("hour"),
                db.func.count(QueryModel.query_id).label("count")
            ).filter(
                QueryModel.workspace_id == workspace_id,
                QueryModel.created_at >= since_date
            ).group_by(db.func.extract('hour', QueryModel.created_at)).all()
            
            return {
                "total_sessions": total_sessions,
                "average_session_time_hours": round(avg_session_time, 2),
                "daily_activity": [
                    {"date": str(d[0]), "queries": d[1]} for d in daily_queries
                ],
                "hourly_activity": [
                    {"hour": int(h[0]), "queries": h[1]} for h in hourly_queries
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting engagement analytics: {e}")
            return {"error": str(e)}
    
    async def generate_insights(
        self,
        workspace_id: str,
        db: Session
    ) -> List[Dict[str, Any]]:
        """Generate actionable insights from workspace data"""
        
        insights = []
        
        try:
            # Document processing insights
            failed_docs = db.query(Document).filter(
                Document.workspace_id == workspace_id,
                Document.status == "failed"
            ).count()
            
            if failed_docs > 0:
                insights.append({
                    "type": "warning",
                    "category": "document_processing",
                    "title": "Document Processing Issues",
                    "description": f"{failed_docs} documents failed to process. Review and reprocess these documents.",
                    "action": "Check failed documents in the document management section",
                    "priority": "medium"
                })
            
            # Knowledge graph completeness
            total_entities = db.query(KnowledgeEntity).filter(
                KnowledgeEntity.workspace_id == workspace_id
            ).count()
            
            total_docs = db.query(Document).filter(
                Document.workspace_id == workspace_id,
                Document.status == "ready"
            ).count()
            
            if total_docs > 0 and total_entities < total_docs * 5:  # Heuristic
                insights.append({
                    "type": "suggestion",
                    "category": "knowledge_extraction",
                    "title": "Low Knowledge Extraction",
                    "description": "Your documents may contain more extractable knowledge. Consider running enhanced knowledge extraction.",
                    "action": "Run knowledge extraction with higher sensitivity settings",
                    "priority": "low"
                })
            
            # Query patterns
            recent_queries = db.query(QueryModel).filter(
                QueryModel.workspace_id == workspace_id,
                QueryModel.created_at >= datetime.utcnow() - timedelta(days=7)
            ).count()
            
            if recent_queries == 0:
                insights.append({
                    "type": "info",
                    "category": "usage",
                    "title": "No Recent Activity",
                    "description": "No queries have been made in the past week. Try exploring your knowledge base!",
                    "action": "Ask questions about your documents to discover insights",
                    "priority": "low"
                })
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return [{"type": "error", "description": "Failed to generate insights"}]
    
    def get_cached_metrics(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached metrics if available and not expired"""
        
        if cache_key in self.metrics_cache:
            cached_data, timestamp = self.metrics_cache[cache_key]
            if datetime.utcnow() - timestamp < self.cache_duration:
                return cached_data
        
        return None
    
    def cache_metrics(self, cache_key: str, data: Dict[str, Any]):
        """Cache metrics data"""
        
        self.metrics_cache[cache_key] = (data, datetime.utcnow())
        
        # Clean old cache entries
        if len(self.metrics_cache) > 100:  # Limit cache size
            oldest_key = min(
                self.metrics_cache.keys(),
                key=lambda k: self.metrics_cache[k][1]
            )
            del self.metrics_cache[oldest_key]
