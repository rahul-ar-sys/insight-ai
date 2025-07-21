from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import desc, asc
import json
import uuid
from collections import defaultdict, Counter
import asyncio

from ..core.logging import logger
from ..core.config import settings
from ..models.database import (
    User, Workspace, Query as QueryModel, ConversationTurn, 
    Document, DocumentChunk
)
from ..models.schemas import (
    MemoryContext, MemoryContextCreate, PersonalizationProfile,
    LongTermMemoryResponse
)
from ..services.vector_store import VectorStoreService


class LongTermMemoryService:
    """Service for managing long-term memory and personalization"""
    
    def __init__(self):
        self.vector_service = VectorStoreService()
        self.memory_store = {}  # In-memory cache for frequently accessed memories
        self.personalization_cache = {}  # Cache for user profiles
        self.cache_duration = timedelta(hours=1)
    
    async def store_conversation_memory(
        self,
        user_id: str,
        workspace_id: str,
        query: str,
        response: str,
        context_data: Dict[str, Any],
        db: Session
    ) -> str:
        """Store important conversation context in long-term memory"""
        
        try:
            # Calculate importance score based on multiple factors
            importance_score = self._calculate_importance_score(
                query, response, context_data
            )
            
            # Only store if importance is above threshold
            if importance_score < 0.3:
                return None
            
            # Create memory context
            memory_content = {
                "query": query,
                "response": response,
                "timestamp": datetime.utcnow().isoformat(),
                "context": context_data,
                "importance": importance_score
            }
            
            # Store in vector database for semantic retrieval
            memory_id = str(uuid.uuid4())
            await self.vector_service.store_embedding(
                collection_name=f"memory_{workspace_id}",
                document_id=memory_id,
                text=f"{query} {response}",
                metadata={
                    "user_id": user_id,
                    "workspace_id": workspace_id,
                    "memory_type": "conversation",
                    "importance": importance_score,
                    "created_at": datetime.utcnow().isoformat(),
                    "content": json.dumps(memory_content)
                }
            )
            
            # Cache high-importance memories
            if importance_score > 0.7:
                self.memory_store[memory_id] = {
                    "content": memory_content,
                    "cached_at": datetime.utcnow()
                }
            
            logger.info(f"Stored conversation memory {memory_id} with importance {importance_score:.2f}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Error storing conversation memory: {e}")
            return None
    
    def _calculate_importance_score(
        self,
        query: str,
        response: str,
        context_data: Dict[str, Any]
    ) -> float:
        """Calculate importance score for memory storage"""
        
        score = 0.0
        
        # Query complexity (longer, more complex queries are more important)
        query_words = len(query.split())
        if query_words > 10:
            score += 0.2
        elif query_words > 5:
            score += 0.1
        
        # Response quality indicators
        response_length = len(response)
        if response_length > 500:
            score += 0.2
        elif response_length > 200:
            score += 0.1
        
        # Source count (more sources = more comprehensive answer)
        sources_count = len(context_data.get("sources", []))
        if sources_count > 3:
            score += 0.2
        elif sources_count > 1:
            score += 0.1
        
        # Confidence score
        confidence = context_data.get("confidence_score", 50)
        if confidence > 80:
            score += 0.2
        elif confidence > 60:
            score += 0.1
        
        # Reasoning complexity
        reasoning_steps = len(context_data.get("reasoning_steps", []))
        if reasoning_steps > 5:
            score += 0.2
        elif reasoning_steps > 2:
            score += 0.1
        
        # User engagement indicators (if available)
        if context_data.get("user_feedback") == "positive":
            score += 0.2
        
        return min(score, 1.0)  # Cap at 1.0
    
    async def retrieve_relevant_memories(
        self,
        user_id: str,
        workspace_id: str,
        current_query: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant memories for current context"""
        
        try:
            # Search vector database for semantically similar memories
            collection_name = f"memory_{workspace_id}"
            
            similar_memories = await self.vector_service.search_similar(
                collection_name=collection_name,
                query_text=current_query,
                limit=limit * 2,  # Get more to filter by user
                score_threshold=0.7
            )
            
            # Filter by user and process results
            relevant_memories = []
            for memory in similar_memories:
                metadata = memory.get("metadata", {})
                if metadata.get("user_id") == user_id:
                    try:
                        content = json.loads(metadata.get("content", "{}"))
                        relevant_memories.append({
                            "memory_id": memory.get("id"),
                            "content": content,
                            "similarity_score": memory.get("score", 0.0),
                            "importance": metadata.get("importance", 0.0),
                            "created_at": metadata.get("created_at")
                        })
                    except json.JSONDecodeError:
                        continue
            
            # Sort by combined score (similarity + importance)
            relevant_memories.sort(
                key=lambda m: (m["similarity_score"] * 0.7 + m["importance"] * 0.3),
                reverse=True
            )
            
            return relevant_memories[:limit]
            
        except Exception as e:
            logger.error(f"Error retrieving relevant memories: {e}")
            return []
    
    async def update_personalization_profile(
        self,
        user_id: str,
        workspace_id: str,
        interaction_data: Dict[str, Any],
        db: Session
    ):
        """Update user personalization profile based on interactions"""
        
        try:
            # Get or create profile
            profile = await self._get_personalization_profile(user_id, workspace_id, db)
            
            # Update interaction patterns
            self._update_interaction_patterns(profile, interaction_data)
            
            # Update topic interests
            await self._update_topic_interests(profile, interaction_data, workspace_id)
            
            # Update learning style preferences
            self._update_learning_style(profile, interaction_data)
            
            # Cache updated profile
            self.personalization_cache[f"{user_id}_{workspace_id}"] = {
                "profile": profile,
                "updated_at": datetime.utcnow()
            }
            
            logger.debug(f"Updated personalization profile for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error updating personalization profile: {e}")
    
    async def _get_personalization_profile(
        self,
        user_id: str,
        workspace_id: str,
        db: Session
    ) -> Dict[str, Any]:
        """Get or create personalization profile"""
        
        cache_key = f"{user_id}_{workspace_id}"
        
        # Check cache first
        if cache_key in self.personalization_cache:
            cached = self.personalization_cache[cache_key]
            if datetime.utcnow() - cached["updated_at"] < self.cache_duration:
                return cached["profile"]
        
        # Initialize default profile
        profile = {
            "user_id": user_id,
            "workspace_id": workspace_id,
            "preferences": {
                "response_length": "medium",  # short, medium, long
                "detail_level": "balanced",   # brief, balanced, detailed
                "include_sources": True,
                "show_reasoning": False
            },
            "interaction_patterns": {
                "query_types": defaultdict(int),
                "time_patterns": defaultdict(int),
                "session_lengths": [],
                "preferred_topics": defaultdict(float)
            },
            "topic_interests": defaultdict(float),
            "query_history_summary": [],
            "learning_style": None,
            "expertise_areas": [],
            "updated_at": datetime.utcnow()
        }
        
        # Load historical data to build profile
        await self._build_profile_from_history(profile, user_id, workspace_id, db)
        
        return profile
    
    async def _build_profile_from_history(
        self,
        profile: Dict[str, Any],
        user_id: str,
        workspace_id: str,
        db: Session
    ):
        """Build personalization profile from historical interactions"""
        
        try:
            # Get recent queries (last 90 days)
            recent_queries = db.query(QueryModel).filter(
                QueryModel.workspace_id == workspace_id,
                QueryModel.created_at >= datetime.utcnow() - timedelta(days=90)
            ).order_by(desc(QueryModel.created_at)).limit(100).all()
            
            # Analyze query patterns
            query_types = defaultdict(int)
            query_lengths = []
            topics = defaultdict(int)
            
            for query in recent_queries:
                query_text = query.query_text
                query_lengths.append(len(query_text.split()))
                
                # Simple query type classification
                if any(word in query_text.lower() for word in ["what", "define", "explain"]):
                    query_types["definitional"] += 1
                elif any(word in query_text.lower() for word in ["how", "steps", "process"]):
                    query_types["procedural"] += 1
                elif any(word in query_text.lower() for word in ["why", "because", "reason"]):
                    query_types["causal"] += 1
                elif any(word in query_text.lower() for word in ["compare", "vs", "difference"]):
                    query_types["comparative"] += 1
                else:
                    query_types["factual"] += 1
                
                # Extract simple topics (this could be enhanced with NLP)
                words = query_text.lower().split()
                for word in words:
                    if len(word) > 4:  # Simple filter for meaningful words
                        topics[word] += 1
            
            # Update profile
            profile["interaction_patterns"]["query_types"] = dict(query_types)
            profile["interaction_patterns"]["avg_query_length"] = sum(query_lengths) / len(query_lengths) if query_lengths else 0
            
            # Set preferences based on patterns
            if profile["interaction_patterns"]["avg_query_length"] > 15:
                profile["preferences"]["detail_level"] = "detailed"
            elif profile["interaction_patterns"]["avg_query_length"] < 8:
                profile["preferences"]["detail_level"] = "brief"
            
            # Determine learning style from query patterns
            dominant_type = max(query_types.items(), key=lambda x: x[1])[0] if query_types else None
            if dominant_type == "procedural":
                profile["learning_style"] = "hands_on"
            elif dominant_type == "definitional":
                profile["learning_style"] = "conceptual"
            elif dominant_type == "comparative":
                profile["learning_style"] = "analytical"
            else:
                profile["learning_style"] = "exploratory"
            
        except Exception as e:
            logger.error(f"Error building profile from history: {e}")
    
    def _update_interaction_patterns(
        self,
        profile: Dict[str, Any],
        interaction_data: Dict[str, Any]
    ):
        """Update interaction patterns in profile"""
        
        patterns = profile["interaction_patterns"]
        
        # Update query type
        query_type = interaction_data.get("query_type", "factual")
        patterns["query_types"][query_type] += 1
        
        # Update time pattern
        hour = datetime.utcnow().hour
        time_period = "morning" if 6 <= hour < 12 else "afternoon" if 12 <= hour < 18 else "evening" if 18 <= hour < 22 else "night"
        patterns["time_patterns"][time_period] += 1
        
        # Update session length if provided
        if "session_length_minutes" in interaction_data:
            patterns["session_lengths"].append(interaction_data["session_length_minutes"])
            # Keep only last 50 sessions
            if len(patterns["session_lengths"]) > 50:
                patterns["session_lengths"] = patterns["session_lengths"][-50:]
    
    async def _update_topic_interests(
        self,
        profile: Dict[str, Any],
        interaction_data: Dict[str, Any],
        workspace_id: str
    ):
        """Update topic interests based on interaction"""
        
        # Extract topics from query and response
        query = interaction_data.get("query", "")
        response = interaction_data.get("response", "")
        
        # Simple topic extraction (could be enhanced with NLP)
        text = f"{query} {response}".lower()
        words = text.split()
        
        # Boost score for meaningful words
        for word in words:
            if len(word) > 4 and word.isalpha():
                current_score = profile["topic_interests"].get(word, 0.0)
                # Decay existing score slightly and add new interest
                new_score = current_score * 0.95 + 0.1
                profile["topic_interests"][word] = min(new_score, 1.0)
        
        # Boost based on user engagement
        engagement_boost = 0.0
        if interaction_data.get("user_feedback") == "positive":
            engagement_boost = 0.2
        elif interaction_data.get("follow_up_questions", 0) > 0:
            engagement_boost = 0.1
        
        if engagement_boost > 0:
            for word in words[-20:]:  # Focus on recent words
                if len(word) > 4 and word.isalpha():
                    current_score = profile["topic_interests"].get(word, 0.0)
                    profile["topic_interests"][word] = min(current_score + engagement_boost, 1.0)
    
    def _update_learning_style(
        self,
        profile: Dict[str, Any],
        interaction_data: Dict[str, Any]
    ):
        """Update learning style preferences"""
        
        # Analyze response preferences
        if interaction_data.get("preferred_sources_visual", False):
            profile["preferences"]["include_visual_sources"] = True
        
        if interaction_data.get("requested_examples", False):
            profile["learning_style"] = "example_driven"
        
        if interaction_data.get("asked_for_details", False):
            profile["preferences"]["detail_level"] = "detailed"
    
    async def generate_personalized_context(
        self,
        user_id: str,
        workspace_id: str,
        current_query: str,
        db: Session
    ) -> Dict[str, Any]:
        """Generate personalized context for query processing"""
        
        try:
            # Get relevant memories
            memories = await self.retrieve_relevant_memories(
                user_id, workspace_id, current_query
            )
            
            # Get personalization profile
            profile = await self._get_personalization_profile(user_id, workspace_id, db)
            
            # Generate insights
            personalization_insights = {
                "preferred_response_style": profile["preferences"],
                "topic_expertise": self._get_top_interests(profile["topic_interests"]),
                "learning_style": profile.get("learning_style"),
                "interaction_history": memories[:3],  # Most relevant memories
                "conversation_continuity": self._analyze_conversation_continuity(memories)
            }
            
            return {
                "relevant_memories": memories,
                "personalization_insights": personalization_insights,
                "conversation_continuity": self._get_conversation_context(memories),
                "learning_adaptations": self._suggest_adaptations(profile, current_query)
            }
            
        except Exception as e:
            logger.error(f"Error generating personalized context: {e}")
            return {
                "relevant_memories": [],
                "personalization_insights": {},
                "conversation_continuity": {},
                "learning_adaptations": []
            }
    
    def _get_top_interests(self, topic_interests: Dict[str, float], limit: int = 10) -> List[Dict[str, Any]]:
        """Get top topic interests"""
        
        sorted_interests = sorted(
            topic_interests.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            {"topic": topic, "interest_score": score}
            for topic, score in sorted_interests[:limit]
        ]
    
    def _analyze_conversation_continuity(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze conversation continuity from memories"""
        
        if not memories:
            return {}
        
        # Look for patterns and themes across conversations
        recent_topics = []
        for memory in memories:
            content = memory.get("content", {})
            query = content.get("query", "")
            recent_topics.extend(query.lower().split())
        
        # Find common themes
        topic_counter = Counter(recent_topics)
        common_themes = [topic for topic, count in topic_counter.most_common(5) if len(topic) > 3]
        
        return {
            "recent_themes": common_themes,
            "conversation_depth": len(memories),
            "last_interaction": memories[0].get("created_at") if memories else None
        }
    
    def _get_conversation_context(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get conversation context for continuity"""
        
        if not memories:
            return {}
        
        latest_memory = memories[0]
        content = latest_memory.get("content", {})
        
        return {
            "previous_query": content.get("query"),
            "previous_response_summary": content.get("response", "")[:200] + "..." if len(content.get("response", "")) > 200 else content.get("response", ""),
            "time_since_last": latest_memory.get("created_at"),
            "conversation_thread": len(memories)
        }
    
    def _suggest_adaptations(
        self,
        profile: Dict[str, Any],
        current_query: str
    ) -> List[str]:
        """Suggest learning adaptations based on profile and query"""
        
        adaptations = []
        
        learning_style = profile.get("learning_style")
        if learning_style == "hands_on":
            adaptations.append("Include step-by-step instructions and practical examples")
        elif learning_style == "conceptual":
            adaptations.append("Provide clear definitions and theoretical background")
        elif learning_style == "analytical":
            adaptations.append("Include comparisons and detailed analysis")
        elif learning_style == "example_driven":
            adaptations.append("Provide concrete examples and use cases")
        
        # Query-specific adaptations
        if len(current_query.split()) > 15:
            adaptations.append("User prefers detailed queries - provide comprehensive responses")
        
        if any(word in current_query.lower() for word in ["compare", "vs", "difference"]):
            adaptations.append("Structure response as comparison with clear contrasts")
        
        return adaptations
    
    async def cleanup_old_memories(self, days_to_keep: int = 90):
        """Clean up old, low-importance memories"""
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            
            # This would need to be implemented based on the vector store's capabilities
            # For now, just clean up the cache
            current_time = datetime.utcnow()
            expired_keys = [
                key for key, value in self.memory_store.items()
                if current_time - value["cached_at"] > timedelta(days=1)
            ]
            
            for key in expired_keys:
                del self.memory_store[key]
            
            logger.info(f"Cleaned up {len(expired_keys)} expired memory cache entries")
            
        except Exception as e:
            logger.error(f"Error cleaning up old memories: {e}")
