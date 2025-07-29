import spacy
import asyncio
from typing import List, Dict, Any, Tuple, Optional
from sqlalchemy.orm import Session
from datetime import datetime
import re
from collections import defaultdict, Counter

from ..core.logging import logger
from ..core.config import settings
from ..models.database import (
    Document, DocumentChunk, KnowledgeEntity, KnowledgeRelationship, Workspace
)
from ..models.schemas import KnowledgeEntityCreate, KnowledgeRelationshipCreate
from .vector_store import VectorStoreService
from ..dependencies import get_vector_store_service


_knowledge_service_instance = None
_topic_service_instance = None


class KnowledgeExtractionService:
    """Service for extracting entities and relationships from documents"""
    
    def __init__(self, vector_service: VectorStoreService):
        self.nlp = None
        self.vector_service = vector_service
    
    async def load_models(self):
        """Asynchronously loads the spaCy model."""
        if self.nlp: return
        await asyncio.to_thread(self._load_nlp_model_sync)
    
    def _load_nlp_model_sync(self):
        """Load spaCy NLP model"""
        try:
            # Try to load English model
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy English model successfully")
        except OSError:
            try:
                # Fallback to smaller model
                self.nlp = spacy.load("en_core_web_md")
                logger.info("Loaded spaCy medium English model")
            except OSError:
                logger.warning("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
                self.nlp = None
    
    async def extract_knowledge_from_document(
        self,
        document_id: str,
        workspace_id: str,
        db: Session
    ) -> Dict[str, Any]:
        """Extract entities and relationships from a document"""
        
        if not self.nlp:
            logger.warning("spaCy model not available, skipping knowledge extraction")
            return {"entities": [], "relationships": []}
        
        try:
            # Get document chunks
            chunks = db.query(DocumentChunk).filter(
                DocumentChunk.document_id == document_id
            ).all()
            
            if not chunks:
                return {"entities": [], "relationships": []}
            
            # Combine all chunk content
            full_text = " ".join([chunk.content for chunk in chunks])
            
            # Extract entities
            entities = await self._extract_entities(full_text, workspace_id, db)
            
            # Extract relationships
            relationships = await self._extract_relationships(full_text, entities, db)
            
            logger.info(f"Extracted {len(entities)} entities and {len(relationships)} relationships from document {document_id}")
            
            return {
                "entities": entities,
                "relationships": relationships,
                "text_length": len(full_text),
                "chunks_processed": len(chunks)
            }
            
        except Exception as e:
            logger.error(f"Error extracting knowledge from document {document_id}: {e}")
            return {"entities": [], "relationships": []}
    
    async def _extract_entities(
        self,
        text: str,
        workspace_id: str,
        db: Session
    ) -> List[KnowledgeEntity]:
        """Extract named entities from text"""
        
        entities = []
        entity_counts = defaultdict(int)
        entity_types = defaultdict(list)
        
        # Process text in chunks to handle memory
        chunk_size = 1000000  # 1MB chunks
        text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        for chunk in text_chunks:
            doc = self.nlp(chunk)
            
            for ent in doc.ents:
                if len(ent.text.strip()) < 2:  # Skip very short entities
                    continue
                
                # Clean entity text
                entity_text = ent.text.strip()
                entity_type = ent.label_
                
                # Skip common stop words and pronouns
                if entity_text.lower() in ['the', 'this', 'that', 'these', 'those', 'it', 'he', 'she', 'they']:
                    continue
                
                # Count occurrences
                entity_key = (entity_text.lower(), entity_type)
                entity_counts[entity_key] += 1
                entity_types[entity_key].append({
                    "text": entity_text,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "confidence": 0.8  # spaCy doesn't provide confidence scores directly
                })
        
        # Create entity objects for frequently mentioned entities
        for (entity_text, entity_type), count in entity_counts.items():
            if count >= 2:  # Only keep entities mentioned at least twice
                # Check if entity already exists
                existing_entity = db.query(KnowledgeEntity).filter(
                    KnowledgeEntity.workspace_id == workspace_id,
                    KnowledgeEntity.name.ilike(entity_text),
                    KnowledgeEntity.entity_type == entity_type
                ).first()
                
                if existing_entity:
                    # Update mention count
                    existing_entity.mention_count += count
                    existing_entity.updated_at = datetime.utcnow()
                    entities.append(existing_entity)
                else:
                    # Create new entity
                    new_entity = KnowledgeEntity(
                        workspace_id=workspace_id,
                        name=entity_text,
                        entity_type=entity_type,
                        description=self._generate_entity_description(entity_text, entity_type),
                        confidence_score=min(int(count * 10), 100),  # Confidence based on frequency
                        mention_count=count,
                        metadata={
                            "occurrences": entity_types[(entity_text, entity_type)][:10],  # Store first 10 occurrences
                            "extraction_method": "spacy_ner"
                        }
                    )
                    
                    db.add(new_entity)
                    entities.append(new_entity)
        
        db.commit()
        return entities
    
    def _generate_entity_description(self, entity_text: str, entity_type: str) -> str:
        """Generate a description for an entity"""
        type_descriptions = {
            "PERSON": f"A person named {entity_text}",
            "ORG": f"An organization called {entity_text}",
            "GPE": f"A geopolitical entity: {entity_text}",
            "LOCATION": f"A location: {entity_text}",
            "PRODUCT": f"A product or service: {entity_text}",
            "EVENT": f"An event: {entity_text}",
            "DATE": f"A date or time reference: {entity_text}",
            "MONEY": f"A monetary amount: {entity_text}",
            "PERCENT": f"A percentage: {entity_text}",
            "CARDINAL": f"A cardinal number: {entity_text}",
            "ORDINAL": f"An ordinal number: {entity_text}"
        }
        
        return type_descriptions.get(entity_type, f"An entity of type {entity_type}: {entity_text}")
    
    async def _extract_relationships(
        self,
        text: str,
        entities: List[KnowledgeEntity],
        db: Session
    ) -> List[KnowledgeRelationship]:
        """Extract relationships between entities"""
        
        relationships = []
        
        if len(entities) < 2:
            return relationships
        
        # Create entity lookup
        entity_lookup = {entity.name.lower(): entity for entity in entities}
        
        # Process text in sentences
        doc = self.nlp(text[:100000])  # Limit text size for performance
        
        for sent in doc.sents:
            sentence_text = sent.text.lower()
            
            # Find entities in this sentence
            sentence_entities = []
            for entity_name, entity_obj in entity_lookup.items():
                if entity_name in sentence_text:
                    sentence_entities.append(entity_obj)
            
            # Extract relationships between entities in the same sentence
            if len(sentence_entities) >= 2:
                relationships.extend(
                    await self._extract_sentence_relationships(sent.text, sentence_entities, db)
                )
        
        return relationships
    
    async def _extract_sentence_relationships(
        self,
        sentence: str,
        entities: List[KnowledgeEntity],
        db: Session
    ) -> List[KnowledgeRelationship]:
        """Extract relationships from a single sentence"""
        
        relationships = []
        
        # Simple pattern-based relationship extraction
        relationship_patterns = [
            (r'\b(\w+)\s+(?:is|was|are|were)\s+(?:a|an|the)?\s*(\w+)', "is_a"),
            (r'\b(\w+)\s+(?:works for|employed by|part of)\s+(\w+)', "employed_by"),
            (r'\b(\w+)\s+(?:owns|founded|created|established)\s+(\w+)', "owns"),
            (r'\b(\w+)\s+(?:located in|based in|from)\s+(\w+)', "located_in"),
            (r'\b(\w+)\s+(?:married to|spouse of)\s+(\w+)', "married_to"),
            (r'\b(\w+)\s+(?:brother of|sister of|sibling of)\s+(\w+)', "sibling_of"),
            (r'\b(\w+)\s+(?:parent of|father of|mother of)\s+(\w+)', "parent_of"),
            (r'\b(\w+)\s+(?:acquired|bought|purchased)\s+(\w+)', "acquired"),
            (r'\b(\w+)\s+(?:invested in|funded)\s+(\w+)', "invested_in"),
            (r'\b(\w+)\s+(?:partnered with|collaborated with)\s+(\w+)', "partnered_with")
        ]
        
        sentence_lower = sentence.lower()
        
        for pattern, relation_type in relationship_patterns:
            matches = re.finditer(pattern, sentence_lower)
            
            for match in matches:
                subj_text = match.group(1)
                obj_text = match.group(2)
                
                # Find corresponding entities
                subject_entity = None
                object_entity = None
                
                for entity in entities:
                    if entity.name.lower() in subj_text or subj_text in entity.name.lower():
                        subject_entity = entity
                    if entity.name.lower() in obj_text or obj_text in entity.name.lower():
                        object_entity = entity
                
                if subject_entity and object_entity and subject_entity.entity_id != object_entity.entity_id:
                    # Check if relationship already exists
                    existing_rel = db.query(KnowledgeRelationship).filter(
                        KnowledgeRelationship.subject_id == subject_entity.entity_id,
                        KnowledgeRelationship.object_id == object_entity.entity_id,
                        KnowledgeRelationship.predicate == relation_type
                    ).first()
                    
                    if existing_rel:
                        existing_rel.evidence_count += 1
                        existing_rel.updated_at = datetime.utcnow()
                        relationships.append(existing_rel)
                    else:
                        # Create new relationship
                        new_relationship = KnowledgeRelationship(
                            subject_id=subject_entity.entity_id,
                            object_id=object_entity.entity_id,
                            predicate=relation_type,
                            confidence_score=70,  # Medium confidence for pattern-based extraction
                            evidence_count=1,
                            metadata={
                                "extraction_method": "pattern_based",
                                "pattern": pattern,
                                "sentence": sentence[:200],  # Store context
                                "match_text": match.group(0)
                            }
                        )
                        
                        db.add(new_relationship)
                        relationships.append(new_relationship)
        
        db.commit()
        return relationships
    
    async def process_workspace_knowledge(self, workspace_id: str, db: Session) -> Dict[str, Any]:
        """Process all documents in a workspace for knowledge extraction"""
        
        try:
            # Get all ready documents in workspace
            documents = db.query(Document).filter(
                Document.workspace_id == workspace_id,
                Document.status == "ready"
            ).all()
            
            total_entities = 0
            total_relationships = 0
            processed_documents = 0
            
            for document in documents:
                try:
                    result = await self.extract_knowledge_from_document(
                        str(document.document_id),
                        workspace_id,
                        db
                    )
                    
                    total_entities += len(result.get("entities", []))
                    total_relationships += len(result.get("relationships", []))
                    processed_documents += 1
                    
                except Exception as e:
                    logger.error(f"Error processing document {document.document_id}: {e}")
                    continue
            
            # Calculate knowledge graph statistics
            workspace_entities = db.query(KnowledgeEntity).filter(
                KnowledgeEntity.workspace_id == workspace_id
            ).count()
            
            workspace_relationships = db.query(KnowledgeRelationship).join(
                KnowledgeEntity,
                KnowledgeRelationship.subject_id == KnowledgeEntity.entity_id
            ).filter(
                KnowledgeEntity.workspace_id == workspace_id
            ).count()
            
            summary = {
                "processed_documents": processed_documents,
                "total_documents": len(documents),
                "entities_extracted": total_entities,
                "relationships_extracted": total_relationships,
                "total_workspace_entities": workspace_entities,
                "total_workspace_relationships": workspace_relationships,
                "processing_timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Knowledge extraction complete for workspace {workspace_id}: {summary}")
            return summary
            
        except Exception as e:
            logger.error(f"Error processing workspace knowledge: {e}")
            return {
                "error": str(e),
                "processed_documents": 0,
                "total_documents": 0
            }
    
    async def update_entity_embeddings(self, workspace_id: str, db: Session):
        """Generate embeddings for entities for semantic search"""
        
        try:
            entities = db.query(KnowledgeEntity).filter(
                KnowledgeEntity.workspace_id == workspace_id
            ).all()
            
            # This would require a separate embedding model for entities
            # For now, we'll skip this but it's useful for entity-based search
            logger.info(f"Entity embedding update placeholder for {len(entities)} entities")
            
        except Exception as e:
            logger.error(f"Error updating entity embeddings: {e}")


class TopicModelingService:
    """Service for topic modeling and theme extraction"""
    
    def __init__(self):
        self.topic_model = None
    
    async def load_models(self):
        """Asynchronously loads the BERTopic model."""
        if self.topic_model: return
        await asyncio.to_thread(self._initialize_topic_model_sync)
    
    def _initialize_topic_model_sync(self):
        """Initialize BERTopic model"""
        try:
            from bertopic import BERTopic
            from sentence_transformers import SentenceTransformer
            
            # Initialize with a lightweight sentence transformer
            sentence_model = SentenceTransformer(settings.embedding_model)
            self.topic_model = BERTopic(
                embedding_model=sentence_model,
                min_topic_size=5,  # Minimum documents per topic
                verbose=False
            )
            logger.info("BERTopic model initialized successfully")
            
        except ImportError:
            logger.warning("BERTopic not available. Install with: pip install bertopic")
            self.topic_model = None
        except Exception as e:
            logger.error(f"Error initializing BERTopic: {e}")
            self.topic_model = None
    
    async def extract_topics_from_workspace(
        self,
        workspace_id: str,
        db: Session
    ) -> Dict[str, Any]:
        """Extract topics from all documents in a workspace"""
        
        if not self.topic_model:
            return {"topics": [], "error": "Topic modeling not available"}
        
        try:
            # Get all document chunks
            chunks = db.query(DocumentChunk).join(Document).filter(
                Document.workspace_id == workspace_id,
                Document.status == "ready"
            ).all()
            
            if len(chunks) < 10:  # Need minimum documents for topic modeling
                return {
                    "topics": [],
                    "message": "Insufficient documents for topic modeling (minimum 10 required)",
                    "document_count": len(chunks)
                }
            
            # Prepare documents
            documents = [chunk.content for chunk in chunks]
            
            # Fit topic model
            topics, probabilities = self.topic_model.fit_transform(documents)
            
            # Get topic information
            topic_info = self.topic_model.get_topic_info()
            
            # Format results
            topic_results = []
            for _, row in topic_info.iterrows():
                if row['Topic'] != -1:  # Skip outlier topic
                    topic_words = self.topic_model.get_topic(row['Topic'])
                    topic_results.append({
                        "topic_id": int(row['Topic']),
                        "size": int(row['Count']),
                        "keywords": [word for word, score in topic_words[:10]],
                        "representative_docs": self.topic_model.get_representative_docs(row['Topic'])[:3]
                    })
            
            return {
                "topics": topic_results,
                "total_documents": len(documents),
                "num_topics": len(topic_results),
                "outliers": int(topic_info[topic_info['Topic'] == -1]['Count'].sum()) if len(topic_info[topic_info['Topic'] == -1]) > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error extracting topics: {e}")
            return {"topics": [], "error": str(e)}


def get_knowledge_extraction_service():
    """Dependency function to get the singleton instance of the KnowledgeExtractionService."""
    global _knowledge_service_instance
    if _knowledge_service_instance is None:
        vector_service = get_vector_store_service()
        _knowledge_service_instance = KnowledgeExtractionService(vector_service=vector_service)
    return _knowledge_service_instance


def get_topic_modeling_service():
    """Dependency function to get the singleton instance of the TopicModelingService."""
    global _topic_service_instance
    if _topic_service_instance is None:
        _topic_service_instance = TopicModelingService()
    return _topic_service_instance
