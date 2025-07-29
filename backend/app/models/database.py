from sqlalchemy import Column, String, DateTime, Text, JSON, Boolean, Integer, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from ..core.database import Base


class User(Base):
    """User model for authentication and profile management"""
    __tablename__ = "users"
    
    user_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    hashed_password = Column(String(255), nullable=True)  # Nullable for Firebase users
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    is_active = Column(Boolean, default=True)
    preferences = Column(JSON, default={})
    
    # Relationships
    workspaces = relationship("Workspace", back_populates="owner")
    conversation_turns = relationship("ConversationTurn", back_populates="user")
    
    # Indexes
    __table_args__ = (
        Index('idx_user_email', 'email'),
        Index('idx_user_active', 'is_active'),
    )


class Workspace(Base):
    """Workspace model for organizing documents and conversations"""
    __tablename__ = "workspaces"
    
    workspace_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    is_active = Column(Boolean, default=True)
    settings = Column(JSON, default={})
    
    # Relationships
    owner = relationship("User", back_populates="workspaces")
    documents = relationship("Document", back_populates="workspace", cascade="all, delete-orphan")
    sessions = relationship("Session", back_populates="workspace", cascade="all, delete-orphan")
    knowledge_entities = relationship("KnowledgeEntity", back_populates="workspace", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_workspace_owner', 'owner_id'),
        Index('idx_workspace_active', 'is_active'),
    )


class Document(Base):
    """Document model for uploaded files and their metadata"""
    __tablename__ = "documents"
    
    document_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.workspace_id"), nullable=False)
    file_name = Column(String(500), nullable=False)
    original_name = Column(String(500), nullable=False)
    file_type = Column(String(50), nullable=False)
    file_size = Column(Integer, nullable=False)
    storage_path = Column(String(1000), nullable=True)
    status = Column(String(50), default='processing')  # processing, ready, error
    error_message = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    doc_metadata = Column(JSON, default={})
    
    # Processing Information
    total_chunks = Column(Integer, default=0)
    ocr_applied = Column(Boolean, default=False)
    embeddings_generated = Column(Boolean, default=False)
    
    # Relationships
    workspace = relationship("Workspace", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_document_workspace', 'workspace_id'),
        Index('idx_document_status', 'status'),
        Index('idx_document_created', 'created_at'),
    )


class DocumentChunk(Base):
    """Document chunks for vector storage and retrieval"""
    __tablename__ = "document_chunks"
    
    chunk_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.document_id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    vector_id = Column(String(255))  # ID in vector database
    page_number = Column(Integer)
    start_position = Column(Integer)
    end_position = Column(Integer)
    chunk_metadata = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    document = relationship("Document", back_populates="chunks")
    
    # Indexes
    __table_args__ = (
        Index('idx_chunk_document', 'document_id'),
        Index('idx_chunk_vector', 'vector_id'),
        Index('idx_chunk_page', 'page_number'),
    )


class Session(Base):
    """Session model for conversation management"""
    __tablename__ = "sessions"
    
    session_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.workspace_id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False)
    title = Column(String(500))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    is_active = Column(Boolean, default=True)
    session_summary = Column(Text)
    
    # Relationships
    workspace = relationship("Workspace", back_populates="sessions")
    user = relationship("User")
    conversation_turns = relationship("ConversationTurn", back_populates="session", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_session_workspace', 'workspace_id'),
        Index('idx_session_user', 'user_id'),
        Index('idx_session_active', 'is_active'),
    )


class ConversationTurn(Base):
    """Individual conversation turns within a session"""
    __tablename__ = "conversation_turns"
    
    turn_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.session_id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False)
    query_text = Column(Text, nullable=False)
    response_text = Column(Text, nullable=False)
    agent_trace = Column(JSON, default={})  # Multi-step plan execution trace
    sources = Column(JSON, default=[])  # Cited sources
    reasoning_steps = Column(JSON, default=[])  # Transparent reasoning
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    processing_time_ms = Column(Integer)
    confidence_score = Column(Integer)  # 0-100
    
    # Relationships
    session = relationship("Session", back_populates="conversation_turns")
    user = relationship("User", back_populates="conversation_turns")
    
    # Indexes
    __table_args__ = (
        Index('idx_turn_session', 'session_id'),
        Index('idx_turn_user', 'user_id'),
        Index('idx_turn_timestamp', 'timestamp'),
    )


class KnowledgeEntity(Base):
    """Knowledge graph entities extracted from documents"""
    __tablename__ = "knowledge_entities"
    
    entity_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.workspace_id"), nullable=False)
    name = Column(String(500), nullable=False)
    entity_type = Column(String(100), nullable=False)  # PERSON, ORGANIZATION, LOCATION, etc.
    description = Column(Text)
    confidence_score = Column(Integer)  # 0-100
    mention_count = Column(Integer, default=1)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    entity_metadata = Column(JSON, default={})
    
    # Relationships
    workspace = relationship("Workspace", back_populates="knowledge_entities")
    relationships_as_subject = relationship("KnowledgeRelationship", foreign_keys="KnowledgeRelationship.subject_id", back_populates="subject")
    relationships_as_object = relationship("KnowledgeRelationship", foreign_keys="KnowledgeRelationship.object_id", back_populates="object")
    
    # Indexes
    __table_args__ = (
        Index('idx_entity_workspace', 'workspace_id'),
        Index('idx_entity_type', 'entity_type'),
        Index('idx_entity_name', 'name'),
    )


class KnowledgeRelationship(Base):
    """Knowledge graph relationships between entities"""
    __tablename__ = "knowledge_relationships"
    
    relationship_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    subject_id = Column(UUID(as_uuid=True), ForeignKey("knowledge_entities.entity_id"), nullable=False)
    predicate = Column(String(200), nullable=False)  # relationship type
    object_id = Column(UUID(as_uuid=True), ForeignKey("knowledge_entities.entity_id"), nullable=False)
    confidence_score = Column(Integer)  # 0-100
    evidence_count = Column(Integer, default=1)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    relationship_metadata = Column(JSON, default={})
    
    # Relationships
    subject = relationship("KnowledgeEntity", foreign_keys=[subject_id], back_populates="relationships_as_subject")
    object = relationship("KnowledgeEntity", foreign_keys=[object_id], back_populates="relationships_as_object")
    
    # Indexes
    __table_args__ = (
        Index('idx_rel_subject', 'subject_id'),
        Index('idx_rel_object', 'object_id'),
        Index('idx_rel_predicate', 'predicate'),
    )


class Contradiction(Base):
    """Detected contradictions between document statements"""
    __tablename__ = "contradictions"
    
    contradiction_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.workspace_id"), nullable=False)
    chunk1_id = Column(UUID(as_uuid=True), ForeignKey("document_chunks.chunk_id"), nullable=False)
    chunk2_id = Column(UUID(as_uuid=True), ForeignKey("document_chunks.chunk_id"), nullable=False)
    contradiction_type = Column(String(100), nullable=False)  # factual, temporal, logical
    confidence_score = Column(Integer)  # 0-100
    description = Column(Text, nullable=False)
    resolution_status = Column(String(50), default='unresolved')  # unresolved, resolved, dismissed
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    workspace = relationship("Workspace")
    chunk1 = relationship("DocumentChunk", foreign_keys=[chunk1_id])
    chunk2 = relationship("DocumentChunk", foreign_keys=[chunk2_id])
    
    # Indexes
    __table_args__ = (
        Index('idx_contradiction_workspace', 'workspace_id'),
        Index('idx_contradiction_status', 'resolution_status'),
        Index('idx_contradiction_type', 'contradiction_type'),
    )
