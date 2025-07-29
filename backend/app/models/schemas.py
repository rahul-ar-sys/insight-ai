from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from uuid import UUID
from enum import Enum


class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"


class ContradictionStatus(str, Enum):
    UNRESOLVED = "unresolved"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


class AgentRole(str, Enum):
    PLANNER = "planner"
    RETRIEVER = "retriever"
    ANALYZER = "analyzer"
    SYNTHESIZER = "synthesizer"
    KNOWLEDGE_EXTRACTOR = "knowledge_extractor"
    CONTRADICTION_DETECTOR = "contradiction_detector"


# Base Schemas
class BaseSchema(BaseModel):
    class Config:
        from_attributes = True
        use_enum_values = True


# User Schemas
class UserBase(BaseSchema):
    email: str = Field(..., description="User email address")
    name: str = Field(..., description="User full name")
    preferences: Optional[Dict[str, Any]] = Field(default={}, description="User preferences")


class UserCreate(UserBase):
    pass


class UserUpdate(BaseSchema):
    name: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None


class User(UserBase):
    user_id: UUID
    created_at: datetime
    updated_at: Optional[datetime] = None
    is_active: bool


# Workspace Schemas
class WorkspaceBase(BaseSchema):
    name: str = Field(..., description="Workspace name")
    description: Optional[str] = Field(None, description="Workspace description")
    settings: Optional[Dict[str, Any]] = Field(default={}, description="Workspace settings")


class WorkspaceCreate(WorkspaceBase):
    pass


class WorkspaceUpdate(BaseSchema):
    name: Optional[str] = None
    description: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None


class Workspace(WorkspaceBase):
    workspace_id: UUID
    owner_id: UUID
    created_at: datetime
    updated_at: Optional[datetime] = None
    is_active: bool


class WorkspaceWithStats(Workspace):
    document_count: int = 0
    session_count: int = 0
    total_chunks: int = 0


# Document Schemas
class DocumentBase(BaseSchema):
    file_name: str = Field(..., description="Document filename")
    original_name: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="File MIME type")
    file_size: int = Field(..., description="File size in bytes")


class DocumentCreate(DocumentBase):
    workspace_id: UUID


class DocumentUpdate(BaseSchema):
    status: Optional[DocumentStatus] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(None, alias="doc_metadata")


class Document(DocumentBase):
    document_id: UUID
    workspace_id: UUID
    storage_path: str
    status: DocumentStatus
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(alias="doc_metadata")
    total_chunks: int
    ocr_applied: bool
    embeddings_generated: bool
    
    class Config:
        from_attributes = True
        populate_by_name = True


# Document Chunk Schemas
class DocumentChunkBase(BaseSchema):
    chunk_index: int = Field(..., description="Chunk sequence number")
    content: str = Field(..., description="Chunk text content")
    page_number: Optional[int] = Field(None, description="Source page number")
    start_position: Optional[int] = Field(None, description="Start character position")
    end_position: Optional[int] = Field(None, description="End character position")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Chunk metadata")


class DocumentChunkCreate(DocumentChunkBase):
    document_id: UUID
    vector_id: Optional[str] = None


class DocumentChunk(DocumentChunkBase):
    chunk_id: UUID
    document_id: UUID
    vector_id: Optional[str] = None
    created_at: datetime


# Session Schemas
class SessionBase(BaseSchema):
    title: Optional[str] = Field(None, description="Session title")


class SessionCreate(SessionBase):
    workspace_id: UUID


class SessionUpdate(BaseSchema):
    title: Optional[str] = None
    session_summary: Optional[str] = None
    is_active: Optional[bool] = None


class Session(SessionBase):
    session_id: UUID
    workspace_id: UUID
    user_id: UUID
    created_at: datetime
    updated_at: Optional[datetime] = None
    is_active: bool
    session_summary: Optional[str] = None


# Conversation Turn Schemas
class ConversationTurnBase(BaseSchema):
    query_text: str = Field(..., description="User query")
    response_text: str = Field(..., description="AI response")
    agent_trace: Optional[Dict[str, Any]] = Field(default={}, description="Agent execution trace")
    sources: Optional[List[Dict[str, Any]]] = Field(default=[], description="Response sources")
    reasoning_steps: Optional[List[Dict[str, Any]]] = Field(default=[], description="Reasoning steps")
    processing_time_ms: Optional[int] = Field(None, description="Processing time in milliseconds")
    confidence_score: Optional[int] = Field(None, ge=0, le=100, description="Response confidence (0-100)")


class ConversationTurnCreate(ConversationTurnBase):
    session_id: UUID


class ConversationTurn(ConversationTurnBase):
    turn_id: UUID
    session_id: UUID
    user_id: UUID
    timestamp: datetime


# Query and Response Schemas
class QueryRequest(BaseSchema):
    query: str = Field(..., description="User question")
    workspace_id: UUID = Field(..., description="Target workspace")
    session_id: Optional[UUID] = Field(None, description="Existing session ID")
    use_web_search: bool = Field(default=False, description="Enable web search")
    max_sources: int = Field(default=5, ge=1, le=20, description="Maximum sources to retrieve")


class QueryResponse(BaseSchema):
    response: str = Field(..., description="AI-generated response")
    sources: List[Dict[str, Any]] = Field(default=[], description="Source citations")
    reasoning_steps: List[Dict[str, Any]] = Field(default=[], description="Reasoning process")
    session_id: UUID = Field(..., description="Session ID")
    turn_id: UUID = Field(..., description="Conversation turn ID")
    processing_time_ms: int = Field(..., description="Processing time")
    confidence_score: int = Field(..., ge=0, le=100, description="Response confidence")


# Agent Schemas
class AgentStep(BaseSchema):
    agent_name: str = Field(..., description="Agent identifier")
    role: AgentRole = Field(..., description="Agent role")
    action: str = Field(..., description="Action performed")
    input_data: Dict[str, Any] = Field(..., description="Input to the agent")
    output_data: Dict[str, Any] = Field(..., description="Agent output")
    execution_time_ms: int = Field(..., description="Step execution time")
    timestamp: datetime = Field(..., description="Step timestamp")


class AgentTrace(BaseSchema):
    plan_id: str = Field(..., description="Execution plan identifier")
    query: str = Field(..., description="Original user query")
    steps: List[AgentStep] = Field(..., description="Execution steps")
    total_time_ms: int = Field(..., description="Total execution time")
    success: bool = Field(..., description="Execution success")
    error_message: Optional[str] = Field(None, description="Error message if failed")


# Knowledge Graph Schemas
class KnowledgeEntityBase(BaseSchema):
    name: str = Field(..., description="Entity name")
    entity_type: str = Field(..., description="Entity type (PERSON, ORG, etc.)")
    description: Optional[str] = Field(None, description="Entity description")
    confidence_score: Optional[int] = Field(None, ge=0, le=100, description="Extraction confidence")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Entity metadata")


class KnowledgeEntityCreate(KnowledgeEntityBase):
    workspace_id: UUID


class KnowledgeEntity(KnowledgeEntityBase):
    entity_id: UUID
    workspace_id: UUID
    mention_count: int
    created_at: datetime
    updated_at: Optional[datetime] = None


class KnowledgeRelationshipBase(BaseSchema):
    predicate: str = Field(..., description="Relationship type")
    confidence_score: Optional[int] = Field(None, ge=0, le=100, description="Relationship confidence")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Relationship metadata")


class KnowledgeRelationshipCreate(KnowledgeRelationshipBase):
    subject_id: UUID
    object_id: UUID


class KnowledgeRelationship(KnowledgeRelationshipBase):
    relationship_id: UUID
    subject_id: UUID
    object_id: UUID
    evidence_count: int
    created_at: datetime
    updated_at: Optional[datetime] = None


class KnowledgeGraphNode(BaseSchema):
    id: str = Field(..., description="Node ID")
    label: str = Field(..., description="Node label")
    type: str = Field(..., description="Node type")
    properties: Dict[str, Any] = Field(..., description="Node properties")


class KnowledgeGraphEdge(BaseSchema):
    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    relationship: str = Field(..., description="Relationship type")
    properties: Dict[str, Any] = Field(..., description="Edge properties")


class KnowledgeGraph(BaseSchema):
    nodes: List[KnowledgeGraphNode] = Field(..., description="Graph nodes")
    edges: List[KnowledgeGraphEdge] = Field(..., description="Graph edges")


# Contradiction Schemas
class ContradictionBase(BaseSchema):
    contradiction_type: str = Field(..., description="Type of contradiction")
    confidence_score: int = Field(..., ge=0, le=100, description="Detection confidence")
    description: str = Field(..., description="Contradiction description")


class ContradictionCreate(ContradictionBase):
    workspace_id: UUID
    chunk1_id: UUID
    chunk2_id: UUID


class Contradiction(ContradictionBase):
    contradiction_id: UUID
    workspace_id: UUID
    chunk1_id: UUID
    chunk2_id: UUID
    resolution_status: ContradictionStatus
    created_at: datetime
    updated_at: Optional[datetime] = None


# File Upload Schema
class FileUploadResponse(BaseSchema):
    document_id: UUID = Field(..., description="Created document ID")
    message: str = Field(..., description="Upload status message")
    processing_started: bool = Field(..., description="Whether processing has started")


# Analytics Schemas
class WorkspaceAnalytics(BaseSchema):
    total_documents: int
    total_chunks: int
    total_conversations: int
    total_entities: int
    total_relationships: int
    total_contradictions: int
    recent_activity: List[Dict[str, Any]]
    topic_distribution: Dict[str, int]
    entity_types: Dict[str, int]


class InsightData(BaseSchema):
    key_themes: List[str] = Field(..., description="Main document themes")
    frequent_entities: List[Dict[str, Any]] = Field(..., description="Most mentioned entities")
    trend_summaries: List[str] = Field(..., description="Identified trends")
    contradiction_summary: Dict[str, Any] = Field(..., description="Contradiction analysis")
    recommendations: List[str] = Field(..., description="Proactive recommendations")


# Error Schemas
class ErrorResponse(BaseSchema):
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")


# Phase 2: Enhanced Knowledge Graph Schemas
class KnowledgeEntityResponse(KnowledgeEntity):
    """Enhanced entity response for Phase 2"""
    pass


class KnowledgeRelationshipResponse(KnowledgeRelationship):
    """Enhanced relationship response for Phase 2"""
    pass


class KnowledgeGraphResponse(BaseSchema):
    """Enhanced knowledge graph response"""
    entities: List[KnowledgeEntityResponse] = Field(..., description="Graph entities")
    relationships: List[KnowledgeRelationshipResponse] = Field(..., description="Graph relationships")
    statistics: Dict[str, Any] = Field(..., description="Graph statistics")
    metadata: Dict[str, Any] = Field(..., description="Graph metadata")


class EntitySearchResponse(BaseSchema):
    """Entity search response"""
    query: str = Field(..., description="Search query")
    entities: List[KnowledgeEntityResponse] = Field(..., description="Matching entities")
    related_entities: List[KnowledgeEntityResponse] = Field(..., description="Related entities")
    total_results: int = Field(..., description="Total search results")
    search_metadata: Dict[str, Any] = Field(..., description="Search metadata")


class TopicModelingResponse(BaseSchema):
    """Topic modeling response"""
    topics: List[Dict[str, Any]] = Field(..., description="Discovered topics")
    statistics: Dict[str, Any] = Field(..., description="Topic modeling statistics")
    metadata: Dict[str, Any] = Field(..., description="Processing metadata")


# Phase 2: Reasoning Visualization Schemas
class ReasoningStepResponse(BaseSchema):
    """Individual reasoning step response"""
    step_id: str = Field(..., description="Step identifier")
    step_type: str = Field(..., description="Type of reasoning step")
    timestamp: str = Field(..., description="Step timestamp")
    description: str = Field(..., description="Step description")
    agent_name: Optional[str] = Field(None, description="Agent that performed the step")
    confidence: str = Field(..., description="Confidence level")
    execution_time_ms: float = Field(..., description="Execution time")
    tools_used: List[str] = Field(default=[], description="Tools used in this step")
    input_data: Dict[str, Any] = Field(..., description="Step input")
    output_data: Dict[str, Any] = Field(..., description="Step output")
    dependencies: List[str] = Field(default=[], description="Dependent step IDs")
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata")


class ReasoningTraceResponse(BaseSchema):
    """Complete reasoning trace response"""
    trace_id: str = Field(..., description="Trace identifier")
    query: str = Field(..., description="Original query")
    workspace_id: str = Field(..., description="Workspace ID")
    started_at: str = Field(..., description="Start timestamp")
    completed_at: Optional[str] = Field(None, description="Completion timestamp")
    final_answer: Optional[str] = Field(None, description="Final answer")
    overall_confidence: Optional[str] = Field(None, description="Overall confidence level")
    steps: List[ReasoningStepResponse] = Field(..., description="Reasoning steps")
    metadata: Dict[str, Any] = Field(default={}, description="Trace metadata")


class ReasoningGraphNode(BaseSchema):
    """Node in reasoning graph visualization"""
    id: str = Field(..., description="Node ID")
    label: str = Field(..., description="Node label")
    type: str = Field(..., description="Node type")
    agent: Optional[str] = Field(None, description="Agent name")
    confidence: Optional[str] = Field(None, description="Confidence level")
    execution_time: Optional[float] = Field(None, description="Execution time")
    tools: List[str] = Field(default=[], description="Tools used")
    timestamp: str = Field(..., description="Timestamp")
    position: Optional[int] = Field(None, description="Position in sequence")


class ReasoningGraphEdge(BaseSchema):
    """Edge in reasoning graph visualization"""
    from_node: str = Field(..., alias="from", description="Source node ID")
    to: str = Field(..., description="Target node ID")
    type: str = Field(..., description="Edge type")


class ReasoningGraphResponse(BaseSchema):
    """Reasoning graph visualization response"""
    trace_id: str = Field(..., description="Trace ID")
    nodes: List[ReasoningGraphNode] = Field(..., description="Graph nodes")
    edges: List[ReasoningGraphEdge] = Field(..., description="Graph edges")
    metadata: Dict[str, Any] = Field(..., description="Graph metadata")


class ReasoningAnalyticsResponse(BaseSchema):
    """Reasoning pattern analytics response"""
    analysis_period: Dict[str, Any] = Field(..., description="Analysis period info")
    step_types_frequency: Dict[str, int] = Field(..., description="Step type frequencies")
    agent_usage: Dict[str, int] = Field(..., description="Agent usage statistics")
    tool_usage: Dict[str, int] = Field(..., description="Tool usage statistics")
    execution_times: Dict[str, Dict[str, float]] = Field(..., description="Execution time analytics")
    confidence_distribution: Dict[str, int] = Field(..., description="Confidence level distribution")
    common_step_sequences: List[Dict[str, Any]] = Field(..., description="Common reasoning patterns")
    generated_at: str = Field(..., description="Generation timestamp")


# Phase 2: Multi-Modal Q&A Schemas
class MultiModalQueryRequest(QueryRequest):
    """Multi-modal query request with vision support"""
    image_urls: Optional[List[str]] = Field(None, description="Image URLs to analyze")
    image_descriptions: Optional[List[str]] = Field(None, description="Image descriptions")
    analyze_charts: bool = Field(default=False, description="Enable chart/table analysis")
    extract_text: bool = Field(default=True, description="Extract text from images")


class ImageAnalysisResult(BaseSchema):
    """Image analysis result"""
    image_url: str = Field(..., description="Image URL")
    description: str = Field(..., description="Generated description")
    extracted_text: Optional[str] = Field(None, description="OCR extracted text")
    chart_data: Optional[Dict[str, Any]] = Field(None, description="Chart/table data if detected")
    confidence_score: float = Field(..., description="Analysis confidence")
    analysis_metadata: Dict[str, Any] = Field(default={}, description="Analysis metadata")


class MultiModalQueryResponse(QueryResponse):
    """Multi-modal query response"""
    image_analyses: List[ImageAnalysisResult] = Field(default=[], description="Image analysis results")
    visual_evidence: List[Dict[str, Any]] = Field(default=[], description="Visual evidence citations")


# Phase 2: Long-Term Memory Schemas
class MemoryContextBase(BaseSchema):
    """Base memory context schema"""
    context_type: str = Field(..., description="Type of memory context")
    content: str = Field(..., description="Memory content")
    importance_score: float = Field(..., ge=0.0, le=1.0, description="Memory importance score")
    access_count: int = Field(default=0, description="Number of times accessed")
    metadata: Dict[str, Any] = Field(default={}, description="Context metadata")


class MemoryContextCreate(MemoryContextBase):
    """Create memory context"""
    workspace_id: UUID = Field(..., description="Workspace ID")
    user_id: UUID = Field(..., description="User ID")


class MemoryContext(MemoryContextBase):
    """Memory context response"""
    memory_id: UUID = Field(..., description="Memory ID")
    workspace_id: UUID = Field(..., description="Workspace ID")
    user_id: UUID = Field(..., description="User ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    last_accessed: Optional[datetime] = Field(None, description="Last access timestamp")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")


class PersonalizationProfile(BaseSchema):
    """User personalization profile"""
    user_id: UUID = Field(..., description="User ID")
    preferences: Dict[str, Any] = Field(default={}, description="User preferences")
    interaction_patterns: Dict[str, Any] = Field(default={}, description="Interaction patterns")
    topic_interests: Dict[str, float] = Field(default={}, description="Topic interest scores")
    query_history_summary: List[str] = Field(default=[], description="Query pattern summaries")
    learning_style: Optional[str] = Field(None, description="Preferred learning style")
    expertise_areas: List[str] = Field(default=[], description="Areas of expertise")
    updated_at: datetime = Field(..., description="Last update timestamp")


class LongTermMemoryResponse(BaseSchema):
    """Long-term memory system response"""
    relevant_memories: List[MemoryContext] = Field(..., description="Relevant past contexts")
    personalization_insights: Dict[str, Any] = Field(..., description="Personalization insights")
    conversation_continuity: Dict[str, Any] = Field(..., description="Conversation continuity data")
    learning_adaptations: List[str] = Field(default=[], description="Learning adaptations made")
