from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import uuid
from dataclasses import dataclass, asdict
from enum import Enum

from ..core.logging import logger


class ReasoningStepType(Enum):
    """Types of reasoning steps"""
    QUERY_ANALYSIS = "query_analysis"
    DOCUMENT_RETRIEVAL = "document_retrieval"
    KNOWLEDGE_LOOKUP = "knowledge_lookup"
    WEB_SEARCH = "web_search"
    REASONING = "reasoning"
    CONTRADICTION_CHECK = "contradiction_check"
    SYNTHESIS = "synthesis"
    CONFIDENCE_ASSESSMENT = "confidence_assessment"


class ConfidenceLevel(Enum):
    """Confidence levels for reasoning steps"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class ReasoningStep:
    """Represents a single step in the reasoning process"""
    step_id: str
    step_type: ReasoningStepType
    timestamp: datetime
    description: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    confidence: ConfidenceLevel
    execution_time_ms: float
    agent_name: Optional[str] = None
    tools_used: List[str] = None
    dependencies: List[str] = None  # IDs of previous steps this depends on
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.tools_used is None:
            self.tools_used = []
        if self.dependencies is None:
            self.dependencies = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ReasoningTrace:
    """Complete reasoning trace for a query"""
    trace_id: str
    query: str
    workspace_id: str
    user_id: str
    started_at: datetime
    completed_at: Optional[datetime]
    steps: List[ReasoningStep]
    final_answer: Optional[str]
    overall_confidence: Optional[ConfidenceLevel]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ReasoningVisualizer:
    """Service for capturing and visualizing AI reasoning processes"""
    
    def __init__(self):
        self.active_traces: Dict[str, ReasoningTrace] = {}
        self.completed_traces: Dict[str, ReasoningTrace] = {}
        self.max_traces = 1000  # Limit memory usage
    
    def start_reasoning_trace(
        self,
        query: str,
        workspace_id: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Start a new reasoning trace"""
        
        trace_id = str(uuid.uuid4())
        
        trace = ReasoningTrace(
            trace_id=trace_id,
            query=query,
            workspace_id=workspace_id,
            user_id=user_id,
            started_at=datetime.utcnow(),
            completed_at=None,
            steps=[],
            final_answer=None,
            overall_confidence=None,
            metadata=metadata or {}
        )
        
        self.active_traces[trace_id] = trace
        
        logger.info(f"Started reasoning trace {trace_id} for query: {query[:100]}...")
        return trace_id
    
    def add_reasoning_step(
        self,
        trace_id: str,
        step_type: ReasoningStepType,
        description: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        confidence: ConfidenceLevel,
        execution_time_ms: float,
        agent_name: Optional[str] = None,
        tools_used: Optional[List[str]] = None,
        dependencies: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a reasoning step to an active trace"""
        
        if trace_id not in self.active_traces:
            logger.warning(f"Trace {trace_id} not found in active traces")
            return None
        
        step_id = str(uuid.uuid4())
        
        step = ReasoningStep(
            step_id=step_id,
            step_type=step_type,
            timestamp=datetime.utcnow(),
            description=description,
            input_data=input_data,
            output_data=output_data,
            confidence=confidence,
            execution_time_ms=execution_time_ms,
            agent_name=agent_name,
            tools_used=tools_used or [],
            dependencies=dependencies or [],
            metadata=metadata or {}
        )
        
        self.active_traces[trace_id].steps.append(step)
        
        logger.debug(f"Added reasoning step {step_id} to trace {trace_id}: {description}")
        return step_id
    
    def complete_reasoning_trace(
        self,
        trace_id: str,
        final_answer: str,
        overall_confidence: ConfidenceLevel,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Complete a reasoning trace"""
        
        if trace_id not in self.active_traces:
            logger.warning(f"Trace {trace_id} not found in active traces")
            return
        
        trace = self.active_traces[trace_id]
        trace.completed_at = datetime.utcnow()
        trace.final_answer = final_answer
        trace.overall_confidence = overall_confidence
        
        if metadata:
            trace.metadata.update(metadata)
        
        # Move to completed traces
        self.completed_traces[trace_id] = trace
        del self.active_traces[trace_id]
        
        # Limit memory usage
        if len(self.completed_traces) > self.max_traces:
            oldest_trace_id = min(
                self.completed_traces.keys(),
                key=lambda tid: self.completed_traces[tid].started_at
            )
            del self.completed_traces[oldest_trace_id]
        
        total_time = (trace.completed_at - trace.started_at).total_seconds() * 1000
        logger.info(f"Completed reasoning trace {trace_id} in {total_time:.2f}ms with {len(trace.steps)} steps")
    
    def get_reasoning_trace(self, trace_id: str) -> Optional[ReasoningTrace]:
        """Get a reasoning trace by ID"""
        
        if trace_id in self.active_traces:
            return self.active_traces[trace_id]
        elif trace_id in self.completed_traces:
            return self.completed_traces[trace_id]
        else:
            return None
    
    def get_traces_for_workspace(
        self,
        workspace_id: str,
        limit: int = 50,
        include_active: bool = True
    ) -> List[ReasoningTrace]:
        """Get all traces for a workspace"""
        
        traces = []
        
        # Get completed traces
        for trace in self.completed_traces.values():
            if trace.workspace_id == workspace_id:
                traces.append(trace)
        
        # Get active traces if requested
        if include_active:
            for trace in self.active_traces.values():
                if trace.workspace_id == workspace_id:
                    traces.append(trace)
        
        # Sort by start time (newest first)
        traces.sort(key=lambda t: t.started_at, reverse=True)
        
        return traces[:limit]
    
    def export_trace_to_dict(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Export a reasoning trace to dictionary format"""
        
        trace = self.get_reasoning_trace(trace_id)
        if not trace:
            return None
        
        # Convert enums to strings for JSON serialization
        trace_dict = asdict(trace)
        
        # Convert datetime objects to ISO strings
        trace_dict["started_at"] = trace.started_at.isoformat()
        if trace.completed_at:
            trace_dict["completed_at"] = trace.completed_at.isoformat()
        
        # Convert enum values
        if trace.overall_confidence:
            trace_dict["overall_confidence"] = trace.overall_confidence.value
        
        # Process steps
        for i, step in enumerate(trace_dict["steps"]):
            step["timestamp"] = trace.steps[i].timestamp.isoformat()
            step["step_type"] = trace.steps[i].step_type.value
            step["confidence"] = trace.steps[i].confidence.value
        
        return trace_dict
    
    def generate_reasoning_graph(self, trace_id: str) -> Dict[str, Any]:
        """Generate a graph representation of the reasoning process"""
        
        trace = self.get_reasoning_trace(trace_id)
        if not trace:
            return {"error": "Trace not found"}
        
        # Create nodes for each step
        nodes = []
        edges = []
        
        for i, step in enumerate(trace.steps):
            # Create node
            node = {
                "id": step.step_id,
                "label": step.description[:50] + "..." if len(step.description) > 50 else step.description,
                "type": step.step_type.value,
                "agent": step.agent_name,
                "confidence": step.confidence.value,
                "execution_time": step.execution_time_ms,
                "tools": step.tools_used,
                "timestamp": step.timestamp.isoformat(),
                "position": i  # For ordering
            }
            nodes.append(node)
            
            # Create edges based on dependencies
            for dep_id in step.dependencies:
                edges.append({
                    "from": dep_id,
                    "to": step.step_id,
                    "type": "dependency"
                })
            
            # Create sequential edges (if no explicit dependencies)
            if not step.dependencies and i > 0:
                prev_step = trace.steps[i-1]
                edges.append({
                    "from": prev_step.step_id,
                    "to": step.step_id,
                    "type": "sequence"
                })
        
        # Add query and answer nodes
        query_node = {
            "id": "query",
            "label": trace.query[:100] + "..." if len(trace.query) > 100 else trace.query,
            "type": "query",
            "timestamp": trace.started_at.isoformat()
        }
        nodes.insert(0, query_node)
        
        if trace.final_answer:
            answer_node = {
                "id": "answer",
                "label": trace.final_answer[:100] + "..." if len(trace.final_answer) > 100 else trace.final_answer,
                "type": "answer",
                "confidence": trace.overall_confidence.value if trace.overall_confidence else None,
                "timestamp": trace.completed_at.isoformat() if trace.completed_at else None
            }
            nodes.append(answer_node)
            
            # Connect last step to answer
            if trace.steps:
                edges.append({
                    "from": trace.steps[-1].step_id,
                    "to": "answer",
                    "type": "output"
                })
        
        # Connect query to first step
        if trace.steps:
            edges.append({
                "from": "query",
                "to": trace.steps[0].step_id,
                "type": "input"
            })
        
        return {
            "trace_id": trace_id,
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "total_steps": len(trace.steps),
                "total_time_ms": (trace.completed_at - trace.started_at).total_seconds() * 1000 if trace.completed_at else None,
                "agents_involved": list(set(step.agent_name for step in trace.steps if step.agent_name)),
                "tools_used": list(set(tool for step in trace.steps for tool in step.tools_used))
            }
        }
    
    def analyze_reasoning_patterns(self, workspace_id: str) -> Dict[str, Any]:
        """Analyze reasoning patterns across multiple traces"""
        
        traces = self.get_traces_for_workspace(workspace_id, limit=100, include_active=False)
        
        if not traces:
            return {"message": "No completed traces found"}
        
        # Analyze step patterns
        step_types_freq = {}
        agent_usage = {}
        tool_usage = {}
        avg_execution_times = {}
        confidence_distribution = {}
        
        total_steps = 0
        total_traces = len(traces)
        
        for trace in traces:
            for step in trace.steps:
                total_steps += 1
                
                # Step type frequency
                step_type = step.step_type.value
                step_types_freq[step_type] = step_types_freq.get(step_type, 0) + 1
                
                # Agent usage
                if step.agent_name:
                    agent_usage[step.agent_name] = agent_usage.get(step.agent_name, 0) + 1
                
                # Tool usage
                for tool in step.tools_used:
                    tool_usage[tool] = tool_usage.get(tool, 0) + 1
                
                # Execution times
                if step_type not in avg_execution_times:
                    avg_execution_times[step_type] = []
                avg_execution_times[step_type].append(step.execution_time_ms)
                
                # Confidence distribution
                conf = step.confidence.value
                confidence_distribution[conf] = confidence_distribution.get(conf, 0) + 1
        
        # Calculate averages
        for step_type in avg_execution_times:
            times = avg_execution_times[step_type]
            avg_execution_times[step_type] = {
                "average": sum(times) / len(times),
                "min": min(times),
                "max": max(times),
                "count": len(times)
            }
        
        # Identify common patterns
        common_sequences = self._find_common_step_sequences(traces)
        
        return {
            "analysis_period": {
                "total_traces": total_traces,
                "total_steps": total_steps,
                "average_steps_per_trace": total_steps / total_traces if total_traces > 0 else 0
            },
            "step_types_frequency": step_types_freq,
            "agent_usage": agent_usage,
            "tool_usage": tool_usage,
            "execution_times": avg_execution_times,
            "confidence_distribution": confidence_distribution,
            "common_step_sequences": common_sequences,
            "generated_at": datetime.utcnow().isoformat()
        }
    
    def _find_common_step_sequences(self, traces: List[ReasoningTrace], min_length: int = 3) -> List[Dict[str, Any]]:
        """Find common sequences of reasoning steps"""
        
        sequences = []
        
        for trace in traces:
            if len(trace.steps) >= min_length:
                step_types = [step.step_type.value for step in trace.steps]
                
                # Generate all subsequences of minimum length
                for i in range(len(step_types) - min_length + 1):
                    seq = tuple(step_types[i:i + min_length])
                    sequences.append(seq)
        
        # Count sequence frequencies
        from collections import Counter
        seq_counter = Counter(sequences)
        
        # Return most common sequences
        common_sequences = []
        for seq, count in seq_counter.most_common(10):
            if count > 1:  # Only return sequences that appear more than once
                common_sequences.append({
                    "sequence": list(seq),
                    "frequency": count,
                    "percentage": (count / len(sequences)) * 100 if sequences else 0
                })
        
        return common_sequences
    
    def get_step_details(self, trace_id: str, step_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific reasoning step"""
        
        trace = self.get_reasoning_trace(trace_id)
        if not trace:
            return None
        
        step = next((s for s in trace.steps if s.step_id == step_id), None)
        if not step:
            return None
        
        # Convert to detailed dict
        step_dict = asdict(step)
        step_dict["timestamp"] = step.timestamp.isoformat()
        step_dict["step_type"] = step.step_type.value
        step_dict["confidence"] = step.confidence.value
        
        # Add context from dependencies
        dependencies_info = []
        for dep_id in step.dependencies:
            dep_step = next((s for s in trace.steps if s.step_id == dep_id), None)
            if dep_step:
                dependencies_info.append({
                    "step_id": dep_id,
                    "description": dep_step.description,
                    "type": dep_step.step_type.value,
                    "output_summary": str(dep_step.output_data)[:200] + "..." if len(str(dep_step.output_data)) > 200 else str(dep_step.output_data)
                })
        
        step_dict["dependencies_info"] = dependencies_info
        
        return step_dict


# Global instance
reasoning_visualizer = ReasoningVisualizer()
