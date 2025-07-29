from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional

from ..core.database import get_db
from ..core.auth import get_current_user, verify_workspace_access
from ..core.logging import logger
from ..models.database import User, Workspace
from ..services.reasoning_visualizer import reasoning_visualizer, ReasoningStepType, ConfidenceLevel

router = APIRouter(prefix="/reasoning", tags=["Reasoning Visualization"])


@router.get("/{workspace_id}/traces")
async def get_reasoning_traces(
    workspace_id: str,
    limit: int = Query(50, ge=1, le=200),
    include_active: bool = Query(True),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get reasoning traces for a workspace"""
    
    try:
        await verify_workspace_access(workspace_id, current_user, db)
        
        traces = reasoning_visualizer.get_traces_for_workspace(
            workspace_id, 
            limit=limit, 
            include_active=include_active
        )
        
        # Convert traces to response format
        trace_summaries = []
        for trace in traces:
            summary = {
                "trace_id": trace.trace_id,
                "query": trace.query,
                "started_at": trace.started_at.isoformat(),
                "completed_at": trace.completed_at.isoformat() if trace.completed_at else None,
                "status": "completed" if trace.completed_at else "active",
                "steps_count": len(trace.steps),
                "overall_confidence": trace.overall_confidence.value if trace.overall_confidence else None,
                "final_answer_preview": trace.final_answer[:200] + "..." if trace.final_answer and len(trace.final_answer) > 200 else trace.final_answer,
                "agents_involved": list(set(step.agent_name for step in trace.steps if step.agent_name)),
                "total_time_ms": (trace.completed_at - trace.started_at).total_seconds() * 1000 if trace.completed_at else None
            }
            trace_summaries.append(summary)
        
        return {
            "traces": trace_summaries,
            "total_count": len(trace_summaries),
            "workspace_id": workspace_id,
            "filters": {
                "limit": limit,
                "include_active": include_active
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting reasoning traces: {e}")
        raise HTTPException(status_code=500, detail="Failed to get reasoning traces")


@router.get("/{workspace_id}/traces/{trace_id}")
async def get_reasoning_trace_details(
    workspace_id: str,
    trace_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get detailed information about a specific reasoning trace"""
    
    try:
        await verify_workspace_access(workspace_id, current_user, db)
        
        trace = reasoning_visualizer.get_reasoning_trace(trace_id)
        if not trace:
            raise HTTPException(status_code=404, detail="Reasoning trace not found")
        
        # Verify trace belongs to workspace
        if trace.workspace_id != workspace_id:
            raise HTTPException(status_code=403, detail="Access denied to this trace")
        
        # Export full trace details
        trace_dict = reasoning_visualizer.export_trace_to_dict(trace_id)
        
        return trace_dict
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting trace details: {e}")
        raise HTTPException(status_code=500, detail="Failed to get trace details")


@router.get("/{workspace_id}/traces/{trace_id}/graph")
async def get_reasoning_graph(
    workspace_id: str,
    trace_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get graph representation of reasoning process"""
    
    try:
        await verify_workspace_access(workspace_id, current_user, db)
        
        trace = reasoning_visualizer.get_reasoning_trace(trace_id)
        if not trace:
            raise HTTPException(status_code=404, detail="Reasoning trace not found")
        
        # Verify trace belongs to workspace
        if trace.workspace_id != workspace_id:
            raise HTTPException(status_code=403, detail="Access denied to this trace")
        
        graph_data = reasoning_visualizer.generate_reasoning_graph(trace_id)
        
        return graph_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating reasoning graph: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate reasoning graph")


@router.get("/{workspace_id}/traces/{trace_id}/steps/{step_id}")
async def get_step_details(
    workspace_id: str,
    trace_id: str,
    step_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get detailed information about a specific reasoning step"""
    
    try:
        await verify_workspace_access(workspace_id, current_user, db)
        
        trace = reasoning_visualizer.get_reasoning_trace(trace_id)
        if not trace:
            raise HTTPException(status_code=404, detail="Reasoning trace not found")
        
        # Verify trace belongs to workspace
        if trace.workspace_id != workspace_id:
            raise HTTPException(status_code=403, detail="Access denied to this trace")
        
        step_details = reasoning_visualizer.get_step_details(trace_id, step_id)
        if not step_details:
            raise HTTPException(status_code=404, detail="Reasoning step not found")
        
        return step_details
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting step details: {e}")
        raise HTTPException(status_code=500, detail="Failed to get step details")


@router.get("/{workspace_id}/analytics/reasoning")
async def get_reasoning_analytics(
    workspace_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get analytics about reasoning patterns in the workspace"""
    
    try:
        await verify_workspace_access(workspace_id, current_user, db)
        
        analytics = reasoning_visualizer.analyze_reasoning_patterns(workspace_id)
        
        return analytics
        
    except Exception as e:
        logger.error(f"Error getting reasoning analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get reasoning analytics")


@router.get("/{workspace_id}/step-types")
async def get_available_step_types():
    """Get available reasoning step types and confidence levels"""
    
    step_types = [
        {
            "value": step_type.value,
            "name": step_type.name,
            "description": _get_step_type_description(step_type)
        }
        for step_type in ReasoningStepType
    ]
    
    confidence_levels = [
        {
            "value": conf.value,
            "name": conf.name,
            "description": _get_confidence_description(conf)
        }
        for conf in ConfidenceLevel
    ]
    
    return {
        "step_types": step_types,
        "confidence_levels": confidence_levels
    }


def _get_step_type_description(step_type: ReasoningStepType) -> str:
    """Get description for reasoning step type"""
    descriptions = {
        ReasoningStepType.QUERY_ANALYSIS: "Analyzing and understanding the user's query",
        ReasoningStepType.DOCUMENT_RETRIEVAL: "Retrieving relevant documents from the knowledge base",
        ReasoningStepType.KNOWLEDGE_LOOKUP: "Looking up entities and relationships in the knowledge graph",
        ReasoningStepType.WEB_SEARCH: "Searching the web for additional information",
        ReasoningStepType.REASONING: "Applying logical reasoning to the available information",
        ReasoningStepType.CONTRADICTION_CHECK: "Checking for contradictions in the information",
        ReasoningStepType.SYNTHESIS: "Synthesizing information from multiple sources",
        ReasoningStepType.CONFIDENCE_ASSESSMENT: "Assessing confidence in the reasoning and conclusions"
    }
    return descriptions.get(step_type, "Unknown step type")


def _get_confidence_description(confidence: ConfidenceLevel) -> str:
    """Get description for confidence level"""
    descriptions = {
        ConfidenceLevel.VERY_LOW: "Very low confidence (0-20%)",
        ConfidenceLevel.LOW: "Low confidence (21-40%)",
        ConfidenceLevel.MEDIUM: "Medium confidence (41-60%)",
        ConfidenceLevel.HIGH: "High confidence (61-80%)",
        ConfidenceLevel.VERY_HIGH: "Very high confidence (81-100%)"
    }
    return descriptions.get(confidence, "Unknown confidence level")


@router.post("/{workspace_id}/traces/{trace_id}/export")
async def export_reasoning_trace(
    workspace_id: str,
    trace_id: str,
    format: str = Query("json", regex="^(json|markdown|html)$"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Export reasoning trace in different formats"""
    
    try:
        await verify_workspace_access(workspace_id, current_user, db)
        
        trace = reasoning_visualizer.get_reasoning_trace(trace_id)
        if not trace:
            raise HTTPException(status_code=404, detail="Reasoning trace not found")
        
        # Verify trace belongs to workspace
        if trace.workspace_id != workspace_id:
            raise HTTPException(status_code=403, detail="Access denied to this trace")
        
        if format == "json":
            return reasoning_visualizer.export_trace_to_dict(trace_id)
        
        elif format == "markdown":
            markdown_content = _export_trace_to_markdown(trace)
            return {"content": markdown_content, "format": "markdown"}
        
        elif format == "html":
            html_content = _export_trace_to_html(trace)
            return {"content": html_content, "format": "html"}
        
        else:
            raise HTTPException(status_code=400, detail="Unsupported export format")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting trace: {e}")
        raise HTTPException(status_code=500, detail="Failed to export trace")


def _export_trace_to_markdown(trace) -> str:
    """Export reasoning trace to Markdown format"""
    
    lines = []
    lines.append(f"# Reasoning Trace: {trace.trace_id}")
    lines.append(f"**Query:** {trace.query}")
    lines.append(f"**Started:** {trace.started_at.isoformat()}")
    if trace.completed_at:
        lines.append(f"**Completed:** {trace.completed_at.isoformat()}")
        duration = (trace.completed_at - trace.started_at).total_seconds()
        lines.append(f"**Duration:** {duration:.2f} seconds")
    lines.append("")
    
    if trace.final_answer:
        lines.append("## Final Answer")
        lines.append(trace.final_answer)
        if trace.overall_confidence:
            lines.append(f"**Confidence:** {trace.overall_confidence.value}")
        lines.append("")
    
    lines.append("## Reasoning Steps")
    for i, step in enumerate(trace.steps, 1):
        lines.append(f"### Step {i}: {step.step_type.value}")
        lines.append(f"**Description:** {step.description}")
        lines.append(f"**Agent:** {step.agent_name or 'Unknown'}")
        lines.append(f"**Confidence:** {step.confidence.value}")
        lines.append(f"**Execution Time:** {step.execution_time_ms:.2f}ms")
        
        if step.tools_used:
            lines.append(f"**Tools Used:** {', '.join(step.tools_used)}")
        
        if step.input_data:
            lines.append("**Input:**")
            lines.append(f"```json")
            lines.append(str(step.input_data))
            lines.append("```")
        
        if step.output_data:
            lines.append("**Output:**")
            lines.append(f"```json")
            lines.append(str(step.output_data))
            lines.append("```")
        
        lines.append("")
    
    return "\n".join(lines)


def _export_trace_to_html(trace) -> str:
    """Export reasoning trace to HTML format"""
    
    html = f"""
    <html>
    <head>
        <title>Reasoning Trace: {trace.trace_id}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
            .step {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }}
            .step-header {{ font-weight: bold; color: #333; }}
            .confidence {{ color: #007bff; }}
            .agent {{ color: #28a745; }}
            .tools {{ color: #6c757d; }}
            pre {{ background-color: #f8f9fa; padding: 10px; border-radius: 3px; overflow-x: auto; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Reasoning Trace: {trace.trace_id}</h1>
            <p><strong>Query:</strong> {trace.query}</p>
            <p><strong>Started:</strong> {trace.started_at.isoformat()}</p>
    """
    
    if trace.completed_at:
        duration = (trace.completed_at - trace.started_at).total_seconds()
        html += f"""
            <p><strong>Completed:</strong> {trace.completed_at.isoformat()}</p>
            <p><strong>Duration:</strong> {duration:.2f} seconds</p>
        """
    
    html += "</div>"
    
    if trace.final_answer:
        html += f"""
        <div class="step">
            <div class="step-header">Final Answer</div>
            <p>{trace.final_answer}</p>
        """
        if trace.overall_confidence:
            html += f'<p class="confidence"><strong>Confidence:</strong> {trace.overall_confidence.value}</p>'
        html += "</div>"
    
    html += "<h2>Reasoning Steps</h2>"
    
    for i, step in enumerate(trace.steps, 1):
        html += f"""
        <div class="step">
            <div class="step-header">Step {i}: {step.step_type.value}</div>
            <p><strong>Description:</strong> {step.description}</p>
            <p class="agent"><strong>Agent:</strong> {step.agent_name or 'Unknown'}</p>
            <p class="confidence"><strong>Confidence:</strong> {step.confidence.value}</p>
            <p><strong>Execution Time:</strong> {step.execution_time_ms:.2f}ms</p>
        """
        
        if step.tools_used:
            html += f'<p class="tools"><strong>Tools Used:</strong> {", ".join(step.tools_used)}</p>'
        
        if step.input_data:
            html += f"<p><strong>Input:</strong></p><pre>{str(step.input_data)}</pre>"
        
        if step.output_data:
            html += f"<p><strong>Output:</strong></p><pre>{str(step.output_data)}</pre>"
        
        html += "</div>"
    
    html += """
    </body>
    </html>
    """
    
    return html
