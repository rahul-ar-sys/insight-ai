from typing import Dict, List, Any, Optional, TypedDict, Annotated
from datetime import datetime
import asyncio
import uuid
import time
from abc import ABC, abstractmethod

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END, START
from langchain.schema import AgentAction, AgentFinish

from ..core.logging import logger, log_agent_trace
from ..models.schemas import AgentStep, AgentTrace, QueryRequest, QueryResponse
from ..tools import ToolRegistry
from ..services.reasoning_visualizer import (
    reasoning_visualizer, ReasoningStepType, ConfidenceLevel
)


class AgentState(TypedDict):
    """State shared between agents in the workflow"""
    query: str
    workspace_id: str
    session_id: Optional[str]
    user_id: str
    messages: List[BaseMessage]
    plan: List[Dict[str, Any]]
    current_step: int
    retrieved_docs: List[Dict[str, Any]]
    web_results: List[Dict[str, Any]]
    reasoning_steps: List[Dict[str, Any]]
    final_response: Optional[str]
    sources: List[Dict[str, Any]]
    error: Optional[str]
    execution_trace: List[AgentStep]
    confidence_score: int
    reasoning_trace_id: Optional[str]


class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.start_time = None
    
    @abstractmethod
    async def execute(self, state: AgentState) -> AgentState:
        """Execute the agent's task"""
        pass
    
    def start_step_timing(self):
        """Start timing for a step"""
        self.step_start_time = datetime.utcnow()
    
    def _log_step(self, action: str, input_data: Dict[str, Any], output_data: Dict[str, Any], state: AgentState):
        """Log agent execution step"""
        step_start_time = getattr(self, 'step_start_time', datetime.utcnow())
        execution_time = int((datetime.utcnow() - step_start_time).total_seconds() * 1000)
        
        step = AgentStep(
            agent_name=self.name,
            role=self.role,
            action=action,
            input_data=input_data,
            output_data=output_data,
            execution_time_ms=execution_time,
            timestamp=datetime.utcnow()
        )
        
        state["execution_trace"].append(step)
        
        step_type_mapping = {
            "create_plan": ReasoningStepType.QUERY_ANALYSIS,
            "retrieve_documents": ReasoningStepType.DOCUMENT_RETRIEVAL,
            "web_search": ReasoningStepType.WEB_SEARCH,
            "analyze_documents": ReasoningStepType.REASONING,
            "check_contradictions": ReasoningStepType.CONTRADICTION_CHECK,
            "synthesize_response": ReasoningStepType.SYNTHESIS
        }
        
        step_type = step_type_mapping.get(action, ReasoningStepType.REASONING)
        
        confidence_level = ConfidenceLevel.MEDIUM
        if "confidence" in output_data:
            conf_score = output_data.get("confidence", 50)
            if conf_score >= 80:
                confidence_level = ConfidenceLevel.VERY_HIGH
            elif conf_score >= 60:
                confidence_level = ConfidenceLevel.HIGH
            elif conf_score >= 40:
                confidence_level = ConfidenceLevel.MEDIUM
            elif conf_score >= 20:
                confidence_level = ConfidenceLevel.LOW
            else:
                confidence_level = ConfidenceLevel.VERY_LOW
        
        if state.get("reasoning_trace_id"):
            reasoning_visualizer.add_reasoning_step(
                trace_id=state["reasoning_trace_id"],
                step_type=step_type,
                description=f"{self.name}: {action}",
                input_data=input_data,
                output_data=output_data,
                confidence=confidence_level,
                execution_time_ms=execution_time,
                agent_name=self.name,
                tools_used=output_data.get("tools_used", []),
                metadata={"role": self.role, "action": action}
            )
        
        log_agent_trace(
            self.name,
            action,
            {
                "input": input_data,
                "output": output_data,
                "execution_time_ms": execution_time
            }
        )


class PlannerAgent(BaseAgent):
    def __init__(self):
        super().__init__("planner", "planner")
        self.tool_registry = ToolRegistry()
    
    async def execute(self, state: AgentState) -> AgentState:
        self.start_time = datetime.utcnow()
        self.start_step_timing()
        
        try:
            query = state["query"]
            available_tools = self.tool_registry.get_available_tools()
            plan = await self._create_plan(query, available_tools)
            state["plan"] = plan
            state["current_step"] = 0
            
            self._log_step(
                "create_plan",
                {"query": query, "available_tools": [tool.name for tool in available_tools]},
                {"plan": plan, "steps_count": len(plan), "confidence": 80},
                state
            )
            
            logger.info(f"Created execution plan with {len(plan)} steps for query: {query[:100]}...")
            return state
            
        except Exception as e:
            logger.error(f"Planner agent error: {e}", exc_info=True)
            state["error"] = str(e)
            return state
    
    async def _create_plan(self, query: str, available_tools: List[BaseTool]) -> List[Dict[str, Any]]:
        plan_steps = []
        plan_steps.append({
            "step_id": 1, "agent": "retriever", "action": "retrieve_documents",
            "description": "Retrieve relevant documents from the workspace",
            "tools": ["document_retriever"], "input": {"query": query}, "depends_on": []
        })
        
        web_indicators = ["latest", "recent", "current", "today", "news", "update"]
        needs_web_search = any(indicator in query.lower() for indicator in web_indicators)
        
        if needs_web_search:
            plan_steps.append({
                "step_id": 2, "agent": "web_searcher", "action": "web_search",
                "description": "Search the web for additional context",
                "tools": ["web_search"], "input": {"query": query}, "depends_on": []
            })
        
        plan_steps.append({
            "step_id": 3, "agent": "analyzer", "action": "analyze_information",
            "description": "Analyze retrieved information and perform reasoning",
            "tools": ["reasoning_engine", "calculator"], "input": {"query": query},
            "depends_on": [1] + ([2] if needs_web_search else [])
        })
        
        if len(query.split()) > 10:
            plan_steps.append({
                "step_id": 4, "agent": "contradiction_detector", "action": "detect_contradictions",
                "description": "Check for contradictions in the information",
                "tools": ["contradiction_detector"], "input": {"query": query}, "depends_on": [3]
            })
        
        final_step_id = len(plan_steps) + 1
        plan_steps.append({
            "step_id": final_step_id, "agent": "synthesizer", "action": "synthesize_response",
            "description": "Synthesize comprehensive final response",
            "tools": ["response_synthesizer"], "input": {"query": query},
            "depends_on": list(range(1, final_step_id))
        })
        
        return plan_steps


class RetrieverAgent(BaseAgent):
    def __init__(self):
        super().__init__("retriever", "retriever")
        self.tool_registry = ToolRegistry()
    
    async def execute(self, state: AgentState) -> AgentState:
        self.start_time = datetime.utcnow()
        self.start_step_timing()
        
        try:
            query = state["query"]
            workspace_id = state["workspace_id"]
            
            retriever_tool = self.tool_registry.get_tool("document_retriever")
            if not retriever_tool:
                raise Exception("Document retriever tool not available")
            
            results = await retriever_tool.arun({
                "query": query,
                "workspace_id": workspace_id,
                "max_results": 10
            })
            
            state["retrieved_docs"] = results
            
            self._log_step(
                "retrieve_documents",
                {"query": query, "workspace_id": workspace_id},
                {"documents_found": len(results), "confidence": 90},
                state
            )
            
            logger.info(f"Retrieved {len(results)} relevant documents")
            return state
            
        except Exception as e:
            logger.error(f"Retriever agent error: {e}", exc_info=True)
            state["error"] = str(e)
            return state


class WebSearchAgent(BaseAgent):
    def __init__(self):
        super().__init__("web_searcher", "retriever")
        self.tool_registry = ToolRegistry()
    
    async def execute(self, state: AgentState) -> AgentState:
        self.start_time = datetime.utcnow()
        self.start_step_timing()
        
        try:
            query = state["query"]
            
            web_search_tool = self.tool_registry.get_tool("web_search")
            if not web_search_tool:
                logger.warning("Web search tool not available, skipping web search")
                state["web_results"] = []
                return state
            
            results = await web_search_tool.arun({
                "query": query,
                "max_results": 5
            })
            
            state["web_results"] = results
            
            self._log_step(
                "web_search",
                {"query": query},
                {"results_found": len(results), "confidence": 75},
                state
            )
            
            logger.info(f"Found {len(results)} web search results")
            return state
            
        except Exception as e:
            logger.error(f"Web search agent error: {e}", exc_info=True)
            state["web_results"] = []
            return state


class AnalyzerAgent(BaseAgent):
    def __init__(self):
        super().__init__("analyzer", "analyzer")
        self.tool_registry = ToolRegistry()
    
    async def execute(self, state: AgentState) -> AgentState:
        self.start_time = datetime.utcnow()
        self.start_step_timing()
        
        try:
            query = state["query"]
            docs = state.get("retrieved_docs", [])
            web_results = state.get("web_results", [])
            
            reasoning_tool = self.tool_registry.get_tool("reasoning_engine")
            if not reasoning_tool:
                raise Exception("Reasoning engine tool not available")
            
            all_information = {
                "query": query, "documents": docs, "web_results": web_results
            }
            
            reasoning_results = await reasoning_tool.arun({
                "information": all_information,
                "task": "analyze_and_reason"
            })
            
            state["reasoning_steps"] = reasoning_results.get("steps", [])
            
            self._log_step(
                "analyze_information",
                {"query": query, "docs_count": len(docs), "web_results_count": len(web_results)},
                {"reasoning_steps": len(reasoning_results.get("steps", [])), "confidence": 85},
                state
            )
            
            logger.info(f"Completed analysis with {len(reasoning_results.get('steps', []))} reasoning steps")
            return state
            
        except Exception as e:
            logger.error(f"Analyzer agent error: {e}", exc_info=True)
            state["error"] = str(e)
            return state


class ContradictionDetectorAgent(BaseAgent):
    def __init__(self):
        super().__init__("contradiction_detector", "contradiction_detector")
        self.tool_registry = ToolRegistry()
    
    async def execute(self, state: AgentState) -> AgentState:
        self.start_time = datetime.utcnow()
        self.start_step_timing()
        
        try:
            docs = state.get("retrieved_docs", [])
            web_results = state.get("web_results", [])
            
            if len(docs) < 1 and len(web_results) < 1:
                return state
            
            contradiction_tool = self.tool_registry.get_tool("contradiction_detector")
            if not contradiction_tool:
                logger.warning("Contradiction detector tool not available")
                return state
            
            contradictions = await contradiction_tool.arun({
                "documents": docs,
                "web_results": web_results
            })
            
            if contradictions:
                contradiction_step = {
                    "step_type": "contradiction_detection",
                    "contradictions_found": len(contradictions),
                    "details": contradictions,
                    "timestamp": datetime.utcnow().isoformat()
                }
                if "reasoning_steps" not in state: state["reasoning_steps"] = []
                state["reasoning_steps"].append(contradiction_step)
            
            self._log_step(
                "detect_contradictions",
                {"docs_count": len(docs), "web_results_count": len(web_results)},
                {"contradictions_found": len(contradictions) if contradictions else 0, "confidence": 70},
                state
            )
            
            logger.info(f"Contradiction detection completed, found {len(contradictions) if contradictions else 0} potential contradictions")
            return state
            
        except Exception as e:
            logger.error(f"Contradiction detector agent error: {e}", exc_info=True)
            return state


class SynthesizerAgent(BaseAgent):
    def __init__(self):
        super().__init__("synthesizer", "synthesizer")
        self.tool_registry = ToolRegistry()
    
    async def execute(self, state: AgentState) -> AgentState:
        self.start_time = datetime.utcnow()
        self.start_step_timing()
        
        try:
            query = state["query"]
            docs = state.get("retrieved_docs", [])
            web_results = state.get("web_results", [])
            reasoning_steps = state.get("reasoning_steps", [])
            
            synthesizer_tool = self.tool_registry.get_tool("response_synthesizer")
            if not synthesizer_tool:
                raise Exception("Response synthesizer tool not available")
            
            synthesis_input = {
                "query": query,
                "documents": docs,
                "web_results": web_results,
                "reasoning_steps": reasoning_steps
            }
            
            synthesis_result = await synthesizer_tool.arun({"synthesis_data": synthesis_input})
            
            state["final_response"] = synthesis_result.get("response", "")
            state["sources"] = synthesis_result.get("sources", [])
            state["confidence_score"] = synthesis_result.get("confidence_score", 50)
            
            self._log_step(
                "synthesize_response",
                synthesis_input,
                {
                    "response_length": len(synthesis_result.get("response", "")),
                    "sources_count": len(synthesis_result.get("sources", [])),
                    "confidence_score": synthesis_result.get("confidence_score", 50)
                },
                state
            )
            
            logger.info(f"Synthesized final response with {len(synthesis_result.get('sources', []))} sources")
            return state
            
        except Exception as e:
            logger.error(f"Synthesizer agent error: {e}", exc_info=True)
            state["error"] = str(e)
            return state


class AgentOrchestrator:
    """Main orchestrator that manages the multi-agent workflow using LangGraph"""
    
    def __init__(self):
        self.agents = {
            "planner": PlannerAgent(),
            "retriever": RetrieverAgent(),
            "web_searcher": WebSearchAgent(),
            "analyzer": AnalyzerAgent(),
            "contradiction_detector": ContradictionDetectorAgent(),
            "synthesizer": SynthesizerAgent()
        }
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        
        workflow.add_node("planner", self._run_planner)
        workflow.add_node("retriever", self._run_retriever)
        workflow.add_node("web_searcher", self._run_web_searcher)
        workflow.add_node("analyzer", self._run_analyzer)
        workflow.add_node("contradiction_detector", self._run_contradiction_detector)
        workflow.add_node("synthesizer", self._run_synthesizer)
        
        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "retriever")
        
        workflow.add_conditional_edges(
            "retriever",
            self._should_do_web_search,
            {"web_search": "web_searcher", "analyze": "analyzer"}
        )
        
        workflow.add_edge("web_searcher", "analyzer")
        
        workflow.add_conditional_edges(
            "analyzer",
            self._should_check_contradictions,
            {"check_contradictions": "contradiction_detector", "synthesize": "synthesizer"}
        )
        
        workflow.add_edge("contradiction_detector", "synthesizer")
        workflow.add_edge("synthesizer", END)
        
        return workflow.compile()
    
    async def _run_planner(self, state: AgentState) -> AgentState:
        return await self.agents["planner"].execute(state)
    
    async def _run_retriever(self, state: AgentState) -> AgentState:
        return await self.agents["retriever"].execute(state)
    
    async def _run_web_searcher(self, state: AgentState) -> AgentState:
        return await self.agents["web_searcher"].execute(state)
    
    async def _run_analyzer(self, state: AgentState) -> AgentState:
        return await self.agents["analyzer"].execute(state)
    
    async def _run_contradiction_detector(self, state: AgentState) -> AgentState:
        return await self.agents["contradiction_detector"].execute(state)
    
    async def _run_synthesizer(self, state: AgentState) -> AgentState:
        return await self.agents["synthesizer"].execute(state)
    
    def _should_do_web_search(self, state: AgentState) -> str:
        plan = state.get("plan", [])
        for step in plan:
            if step.get("action") == "web_search":
                return "web_search"
        return "analyze"
    
    def _should_check_contradictions(self, state: AgentState) -> str:
        plan = state.get("plan", [])
        for step in plan:
            if step.get("action") == "detect_contradictions":
                return "check_contradictions"
        return "synthesize"
    
    async def execute_query(self, request: QueryRequest, user_id: str) -> QueryResponse:
        start_time = datetime.utcnow()
        reasoning_trace_id = reasoning_visualizer.start_reasoning_trace(
            query=request.query,
            workspace_id=str(request.workspace_id),
            user_id=user_id,
            metadata={"session_id": str(request.session_id) if request.session_id else None}
        )
        
        initial_state: AgentState = {
            "query": request.query,
            "workspace_id": str(request.workspace_id),
            "session_id": str(request.session_id) if request.session_id else None,
            "user_id": user_id,
            "messages": [HumanMessage(content=request.query)],
            "plan": [], "current_step": 0, "retrieved_docs": [], "web_results": [],
            "reasoning_steps": [], "final_response": None, "sources": [], "error": None,
            "execution_trace": [], "confidence_score": 50, "reasoning_trace_id": reasoning_trace_id
        }
        
        try:
            final_state = await self.workflow.ainvoke(initial_state)
            
            end_time = datetime.utcnow()
            total_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            if final_state.get("error"):
                raise Exception(final_state["error"])
            
            response = QueryResponse(
                response=final_state.get("final_response", "I couldn't generate a response."),
                sources=final_state.get("sources", []),
                reasoning_steps=final_state.get("reasoning_steps", []),
                session_id=request.session_id or uuid.uuid4(),
                turn_id=uuid.uuid4(),
                processing_time_ms=total_time_ms,
                confidence_score=final_state.get("confidence_score", 50)
            )
            
            logger.info(f"Query execution completed in {total_time_ms}ms")
            
            if reasoning_trace_id:
                confidence_map = {
                    0: ConfidenceLevel.VERY_LOW, 1: ConfidenceLevel.LOW, 2: ConfidenceLevel.MEDIUM,
                    3: ConfidenceLevel.HIGH, 4: ConfidenceLevel.VERY_HIGH
                }
                confidence_level = confidence_map.get(
                    min(4, max(0, int(final_state.get("confidence_score", 50) / 20))),
                    ConfidenceLevel.MEDIUM
                )
                
                reasoning_visualizer.complete_reasoning_trace(
                    trace_id=reasoning_trace_id,
                    final_answer=final_state.get("final_response", ""),
                    overall_confidence=confidence_level,
                    metadata={
                        "total_execution_time_ms": total_time_ms,
                        "total_steps": len(final_state.get("execution_trace", [])),
                        "sources_count": len(final_state.get("sources", []))
                    }
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}", exc_info=True)
            
            if reasoning_trace_id:
                reasoning_visualizer.complete_reasoning_trace(
                    trace_id=reasoning_trace_id,
                    final_answer=f"Error: {str(e)}",
                    overall_confidence=ConfidenceLevel.VERY_LOW,
                    metadata={"error": str(e)}
                )
            
            return QueryResponse(
                response=f"I apologize, but I encountered an error while processing your query: {str(e)}",
                sources=[],
                reasoning_steps=[],
                session_id=request.session_id or uuid.uuid4(),
                turn_id=uuid.uuid4(),
                processing_time_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000),
                confidence_score=0
            )