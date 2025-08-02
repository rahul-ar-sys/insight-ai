from typing import Dict, List, Any, Optional, Type
from abc import ABC, abstractmethod
import asyncio
import httpx
from datetime import datetime
import json

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
# --- CHANGE: Import ChatOpenAI instead of ChatGoogleGenerativeAI ---
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from ..core.config import settings
from ..core.logging import logger
from ..dependencies import get_vector_store_service


# --- Input Schemas for Tools (No changes here) ---

class DocumentRetrieverInput(BaseModel):
    query: str = Field(description="Search query for document retrieval")
    workspace_id: str = Field(description="Workspace ID to search in")
    max_results: int = Field(default=10, description="Maximum number of results")

class WebSearchInput(BaseModel):
    query: str = Field(description="Search query for web search")
    max_results: int = Field(default=5, description="Maximum number of results")

class ReasoningEngineInput(BaseModel):
    information: Dict[str, Any] = Field(description="Information to reason over")
    task: str = Field(description="Reasoning task to perform")

class ContradictionDetectorInput(BaseModel):
    documents: List[Dict[str, Any]] = Field(description="Documents to check for contradictions")
    web_results: List[Dict[str, Any]] = Field(default=[], description="Web results to include in check")

class ResponseSynthesizerInput(BaseModel):
    synthesis_data: Dict[str, Any] = Field(description="All data to synthesize into a response")


# --- Tool Definitions (Only ResponseSynthesizerTool is changed) ---

class DocumentRetrieverTool(BaseTool):
    # ... (no changes in this class)
    name: str = "document_retriever"
    description: str = "Retrieves relevant document chunks based on semantic similarity"
    args_schema: Type[BaseModel] = DocumentRetrieverInput
    
    _vector_service = None
    _embedding_model = None

    def _get_vector_service(self):
        if self._vector_service is None:
            self._vector_service = get_vector_store_service()
        return self._vector_service
    
    def _get_embedding_model(self):
        if self._embedding_model is None:
            try:
                self._embedding_model = SentenceTransformer(settings.embedding_model)
            except Exception as e:
                logger.error(f"Failed to load embedding model in DocumentRetrieverTool: {e}")
                self._embedding_model = None
        return self._embedding_model
    
    async def _arun(self, query: str, workspace_id: str, max_results: int = 10) -> List[Dict[str, Any]]:
        embedding_model = self._get_embedding_model()
        if not embedding_model: return []
        
        try:
            query_embedding = embedding_model.encode(query, normalize_embeddings=True)
            results = await self._get_vector_service().search_similar(
                query_embedding=query_embedding, limit=max_results, filters={"workspace_id": workspace_id}
            )
            
            for r in results:
                r['content'] = r.get('text', '')
            return results
        except Exception as e:
            logger.error(f"Document retrieval error: {e}", exc_info=True)
            return []
    
    def _run(self, query: str, workspace_id: str, max_results: int = 10) -> List[Dict[str, Any]]:
        return asyncio.run(self._arun(query, workspace_id, max_results))


class WebSearchTool(BaseTool):
    # ... (no changes in this class)
    name: str = "web_search"
    description: str = "Searches the web for additional information and recent updates"
    args_schema: Type[BaseModel] = WebSearchInput
    
    async def _arun(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        if not settings.tavily_api_key:
            logger.warning("Tavily API key not configured, skipping web search")
            return []
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.tavily.com/search",
                    json={"api_key": settings.tavily_api_key, "query": query, "search_depth": "advanced", "max_results": max_results},
                    timeout=30.0
                )
            if response.status_code == 200:
                return response.json().get("results", [])
            else:
                logger.error(f"Web search API error: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            logger.error(f"Web search error: {e}", exc_info=True)
            return []
    
    def _run(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        return asyncio.run(self._arun(query, max_results))


class ReasoningEngineTool(BaseTool):
    # ... (no changes in this class)
    name: str = "reasoning_engine"
    description: str = "Analyzes information and performs logical reasoning (placeholder)"
    args_schema: Type[BaseModel] = ReasoningEngineInput
    
    async def _arun(self, information: Dict[str, Any], task: str) -> Dict[str, Any]:
        return {"steps": [], "confidence": 70}
    
    def _run(self, information: Dict[str, Any], task: str) -> Dict[str, Any]:
        return asyncio.run(self._arun(information, task))


class ContradictionDetectorTool(BaseTool):
    # ... (no changes in this class)
    name: str = "contradiction_detector"
    description: str = "Detects potential contradictions in information (placeholder)"
    args_schema: Type[BaseModel] = ContradictionDetectorInput
    
    async def _arun(self, documents: List[Dict[str, Any]], web_results: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        return []
    
    def _run(self, documents: List[Dict[str, Any]], web_results: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        return asyncio.run(self._arun(documents, web_results))


class ResponseSynthesizerTool(BaseTool):
    name: str = "response_synthesizer"
    description: str = "Synthesizes information from multiple sources into a comprehensive response"
    args_schema: Type[BaseModel] = ResponseSynthesizerInput

    _llm = None

    # --- CHANGE: Updated to initialize Ollama ---
    def _get_llm(self):
        """Lazy initialization of the LLM to connect to the Ollama service."""
        if self._llm is None:
            if not settings.ollama_base_url:
                raise ValueError("OLLAMA_BASE_URL not found in environment settings.")
            
            self._llm = ChatOpenAI(
                model=settings.ollama_model, # e.g., 'llama3:8b'
                base_url=settings.ollama_base_url, # e.g., 'http://ollama:11434/v1'
                api_key="ollama", # Required by the library, but not used by Ollama
                temperature=0.2,
            )
        return self._llm

    # --- No changes needed in the methods below ---
    async def _arun(self, synthesis_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            query = synthesis_data.get("query", "")
            documents = synthesis_data.get("documents", [])
            
            context = "\n\n---\n\n".join([doc.get("content", "") for doc in documents])
            
            response_text = await self._generate_response(query, context)
            
            sources = [
                {
                    "type": "document",
                    "title": doc.get("metadata", {}).get("file_name", "Document"),
                    "relevance_score": doc.get("score", 0),
                } for doc in documents[:5]
            ]
            
            return {"response": response_text, "sources": sources, "confidence_score": 85}
            
        except Exception as e:
            logger.error(f"Response synthesis error: {e}", exc_info=True)
            return {"response": "I apologize, but I encountered an error during synthesis.", "sources": [], "confidence_score": 0}
    
    async def _generate_response(self, query: str, context: str) -> str:
        llm = self._get_llm() # Get the initialized LLM
        
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", 
                 "You are a helpful AI assistant. Provide a comprehensive answer to the user's question.\n"
                 "First, use your general knowledge to give a clear, direct answer.\n"
                 "Then, use the specific context provided below to add details, examples, or correct your general knowledge.\n"
                 "If the context is empty, rely solely on your general knowledge.\n"
                 "Format your response clearly using markdown."),
                ("user", "Context:\n{context}\n\nQuestion: {question}")
            ]
        )
        chain = prompt_template | llm
        response = await chain.ainvoke({"context": context, "question": query})
        return response.content

    def _run(self, synthesis_data: Dict[str, Any]) -> Dict[str, Any]:
        return asyncio.run(self._arun(synthesis_data))

# --- Tool Registry to manage all tools (No changes here) ---

class ToolRegistry:
    # ... (no changes in this class)
    def __init__(self):
        self.tools = {
            "document_retriever": DocumentRetrieverTool(),
            "web_search": WebSearchTool(),
            "reasoning_engine": ReasoningEngineTool(),
            "contradiction_detector": ContradictionDetectorTool(),
            "response_synthesizer": ResponseSynthesizerTool()
        }
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        return self.tools.get(tool_name)
    
    def get_available_tools(self) -> List[BaseTool]:
        return list(self.tools.values())