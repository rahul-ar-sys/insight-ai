from typing import Dict, List, Any, Optional, Type
from abc import ABC, abstractmethod
import asyncio
import httpx
from datetime import datetime
import json

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from ..core.config import settings
from ..core.logging import logger
from ..core.database import get_db
from ..models.database import DocumentChunk, Document, Workspace
from ..services.vector_store import VectorStoreService


class DocumentRetrieverInput(BaseModel):
    """Input schema for document retriever tool"""
    query: str = Field(description="Search query for document retrieval")
    workspace_id: str = Field(description="Workspace ID to search in")
    max_results: int = Field(default=10, description="Maximum number of results")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Additional filters")


class DocumentRetrieverTool(BaseTool):
    """Tool for retrieving relevant documents from the vector database"""
    
    name: str = "document_retriever"
    description: str = "Retrieves relevant document chunks based on semantic similarity"
    args_schema: Type[BaseModel] = DocumentRetrieverInput
    
    def _get_vector_service(self):
        """Lazy initialization of vector service"""
        if not hasattr(self, '_vector_service'):
            self._vector_service = VectorStoreService()
        return self._vector_service
    
    def _get_embedding_model(self):
        """Lazy initialization of embedding model"""
        if not hasattr(self, '_embedding_model'):
            self._load_embedding_model()
        return self._embedding_model
    
    def _load_embedding_model(self):
        """Load the embedding model"""
        try:
            self._embedding_model = SentenceTransformer(settings.embedding_model)
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self._embedding_model = None
    
    async def _arun(
        self,
        query: str,
        workspace_id: str,
        max_results: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant documents asynchronously"""
        
        embedding_model = self._get_embedding_model()
        if not embedding_model:
            return []
        
        try:
            # Generate query embedding
            query_embedding = embedding_model.encode(query)
            
            # Set up filters
            search_filters = {"workspace_id": workspace_id}
            if filters:
                search_filters.update(filters)
            
            # Search vector database
            vector_service = self._get_vector_service()
            results = await vector_service.search_similar(
                query_embedding=query_embedding,
                limit=max_results,
                filters=search_filters
            )
            
            # Format results for agents
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "chunk_id": result["id"],
                    "content": result["text"],
                    "score": result["score"],
                    "metadata": result["metadata"],
                    "source": {
                        "document_id": result["metadata"].get("document_id"),
                        "file_name": result["metadata"].get("file_name"),
                        "chunk_index": result["metadata"].get("chunk_index")
                    }
                })
            
            logger.info(f"Retrieved {len(formatted_results)} documents for query: {query[:50]}...")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Document retrieval error: {e}")
            return []
    
    def _run(self, query: str, workspace_id: str, max_results: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Synchronous version (fallback)"""
        return asyncio.run(self._arun(query, workspace_id, max_results, filters))


class WebSearchInput(BaseModel):
    """Input schema for web search tool"""
    query: str = Field(description="Search query for web search")
    max_results: int = Field(default=5, description="Maximum number of results")


class WebSearchTool(BaseTool):
    """Tool for searching the web using Tavily API"""
    
    name: str = "web_search"
    description: str = "Searches the web for additional information and recent updates"
    args_schema: Type[BaseModel] = WebSearchInput
    
    async def _arun(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search the web asynchronously"""
        
        if not settings.tavily_api_key:
            logger.warning("Tavily API key not configured, skipping web search")
            return []
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": settings.tavily_api_key,
                        "query": query,
                        "search_depth": "advanced",
                        "include_answer": True,
                        "include_raw_content": False,
                        "max_results": max_results,
                        "include_domains": [],
                        "exclude_domains": []
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results = []
                    
                    for item in data.get("results", []):
                        results.append({
                            "title": item.get("title", ""),
                            "content": item.get("content", ""),
                            "url": item.get("url", ""),
                            "score": item.get("score", 0.0),
                            "published_date": item.get("published_date"),
                            "source": "web"
                        })
                    
                    logger.info(f"Found {len(results)} web search results for query: {query[:50]}...")
                    return results
                else:
                    logger.error(f"Web search API error: {response.status_code}")
                    return []
                    
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return []
    
    def _run(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Synchronous version (fallback)"""
        return asyncio.run(self._arun(query, max_results))


class ReasoningEngineInput(BaseModel):
    """Input schema for reasoning engine tool"""
    information: Dict[str, Any] = Field(description="Information to reason over")
    task: str = Field(description="Reasoning task to perform")


class ReasoningEngineTool(BaseTool):
    """Tool for analyzing and reasoning over information"""
    
    name: str = "reasoning_engine"
    description: str = "Analyzes information and performs logical reasoning"
    args_schema: Type[BaseModel] = ReasoningEngineInput
    
    async def _arun(self, information: Dict[str, Any], task: str) -> Dict[str, Any]:
        """Perform reasoning over the provided information"""
        
        try:
            query = information.get("query", "")
            documents = information.get("documents", [])
            web_results = information.get("web_results", [])
            
            reasoning_steps = []
            
            # Step 1: Information Summary
            doc_count = len(documents)
            web_count = len(web_results)
            
            reasoning_steps.append({
                "step_type": "information_summary",
                "description": f"Analyzed {doc_count} documents and {web_count} web results",
                "details": {
                    "document_sources": [doc.get("source", {}).get("file_name", "Unknown") for doc in documents[:5]],
                    "web_sources": [result.get("title", "Unknown") for result in web_results[:3]]
                },
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Step 2: Relevance Analysis
            relevant_docs = self._analyze_relevance(query, documents)
            relevant_web = self._analyze_relevance(query, web_results)
            
            reasoning_steps.append({
                "step_type": "relevance_analysis",
                "description": f"Identified {len(relevant_docs)} relevant documents and {len(relevant_web)} relevant web results",
                "details": {
                    "relevant_document_scores": [doc.get("score", 0) for doc in relevant_docs],
                    "relevant_web_scores": [result.get("score", 0) for result in relevant_web]
                },
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Step 3: Key Information Extraction
            key_information = self._extract_key_information(query, relevant_docs + relevant_web)
            
            reasoning_steps.append({
                "step_type": "key_information_extraction",
                "description": f"Extracted {len(key_information)} key pieces of information",
                "details": {
                    "key_points": key_information[:10]  # Limit to top 10
                },
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Step 4: Logical Connections
            connections = self._find_logical_connections(key_information)
            
            reasoning_steps.append({
                "step_type": "logical_connections",
                "description": f"Found {len(connections)} logical connections between information pieces",
                "details": {
                    "connections": connections[:5]  # Limit to top 5
                },
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return {
                "steps": reasoning_steps,
                "key_information": key_information,
                "connections": connections,
                "confidence": self._calculate_reasoning_confidence(relevant_docs, relevant_web)
            }
            
        except Exception as e:
            logger.error(f"Reasoning engine error: {e}")
            return {"steps": [], "key_information": [], "connections": [], "confidence": 0}
    
    def _analyze_relevance(self, query: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze relevance of items to the query"""
        relevant_items = []
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for item in items:
            content = item.get("content", "") + " " + item.get("title", "")
            content_lower = content.lower()
            content_words = set(content_lower.split())
            
            # Simple relevance scoring based on word overlap
            overlap = len(query_words.intersection(content_words))
            relevance_score = overlap / len(query_words) if query_words else 0
            
            if relevance_score > 0.1:  # Threshold for relevance
                item_copy = item.copy()
                item_copy["relevance_score"] = relevance_score
                relevant_items.append(item_copy)
        
        # Sort by relevance
        relevant_items.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return relevant_items
    
    def _extract_key_information(self, query: str, items: List[Dict[str, Any]]) -> List[str]:
        """Extract key information pieces from relevant items"""
        key_info = []
        
        for item in items[:10]:  # Process top 10 most relevant items
            content = item.get("content", "")
            
            # Simple key information extraction (can be enhanced with NLP)
            sentences = content.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20 and len(sentence) < 200:  # Filter by length
                    # Check if sentence contains query-related words
                    query_words = query.lower().split()
                    if any(word in sentence.lower() for word in query_words):
                        key_info.append(sentence)
        
        return key_info[:20]  # Limit to top 20
    
    def _find_logical_connections(self, key_information: List[str]) -> List[Dict[str, Any]]:
        """Find logical connections between key information pieces"""
        connections = []
        
        # Simple connection detection based on common entities/concepts
        for i, info1 in enumerate(key_information):
            for j, info2 in enumerate(key_information[i+1:], i+1):
                
                # Find common words (excluding common stop words)
                stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
                words1 = set(info1.lower().split()) - stop_words
                words2 = set(info2.lower().split()) - stop_words
                
                common_words = words1.intersection(words2)
                
                if len(common_words) >= 2:  # At least 2 common significant words
                    connections.append({
                        "info1_index": i,
                        "info2_index": j,
                        "common_concepts": list(common_words),
                        "connection_strength": len(common_words) / max(len(words1), len(words2))
                    })
        
        # Sort by connection strength
        connections.sort(key=lambda x: x["connection_strength"], reverse=True)
        return connections[:10]  # Return top 10 connections
    
    def _calculate_reasoning_confidence(self, docs: List[Dict[str, Any]], web_results: List[Dict[str, Any]]) -> int:
        """Calculate confidence in the reasoning process"""
        
        # Base confidence
        confidence = 50
        
        # Boost confidence based on number of sources
        total_sources = len(docs) + len(web_results)
        confidence += min(total_sources * 5, 30)  # Max 30 points from source count
        
        # Boost confidence based on relevance scores
        if docs:
            avg_doc_score = sum(doc.get("score", 0) for doc in docs) / len(docs)
            confidence += int(avg_doc_score * 20)  # Max 20 points from document relevance
        
        return min(confidence, 100)
    
    def _run(self, information: Dict[str, Any], task: str) -> Dict[str, Any]:
        """Synchronous version (fallback)"""
        return asyncio.run(self._arun(information, task))


class ContradictionDetectorInput(BaseModel):
    """Input schema for contradiction detector tool"""
    documents: List[Dict[str, Any]] = Field(description="Documents to check for contradictions")
    web_results: List[Dict[str, Any]] = Field(default=[], description="Web results to include in check")

class ContradictionDetectorTool(BaseTool):
    """Tool for detecting contradictions in information"""
    
    name: str = "contradiction_detector"
    description: str = "Detects potential contradictions between different pieces of information"
    args_schema: Type[BaseModel] = ContradictionDetectorInput
    
    async def _arun(self, documents: List[Dict[str, Any]], web_results: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Detect contradictions in the provided information"""
        
        if web_results is None:
            web_results = []
        
        try:
            all_items = documents + web_results
            contradictions = []
            
            # Simple contradiction detection based on opposing statements
            contradiction_patterns = [
                (["yes", "true", "correct", "confirmed"], ["no", "false", "incorrect", "denied"]),
                (["increase", "rise", "growth", "higher"], ["decrease", "fall", "decline", "lower"]),
                (["before", "earlier", "prior"], ["after", "later", "subsequent"]),
                (["approved", "accepted", "allowed"], ["rejected", "denied", "forbidden"])
            ]
            
            for i, item1 in enumerate(all_items):
                for j, item2 in enumerate(all_items[i+1:], i+1):
                    
                    content1 = item1.get("content", "").lower()
                    content2 = item2.get("content", "").lower()
                    
                    # Check for contradiction patterns
                    for positive_words, negative_words in contradiction_patterns:
                        has_positive_1 = any(word in content1 for word in positive_words)
                        has_negative_1 = any(word in content1 for word in negative_words)
                        has_positive_2 = any(word in content2 for word in positive_words)
                        has_negative_2 = any(word in content2 for word in negative_words)
                        
                        # Check for direct contradiction
                        if (has_positive_1 and has_negative_2) or (has_negative_1 and has_positive_2):
                            
                            # Find common subject/topic
                            words1 = set(content1.split())
                            words2 = set(content2.split())
                            common_words = words1.intersection(words2)
                            
                            if len(common_words) >= 3:  # Must have common context
                                contradictions.append({
                                    "item1_index": i,
                                    "item2_index": j,
                                    "item1_source": item1.get("source", {}),
                                    "item2_source": item2.get("source", {}),
                                    "contradiction_type": "factual_opposition",
                                    "common_context": list(common_words)[:5],
                                    "confidence": min(len(common_words) * 10, 80),
                                    "description": f"Contradictory statements detected between sources"
                                })
            
            logger.info(f"Detected {len(contradictions)} potential contradictions")
            return contradictions
            
        except Exception as e:
            logger.error(f"Contradiction detection error: {e}")
            return []
    
    def _run(self, documents: List[Dict[str, Any]], web_results: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Synchronous version (fallback)"""
        return asyncio.run(self._arun(documents, web_results))


class ResponseSynthesizerInput(BaseModel):
    """Input schema for response synthesizer tool"""
    synthesis_data: Dict[str, Any] = Field(description="All data to synthesize into a response")

class ResponseSynthesizerTool(BaseTool):
    """Tool for synthesizing final comprehensive responses"""
    
    name: str = "response_synthesizer"
    description: str = "Synthesizes information from multiple sources into a comprehensive response"
    args_schema: Type[BaseModel] = ResponseSynthesizerInput
    
    async def _arun(self, synthesis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize comprehensive response from all gathered information"""
        
        try:
            query = synthesis_data.get("query", "")
            documents = synthesis_data.get("documents", [])
            web_results = synthesis_data.get("web_results", [])
            reasoning_steps = synthesis_data.get("reasoning_steps", [])
            
            # Collect all relevant content
            all_content = []
            sources = []
            
            # Process documents
            for doc in documents[:5]:  # Top 5 documents
                all_content.append(doc.get("content", ""))
                sources.append({
                    "type": "document",
                    "id": doc.get("chunk_id", ""),
                    "title": doc.get("source", {}).get("file_name", "Document"),
                    "relevance_score": doc.get("score", 0),
                    "content_preview": doc.get("content", "")[:200] + "..."
                })
            
            # Process web results
            for result in web_results[:3]:  # Top 3 web results
                all_content.append(result.get("content", ""))
                sources.append({
                    "type": "web",
                    "id": result.get("url", ""),
                    "title": result.get("title", "Web Result"),
                    "url": result.get("url", ""),
                    "relevance_score": result.get("score", 0),
                    "content_preview": result.get("content", "")[:200] + "..."
                })
            
            # Generate response based on available information
            response = await self._generate_response(query, all_content, reasoning_steps)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence(documents, web_results, reasoning_steps)
            
            return {
                "response": response,
                "sources": sources,
                "confidence_score": confidence_score,
                "synthesis_metadata": {
                    "total_sources": len(sources),
                    "document_sources": len(documents),
                    "web_sources": len(web_results),
                    "reasoning_steps": len(reasoning_steps)
                }
            }
            
        except Exception as e:
            logger.error(f"Response synthesis error: {e}")
            return {
                "response": "I apologize, but I encountered an error while synthesizing the response.",
                "sources": [],
                "confidence_score": 0
            }
    
    async def _generate_response(self, query: str, content_list: List[str], reasoning_steps: List[Dict[str, Any]]) -> str:
        """Generate comprehensive response from content"""
        
        # This is a simplified response generation
        # In a real implementation, this would use an LLM like GPT-4 or Gemini
        
        if not content_list:
            return "I couldn't find relevant information to answer your question. Please try rephrasing your query or adding more documents to your workspace."
        
        # Extract key points from content
        key_points = []
        for content in content_list:
            sentences = content.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 30 and len(sentence) < 300:
                    # Check relevance to query
                    query_words = set(query.lower().split())
                    sentence_words = set(sentence.lower().split())
                    if len(query_words.intersection(sentence_words)) >= 2:
                        key_points.append(sentence)
        
        # Build response
        response_parts = []
        
        # Introduction
        response_parts.append(f"Based on the available information, here's what I found regarding your question:")
        
        # Main content
        if key_points:
            response_parts.append("\n\n**Key Findings:**")
            for i, point in enumerate(key_points[:5], 1):
                response_parts.append(f"\n{i}. {point}")
        
        # Reasoning insights
        contradiction_steps = [step for step in reasoning_steps if step.get("step_type") == "contradiction_detection"]
        if contradiction_steps and contradiction_steps[0].get("contradictions_found", 0) > 0:
            response_parts.append(f"\n\n**Note:** I detected some potential contradictions in the source materials. Please review the sources carefully to understand different perspectives on this topic.")
        
        # Confidence indicator
        if len(content_list) >= 3:
            response_parts.append(f"\n\n*This response is based on {len(content_list)} sources from your workspace and web search.*")
        else:
            response_parts.append(f"\n\n*This response is based on limited sources ({len(content_list)}). Consider adding more relevant documents for a more comprehensive answer.*")
        
        return "".join(response_parts)
    
    def _calculate_confidence(self, documents: List[Dict[str, Any]], web_results: List[Dict[str, Any]], reasoning_steps: List[Dict[str, Any]]) -> int:
        """Calculate confidence score for the synthesized response"""
        
        confidence = 30  # Base confidence
        
        # Source quality and quantity
        doc_count = len(documents)
        web_count = len(web_results)
        
        confidence += min(doc_count * 10, 30)  # Max 30 points from documents
        confidence += min(web_count * 5, 15)   # Max 15 points from web results
        
        # Reasoning quality
        reasoning_count = len(reasoning_steps)
        confidence += min(reasoning_count * 5, 15)  # Max 15 points from reasoning
        
        # Check for contradictions (reduce confidence)
        contradiction_steps = [step for step in reasoning_steps if step.get("step_type") == "contradiction_detection"]
        if contradiction_steps and contradiction_steps[0].get("contradictions_found", 0) > 0:
            confidence -= 10
        
        return max(min(confidence, 95), 10)  # Keep between 10-95
    
    def _run(self, synthesis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous version (fallback)"""
        return asyncio.run(self._arun(synthesis_data))


class ToolRegistry:
    """Registry for managing available tools"""
    
    def __init__(self):
        self.tools = {
            "document_retriever": DocumentRetrieverTool(),
            "web_search": WebSearchTool(),
            "reasoning_engine": ReasoningEngineTool(),
            "contradiction_detector": ContradictionDetectorTool(),
            "response_synthesizer": ResponseSynthesizerTool()
        }
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Get a tool by name"""
        return self.tools.get(tool_name)
    
    def get_available_tools(self) -> List[BaseTool]:
        """Get list of all available tools"""
        return list(self.tools.values())
    
    def register_tool(self, tool_name: str, tool: BaseTool):
        """Register a new tool"""
        self.tools[tool_name] = tool
    
    def get_tool_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all tools"""
        return {name: tool.description for name, tool in self.tools.items()}
