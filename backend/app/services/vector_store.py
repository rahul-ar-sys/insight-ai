import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import uuid
import asyncio
import json

from ..core.config import settings
from ..core.logging import logger

_vector_store_instance = None


class VectorStoreInterface(ABC):
    """Abstract interface for vector database operations"""
    
    @abstractmethod
    async def connect(self):
        """Connect to the vector store if needed."""
        pass

    @abstractmethod
    async def store_embedding(
        self,
        text: str,
        embedding: np.ndarray,
        metadata: Dict[str, Any]
    ) -> str:
        """Store text embedding and return vector ID"""
        pass
    
    @abstractmethod
    async def search_similar(
        self,
        query_embedding: np.ndarray,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar embeddings"""
        pass
    
    @abstractmethod
    async def delete_embeddings(self, vector_ids: List[str], **kwargs) -> bool:
        """Delete embeddings by IDs"""
        pass
    
    @abstractmethod
    async def delete_document_embeddings(self, document_id: str, **kwargs) -> bool:
        """Delete all embeddings for a document"""
        pass
class ChromaVectorStore(VectorStoreInterface):
    """ChromaDB vector database implementation"""

    def __init__(self):
        self.client = None

    async def connect(self):
        """Initializes the ChromaDB client. To be called at application startup."""
        if self.client:
            logger.info("ChromaDB client already initialized.")
            return
        try:
            import chromadb
            self.client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
            await asyncio.to_thread(self.client.heartbeat)
            logger.info(f"ChromaDB client initialized successfully. Host: {settings.chroma_host}, Port: {settings.chroma_port}")
        except ImportError:
            logger.error("ChromaDB library (chromadb-client) not available. Please install it.")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            self.client = None
            raise

    def _get_or_create_collection(self, workspace_id: str):
        if not self.client:
            raise Exception("ChromaDB client not initialized. The service may have failed to connect at startup.")
        collection_name = f"workspace-{str(workspace_id).replace('-', '')}"
        return self.client.get_or_create_collection(name=collection_name)

    async def store_embedding(self, text: str, embedding: np.ndarray, metadata: Dict[str, Any]) -> str:
        if not self.client: raise Exception("ChromaDB not properly configured")
        workspace_id = metadata.get("workspace_id")
        if not workspace_id: raise ValueError("workspace_id is required in metadata for ChromaDB")
        
        try:
            collection = self._get_or_create_collection(workspace_id)
            vector_id = str(uuid.uuid4())
            clean_metadata = {k: v for k, v in metadata.items() if isinstance(v, (str, int, float, bool))}
            embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
            
            collection.add(ids=[vector_id], embeddings=[embedding_list], metadatas=[clean_metadata], documents=[text])
            
            logger.debug(f"Stored embedding in ChromaDB collection '{collection.name}': {vector_id}")
            return vector_id
        except Exception as e:
            logger.error(f"Error storing embedding in ChromaDB: {e}")
            raise

    async def search_similar(self, query_embedding: np.ndarray, limit: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not self.client or not filters or "workspace_id" not in filters:
            logger.warning("ChromaDB search requires a workspace_id filter.")
            return []
        
        try:
            workspace_id = filters["workspace_id"]
            collection = self._get_or_create_collection(workspace_id)
            where_filter = {"document_id": str(filters["document_id"])} if "document_id" in filters else None
            embedding_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
            
            results = collection.query(query_embeddings=[embedding_list], n_results=limit, where=where_filter)
            
            formatted_results = []
            if results and results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    formatted_results.append({
                        "id": results['ids'][0][i],
                        "score": 1 - results['distances'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "text": results['documents'][0][i]
                    })
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching in ChromaDB: {e}")
            return []

    async def delete_embeddings(self, vector_ids: List[str], **kwargs) -> bool:
        workspace_id = kwargs.get("workspace_id")
        if not self.client or not workspace_id: return False
        try:
            collection = self._get_or_create_collection(workspace_id)
            collection.delete(ids=vector_ids)
            logger.info(f"Deleted {len(vector_ids)} embeddings from ChromaDB for workspace {workspace_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting embeddings from ChromaDB: {e}")
            return False

    async def delete_document_embeddings(self, document_id: str, **kwargs) -> bool:
        workspace_id = kwargs.get("workspace_id")
        if not self.client or not workspace_id: return False
        try:
            collection = self._get_or_create_collection(workspace_id)
            collection.delete(where={"document_id": str(document_id)})
            logger.info(f"Deleted embeddings for document {document_id} from ChromaDB")
            return True
        except Exception as e:
            logger.error(f"Error deleting document embeddings from ChromaDB: {e}")
            return False


class InMemoryVectorStore(VectorStoreInterface):
    """In-memory vector store for development/testing"""
    
    def __init__(self):
        self.vectors = {}
        self.document_vectors = {}

    async def connect(self):
        """In-memory store does not need to connect."""
        logger.debug("In-memory vector store connect() called. No action needed.")
        pass
    
    async def store_embedding(self, text: str, embedding: np.ndarray, metadata: Dict[str, Any]) -> str:
        vector_id = str(uuid.uuid4())
        self.vectors[vector_id] = (embedding, metadata, text)
        document_id = metadata.get("document_id")
        if document_id:
            self.document_vectors.setdefault(document_id, []).append(vector_id)
        logger.debug(f"Stored embedding in memory: {vector_id}")
        return vector_id
    
    async def search_similar(self, query_embedding: np.ndarray, limit: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        similarities = []
        for vector_id, (embedding, metadata, text) in self.vectors.items():
            if filters:
                if "workspace_id" in filters and metadata.get("workspace_id") != filters["workspace_id"]: continue
                if "document_id" in filters and metadata.get("document_id") != filters["document_id"]: continue
            
            similarity = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
            similarities.append({"id": vector_id, "score": float(similarity), "metadata": metadata, "text": text})
        
        similarities.sort(key=lambda x: x["score"], reverse=True)
        return similarities[:limit]
    
    async def delete_embeddings(self, vector_ids: List[str], **kwargs) -> bool:
        try:
            for vector_id in vector_ids:
                if vector_id in self.vectors: del self.vectors[vector_id]
            for doc_id in self.document_vectors:
                self.document_vectors[doc_id] = [vid for vid in self.document_vectors[doc_id] if vid not in vector_ids]
            logger.info(f"Deleted {len(vector_ids)} embeddings from memory")
            return True
        except Exception as e:
            logger.error(f"Error deleting embeddings from memory: {e}")
            return False
    
    async def delete_document_embeddings(self, document_id: str, **kwargs) -> bool:
        try:
            if document_id in self.document_vectors:
                await self.delete_embeddings(self.document_vectors[document_id])
                del self.document_vectors[document_id]
            logger.info(f"Deleted embeddings for document {document_id} from memory")
            return True
        except Exception as e:
            logger.error(f"Error deleting document embeddings from memory: {e}")
            return False


class VectorStoreService:
    """Main vector store service that delegates to the appropriate implementation"""
    
    def __init__(self):
        self.store = self._get_vector_store_implementation()
    
    def _get_vector_store_implementation(self) -> VectorStoreInterface:
        """Get the appropriate vector store implementation based on configuration"""
        db_type = settings.vector_db_type.lower()
        
        if db_type == "chroma":
            logger.info("Using ChromaDB vector store")
            return ChromaVectorStore()
        else:
            logger.warning(f"Unknown or unconfigured vector_db_type '{settings.vector_db_type}'. Using in-memory vector store as fallback.")
            return InMemoryVectorStore()

    async def connect(self):
        """Connects the underlying vector store implementation."""
        await self.store.connect()
    
    async def store_embedding(self, text: str, embedding: np.ndarray, metadata: Dict[str, Any]) -> str:
        return await self.store.store_embedding(text, embedding, metadata)
    
    async def search_similar(self, query_embedding: np.ndarray, limit: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        return await self.store.search_similar(query_embedding, limit, filters)
    
    async def delete_embeddings(self, vector_ids: List[str], **kwargs) -> bool:
        return await self.store.delete_embeddings(vector_ids, **kwargs)
    
    async def delete_document_embeddings(self, document_id: str, **kwargs) -> bool:
        return await self.store.delete_document_embeddings(document_id, **kwargs)


