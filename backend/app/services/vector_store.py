import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import uuid
import json

from ..core.config import settings
from ..core.logging import logger


class VectorStoreInterface(ABC):
    """Abstract interface for vector database operations"""
    
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
    async def delete_embeddings(self, vector_ids: List[str]) -> bool:
        """Delete embeddings by IDs"""
        pass
    
    @abstractmethod
    async def delete_document_embeddings(self, document_id: str) -> bool:
        """Delete all embeddings for a document"""
        pass


class PineconeVectorStore(VectorStoreInterface):
    """Pinecone vector database implementation"""
    
    def __init__(self):
        self.client = None
        self.index = None
        self.index_name = "insight-ai-docs"
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Pinecone client"""
        try:
            import pinecone
            
            pinecone.init(
                api_key=settings.pinecone_api_key,
                environment=settings.pinecone_environment
            )
            
            # Create index if it doesn't exist
            if self.index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=self.index_name,
                    dimension=384,  # Dimension for all-MiniLM-L6-v2
                    metric="cosine"
                )
                logger.info(f"Created Pinecone index: {self.index_name}")
            
            self.index = pinecone.Index(self.index_name)
            logger.info("Pinecone client initialized successfully")
            
        except ImportError:
            logger.error("Pinecone library not available")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
    
    async def store_embedding(
        self,
        text: str,
        embedding: np.ndarray,
        metadata: Dict[str, Any]
    ) -> str:
        """Store embedding in Pinecone"""
        if not self.index:
            raise Exception("Pinecone not properly configured")
        
        try:
            vector_id = str(uuid.uuid4())
            
            # Prepare metadata (Pinecone has limitations on metadata)
            clean_metadata = {
                "document_id": metadata.get("document_id"),
                "workspace_id": metadata.get("workspace_id"),
                "chunk_index": metadata.get("chunk_index"),
                "file_name": metadata.get("file_name", "")[:100],  # Truncate long filenames
                "text": text[:1000]  # Store truncated text for reference
            }
            
            # Convert numpy array to list
            embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
            
            # Upsert to Pinecone
            self.index.upsert([(vector_id, embedding_list, clean_metadata)])
            
            logger.debug(f"Stored embedding in Pinecone: {vector_id}")
            return vector_id
            
        except Exception as e:
            logger.error(f"Error storing embedding in Pinecone: {e}")
            raise
    
    async def search_similar(
        self,
        query_embedding: np.ndarray,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar embeddings in Pinecone"""
        if not self.index:
            raise Exception("Pinecone not properly configured")
        
        try:
            # Convert numpy array to list
            embedding_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
            
            # Prepare filter for Pinecone
            pinecone_filter = {}
            if filters:
                if "workspace_id" in filters:
                    pinecone_filter["workspace_id"] = {"$eq": filters["workspace_id"]}
                if "document_id" in filters:
                    pinecone_filter["document_id"] = {"$eq": filters["document_id"]}
            
            # Query Pinecone
            results = self.index.query(
                vector=embedding_list,
                top_k=limit,
                include_metadata=True,
                filter=pinecone_filter if pinecone_filter else None
            )
            
            # Format results
            formatted_results = []
            for match in results.matches:
                formatted_results.append({
                    "id": match.id,
                    "score": float(match.score),
                    "metadata": dict(match.metadata),
                    "text": match.metadata.get("text", "")
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching in Pinecone: {e}")
            return []
    
    async def delete_embeddings(self, vector_ids: List[str]) -> bool:
        """Delete embeddings by IDs"""
        if not self.index:
            return False
        
        try:
            self.index.delete(ids=vector_ids)
            logger.info(f"Deleted {len(vector_ids)} embeddings from Pinecone")
            return True
        except Exception as e:
            logger.error(f"Error deleting embeddings from Pinecone: {e}")
            return False
    
    async def delete_document_embeddings(self, document_id: str) -> bool:
        """Delete all embeddings for a document"""
        if not self.index:
            return False
        
        try:
            # Delete by filter
            self.index.delete(filter={"document_id": {"$eq": document_id}})
            logger.info(f"Deleted embeddings for document {document_id} from Pinecone")
            return True
        except Exception as e:
            logger.error(f"Error deleting document embeddings from Pinecone: {e}")
            return False


class WeaviateVectorStore(VectorStoreInterface):
    """Weaviate vector database implementation"""
    
    def __init__(self):
        self.client = None
        self.class_name = "DocumentChunk"
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Weaviate client"""
        try:
            import weaviate
            
            self.client = weaviate.Client(
                url=settings.weaviate_url,
                auth_client_secret=weaviate.AuthApiKey(api_key=settings.weaviate_api_key)
            )
            
            # Create schema if it doesn't exist
            self._create_schema()
            logger.info("Weaviate client initialized successfully")
            
        except ImportError:
            logger.error("Weaviate library not available")
        except Exception as e:
            logger.error(f"Failed to initialize Weaviate: {e}")
    
    def _create_schema(self):
        """Create Weaviate schema"""
        try:
            # Check if class exists
            if self.client.schema.exists(self.class_name):
                return
            
            # Define schema
            schema = {
                "class": self.class_name,
                "description": "Document chunks with embeddings",
                "vectorizer": "none",  # We provide our own vectors
                "properties": [
                    {
                        "name": "text",
                        "dataType": ["text"],
                        "description": "The text content of the chunk"
                    },
                    {
                        "name": "documentId",
                        "dataType": ["string"],
                        "description": "ID of the source document"
                    },
                    {
                        "name": "workspaceId",
                        "dataType": ["string"],
                        "description": "ID of the workspace"
                    },
                    {
                        "name": "chunkIndex",
                        "dataType": ["int"],
                        "description": "Index of the chunk within the document"
                    },
                    {
                        "name": "fileName",
                        "dataType": ["string"],
                        "description": "Name of the source file"
                    },
                    {
                        "name": "metadata",
                        "dataType": ["text"],
                        "description": "Additional metadata as JSON string"
                    }
                ]
            }
            
            self.client.schema.create_class(schema)
            logger.info(f"Created Weaviate schema: {self.class_name}")
            
        except Exception as e:
            logger.error(f"Error creating Weaviate schema: {e}")
    
    async def store_embedding(
        self,
        text: str,
        embedding: np.ndarray,
        metadata: Dict[str, Any]
    ) -> str:
        """Store embedding in Weaviate"""
        if not self.client:
            raise Exception("Weaviate not properly configured")
        
        try:
            vector_id = str(uuid.uuid4())
            
            # Prepare properties
            properties = {
                "text": text,
                "documentId": metadata.get("document_id"),
                "workspaceId": metadata.get("workspace_id"),
                "chunkIndex": metadata.get("chunk_index"),
                "fileName": metadata.get("file_name", ""),
                "metadata": json.dumps(metadata)
            }
            
            # Convert numpy array to list
            embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
            
            # Add object to Weaviate
            self.client.data_object.create(
                data_object=properties,
                class_name=self.class_name,
                uuid=vector_id,
                vector=embedding_list
            )
            
            logger.debug(f"Stored embedding in Weaviate: {vector_id}")
            return vector_id
            
        except Exception as e:
            logger.error(f"Error storing embedding in Weaviate: {e}")
            raise
    
    async def search_similar(
        self,
        query_embedding: np.ndarray,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar embeddings in Weaviate"""
        if not self.client:
            raise Exception("Weaviate not properly configured")
        
        try:
            # Convert numpy array to list
            embedding_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
            
            # Build query
            query = self.client.query.get(self.class_name, [
                "text", "documentId", "workspaceId", "chunkIndex", "fileName", "metadata"
            ]).with_near_vector({
                "vector": embedding_list
            }).with_limit(limit).with_additional(["certainty"])
            
            # Add filters
            if filters:
                where_conditions = []
                if "workspace_id" in filters:
                    where_conditions.append({
                        "path": ["workspaceId"],
                        "operator": "Equal",
                        "valueString": filters["workspace_id"]
                    })
                if "document_id" in filters:
                    where_conditions.append({
                        "path": ["documentId"],
                        "operator": "Equal",
                        "valueString": filters["document_id"]
                    })
                
                if where_conditions:
                    if len(where_conditions) == 1:
                        query = query.with_where(where_conditions[0])
                    else:
                        query = query.with_where({
                            "operator": "And",
                            "operands": where_conditions
                        })
            
            # Execute query
            results = query.do()
            
            # Format results
            formatted_results = []
            if "data" in results and "Get" in results["data"] and self.class_name in results["data"]["Get"]:
                for item in results["data"]["Get"][self.class_name]:
                    # Parse metadata
                    metadata = {}
                    if item.get("metadata"):
                        try:
                            metadata = json.loads(item["metadata"])
                        except:
                            pass
                    
                    formatted_results.append({
                        "id": item.get("_additional", {}).get("id", ""),
                        "score": item.get("_additional", {}).get("certainty", 0.0),
                        "metadata": {
                            "document_id": item.get("documentId"),
                            "workspace_id": item.get("workspaceId"),
                            "chunk_index": item.get("chunkIndex"),
                            "file_name": item.get("fileName"),
                            **metadata
                        },
                        "text": item.get("text", "")
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching in Weaviate: {e}")
            return []
    
    async def delete_embeddings(self, vector_ids: List[str]) -> bool:
        """Delete embeddings by IDs"""
        if not self.client:
            return False
        
        try:
            for vector_id in vector_ids:
                self.client.data_object.delete(uuid=vector_id, class_name=self.class_name)
            
            logger.info(f"Deleted {len(vector_ids)} embeddings from Weaviate")
            return True
        except Exception as e:
            logger.error(f"Error deleting embeddings from Weaviate: {e}")
            return False
    
    async def delete_document_embeddings(self, document_id: str) -> bool:
        """Delete all embeddings for a document"""
        if not self.client:
            return False
        
        try:
            # Query for all objects with the document ID
            query = self.client.query.get(self.class_name, ["documentId"]).with_where({
                "path": ["documentId"],
                "operator": "Equal",
                "valueString": document_id
            }).with_additional(["id"])
            
            results = query.do()
            
            # Delete each object
            if "data" in results and "Get" in results["data"] and self.class_name in results["data"]["Get"]:
                for item in results["data"]["Get"][self.class_name]:
                    object_id = item.get("_additional", {}).get("id")
                    if object_id:
                        self.client.data_object.delete(uuid=object_id, class_name=self.class_name)
            
            logger.info(f"Deleted embeddings for document {document_id} from Weaviate")
            return True
        except Exception as e:
            logger.error(f"Error deleting document embeddings from Weaviate: {e}")
            return False


class InMemoryVectorStore(VectorStoreInterface):
    """In-memory vector store for development/testing"""
    
    def __init__(self):
        self.vectors = {}  # vector_id -> (embedding, metadata, text)
        self.document_vectors = {}  # document_id -> [vector_ids]
    
    async def store_embedding(
        self,
        text: str,
        embedding: np.ndarray,
        metadata: Dict[str, Any]
    ) -> str:
        """Store embedding in memory"""
        vector_id = str(uuid.uuid4())
        
        self.vectors[vector_id] = (embedding, metadata, text)
        
        # Track by document
        document_id = metadata.get("document_id")
        if document_id:
            if document_id not in self.document_vectors:
                self.document_vectors[document_id] = []
            self.document_vectors[document_id].append(vector_id)
        
        logger.debug(f"Stored embedding in memory: {vector_id}")
        return vector_id
    
    async def search_similar(
        self,
        query_embedding: np.ndarray,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar embeddings in memory"""
        similarities = []
        
        for vector_id, (embedding, metadata, text) in self.vectors.items():
            # Apply filters
            if filters:
                if "workspace_id" in filters and metadata.get("workspace_id") != filters["workspace_id"]:
                    continue
                if "document_id" in filters and metadata.get("document_id") != filters["document_id"]:
                    continue
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            
            similarities.append({
                "id": vector_id,
                "score": float(similarity),
                "metadata": metadata,
                "text": text
            })
        
        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x["score"], reverse=True)
        return similarities[:limit]
    
    async def delete_embeddings(self, vector_ids: List[str]) -> bool:
        """Delete embeddings by IDs"""
        try:
            for vector_id in vector_ids:
                if vector_id in self.vectors:
                    del self.vectors[vector_id]
            
            # Clean up document tracking
            for doc_id, ids in self.document_vectors.items():
                self.document_vectors[doc_id] = [id for id in ids if id not in vector_ids]
            
            logger.info(f"Deleted {len(vector_ids)} embeddings from memory")
            return True
        except Exception as e:
            logger.error(f"Error deleting embeddings from memory: {e}")
            return False
    
    async def delete_document_embeddings(self, document_id: str) -> bool:
        """Delete all embeddings for a document"""
        try:
            if document_id in self.document_vectors:
                vector_ids = self.document_vectors[document_id]
                await self.delete_embeddings(vector_ids)
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
        # Use in-memory store for development since we don't have Pinecone configured
        # and ChromaDB has issues with the HTTP client
        logger.warning("Using in-memory vector store for development")
        return InMemoryVectorStore()
    
    async def store_embedding(
        self,
        text: str,
        embedding: np.ndarray,
        metadata: Dict[str, Any]
    ) -> str:
        """Store text embedding"""
        return await self.store.store_embedding(text, embedding, metadata)
    
    async def search_similar(
        self,
        query_embedding: np.ndarray,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar embeddings"""
        return await self.store.search_similar(query_embedding, limit, filters)
    
    async def delete_embeddings(self, vector_ids: List[str]) -> bool:
        """Delete embeddings by IDs"""
        return await self.store.delete_embeddings(vector_ids)
    
    async def delete_document_embeddings(self, document_id: str) -> bool:
        """Delete all embeddings for a document"""
        return await self.store.delete_document_embeddings(document_id)
