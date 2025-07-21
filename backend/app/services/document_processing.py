import os
import asyncio
import aiofiles
from typing import List, Optional, Dict, Any, BinaryIO
from uuid import UUID
from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from PIL import Image
import pytesseract
import PyPDF2
from docx import Document as DocxDocument
import io
import tempfile
import hashlib
from datetime import datetime
import concurrent.futures
import torch
import time

from ..core.config import settings
from ..core.logging import logger, log_agent_trace
from ..core.database import get_vector_db
from ..models.database import Document, DocumentChunk, Workspace
from ..models.schemas import DocumentCreate, DocumentUpdate, DocumentChunkCreate
from .storage import StorageService
from .vector_store import VectorStoreService


class DocumentProcessingService:
    """Service for processing uploaded documents"""
    
    def __init__(self):
        self.storage_service = StorageService()
        self.vector_service = VectorStoreService()
        self.embedding_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self._load_embedding_model()
    
    def _load_embedding_model(self):
        """Load sentence transformer model for embeddings with optimizations"""
        try:
            # Use smaller, faster model for better performance
            model_name = settings.embedding_model
            
            self.embedding_model = SentenceTransformer(
                model_name,
                device=self.device
            )
            
            # Optimize for inference if using GPU
            if self.device == "cuda":
                try:
                    self.embedding_model.half()  # Use FP16 for speed
                    logger.info(f"Loaded embedding model with GPU optimization: {model_name}")
                except Exception as e:
                    logger.warning(f"GPU optimization failed, using CPU: {e}")
            else:
                logger.info(f"Loaded embedding model on CPU: {model_name}")
                
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.embedding_model = None
    
    async def process_document(
        self,
        file_content: bytes,
        document: Document,
        db: Session
    ) -> bool:
        """Process uploaded document through the optimized pipeline"""
        processing_start_time = time.time()
        
        try:
            log_agent_trace(
                "document_processor",
                "start_processing",
                {"document_id": str(document.document_id), "file_type": document.file_type}
            )
            
            logger.info(f"🚀 Starting optimized processing for document {document.document_id}")
            
            # Step 1: Parallel file storage and text extraction
            storage_task = self.storage_service.store_file(
                file_content,
                f"{document.workspace_id}/{document.document_id}_{document.file_name}"
            )
            text_task = self._extract_text_optimized(file_content, document.file_type)
            
            storage_path, text_content = await asyncio.gather(storage_task, text_task)
            
            # Update document with storage path
            document.storage_path = storage_path
            db.commit()
            
            if not text_content:
                raise ValueError("No text content extracted from document")
            
            logger.info(f"📖 Text extraction completed: {len(text_content)} characters")
            
            # Step 2: Apply OCR if needed (async)
            if self._needs_ocr(document.file_type, text_content):
                ocr_text = await self._apply_ocr(file_content, document.file_type)
                if ocr_text:
                    text_content = ocr_text
                    document.ocr_applied = True
                    logger.info("🔍 OCR processing completed")
            
            # Step 3: Optimized semantic chunking
            chunks = await self._create_chunks_optimized(text_content, document)
            logger.info(f"✂️ Created {len(chunks)} optimized chunks")
            
            # Step 4: BATCH embedding generation and storage (MAJOR OPTIMIZATION)
            await self._process_chunks_batch(chunks, document, db)
            
            # Step 5: Update document status
            document.status = "ready"
            document.total_chunks = len(chunks)
            document.embeddings_generated = True
            db.commit()
            
            processing_time = time.time() - processing_start_time
            
            log_agent_trace(
                "document_processor",
                "processing_complete",
                {
                    "document_id": str(document.document_id),
                    "chunks_created": len(chunks),
                    "ocr_applied": document.ocr_applied,
                    "processing_time": processing_time
                }
            )
            
            logger.info(f"✅ Successfully processed document {document.document_id} in {processing_time:.2f}s")
            return True
            
        except Exception as e:
            processing_time = time.time() - processing_start_time
            logger.error(f"❌ Error processing document {document.document_id} after {processing_time:.2f}s: {e}")
            document.status = "error"
            document.error_message = str(e)
            db.commit()
            return False
    
    async def _extract_text_optimized(self, file_content: bytes, file_type: str) -> str:
        """Optimized text extraction with parallel processing"""
        loop = asyncio.get_event_loop()
        
        def extract_text():
            if file_type == "application/pdf":
                return self._extract_pdf_optimized(file_content)
            elif file_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
                return self._extract_docx_optimized(file_content)
            elif file_type.startswith("text/"):
                return file_content.decode('utf-8', errors='ignore')
            elif file_type.startswith("image/"):
                return ""  # Will be handled by OCR
            else:
                logger.warning(f"Unsupported file type: {file_type}")
                return ""
        
        return await loop.run_in_executor(self.executor, extract_text)
    
    def _extract_pdf_optimized(self, file_content: bytes) -> str:
        """Optimized PDF text extraction with parallel page processing"""
        try:
            pdf_file = io.BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Parallel page processing for better performance
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                page_futures = [
                    executor.submit(self._extract_page_text, page, i)
                    for i, page in enumerate(pdf_reader.pages)
                ]
                
                pages_text = []
                for future in concurrent.futures.as_completed(page_futures):
                    try:
                        page_num, page_text = future.result()
                        if page_text.strip():
                            pages_text.append((page_num, page_text))
                    except Exception as e:
                        logger.warning(f"Error extracting page: {e}")
                        continue
                
                # Sort by page number and combine
                pages_text.sort(key=lambda x: x[0])
                return "\n\n".join([text for _, text in pages_text])
            
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            return ""
    
    def _extract_page_text(self, page, page_num: int) -> tuple:
        """Extract text from a single PDF page"""
        try:
            page_text = page.extract_text()
            if page_text.strip():
                return page_num, f"--- Page {page_num + 1} ---\n\n{page_text}"
            return page_num, ""
        except Exception as e:
            logger.warning(f"Error extracting page {page_num}: {e}")
            return page_num, ""
    
    def _extract_docx_optimized(self, file_content: bytes) -> str:
        """Optimized DOCX text extraction"""
        try:
            docx_file = io.BytesIO(file_content)
            doc = DocxDocument(docx_file)
            
            # Use list comprehension for better performance
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            
            # Extract table text efficiently
            table_texts = []
            for table in doc.tables:
                for row in table.rows:
                    row_text = "\t".join([cell.text for cell in row.cells])
                    if row_text.strip():
                        table_texts.append(row_text)
            
            # Combine all text
            all_text = "\n".join(paragraphs)
            if table_texts:
                all_text += "\n\n--- Tables ---\n\n" + "\n".join(table_texts)
            
            return all_text
            
        except Exception as e:
            logger.error(f"Error extracting DOCX text: {e}")
            return ""

    async def _extract_text(self, file_content: bytes, file_type: str) -> str:
        """Legacy text extraction method - kept for compatibility"""
        return await self._extract_text_optimized(file_content, file_type)
    
    async def _extract_pdf_text(self, file_content: bytes) -> str:
        """Extract text from PDF"""
        try:
            pdf_file = io.BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text.strip():
                    text += f"\n\n--- Page {page_num + 1} ---\n\n{page_text}"
            
            return text
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            return ""
    
    async def _extract_docx_text(self, file_content: bytes) -> str:
        """Extract text from DOCX"""
        try:
            docx_file = io.BytesIO(file_content)
            doc = DocxDocument(docx_file)
            text = ""
            
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            # Extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        text += cell.text + "\t"
                    text += "\n"
            
            return text
        except Exception as e:
            logger.error(f"Error extracting DOCX text: {e}")
            return ""
    
    def _needs_ocr(self, file_type: str, text_content: str) -> bool:
        """Determine if OCR is needed"""
        # Apply OCR if it's an image or if PDF has very little text
        if file_type.startswith("image/"):
            return True
        
        if file_type == "application/pdf" and len(text_content.strip()) < 100:
            return True
        
        return False
    
    async def _apply_ocr(self, file_content: bytes, file_type: str) -> Optional[str]:
        """Apply OCR to extract text from images or image-based PDFs"""
        try:
            if settings.google_vision_api_key:
                return await self._google_vision_ocr(file_content)
            else:
                return await self._tesseract_ocr(file_content, file_type)
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return None
    
    async def _google_vision_ocr(self, file_content: bytes) -> str:
        """Use Google Cloud Vision API for OCR"""
        try:
            from google.cloud import vision
            
            client = vision.ImageAnnotatorClient()
            image = vision.Image(content=file_content)
            response = client.text_detection(image=image)
            
            if response.error.message:
                raise Exception(f"Google Vision API error: {response.error.message}")
            
            texts = response.text_annotations
            if texts:
                return texts[0].description
            return ""
            
        except ImportError:
            logger.error("Google Cloud Vision API not available")
            return ""
        except Exception as e:
            logger.error(f"Google Vision OCR error: {e}")
            return ""
    
    async def _tesseract_ocr(self, file_content: bytes, file_type: str) -> str:
        """Use Tesseract for OCR"""
        try:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                if file_type.startswith("image/"):
                    temp_file.write(file_content)
                else:
                    # Convert PDF to image first (simplified - would need pdf2image)
                    logger.warning("PDF to image conversion not implemented for Tesseract")
                    return ""
                
                temp_file.flush()
                
                # Apply OCR
                if settings.tesseract_path:
                    pytesseract.pytesseract.tesseract_cmd = settings.tesseract_path
                
                image = Image.open(temp_file.name)
                text = pytesseract.image_to_string(image)
                
                # Clean up
                os.unlink(temp_file.name)
                
                return text
                
        except Exception as e:
            logger.error(f"Tesseract OCR error: {e}")
            return ""
    
    async def _create_chunks_optimized(self, text: str, document: Document) -> List[Dict[str, Any]]:
        """Create optimized semantic chunks from text"""
        chunks = []
        
        # Use paragraph-aware chunking for better semantic coherence
        paragraphs = text.split('\n\n')
        chunk_index = 0
        current_position = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If paragraph is within chunk size, use as-is
            if len(para) <= settings.chunk_size:
                chunks.append({
                    "chunk_index": chunk_index,
                    "content": para,
                    "start_position": current_position,
                    "end_position": current_position + len(para),
                    "metadata": {
                        "type": "paragraph",
                        "word_count": len(para.split()),
                        "char_count": len(para),
                        "sentence_count": para.count('.') + para.count('!') + para.count('?')
                    }
                })
                chunk_index += 1
                current_position += len(para) + 2  # +2 for \n\n
            
            else:
                # Split large paragraphs by sentences
                sentences = [s.strip() for s in para.replace('!', '.').replace('?', '.').split('.') if s.strip()]
                current_chunk = ""
                
                for sentence in sentences:
                    # Check if adding sentence exceeds chunk size
                    test_chunk = current_chunk + ". " + sentence if current_chunk else sentence
                    
                    if len(test_chunk) > settings.chunk_size:
                        if current_chunk:
                            # Save current chunk
                            chunks.append({
                                "chunk_index": chunk_index,
                                "content": current_chunk.strip(),
                                "start_position": current_position,
                                "end_position": current_position + len(current_chunk),
                                "metadata": {
                                    "type": "sentence_group",
                                    "word_count": len(current_chunk.split()),
                                    "char_count": len(current_chunk),
                                    "sentence_count": current_chunk.count('.') + current_chunk.count('!') + current_chunk.count('?')
                                }
                            })
                            chunk_index += 1
                            current_position += len(current_chunk)
                            current_chunk = sentence
                        else:
                            # Single sentence too long, truncate
                            current_chunk = sentence[:settings.chunk_size]
                    else:
                        current_chunk = test_chunk
                
                # Add remaining chunk
                if current_chunk:
                    chunks.append({
                        "chunk_index": chunk_index,
                        "content": current_chunk.strip(),
                        "start_position": current_position,
                        "end_position": current_position + len(current_chunk),
                        "metadata": {
                            "type": "sentence_group",
                            "word_count": len(current_chunk.split()),
                            "char_count": len(current_chunk),
                            "sentence_count": current_chunk.count('.') + current_chunk.count('!') + current_chunk.count('?')
                        }
                    })
                    chunk_index += 1
                    current_position += len(current_chunk)
        
        logger.info(f"📄 Created {len(chunks)} optimized chunks (avg: {sum(len(c['content']) for c in chunks) // len(chunks) if chunks else 0} chars/chunk)")
        return chunks

    async def _create_chunks(self, text: str, document: Document) -> List[Dict[str, Any]]:
        """Legacy chunking method - kept for compatibility"""
        return await self._create_chunks_optimized(text, document)
    
    async def _process_chunks_batch(
        self,
        chunks: List[Dict[str, Any]],
        document: Document,
        db: Session
    ):
        """OPTIMIZED: Process chunks with batch embedding generation"""
        if not self.embedding_model:
            raise Exception("Embedding model not available")
        
        batch_start_time = time.time()
        logger.info(f"🔗 Starting batch embedding generation for {len(chunks)} chunks")
        
        # Extract all chunk texts for batch processing
        chunk_texts = [chunk["content"] for chunk in chunks]
        
        # BATCH EMBEDDING GENERATION (MAJOR OPTIMIZATION)
        try:
            embedding_start_time = time.time()
            
            # Process embeddings in batches to manage memory
            batch_size = 32  # Optimal batch size for most GPUs
            all_embeddings = []
            
            for i in range(0, len(chunk_texts), batch_size):
                batch_texts = chunk_texts[i:i + batch_size]
                
                # Generate embeddings for entire batch at once
                batch_embeddings = self.embedding_model.encode(
                    batch_texts,
                    batch_size=len(batch_texts),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True  # Normalize for better similarity search
                )
                
                all_embeddings.extend(batch_embeddings)
                
                batch_num = (i // batch_size) + 1
                total_batches = (len(chunk_texts) + batch_size - 1) // batch_size
                logger.info(f"   ✅ Processed embedding batch {batch_num}/{total_batches}")
            
            embedding_time = time.time() - embedding_start_time
            logger.info(f"🔗 Batch embedding generation completed in {embedding_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Batch embedding generation failed: {e}")
            # Fallback to sequential processing
            return await self._process_chunks_sequential(chunks, document, db)
        
        # BATCH VECTOR STORAGE AND DATABASE OPERATIONS
        try:
            storage_start_time = time.time()
            
            # Prepare batch operations
            vector_operations = []
            chunk_objects = []
            
            for chunk_data, embedding in zip(chunks, all_embeddings):
                # Prepare vector storage operation
                vector_operations.append({
                    "text": chunk_data["content"],
                    "embedding": embedding,
                    "metadata": {
                        "document_id": str(document.document_id),
                        "workspace_id": str(document.workspace_id),
                        "chunk_index": chunk_data["chunk_index"],
                        "file_name": document.file_name,
                        **chunk_data["metadata"]
                    }
                })
            
            # Batch store in vector database
            logger.info("💾 Batch storing embeddings in vector database...")
            vector_ids = await self._batch_store_vectors(vector_operations)
            
            # Prepare database chunk objects
            for chunk_data, vector_id in zip(chunks, vector_ids):
                chunk = DocumentChunk(
                    document_id=document.document_id,
                    chunk_index=chunk_data["chunk_index"],
                    content=chunk_data["content"],
                    vector_id=vector_id,
                    start_position=chunk_data["start_position"],
                    end_position=chunk_data["end_position"],
                    metadata=chunk_data["metadata"]
                )
                chunk_objects.append(chunk)
            
            # Batch insert into database
            logger.info("🗄️ Batch inserting chunks into database...")
            db.add_all(chunk_objects)
            db.commit()
            
            storage_time = time.time() - storage_start_time
            total_time = time.time() - batch_start_time
            
            logger.info(f"💾 Batch storage completed in {storage_time:.2f}s")
            logger.info(f"✅ Total batch processing completed in {total_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Batch storage failed: {e}")
            # Rollback on failure
            db.rollback()
            raise e
    
    async def _batch_store_vectors(self, vector_operations: List[Dict]) -> List[str]:
        """Batch store vectors in vector database"""
        try:
            vector_ids = []
            
            # Process in batches to avoid memory issues
            batch_size = 50
            
            for i in range(0, len(vector_operations), batch_size):
                batch_ops = vector_operations[i:i + batch_size]
                
                # Use vector service batch operations if available
                batch_ids = []
                for op in batch_ops:
                    vector_id = await self.vector_service.store_embedding(
                        text=op["text"],
                        embedding=op["embedding"],
                        metadata=op["metadata"]
                    )
                    batch_ids.append(vector_id)
                
                vector_ids.extend(batch_ids)
                
                batch_num = (i // batch_size) + 1
                total_batches = (len(vector_operations) + batch_size - 1) // batch_size
                logger.info(f"   🔍 Stored vector batch {batch_num}/{total_batches}")
            
            return vector_ids
            
        except Exception as e:
            logger.error(f"❌ Batch vector storage failed: {e}")
            raise e
    
    async def _process_chunks_sequential(
        self,
        chunks: List[Dict[str, Any]],
        document: Document,
        db: Session
    ):
        """Fallback: Sequential chunk processing (original method)"""
        logger.warning("⚠️ Using fallback sequential processing")
        
        for chunk_data in chunks:
            try:
                # Generate embedding
                embedding = self.embedding_model.encode(chunk_data["content"])
                
                # Store in vector database
                vector_id = await self.vector_service.store_embedding(
                    text=chunk_data["content"],
                    embedding=embedding,
                    metadata={
                        "document_id": str(document.document_id),
                        "workspace_id": str(document.workspace_id),
                        "chunk_index": chunk_data["chunk_index"],
                        "file_name": document.file_name,
                        **chunk_data["metadata"]
                    }
                )
                
                # Store chunk in database
                chunk = DocumentChunk(
                    document_id=document.document_id,
                    chunk_index=chunk_data["chunk_index"],
                    content=chunk_data["content"],
                    vector_id=vector_id,
                    start_position=chunk_data["start_position"],
                    end_position=chunk_data["end_position"],
                    metadata=chunk_data["metadata"]
                )
                
                db.add(chunk)
                
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_data['chunk_index']}: {e}")
                continue
        
        db.commit()

    async def _process_chunks(
        self,
        chunks: List[Dict[str, Any]],
        document: Document,
        db: Session
    ):
        """Legacy method - redirects to optimized batch processing"""
        return await self._process_chunks_batch(chunks, document, db)
    
    async def reprocess_document(self, document_id: UUID, db: Session) -> bool:
        """Reprocess an existing document"""
        document = db.query(Document).filter(Document.document_id == document_id).first()
        if not document:
            return False
        
        try:
            # Delete existing chunks
            db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).delete()
            
            # Delete from vector store
            await self.vector_service.delete_document_embeddings(str(document_id))
            
            # Reset document status
            document.status = "processing"
            document.total_chunks = 0
            document.embeddings_generated = False
            document.error_message = None
            db.commit()
            
            # Retrieve file content
            file_content = await self.storage_service.get_file(document.storage_path)
            
            # Reprocess
            return await self.process_document(file_content, document, db)
            
        except Exception as e:
            logger.error(f"Error reprocessing document {document_id}: {e}")
            document.status = "error"
            document.error_message = str(e)
            db.commit()
            return False
    
    async def delete_document(self, document_id: UUID, db: Session) -> bool:
        """Delete document and all associated data"""
        try:
            document = db.query(Document).filter(Document.document_id == document_id).first()
            if not document:
                return False
            
            # Delete from vector store
            await self.vector_service.delete_document_embeddings(str(document_id))
            
            # Delete from storage
            await self.storage_service.delete_file(document.storage_path)
            
            # Delete chunks from database
            db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).delete()
            
            # Delete document
            db.delete(document)
            db.commit()
            
            logger.info(f"Successfully deleted document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return False
    
    def __del__(self):
        """Cleanup thread pool on destruction"""
        try:
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=False)
        except Exception:
            pass  # Ignore cleanup errors
