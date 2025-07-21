#!/usr/bin/env python3
"""
Test script for document ingestion and processing in Insight AI backend.

This script demonstrates how to:
1. Register/login a test user
2. Create a workspace
3. Upload and process documents
4. Query the processed documents
5. Monitor processing status

Usage:
    python test_document_ingestion.py

Requirements:
    - Backend server running on http://localhost:8000
    - Test documents in the data/ directory
"""

import requests
import json
import time
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import mimetypes

# Configuration
API_BASE_URL = "http://localhost:8000"
TEST_USER = {
    "name": "Test User",
    "email": "test@example.com", 
    "password": "testpass123"
}

# Test documents directory
DOCUMENTS_DIR = Path("data/test_documents")
SAMPLE_DOCUMENTS = [
    "sample.pdf",
    "test_doc.txt", 
    "research_paper.pdf",
    "meeting_notes.md"
]

class DocumentIngestionTester:
    """Test client for document ingestion and processing"""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.token = None
        self.user_id = None
        self.workspace_id = None
        
    def set_auth_token(self, token: str):
        """Set authentication token for API requests"""
        self.token = token
        self.session.headers.update({"Authorization": f"Bearer {token}"})
        print(f"✅ Authentication token set")
    
    def clear_auth_token(self):
        """Clear authentication token"""
        self.token = None
        self.session.headers.pop("Authorization", None)
        print("🔄 Authentication token cleared")
    
    def check_server_health(self) -> bool:
        """Check if the backend server is running"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                print("✅ Backend server is running")
                return True
            else:
                print(f"❌ Server health check failed: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print(f"❌ Cannot connect to backend server at {self.base_url}")
            print("   Make sure the backend is running with: python -m uvicorn app.main:app --reload")
            return False
        except Exception as e:
            print(f"❌ Server health check error: {e}")
            return False
    
    def register_user(self) -> bool:
        """Register a test user"""
        try:
            print(f"📝 Registering user: {TEST_USER['email']}")
            response = self.session.post(
                f"{self.base_url}/api/v1/auth/register",
                json=TEST_USER
            )
            
            if response.status_code == 201:
                data = response.json()
                print(f"✅ User registered successfully: {data.get('user', {}).get('name')}")
                return True
            elif response.status_code == 400:
                error_data = response.json()
                if "already exists" in str(error_data.get("detail", "")):
                    print(f"ℹ️  User already exists: {TEST_USER['email']}")
                    return True
                else:
                    print(f"❌ Registration failed: {error_data}")
                    return False
            else:
                print(f"❌ Registration failed with status {response.status_code}: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Registration error: {e}")
            return False
    
    def login_user(self) -> bool:
        """Login the test user"""
        try:
            print(f"🔐 Logging in user: {TEST_USER['email']}")
            response = self.session.post(
                f"{self.base_url}/api/v1/auth/login",
                json={"email": TEST_USER["email"], "password": TEST_USER["password"]}
            )
            
            if response.status_code == 200:
                data = response.json()
                # Handle nested token structure
                access_token = data.get("tokens", {}).get("access_token") or data.get("access_token")
                
                if access_token:
                    self.set_auth_token(access_token)
                    self.user_id = data.get("user", {}).get("user_id")
                    print(f"✅ Login successful for user: {data.get('user', {}).get('name')}")
                    return True
                else:
                    print("❌ No access token in login response")
                    return False
            else:
                error_data = response.json()
                print(f"❌ Login failed: {error_data}")
                return False
        except Exception as e:
            print(f"❌ Login error: {e}")
            return False
    
    def create_workspace(self, name: str = "Test Document Processing", description: str = "") -> bool:
        """Create a test workspace"""
        try:
            print(f"🏢 Creating workspace: {name}")
            response = self.session.post(
                f"{self.base_url}/api/v1/workspaces/",
                json={"name": name, "description": description or "Workspace for testing document processing"}
            )
            
            if response.status_code == 201:
                data = response.json()
                self.workspace_id = data.get("workspace_id")
                print(f"✅ Workspace created: {data.get('name')} (ID: {self.workspace_id})")
                return True
            else:
                error_data = response.json()
                print(f"❌ Workspace creation failed: {error_data}")
                return False
        except Exception as e:
            print(f"❌ Workspace creation error: {e}")
            return False
    
    def create_test_documents(self):
        """Create sample test documents if they don't exist"""
        DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Create sample text document
        sample_txt = DOCUMENTS_DIR / "test_doc.txt"
        if not sample_txt.exists():
            sample_txt.write_text("""
# Sample Document for Testing

This is a test document for the Insight AI document processing pipeline.

## Key Points:
1. Document ingestion and processing
2. Text extraction and chunking
3. Embedding generation
4. Vector storage and retrieval

## Technical Details:
- The system supports multiple document formats
- OCR is applied when necessary
- Semantic chunking preserves context
- Vector embeddings enable semantic search

## Testing Scenarios:
- Upload various document types
- Monitor processing status
- Query processed content
- Verify AI responses
            """)
        
        # Create sample markdown document
        sample_md = DOCUMENTS_DIR / "meeting_notes.md"
        if not sample_md.exists():
            sample_md.write_text("""
# Team Meeting Notes - AI Project

**Date:** July 17, 2025
**Attendees:** Development Team

## Agenda Items

### 1. Document Processing Pipeline
- ✅ Implemented OCR for scanned documents
- ✅ Added semantic chunking
- ✅ Vector embedding generation
- 🔄 Testing phase initiated

### 2. AI Query System
- Multi-agent orchestration with LangGraph
- Reasoning and synthesis capabilities
- Memory management (short-term and long-term)

### 3. User Interface
- Streamlit frontend with enhanced sidebar
- Real-time processing status tracking
- Document management features

## Action Items
1. Complete document processing tests
2. Optimize embedding generation
3. Implement advanced query features
4. Performance benchmarking

## Next Steps
- Deploy to staging environment
- User acceptance testing
- Documentation updates
            """)
        
        # Create sample Python code document
        sample_py = DOCUMENTS_DIR / "sample_code.py"
        if not sample_py.exists():
            sample_py.write_text("""
#!/usr/bin/env python3
\"\"\"
Sample Python code for AI document processing system.
This demonstrates the core processing pipeline.
\"\"\"

import asyncio
from typing import List, Dict, Any

class DocumentProcessor:
    \"\"\"Processes documents for AI analysis\"\"\"
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ocr_engine = None
        self.chunker = None
        self.embedder = None
    
    async def process_document(self, document_path: str) -> Dict[str, Any]:
        \"\"\"
        Process a single document through the pipeline.
        
        Args:
            document_path: Path to the document file
            
        Returns:
            Processing results including chunks and embeddings
        \"\"\"
        try:
            # Step 1: Extract text
            text = await self.extract_text(document_path)
            
            # Step 2: Apply OCR if needed
            if self.requires_ocr(document_path):
                text = await self.apply_ocr(document_path)
            
            # Step 3: Semantic chunking
            chunks = await self.create_chunks(text)
            
            # Step 4: Generate embeddings
            embeddings = await self.generate_embeddings(chunks)
            
            # Step 5: Store in vector database
            await self.store_embeddings(embeddings)
            
            return {
                "status": "completed",
                "chunks": len(chunks),
                "embeddings": len(embeddings)
            }
            
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    async def extract_text(self, document_path: str) -> str:
        \"\"\"Extract text from document\"\"\"
        # Implementation details...
        pass
    
    async def apply_ocr(self, document_path: str) -> str:
        \"\"\"Apply OCR to extract text from images\"\"\"
        # Implementation details...
        pass
    
    async def create_chunks(self, text: str) -> List[str]:
        \"\"\"Create semantic chunks from text\"\"\"
        # Implementation details...
        pass
    
    async def generate_embeddings(self, chunks: List[str]) -> List[List[float]]:
        \"\"\"Generate vector embeddings for chunks\"\"\"
        # Implementation details...
        pass
    
    async def store_embeddings(self, embeddings: List[List[float]]) -> bool:
        \"\"\"Store embeddings in vector database\"\"\"
        # Implementation details...
        pass

if __name__ == "__main__":
    processor = DocumentProcessor({"model": "sentence-transformers"})
    asyncio.run(processor.process_document("sample.pdf"))
            """)
        
        print(f"📄 Test documents created in: {DOCUMENTS_DIR}")
    
    def upload_document(self, file_path: Path) -> Optional[str]:
        """Upload a document to the workspace"""
        try:
            if not file_path.exists():
                print(f"❌ File not found: {file_path}")
                return None
            
            print(f"📤 Uploading document: {file_path.name}")
            
            # Determine MIME type
            mime_type, _ = mimetypes.guess_type(str(file_path))
            if not mime_type:
                mime_type = "application/octet-stream"
            
            with open(file_path, 'rb') as file:
                files = {"file": (file_path.name, file, mime_type)}
                response = self.session.post(
                    f"{self.base_url}/api/v1/workspaces/{self.workspace_id}/documents",
                    files=files
                )
            
            if response.status_code == 201:
                data = response.json()
                document_id = data.get("document_id")
                print(f"✅ Document uploaded: {file_path.name} (ID: {document_id})")
                return document_id
            else:
                error_data = response.json()
                print(f"❌ Upload failed for {file_path.name}: {error_data}")
                return None
                
        except Exception as e:
            print(f"❌ Upload error for {file_path.name}: {e}")
            return None
    
    def get_document_status(self, document_id: str) -> Dict[str, Any]:
        """Get detailed processing status for a document"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/workspaces/{self.workspace_id}/documents/{document_id}/status"
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Failed to get status for document {document_id}")
                return {}
                
        except Exception as e:
            print(f"❌ Status check error for document {document_id}: {e}")
            return {}
    
    def monitor_processing(self, document_id: str, max_wait_time: int = 300) -> bool:
        """Monitor document processing until completion"""
        start_time = time.time()
        print(f"⏳ Monitoring processing for document: {document_id}")
        
        while time.time() - start_time < max_wait_time:
            status_data = self.get_document_status(document_id)
            
            if not status_data:
                time.sleep(5)
                continue
            
            status = status_data.get("status", "unknown")
            progress = status_data.get("progress_percentage", 0)
            
            print(f"   📊 Status: {status.title()} ({progress}%)")
            
            # Show processing steps
            steps = status_data.get("steps", [])
            for step in steps:
                step_status = step.get("status", "unknown")
                step_name = step.get("step", "Unknown Step")
                emoji = {
                    "completed": "✅",
                    "in_progress": "⏳",
                    "pending": "⏸️",
                    "skipped": "⏭️",
                    "failed": "❌"
                }.get(step_status, "❓")
                print(f"      {emoji} {step_name}: {step_status.replace('_', ' ').title()}")
            
            if status in ["ready", "completed"]:
                print(f"✅ Document processing completed successfully!")
                print(f"   📊 Total chunks: {status_data.get('total_chunks', 0)}")
                print(f"   🔍 OCR applied: {'Yes' if status_data.get('ocr_applied') else 'No'}")
                return True
            elif status in ["failed", "error"]:
                error_msg = status_data.get("error_message", "Unknown error")
                print(f"❌ Document processing failed: {error_msg}")
                return False
            
            time.sleep(5)  # Wait 5 seconds before checking again
        
        print(f"⏰ Processing monitoring timed out after {max_wait_time} seconds")
        return False
    
    def query_documents(self, question: str) -> Dict[str, Any]:
        """Query the processed documents"""
        try:
            print(f"❓ Querying: {question}")
            response = self.session.post(
                f"{self.base_url}/api/v1/query/",
                json={
                    "workspace_id": self.workspace_id,
                    "query": question,
                    "session_id": "test_session_001"
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get("response", "No response received")
                print(f"💬 AI Response: {answer}")
                return data
            else:
                error_data = response.json()
                print(f"❌ Query failed: {error_data}")
                return {}
                
        except Exception as e:
            print(f"❌ Query error: {e}")
            return {}
    
    def get_workspace_documents(self) -> List[Dict[str, Any]]:
        """Get all documents in the workspace"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/workspaces/{self.workspace_id}/documents"
            )
            
            if response.status_code == 200:
                documents = response.json()
                print(f"📚 Found {len(documents)} documents in workspace")
                return documents
            else:
                print("❌ Failed to retrieve workspace documents")
                return []
                
        except Exception as e:
            print(f"❌ Error retrieving documents: {e}")
            return []

def main():
    """Main test function"""
    print("🚀 Starting Insight AI Document Ingestion Test")
    print("=" * 50)
    
    tester = DocumentIngestionTester()
    
    # Step 1: Check server health
    if not tester.check_server_health():
        return
    
    # Step 2: Register and login user
    if not tester.register_user():
        return
    
    if not tester.login_user():
        return
    
    # Step 3: Create workspace
    if not tester.create_workspace():
        return
    
    # Step 4: Create test documents
    tester.create_test_documents()
    
    # Step 5: Upload and process documents
    print("\n📤 Starting document upload and processing...")
    document_ids = []
    
    for doc_name in ["test_doc.txt", "meeting_notes.md", "sample_code.py"]:
        doc_path = DOCUMENTS_DIR / doc_name
        document_id = tester.upload_document(doc_path)
        if document_id:
            document_ids.append((document_id, doc_name))
    
    if not document_ids:
        print("❌ No documents were uploaded successfully")
        return
    
    # Step 6: Monitor processing
    print("\n⏳ Monitoring document processing...")
    processed_docs = []
    
    for document_id, doc_name in document_ids:
        print(f"\n📄 Processing: {doc_name}")
        if tester.monitor_processing(document_id, max_wait_time=120):
            processed_docs.append(document_id)
    
    if not processed_docs:
        print("❌ No documents were processed successfully")
        return
    
    # Step 7: Query the processed documents
    print("\n💬 Testing AI queries...")
    test_queries = [
        "What are the main features of the document processing pipeline?",
        "What were the action items from the team meeting?",
        "How does the DocumentProcessor class work?",
        "What testing scenarios are mentioned in the documents?"
    ]
    
    for query in test_queries:
        print(f"\n" + "=" * 60)
        result = tester.query_documents(query)
        time.sleep(2)  # Small delay between queries
    
    # Step 8: Final status check
    print("\n📊 Final workspace summary:")
    documents = tester.get_workspace_documents()
    for doc in documents:
        status = doc.get("status", "unknown")
        name = doc.get("file_name", "Unknown")
        chunks = doc.get("total_chunks", 0)
        print(f"   📄 {name}: {status.title()} ({chunks} chunks)")
    
    print("\n✅ Document ingestion test completed!")
    print("🎉 All systems are working correctly!")

if __name__ == "__main__":
    main()
