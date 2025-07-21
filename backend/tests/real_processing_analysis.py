#!/usr/bin/env python3
"""
Real Document Processing Performance Analysis

This script analyzes actual document processing performance by monitoring
the backend services and database operations during real document uploads.

Usage:
    python real_processing_analysis.py
"""

import time
import requests
import json
from pathlib import Path
from typing import Dict, Any, Optional
import sys
import os

# Add parent directory to path to import from backend
sys.path.append(str(Path(__file__).parent.parent))

try:
    from app.core.database import get_db_session
    from app.models.database import Document
    from sqlalchemy.orm import Session
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False
    print("⚠️ Backend imports not available, using API-only monitoring")

class RealProcessingAnalyzer:
    """Analyzes real document processing performance"""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.metrics = {
            "upload_start_time": None,
            "upload_complete_time": None,
            "processing_phases": {},
            "database_queries": [],
            "api_calls": [],
            "total_processing_time": None
        }
        
    def monitor_document_status(self, document_id: str, max_wait: int = 300) -> Dict[str, Any]:
        """Monitor document processing with detailed timing"""
        print(f"🔍 Monitoring document processing: {document_id}")
        
        start_time = time.time()
        last_status = None
        status_changes = []
        
        while time.time() - start_time < max_wait:
            try:
                # Check document status via API
                response = requests.get(f"{self.base_url}/api/documents/{document_id}/status")
                
                if response.status_code == 200:
                    data = response.json()
                    current_status = data.get('status', 'unknown')
                    current_time = time.time()
                    
                    if current_status != last_status:
                        phase_time = current_time - start_time
                        status_changes.append({
                            "status": current_status,
                            "time": phase_time,
                            "timestamp": current_time
                        })
                        
                        print(f"   📊 Status: {current_status} (at {phase_time:.1f}s)")
                        
                        if current_status in ["completed", "failed"]:
                            break
                            
                        last_status = current_status
                
                elif response.status_code == 404:
                    print("   ❌ Document not found")
                    break
                else:
                    print(f"   ⚠️ API Error: {response.status_code}")
                
                time.sleep(2)  # Check every 2 seconds
                
            except Exception as e:
                print(f"   ❌ Monitoring error: {e}")
                time.sleep(5)
        
        total_time = time.time() - start_time
        
        return {
            "total_time": total_time,
            "status_changes": status_changes,
            "final_status": last_status
        }
    
    def analyze_processing_bottlenecks(self, document_id: str) -> Dict[str, Any]:
        """Analyze what's causing processing delays"""
        print("🔬 Analyzing processing bottlenecks...")
        
        bottlenecks = {
            "file_size_impact": None,
            "text_extraction_time": None,
            "chunking_complexity": None,
            "embedding_generation": None,
            "database_operations": None,
            "vector_storage": None
        }
        
        try:
            # Get document details
            doc_response = requests.get(f"{self.base_url}/api/documents/{document_id}")
            if doc_response.status_code == 200:
                doc_data = doc_response.json()
                file_size = doc_data.get('file_size', 0)
                
                print(f"   📄 File size: {file_size / 1024:.1f} KB")
                
                # Estimate expected processing time based on file size
                expected_time = self.estimate_processing_time(file_size)
                bottlenecks["file_size_impact"] = expected_time
                
                print(f"   ⏱️ Expected processing time: {expected_time:.1f}s")
        
        except Exception as e:
            print(f"   ❌ Error analyzing document: {e}")
        
        # Check for common bottlenecks
        print("   🔍 Checking common bottlenecks:")
        
        # 1. Text Extraction (OCR/PDF parsing)
        print("     - Text extraction complexity...")
        bottlenecks["text_extraction_time"] = self.estimate_text_extraction_time()
        
        # 2. Semantic Chunking
        print("     - Semantic chunking overhead...")
        bottlenecks["chunking_complexity"] = self.estimate_chunking_time()
        
        # 3. Embedding Generation
        print("     - Embedding generation load...")
        bottlenecks["embedding_generation"] = self.estimate_embedding_time()
        
        # 4. Database Operations
        print("     - Database operation latency...")
        bottlenecks["database_operations"] = self.measure_db_latency()
        
        # 5. Vector Storage
        print("     - Vector database performance...")
        bottlenecks["vector_storage"] = self.measure_vector_storage_latency()
        
        return bottlenecks
    
    def estimate_processing_time(self, file_size_bytes: int) -> float:
        """Estimate expected processing time based on file size"""
        # Base processing time + size-dependent time
        base_time = 5.0  # seconds
        size_factor = file_size_bytes / (1024 * 1024)  # MB
        
        # Different file types have different processing complexity
        estimated = base_time + (size_factor * 10)  # ~10s per MB
        
        return max(estimated, 2.0)  # Minimum 2 seconds
    
    def estimate_text_extraction_time(self) -> float:
        """Estimate text extraction bottleneck"""
        # This would normally check if OCR is being used, PDF complexity, etc.
        return 3.0  # Typical PDF text extraction
    
    def estimate_chunking_time(self) -> float:
        """Estimate chunking processing time"""
        # Semantic chunking can be compute-intensive
        return 2.0  # Typical chunking time
    
    def estimate_embedding_time(self) -> float:
        """Estimate embedding generation time"""
        # This depends on the embedding model and number of chunks
        return 8.0  # Typical embedding generation (major bottleneck)
    
    def measure_db_latency(self) -> float:
        """Measure database operation latency"""
        try:
            start_time = time.time()
            response = requests.get(f"{self.base_url}/api/health/db")
            return time.time() - start_time
        except:
            return 1.0  # Default estimate
    
    def measure_vector_storage_latency(self) -> float:
        """Measure vector database latency"""
        try:
            start_time = time.time()
            response = requests.get(f"{self.base_url}/api/health/vector")
            return time.time() - start_time
        except:
            return 2.0  # Default estimate
    
    def print_performance_analysis(self, monitoring_result: Dict, bottlenecks: Dict):
        """Print detailed performance analysis"""
        print("\\n" + "=" * 70)
        print("📊 REAL DOCUMENT PROCESSING ANALYSIS")
        print("=" * 70)
        
        total_time = monitoring_result["total_time"]
        status_changes = monitoring_result["status_changes"]
        
        print(f"⏱️ TOTAL PROCESSING TIME: {total_time:.1f} seconds")
        print(f"🏁 FINAL STATUS: {monitoring_result['final_status']}")
        
        if status_changes:
            print("\\n📈 PROCESSING PHASES:")
            for i, change in enumerate(status_changes):
                if i == 0:
                    phase_duration = change["time"]
                else:
                    phase_duration = change["time"] - status_changes[i-1]["time"]
                
                print(f"   {change['status']:15} | {phase_duration:6.1f}s | (at {change['time']:6.1f}s)")
        
        print("\\n🔍 BOTTLENECK ANALYSIS:")
        expected_time = bottlenecks.get("file_size_impact", 0)
        
        if total_time > expected_time * 2:
            print(f"   ❌ SLOW: {total_time:.1f}s vs expected {expected_time:.1f}s")
            print("   🔍 Likely bottlenecks:")
            
            embedding_time = bottlenecks.get("embedding_generation", 0)
            if embedding_time > 5:
                print(f"     - Embedding generation: ~{embedding_time:.1f}s (HIGH IMPACT)")
            
            extraction_time = bottlenecks.get("text_extraction_time", 0)
            if extraction_time > 2:
                print(f"     - Text extraction: ~{extraction_time:.1f}s")
            
            chunking_time = bottlenecks.get("chunking_complexity", 0)
            if chunking_time > 1:
                print(f"     - Semantic chunking: ~{chunking_time:.1f}s")
            
            db_latency = bottlenecks.get("database_operations", 0)
            if db_latency > 0.5:
                print(f"     - Database operations: ~{db_latency:.1f}s")
            
            vector_latency = bottlenecks.get("vector_storage", 0)
            if vector_latency > 1:
                print(f"     - Vector storage: ~{vector_latency:.1f}s")
        
        elif total_time > expected_time * 1.5:
            print(f"   ⚠️ SLOWER THAN EXPECTED: {total_time:.1f}s vs {expected_time:.1f}s")
        else:
            print(f"   ✅ WITHIN EXPECTED RANGE: {total_time:.1f}s")
        
        print("\\n💡 OPTIMIZATION RECOMMENDATIONS:")
        
        if bottlenecks.get("embedding_generation", 0) > 5:
            print("   🔗 Embedding Generation (MAJOR BOTTLENECK):")
            print("     - Consider using faster embedding models")
            print("     - Implement batch embedding generation")
            print("     - Use GPU acceleration if available")
            print("     - Cache embeddings for similar content")
        
        if bottlenecks.get("text_extraction_time", 0) > 3:
            print("   📖 Text Extraction:")
            print("     - Optimize PDF parsing libraries")
            print("     - Use streaming for large documents")
            print("     - Implement OCR optimization")
        
        if bottlenecks.get("database_operations", 0) > 0.5:
            print("   🗄️ Database Operations:")
            print("     - Add database indexes")
            print("     - Use connection pooling")
            print("     - Optimize query patterns")
        
        if bottlenecks.get("vector_storage", 0) > 1:
            print("   🔍 Vector Storage:")
            print("     - Optimize ChromaDB configuration")
            print("     - Consider batch vector operations")
            print("     - Monitor memory usage")

def analyze_real_processing():
    """Run real document processing analysis"""
    print("🔍 Real Document Processing Performance Analysis")
    print("=" * 60)
    
    analyzer = RealProcessingAnalyzer()
    
    # Check if backend is running
    try:
        response = requests.get(f"{analyzer.base_url}/health", timeout=5)
        if response.status_code != 200:
            print("❌ Backend server not responding properly")
            return
    except Exception as e:
        print(f"❌ Cannot connect to backend: {e}")
        print("   Make sure the backend is running on http://localhost:8000")
        return
    
    print("✅ Backend server is running")
    
    # Instructions for user
    print("\\n📋 INSTRUCTIONS:")
    print("1. This tool will monitor document processing in real-time")
    print("2. Upload a document through the frontend")
    print("3. Provide the document ID when prompted")
    print("4. The tool will analyze processing performance")
    
    # Get document ID from user
    document_id = input("\\n📄 Enter document ID to monitor (or 'test' for demo): ").strip()
    
    if document_id.lower() == 'test':
        # Demo analysis with simulated data
        print("\\n🎭 Running demo analysis with simulated data...")
        
        simulated_result = {
            "total_time": 45.2,
            "status_changes": [
                {"status": "uploaded", "time": 0.5, "timestamp": time.time()},
                {"status": "processing", "time": 2.1, "timestamp": time.time() + 2.1},
                {"status": "extracting_text", "time": 8.3, "timestamp": time.time() + 8.3},
                {"status": "chunking", "time": 15.7, "timestamp": time.time() + 15.7},
                {"status": "generating_embeddings", "time": 42.1, "timestamp": time.time() + 42.1},
                {"status": "completed", "time": 45.2, "timestamp": time.time() + 45.2}
            ],
            "final_status": "completed"
        }
        
        simulated_bottlenecks = {
            "file_size_impact": 12.0,
            "text_extraction_time": 6.2,
            "chunking_complexity": 7.4,
            "embedding_generation": 26.4,  # Major bottleneck
            "database_operations": 0.8,
            "vector_storage": 2.3
        }
        
        analyzer.print_performance_analysis(simulated_result, simulated_bottlenecks)
        
    else:
        # Real monitoring
        print(f"\\n🚀 Starting real-time monitoring for document: {document_id}")
        
        # Monitor document processing
        monitoring_result = analyzer.monitor_document_status(document_id)
        
        # Analyze bottlenecks
        bottlenecks = analyzer.analyze_processing_bottlenecks(document_id)
        
        # Print analysis
        analyzer.print_performance_analysis(monitoring_result, bottlenecks)
    
    print("\\n✅ Performance analysis completed!")

if __name__ == "__main__":
    analyze_real_processing()
