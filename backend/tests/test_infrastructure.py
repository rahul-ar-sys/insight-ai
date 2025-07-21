#!/usr/bin/env python3
"""
Complete Infrastructure Test for Insight AI Platform
Tests all components: Database, Redis, ChromaDB, Google AI, Storage, OCR
"""
import os
import sys
import asyncio
from pathlib import Path
import json

# Test results storage
test_results = {
    "database": {"status": "pending", "details": ""},
    "redis": {"status": "pending", "details": ""},
    "chromadb": {"status": "pending", "details": ""},
    "google_ai": {"status": "pending", "details": ""},
    "storage": {"status": "pending", "details": ""},
    "ocr": {"status": "pending", "details": ""},
    "nlp": {"status": "pending", "details": ""}
}

def test_database():
    """Test PostgreSQL database connection"""
    try:
        import psycopg2
        
        print("🗄️ Testing PostgreSQL database...")
        
        conn = psycopg2.connect(
            host="localhost",
            database="insight_ai",
            user="insight_ai_user",
            password="insight_ai_password"
        )
        
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        test_results["database"]["status"] = "✅ PASS"
        test_results["database"]["details"] = f"Connected to PostgreSQL: {version.split()[0]} {version.split()[1]}"
        print(f"✅ Database: {test_results['database']['details']}")
        return True
        
    except Exception as e:
        test_results["database"]["status"] = "❌ FAIL"
        test_results["database"]["details"] = str(e)
        print(f"❌ Database failed: {e}")
        return False

def test_redis():
    """Test Redis cache connection"""
    try:
        import redis
        
        print("🔴 Testing Redis cache...")
        
        r = redis.Redis(host='localhost', port=6379, db=0)
        
        # Test basic operations
        r.set('test_key', 'test_value')
        value = r.get('test_key').decode('utf-8')
        r.delete('test_key')
        
        if value == 'test_value':
            info = r.info()
            version = info['redis_version']
            test_results["redis"]["status"] = "✅ PASS"
            test_results["redis"]["details"] = f"Redis {version} - Read/Write operations working"
            print(f"✅ Redis: {test_results['redis']['details']}")
            return True
        else:
            raise Exception("Read/Write test failed")
            
    except Exception as e:
        test_results["redis"]["status"] = "❌ FAIL"
        test_results["redis"]["details"] = str(e)
        print(f"❌ Redis failed: {e}")
        return False

def test_chromadb():
    """Test ChromaDB vector database with persistent client"""
    try:
        import chromadb
        from pathlib import Path
        
        print("🧬 Testing ChromaDB vector database...")
        
        # Create persist directory
        persist_dir = Path("./data/chroma")
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Use persistent client (more reliable for development)
        client = chromadb.PersistentClient(path=str(persist_dir))
        
        # Test collection creation and operations
        collection_name = "test_collection"
        
        # Delete if exists
        try:
            client.delete_collection(collection_name)
        except:
            pass
        
        collection = client.create_collection(collection_name)
        
        # Add test documents
        collection.add(
            documents=["This is a test document", "Another test document"],
            metadatas=[{"source": "test1"}, {"source": "test2"}],
            ids=["id1", "id2"]
        )
        
        # Query test
        results = collection.query(
            query_texts=["test document"],
            n_results=1
        )
        
        # Cleanup
        client.delete_collection(collection_name)
        
        if results['documents']:
            test_results["chromadb"]["status"] = "✅ PASS"
            test_results["chromadb"]["details"] = f"ChromaDB persistent client working - {persist_dir.absolute()}"
            print(f"✅ ChromaDB: {test_results['chromadb']['details']}")
            return True
        else:
            raise Exception("Query returned no results")
            
    except Exception as e:
        test_results["chromadb"]["status"] = "❌ FAIL"
        test_results["chromadb"]["details"] = str(e)
        print(f"❌ ChromaDB failed: {e}")
        return False

def test_google_ai():
    """Test Google AI (Gemini) API"""
    try:
        import google.generativeai as genai
        
        print("🤖 Testing Google AI (Gemini)...")
        
        genai.configure(api_key="AIzaSyDnVcMsUDUrn3JQSexBCE3EwZp4xOxInyc")
        
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content("Reply with exactly: 'AI Test Successful'")
        
        if "AI Test Successful" in response.text:
            test_results["google_ai"]["status"] = "✅ PASS"
            test_results["google_ai"]["details"] = f"Gemini 2.0 Flash responding correctly"
            print(f"✅ Google AI: {test_results['google_ai']['details']}")
            return True
        else:
            raise Exception(f"Unexpected response: {response.text}")
            
    except Exception as e:
        test_results["google_ai"]["status"] = "❌ FAIL"
        test_results["google_ai"]["details"] = str(e)
        print(f"❌ Google AI failed: {e}")
        return False

def test_local_storage():
    """Test local file storage"""
    try:
        print("💾 Testing local file storage...")
        
        storage_path = Path("./data/uploads")
        storage_path.mkdir(parents=True, exist_ok=True)
        
        # Test file operations
        test_file = storage_path / "infrastructure_test.txt"
        test_content = "Infrastructure test - file storage working!"
        
        test_file.write_text(test_content)
        read_content = test_file.read_text()
        test_file.unlink()
        
        if read_content == test_content:
            test_results["storage"]["status"] = "✅ PASS"
            test_results["storage"]["details"] = f"Local storage at {storage_path.absolute()}"
            print(f"✅ Storage: {test_results['storage']['details']}")
            return True
        else:
            raise Exception("File content mismatch")
            
    except Exception as e:
        test_results["storage"]["status"] = "❌ FAIL"
        test_results["storage"]["details"] = str(e)
        print(f"❌ Storage failed: {e}")
        return False

def test_ocr():
    """Test Tesseract OCR functionality"""
    try:
        import pytesseract
        from PIL import Image, ImageDraw, ImageFont
        
        print("👀 Testing Tesseract OCR...")
        
        # Configure tesseract path
        pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
        
        # Create test image
        img = Image.new('RGB', (300, 80), color='white')
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        draw.text((10, 25), "OCR Infrastructure Test", fill='black', font=font)
        
        # Perform OCR
        extracted_text = pytesseract.image_to_string(img).strip()
        
        if "OCR" in extracted_text and "Test" in extracted_text:
            test_results["ocr"]["status"] = "✅ PASS"
            test_results["ocr"]["details"] = f"Tesseract extracting text correctly: '{extracted_text}'"
            print(f"✅ OCR: {test_results['ocr']['details']}")
            return True
        else:
            raise Exception(f"OCR extraction failed: '{extracted_text}'")
            
    except Exception as e:
        test_results["ocr"]["status"] = "❌ FAIL"
        test_results["ocr"]["details"] = str(e)
        print(f"❌ OCR failed: {e}")
        return False

def test_nlp_libraries():
    """Test NLP libraries (spaCy, sentence-transformers)"""
    try:
        print("🧠 Testing NLP libraries...")
        
        # Test spaCy
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("This is a test sentence for NLP processing.")
        entities = [ent.text for ent in doc.ents]
        
        # Test sentence transformers
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(["Test sentence for embeddings"])
        
        if len(embeddings[0]) == 384:  # Expected embedding dimension
            test_results["nlp"]["status"] = "✅ PASS"
            test_results["nlp"]["details"] = "spaCy and SentenceTransformers working correctly"
            print(f"✅ NLP: {test_results['nlp']['details']}")
            return True
        else:
            raise Exception(f"Unexpected embedding dimension: {len(embeddings[0])}")
            
    except Exception as e:
        test_results["nlp"]["status"] = "❌ FAIL"
        test_results["nlp"]["details"] = str(e)
        print(f"❌ NLP failed: {e}")
        return False

def generate_report():
    """Generate final test report"""
    print("\n" + "="*60)
    print("🎯 INFRASTRUCTURE TEST REPORT")
    print("="*60)
    
    passed = 0
    total = len(test_results)
    
    for component, result in test_results.items():
        status = result["status"]
        details = result["details"]
        print(f"{status} {component.upper()}: {details}")
        if "✅" in status:
            passed += 1
    
    print("="*60)
    print(f"📊 SUMMARY: {passed}/{total} components working")
    
    if passed == total:
        print("🎉 ALL SYSTEMS OPERATIONAL!")
        print("🚀 Ready to start the Insight AI application!")
        return True
    else:
        print("⚠️ Some components need attention before starting the application")
        return False

def main():
    """Run complete infrastructure test"""
    print("🚀 INSIGHT AI INFRASTRUCTURE TEST")
    print("="*50)
    
    # Run all tests
    test_database()
    test_redis()
    test_chromadb()
    test_google_ai()
    test_local_storage()
    test_ocr()
    test_nlp_libraries()
    
    # Generate report
    all_passed = generate_report()
    
    if all_passed:
        print("\n✅ Infrastructure is ready!")
        print("🎯 Next step: Start the FastAPI application")
        print("   Command: python -m uvicorn app.main:app --reload")
    else:
        print("\n❌ Please fix failing components before proceeding")

if __name__ == "__main__":
    main()
