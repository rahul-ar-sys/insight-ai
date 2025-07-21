# 🚀 Insight AI - Setup Guide (Google-Focused Tech Stack)

This guide covers setting up all required APIs and services for the Insight AI platform using Google services as the primary provider.

## 📋 **Tech Stack Overview**

### **Core Infrastructure**
- **Database**: PostgreSQL + Redis
- **Vector Store**: ChromaDB
- **AI Provider**: Google AI (Gemini Pro + Vision)
- **Authentication**: Firebase Auth
- **Storage**: Google Cloud Storage
- **NLP**: spaCy (local) + Sentence Transformers

### **Excluded Services**
- ❌ OpenAI API (not using)
- ❌ Anthropic Claude (not using)  
- ❌ AWS S3 (not using)

## 🛠️ **Step-by-Step Setup**

### **1. Google Cloud Platform Setup**

#### **1.1 Create Google Cloud Project**
```bash
# Visit Google Cloud Console
https://console.cloud.google.com/

# Create new project or select existing
# Note your PROJECT_ID for later use
```

#### **1.2 Enable Required APIs**
```bash
# Enable these APIs in Google Cloud Console:
- AI Platform API
- Cloud Storage API
- Firebase Authentication API
- Cloud Vision API (for document OCR)
```

#### **1.3 Create Service Account**
```bash
# Go to IAM & Admin > Service Accounts
# Create service account with roles:
- Storage Admin
- AI Platform User
- Firebase Admin

# Download JSON key file
# Save as: ./backend/credentials/gcs-service-account.json
```

### **2. Google AI Studio Setup**

#### **2.1 Get API Key**
```bash
# Visit Google AI Studio
https://makersuite.google.com/app/apikey

# Create API key for Gemini Pro access
# Copy API key for .env file
```

#### **2.2 Test API Access**
```bash
# Test your API key
curl -H "Content-Type: application/json" \
-d '{"contents":[{"parts":[{"text":"Hello"}]}]}' \
"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=YOUR_API_KEY"
```

### **3. Firebase Authentication Setup**

#### **3.1 Create Firebase Project**
```bash
# Visit Firebase Console
https://console.firebase.google.com/

# Create project (can use same as GCP project)
# Enable Authentication with desired providers:
- Email/Password
- Google Sign-in
- Others as needed
```

#### **3.2 Get Firebase Credentials**
```bash
# Go to Project Settings > Service Accounts
# Generate new private key
# Copy the JSON contents to your .env file

# Key fields needed:
- project_id
- private_key_id  
- private_key
- client_email
- client_id
```

### **4. Google Cloud Storage Setup**

#### **4.1 Create Storage Bucket**
```bash
# Using gcloud CLI (install if needed)
gsutil mb gs://your-insight-ai-bucket

# Or via Cloud Console:
# Cloud Storage > Create Bucket
# Choose region, storage class, etc.
```

#### **4.2 Set Bucket Permissions**
```bash
# Make bucket accessible to your service account
gsutil iam ch serviceAccount:your-service-account@project.iam.gserviceaccount.com:objectAdmin gs://your-bucket-name
```

### **5. Local Infrastructure Setup**

#### **5.1 PostgreSQL Database**
```bash
# Install PostgreSQL
# Ubuntu/Debian:
sudo apt update
sudo apt install postgresql postgresql-contrib

# macOS:
brew install postgresql
brew services start postgresql

# Windows:
# Download from https://www.postgresql.org/download/windows/

# Create database
sudo -u postgres createdb insight_ai
sudo -u postgres createuser --interactive your_username
```

#### **5.2 Redis Cache**
```bash
# Install Redis
# Ubuntu/Debian:
sudo apt install redis-server
sudo systemctl start redis-server

# macOS:
brew install redis
brew services start redis

# Windows:
# Download from https://github.com/microsoftarchive/redis/releases

# Test Redis
redis-cli ping
# Should return: PONG
```

#### **5.3 ChromaDB Vector Store**
```bash
# ChromaDB will be installed via pip
# No separate installation needed
# Will create local database automatically
```

### **6. Python Environment Setup**

#### **6.1 Create Virtual Environment**
```bash
cd backend
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

#### **6.2 Install Dependencies**
```bash
pip install -r requirements.txt
```

#### **6.3 Download NLP Models**
```bash
# Download spaCy English model
python -m spacy download en_core_web_sm

# Sentence transformers will download automatically on first use
```

#### **6.4 Install Tesseract OCR**
```bash
# Ubuntu/Debian:
sudo apt install tesseract-ocr

# macOS:
brew install tesseract

# Windows:
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
# Add to PATH: C:\Program Files\Tesseract-OCR
```

### **7. Environment Configuration**

#### **7.1 Create .env File**
```bash
cp .env.example .env
```

#### **7.2 Fill in Google Credentials**
```bash
# Update .env with your actual values:

# Google AI
GOOGLE_API_KEY="your-actual-api-key"

# Firebase (from service account JSON)
FIREBASE_PROJECT_ID="your-project-id"
FIREBASE_PRIVATE_KEY_ID="your-private-key-id"
FIREBASE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\nyour-actual-private-key\n-----END PRIVATE KEY-----"
FIREBASE_CLIENT_EMAIL="your-service-account@project.iam.gserviceaccount.com"
FIREBASE_CLIENT_ID="your-client-id"

# Google Cloud Storage
GCS_BUCKET_NAME="your-actual-bucket-name"
GCS_PROJECT_ID="your-project-id"

# Database
DATABASE_URL="postgresql://username:password@localhost:5432/insight_ai"
REDIS_URL="redis://localhost:6379/0"
```

### **8. Database Initialization**

#### **8.1 Run Database Migrations**
```bash
cd backend
alembic upgrade head
```

#### **8.2 Verify Database Setup**
```bash
# Test database connection
python -c "
from app.core.database import engine
from sqlalchemy import text
with engine.connect() as conn:
    result = conn.execute(text('SELECT 1'))
    print('Database connected successfully!')
"
```

### **9. Test the Setup**

#### **9.1 Start the Application**
```bash
cd backend
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

#### **9.2 Verify Services**
```bash
# Check API documentation
http://localhost:8000/docs

# Test health endpoint
curl http://localhost:8000/health

# Check ChromaDB
curl http://localhost:8000/api/v1/health/vector-store
```

## 🔧 **Service Dependencies Summary**

### **External APIs (Require Keys)**
1. **Google AI API** - Gemini Pro models
2. **Firebase Authentication** - User management
3. **Google Cloud Storage** - File storage

### **Local Services (Self-hosted)**
1. **PostgreSQL** - Main database
2. **Redis** - Caching and sessions
3. **ChromaDB** - Vector embeddings
4. **Tesseract OCR** - Text extraction

### **Local Libraries (No setup required)**
1. **spaCy** - NLP processing
2. **Sentence Transformers** - Embeddings
3. **BERTopic** - Topic modeling

## 🐳 **Ready for Docker**

Once you complete this setup, the system will be ready for dockerization with:
- All Google services properly configured
- Local databases running
- API keys and credentials in place
- Dependencies installed and tested

Would you like me to create the Docker configuration files next?
