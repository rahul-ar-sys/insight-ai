# Insight AI: Advanced Document Analysis and Retrieval System

## 1. Project Overview

Insight AI is a comprehensive, multi-service application designed to ingest, process, and analyze large volumes of documents. It uses a Retrieval-Augmented Generation (RAG) architecture to provide accurate, context-aware answers based on the content of a provided document library.

The system is fully containerized using Docker and Docker Compose, allowing for a consistent, one-command startup process across different development environments. It leverages a local, open-source Large Language Model (LLM) to ensure data privacy and eliminate the need for external API keys or manual configuration.

---

## 2. System Architecture

The application is composed of several interconnected microservices, each playing a specific role:

* **Frontend (Streamlit):** A web-based user interface built with Streamlit. This is the primary entry point for users to upload documents, ask questions, and view the AI-generated responses and their sources.

* **Backend (FastAPI):** The central nervous system of the application. This Python-based API handles business logic, orchestrates the flow of data between services, manages document processing pipelines, and exposes endpoints for the frontend to consume.

* **Ollama (LLM Service):** Provides local, open-source LLM capabilities. Instead of relying on external APIs like Gemini or OpenAI, Ollama runs models such as Llama 3 directly within the Docker environment. This service is responsible for the final synthesis and reasoning steps.

* **ChromaDB (Vector Database):** The core of the retrieval system. When documents are processed, their text is broken into chunks and converted into numerical representations (embeddings). ChromaDB stores these embeddings and allows for lightning-fast semantic similarity searches to find the most relevant context for a given query.

* **PostgreSQL (Relational Database):** The primary store for structured data. This includes user information, document metadata (filenames, upload dates, processing status), chat history, and any other relational data required by the application.

* **Redis (In-Memory Cache):** A high-speed cache used to store frequently accessed data, reducing latency and offloading work from the database. It can also be used as a message broker for managing background tasks in more complex workflows.

* **Document Processors (Tesseract & Sentence-Transformers):** These are not services but critical libraries baked into the backend's Docker image.
    * **Tesseract OCR:** An optical character recognition engine used to extract text from scanned documents and images.
    * **Sentence-Transformers:** A library used to generate the high-quality semantic embeddings from text chunks that are stored in ChromaDB.

---

## 3. Prerequisites

Before starting, ensure the following software is installed on your machine:

1.  **Docker Desktop:** The core engine for running containers. The application has been configured and tested with Docker.
2.  An application to **unzip files**.

---

## 4. Getting Started: One-Command Launch

Follow these steps to set up and run the entire application stack. The project includes a pre-configured `.env` file, so no manual configuration is needed.

### Step 1: Download and Unzip the Project

1.  Download the project ZIP file.
2.  Unzip the contents to a location on your computer.
3.  Open your terminal or PowerShell and navigate into the unzipped project folder (e.g., `cd C:\path\to\insight-ai`).

### Step 2: Run the Application

A startup script is provided for convenience. This script will check that Docker is running and then launch all services defined in the `docker-compose.yml` file.

**For Windows Users:**

1.  Open PowerShell.
2.  Navigate to the project's root directory (if you are not already there).
3.  Run the startup script:
    ```powershell
    .\start.ps1
    ```
    *Note: If you encounter an execution policy error, you may need to run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` in PowerShell and try again.*

**For macOS / Linux Users:**

1.  Open your terminal.
2.  Navigate to the project's root directory (if you are not already there).
3.  Run the `docker-compose` command directly:
    ```bash
    docker-compose up --build -d
    ```

The first time you run this command, it will take several minutes to download all the necessary Docker images and build the custom images for the backend and frontend. The `init-ollama` service will also download the `llama3:8b` model, which is a multi-gigabyte download. Subsequent startups will be much faster.

### Step 3: Access the Application

Once the startup process is complete, the services will be available at the following local URLs:

* **Frontend (Streamlit UI):** `http://localhost:8501`
* **Backend (API Docs):** `http://localhost:5000/docs`
* **ChromaDB UI:** `http://localhost:8000`
* **Ollama API:** `http://localhost:11434`

---

## 5. How to Use the Application

1.  Open your web browser and navigate to the **Frontend UI** at `http://localhost:8501`.
2.  Use the interface to upload a document (e.g., a PDF file). The backend will automatically process it, perform OCR if necessary, generate embeddings, and store them in ChromaDB.
3.  Once the document is processed, use the chat interface to ask a question related to its content.
4.  The backend will retrieve the most relevant context from ChromaDB and use the local Ollama LLM to generate a comprehensive answer, which will be displayed in the UI along with its sources.

---

## 6. Alternative Setup: Using Git

If you prefer to use Git, you can clone the repository instead of downloading the ZIP file.

1.  **Prerequisite:** Ensure **Git** is installed on your machine.
2.  **Clone the repository:**
    ```bash
    git clone https://github.com/rahul-ar-sys/insight-ai
    ```
3.  **Navigate into the directory:**
    ```bash
    cd insight-ai
    ```
4.  Proceed to **Step 2: Run the Application** from the main instructions above.
