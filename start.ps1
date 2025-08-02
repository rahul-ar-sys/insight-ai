# start.ps1 - Starts the Insight AI application stack using Docker Compose.

# Set text color for better readability
$InfoColor = "Green"
$WarnColor = "Yellow"
$ErrorColor = "Red"

# 1. Check if Docker Desktop is running
Write-Host "Checking if Docker is running..." -ForegroundColor $InfoColor
if (-not (docker info 2>$null)) {
    Write-Host "Docker is not running. Please start Docker Desktop and try again." -ForegroundColor $ErrorColor
    # Pause to allow user to read the message before the window closes
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "Docker is running." -ForegroundColor $InfoColor
Write-Host ""

# 2. Check for the .env file
$envFile = ".env"
if (-not (Test-Path $envFile)) {
    Write-Host "The '.env' file was not found." -ForegroundColor $WarnColor
    Write-Host "Please copy 'env.example' to '.env' and fill in the required values." -ForegroundColor $WarnColor
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "'.env' file found." -ForegroundColor $InfoColor
Write-Host ""

# 3. Start the Docker Compose stack
# CHANGED: Added 'ollama' to the list of services for clarity.
Write-Host "Starting all services (backend, frontend, postgres, redis, chromadb, ollama)..." -ForegroundColor $InfoColor
Write-Host "This might take a few minutes on the first run." -ForegroundColor $InfoColor

# Use --build to rebuild images if code has changed
# Use -d to run in detached mode (in the background)
docker-compose up --build -d

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "------------------------------------------------------" -ForegroundColor $InfoColor
    Write-Host "✅ Application started successfully!" -ForegroundColor $InfoColor
    Write-Host ""
    Write-Host "   - Frontend (Streamlit): http://localhost:8501"
    Write-Host "   - Backend API (docs):   http://localhost:5000/docs"
    Write-Host "   - ChromaDB UI:          http://localhost:8000"
    # CHANGED: Added Ollama API endpoint for debugging.
    Write-Host "   - Ollama API:           http://localhost:11434"
    Write-Host "------------------------------------------------------" -ForegroundColor $InfoColor
    Write-Host "To stop the application, run the 'stop.ps1' script."
} else {
    Write-Host ""
    Write-Host "❌ There was an error starting the application with Docker Compose." -ForegroundColor $ErrorColor
    Write-Host "Check the output above for more details."
}

# Keep the window open to see the final status
Read-Host "Press Enter to exit"