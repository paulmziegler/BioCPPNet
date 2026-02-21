<#
.SYNOPSIS
    Builds the BioCPPNet Docker container, downloads external training data, trains the model, and evaluates it.

.DESCRIPTION
    This script automates the entire end-to-end process:
    1. Tears down any existing containers to ensure a clean state.
    2. Builds the Docker image via docker-compose.
    3. Starts the container in detached mode.
    4. Downloads a specified amount of external data from the Earth Species Project.
    5. Executes the model training loop.
    6. Runs the evaluation/testing metric.
    7. Cleans up by stopping the container.

.PARAMETER NumTrainingFiles
    The number of mono audio files to download from the Earth Species Project for training.
    Default is 50. Keep in mind that downloading large amounts of data may take time.
#>

param (
    [int]$NumTrainingFiles = 50
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " BioCPPNet: Docker Build & Train Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 1. Clean up existing containers
Write-Host "`n[1/5] Tearing down existing containers..." -ForegroundColor Yellow
docker-compose down

# 2. Build the Docker image
Write-Host "`n[2/5] Building Docker image..." -ForegroundColor Yellow
docker-compose build

# 3. Start the container in detached mode
Write-Host "`n[3/5] Starting container..." -ForegroundColor Yellow
docker-compose up -d

# Give the container a second to initialize
Start-Sleep -Seconds 2

try {
    # 4. Download External Data
    Write-Host "`n[4/5] Downloading $NumTrainingFiles external audio files from Earth Species Project..." -ForegroundColor Yellow
    # Ensure the data directory is writable by the container
    docker-compose exec -u root biocppnet mkdir -p data/raw
    docker-compose exec -u root biocppnet chown -R appuser:appuser data/
    
    # Download the data using the CLI
    docker-compose exec biocppnet python manage.py download-data --split "train" --limit $NumTrainingFiles

    # 5. Train the Model
    Write-Host "`n[5/5] Training the model..." -ForegroundColor Yellow
    # Assuming 'manage.py train' looks for files in data/raw
    docker-compose exec biocppnet python manage.py train

    # 6. Evaluate / Test
    Write-Host "`n[6/5] Evaluating the trained model..." -ForegroundColor Yellow
    docker-compose exec biocppnet python manage.py evaluate

} catch {
    Write-Host "`nAn error occurred during execution:" -ForegroundColor Red
    Write-Error $_
} finally {
    # 7. Teardown
    Write-Host "`n[Cleanup] Stopping and removing container..." -ForegroundColor Yellow
    docker-compose down
}

Write-Host "`n========================================" -ForegroundColor Green
Write-Host " Finished Successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
