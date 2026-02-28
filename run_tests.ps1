$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " BioCPPNet: Docker Test Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

docker-compose down
docker-compose build
docker-compose up -d

try {
    Write-Host "`n[1/2] Installing dev dependencies..." -ForegroundColor Yellow
    docker-compose exec -u root biocppnet pip install -r requirements-dev.txt
    
    Write-Host "`n[2/2] Running PyTest..." -ForegroundColor Yellow
    docker-compose exec biocppnet pytest tests/test_cocktail_mixer.py tests/test_online_dataset.py
} finally {
    docker-compose down
}
Write-Host "`n========================================" -ForegroundColor Green
Write-Host " Finished Successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
