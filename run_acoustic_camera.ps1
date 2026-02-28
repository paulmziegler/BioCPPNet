<#
.SYNOPSIS
    Runs the acoustic camera generation script inside the Docker container.
.DESCRIPTION
    This script executes `generate_acoustic_camera.py` within the 'biocppnet' Docker container.
    It processes synchronized audio/video files to generate acoustic heatmaps overlayed on video frames.
#>

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " BioCPPNet: Acoustic Camera Generator" -ForegroundColor Cyan
Write-Host "========================================"

Write-Host "`nStarting Acoustic Camera generation via Docker..." -ForegroundColor Yellow

try {
    # Ensure the container environment is ready and run the script inside it.
    # The --rm flag ensures the temporary run container is removed after it finishes.
    docker-compose run --rm biocppnet python generate_acoustic_camera.py
    Write-Host "`nAcoustic Camera generation completed successfully." -ForegroundColor Green
    Write-Host "Check the 'results/acoustic_camera/' directory for your output images." -ForegroundColor Green
} catch {
    Write-Host "`nAn error occurred while running the script in Docker." -ForegroundColor Red
    Write-Error $_
    exit 1
}
