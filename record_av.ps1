<#
.SYNOPSIS
    Records synchronized, multichannel audio and video from user-selected devices using ffmpeg.

.DESCRIPTION
    This script automates the process of capturing high-quality, synchronized audio and video, which is ideal for bioacoustic research.
    1.  It first checks if ffmpeg is installed and accessible.
    2.  It lists all available audio input and video capture devices found on the system.
    3.  It prompts the user to select the desired audio and video device from the lists.
    4.  It constructs and runs an ffmpeg command to record simultaneously from both devices.
    5.  Audio is saved as a high-quality, multichannel WAV file.
    6.  Video is saved as a high-quality MKV file.
    7.  The script runs until the user presses 'q' in the ffmpeg window to stop the recording.

.NOTES
    Prerequisite: ffmpeg must be installed on the system and accessible via the system's PATH.
    You can download ffmpeg from https://ffmpeg.org/download.html
#>

$ErrorActionPreference = "Stop"

# --- Configuration ---
$OutputDirectory = "recordings"

# --- Main Script ---

# 1. Verify ffmpeg is installed
try {
    Write-Host "Checking for ffmpeg installation..."
    Get-Command ffmpeg -ErrorAction Stop | Out-Null
    Write-Host "ffmpeg found." -ForegroundColor Green
}
catch {
    Write-Host "Error: ffmpeg not found." -ForegroundColor Red
    Write-Host "Please install ffmpeg from https://ffmpeg.org/download.html and ensure it's in your system's PATH." -ForegroundColor Red
    exit 1
}

# Create output directory if it doesn't exist
if (-not (Test-Path -Path $OutputDirectory)) {
    New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host " BioCPPNet: Audio/Video Recorder" -ForegroundColor Cyan
Write-Host "========================================"

# 2. Detect Devices
Write-Host "`n[1/3] Detecting available audio and video devices..." -ForegroundColor Yellow
try {
    # Use ffmpeg to list DirectShow devices. The output is sent to the error stream, so we redirect it.
    $deviceList = (ffmpeg -list_devices true -f dshow -i dummy 2>&1)
}
catch {
    Write-Host "Error running ffmpeg to list devices. Ensure it is working correctly." -ForegroundColor Red
    exit 1
}

# Parse the output to find device names
$audioDevices = [System.Collections.ArrayList]@()
$videoDevices = [System.Collections.ArrayList]@()
$isAudioSection = $false
$isVideoSection = $false

foreach ($line in ($deviceList -split '[
]+')) {
    if ($line -like "*DirectShow audio devices*") { $isAudioSection = $true; $isVideoSection = $false; continue }
    if ($line -like "*DirectShow video devices*") { $isAudioSection = $false; $isVideoSection = $true; continue }

    if ($line -match '"([^"]+)"') {
        $deviceName = $matches[1]
        if ($isAudioSection -and $deviceName -notlike "*audio capture sources*") {
            $null = $audioDevices.Add($deviceName)
        }
        if ($isVideoSection -and $deviceName -notlike "*video capture sources*") {
            $null = $videoDevices.Add($deviceName)
        }
    }
}

# 3. Prompt User for Selection
Write-Host "`n[2/3] Please select your recording devices." -ForegroundColor Yellow

# --- Audio Device Selection ---
Write-Host "`nAvailable Audio Devices:" -ForegroundColor Green
for ($i = 0; $i -lt $audioDevices.Count; $i++) {
    Write-Host "  [$($i+1)] $($audioDevices[$i])"
}
$audioChoice = 0
while ($audioChoice -lt 1 -or $audioChoice -gt $audioDevices.Count) {
    try {
        $audioChoice = [int](Read-Host "`nEnter the number for your audio device")
    }
    catch {
        Write-Host "Invalid input. Please enter a number." -ForegroundColor Red
    }
}
$selectedAudioDevice = $audioDevices[$audioChoice - 1]

# --- Video Device Selection ---
Write-Host "`nAvailable Video Devices:" -ForegroundColor Green
for ($i = 0; $i -lt $videoDevices.Count; $i++) {
    Write-Host "  [$($i+1)] $($videoDevices[$i])"
}
$videoChoice = 0
while ($videoChoice -lt 1 -or $videoChoice -gt $videoDevices.Count) {
    try {
        $videoChoice = [int](Read-Host "`nEnter the number for your video device")
    }
    catch {
        Write-Host "Invalid input. Please enter a number." -ForegroundColor Red
    }
}
$selectedVideoDevice = $videoDevices[$videoChoice - 1]

# 4. Construct and Execute ffmpeg Command
Write-Host "`n[3/3] Preparing to record..." -ForegroundColor Yellow
Write-Host "  Audio Device: $selectedAudioDevice"
Write-Host "  Video Device: $selectedVideoDevice"

$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$audioOutputFile = Join-Path -Path $OutputDirectory -ChildPath "recording_audio_$timestamp.wav"
$videoOutputFile = Join-Path -Path $OutputDirectory -ChildPath "recording_video_$timestamp.mkv"

Write-Host "`nAudio will be saved to: $audioOutputFile"
Write-Host "Video will be saved to: $videoOutputFile"

Write-Host "`n-----------------------------------------------------" -ForegroundColor Magenta
Write-Host "  Press 'q' in THIS window to stop recording." -ForegroundColor Magenta
Write-Host "-----------------------------------------------------`n"

$ffmpegArgs = @(
    "-f", "dshow",
    "-i", "audio=`"$selectedAudioDevice`"",
    "-f", "dshow",
    "-i", "video=`"$selectedVideoDevice`"",
    "-c:a", "pcm_s32le", # High-quality uncompressed audio for WAV
    "`"$audioOutputFile`"",
    "-c:v", "libx264",   # High-compatibility video codec
    "-preset", "ultrafast",
    "-crf", "18",        # Visually lossless quality
    "`"$videoOutputFile`""
)

# Start ffmpeg process
try {
    Start-Process ffmpeg -ArgumentList $ffmpegArgs -Wait
    Write-Host "`nRecording stopped." -ForegroundColor Green
}
catch {
    Write-Host "`nAn error occurred while running ffmpeg:" -ForegroundColor Red
    Write-Error $_
}

Write-Host "`n========================================" -ForegroundColor Green
Write-Host " Recording complete."
Write-Host "========================================"
