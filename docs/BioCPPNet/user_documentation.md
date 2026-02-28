# User Documentation

This guide explains how to set up and utilize the various scripts and tools within the **BioCPPNet** project.

---

## 📋 Prerequisites

Before running the scripts, ensure you have the following installed:

1.  **Python 3.11+**
2.  **ffmpeg:** Required for all audio/video recording and processing.
    -   Download from: [ffmpeg.org/download.html](https://ffmpeg.org/download.html)
    -   Ensure it is added to your system's `PATH`.
3.  **Docker & Docker Compose:** Recommended for training and consistent environments.

---

## 🛠️ Local Setup

1.  **Clone the repository** and navigate to the project root.
2.  **Install dependencies:**
    ```powershell
    # Core requirements for pipeline and models
    pip install -r requirements.txt

    # Developer requirements (Recording tools, UI, Testing, Linting)
    pip install -r requirements-dev.txt
    ```

---

## 🚀 Script Reference

### 1. Management CLI (`manage.py`)
The primary entry point for project tasks. Use `python manage.py [command]`.

| Command | Description |
| :--- | :--- |
| `demo` | Launches the interactive Gradio web demo on `localhost:8502`. |
| `download_data` | Downloads isolated vocalizations from the Earth Species Project. |
| `train` | Starts the model training loop based on `project_config.yaml`. |
| `evaluate` | Runs a synthetic SI-SDR benchmark to test the current model weights. |
| `mix_data` | Generates synthetic multichannel mixtures from raw audio. |
| `test` | Executes the unit test suite and saves results to `unit test results/`. |
| `lint` | Runs `ruff` to check code quality and formatting. |

### 2. Interactive Demo (`app.py`)
You can launch the demo directly or via the CLI. It allows you to upload multichannel files and visualize the separation process.
```bash
python manage.py demo
```

### 3. Build & Train Automation (`build_and_train.ps1`)
A PowerShell script that handles the entire Docker workflow:
1.  Rebuilds the Docker image.
2.  Downloads 50 training files from the Earth Species Project.
3.  Executes the training loop inside the container.
4.  Runs final evaluation.
```powershell
.\build_and_train.ps1 -NumTrainingFiles 100
```

### 4. Acoustic Camera Generation (`run_acoustic_camera.ps1`)
A PowerShell script that processes synchronized `.wav` and `.avi` files in the `recordings/outside` directory. It calculates a 2D spatial acoustic heatmap for each second of audio and cleanly overlays it onto the corresponding video frame.
- **Environment:** Executes securely inside the `biocppnet` Docker container using `docker-compose run`.
- **Output:** Annotated video frames are saved as JPG images in `results/acoustic_camera/`.
```powershell
.\run_acoustic_camera.ps1
```

---

## 🎤 Data Collection Tools

The project provides two methods for capturing synchronized multichannel audio and video.

### A. Advanced GUI Recorder (`recorder_app.py`)
A PyQt6-based application designed for real-time monitoring and high-precision recording.

**Key Features:**
-   **Live Video Preview:** Monitor your camera feed.
-   **Multichannel Support:** Automatically detects and displays the number of input channels (e.g., for 16-channel arrays).
-   **High-Resolution Plots:** Real-time scrolling waveform for up to 16 channels with distinct color coding, and FFT Spectrum Analyzer.
-   **Auto-Scale Y-Axis:** Optional dynamic vertical scaling for inspecting low-amplitude signals.
-   **Configurable Output:** Select your output directory and devices via a graphical interface.

**Requirements:**
-   `PyQt6`: GUI framework.
-   `sounddevice`: Audio input streaming.
-   `opencv-python`: Video capture.
-   `pyqtgraph`: High-performance plotting.
-   `numpy`: Data processing.

**Usage:**
```bash
python recorder_app.py
```

### B. Simple Command-Line Recorder (`record_av.ps1`)
A lightweight PowerShell wrapper around `ffmpeg` for quick, synchronized capture.

**Usage:**
1.  Open PowerShell in the project root.
2.  Run the script: `.ecord_av.ps1`.
3.  Follow the prompts to select your audio and video devices.
4.  Press **'q'** in the terminal to stop and save files to the `recordings/` folder.

---

## ⚙️ Configuration
Most scripts rely on `project_config.yaml`. You can modify this file to change:
-   **Sample Rate:** (e.g., 48,000 or 250,000 Hz).
-   **Array Geometry:** Microphone coordinates for beamforming.
-   **Training Parameters:** Epochs, batch size, and learning rate.
-   **Paths:** Directories for logs, results, and data.
