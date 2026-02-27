# Welcome to BioCPPNet

**BioCPPNet** is a bioacoustic source separation project designed to solve the Cocktail Party Problem for non-human vocalizations. This vault serves as the central knowledge base for the project.

## 📂 Navigation

-   [[architecture|Architecture Overview]]: Understanding the system design, folder structure, and Python modules.
-   [[Signal_Specification|Signal Specification]]: Sampling rates (250kHz) and data formats.
-   [[Data_Augmentation_Strategy|Data Augmentation Strategy]]: Closing the reality gap with reverb and noise.
-   [[KANBAN|Project Kanban Board]]: Track tasks, bugs, and feature requests.
-   [[Evaluation_Report|Project Evaluation Report]]: Read the latest performance metrics and training outcomes.

### Core Models & Algorithms
-   [[Model_UNet|BioCPPNet U-Net]]: The main source separation model.
-   [[Model_DAE|Denoising Autoencoder]]: The noise reduction pre-processor.
-   [[Algorithm_GCC_PHAT|GCC-PHAT]]: The blind beamforming and TDOA algorithm.
-   [[Algorithm_MUSIC|MUSIC Algorithm]]: High-resolution direction of arrival estimation.

## 🚀 Getting Started

### Prerequisites
- **ffmpeg:** You must install ffmpeg and add it to your system's PATH. Download it from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html).
- Docker & Docker Compose
- Python 3.11+ (for local development)

### Recording Real-World Data
The project includes multiple tools for capturing synchronized, multichannel audio and video:

#### 1. Simple Command-Line Recording (using `record_av.ps1`)
This script uses `ffmpeg` for straightforward audio/video capture without a live preview.
1.  **Open PowerShell:** Navigate to the project's root directory.
2.  **Run the script:**
    ```powershell
    .\\record_av.ps1
    ```
3.  **Select Devices:** Follow the on-screen prompts to choose your audio interface and camera.
4.  **Stop Recording:** Press 'q' in the terminal window to stop. Your files will be saved in the `recordings/` directory.

#### 2. Advanced GUI Recording with Real-time Previews (using `recorder_app.py`)
This PyQt6 application provides a comprehensive real-time monitoring solution for your recording sessions.
- **Live Video Preview:** See the feed from your selected camera to ensure it's positioned correctly.
- **Dual Audio Plots:** Monitor the incoming audio signals in both the time domain (scrolling waveform) and the frequency domain (real-time FFT spectrum analyzer).
- **Configurable Output:** Use the "Browse" button to select a custom folder for your recordings.

**Instructions:**
1.  **Install Dev Dependencies:** Ensure you've run `pip install -r requirements-dev.txt` to install `PyQt6`, `sounddevice`, `opencv-python`, and `pyqtgraph`.
2.  **Run the Application:**
    ```bash
    python recorder_app.py
    ```
3.  **Select Devices & Folder:** Use the dropdown menus to select your devices and choose an output directory.
4.  **Start/Stop Recording:** Click the 'Start Recording' button. Click 'Stop Recording' to save your synchronized audio and video files.

### Quick Start (Docker)
Run the application using Docker Compose:
```bash
docker-compose up --build -d
```

### Local Development
1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```
2. **Setup Pre-commit Hooks:**
   ```bash
   pre-commit install
   ```
3. **Run the Interactive Demo:**
   ```bash
   python manage.py demo
   ```
   Then open your browser to `http://localhost:8502`.

## 🛠️ Development Tools

- **Linting:** We use `ruff` for fast Python linting.
  ```bash
  docker-compose run biocppnet python manage.py lint
  ```
- **Testing:** We use `pytest` for unit testing.
  ```bash
  docker-compose run biocppnet python manage.py test
  ```
- **CLI Management:** The `manage.py` script provides project-specific commands (e.g., `mix_data`, `train`).

## 📝 Notes
- This documentation is built to be viewed in **Obsidian**.
- The `docs/` folder is mapped directly to this vault.
