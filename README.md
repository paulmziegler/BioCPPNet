# Bioacoustic Cocktail Party Problem Solver (BioCPPNet)

This project tackles the **Bioacoustic Cocktail Party Problem** using a "local-first" containerized approach. It aims to separate overlapping animal vocalizations (e.g., rhesus macaques, Egyptian fruit bats) using deep learning and microphone array beamforming.

## 🚀 Project Overview

The system integrates:
1.  **Synthetic Data Pipeline:** Mixing isolated calls at varying SNRs.
2.  **BioCPPNet Core:** A U-Net architecture for source separation in the time-frequency domain.
3.  **Spatial Extension:** DoA estimation (MUSIC) and Delay-and-Sum beamforming for 16-channel arrays.
4.  **Validation:** SI-SDR metrics and downstream classification.

## 🛠️ Tech Stack

-   **Language:** Python 3.11
-   **Core Libraries:** `librosa`, `torch`, `scipy.signal`, `numpy`
-   **Containerization:** Docker & Docker Compose
-   **Configuration:** YAML (`project_config.yaml`)
-   **Management:** CLI (`manage.py`)

## 📂 Directory Structure

-   `src/`: Source code (`data_mixer.py`, `models/`, `spatial/`, `metrics/`)
-   `tests/`: Unit tests
-   `docs/`: Project documentation & Kanban board
-   `results/`: General project output and results
-   `data/`: Raw and processed audio data

## 🚦 Getting Started

### Prerequisites
-   Docker & Docker Compose
-   Python 3.11+ (optional for local dev)

### Quick Start (Docker)

1.  **Build the container:**
    ```bash
    docker-compose up --build -d
    ```

2.  **Run Management Commands:**
    ```bash
    # Run linter
    docker-compose run biocppnet python manage.py lint

    # Run tests
    docker-compose run biocppnet python manage.py test

    # Download data (placeholder)
    docker-compose run biocppnet python manage.py download_data

    # Generate synthetic mixtures
    docker-compose run biocppnet python manage.py mix_data

    # Train the model
    docker-compose run biocppnet python manage.py train --config project_config.yaml
    ```

## 📅 Roadmap (8-Week Plan)

-   **Week 1-2:** Data Engineering (Isolated call extraction & synthetic mixing)
-   **Week 3-4:** BioCPPNet Core (U-Net implementation & loss functions)
-   **Week 5-6:** Spatial Extension (DoA estimation & Beamforming)
-   **Week 7:** Integration (Beamformed signals -> U-Net)
-   **Week 8:** Evaluation & Documentation

## 📝 Documentation
Documentation is built for Obsidian:
-   [[KANBAN]]: Project tracking board.
-   [[architecture]]: System design.
