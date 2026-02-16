# BioCPPNet Architecture

## Goals
-   Source separation of overlapping bioacoustic signals.
-   Handling high-frequency sampling rates (250kHz).
-   Integration of spatial filtering (Beamforming) with deep learning (U-Net/DAE).

## System Overview

```mermaid
graph TD
    Data[Raw Audio Data] --> Mixer[DataMixer: Synthetic Mixing]
    Mixer --> Mixed[Mixed Audio]
    Mixed --> Beam[Beamformer: Spatial Filtering]
    Beam --> PreProcess[Denoising Autoencoder (DAE)]
    PreProcess --> Clean[Denoised Signal]
    Clean --> Model[BioCPPNet: U-Net Separation]
    Model --> Output[Isolated Sources]
    Output --> Metric[SI-SDR Metric]
    Clean --> Metric
```

## Python Module Diagram

```mermaid
classDiagram
    class DataMixer {
        +mix_signals(signal1, signal2, snr_db)
        +add_noise(signal, type, snr)
        +spatialise_signal(mono, azimuth)
    }

    class BioCPPNet {
        +forward(x)
        -_conv_block(in_ch, out_ch)
    }
    
    class DAE {
        +forward(x)
        +denoise(noisy_signal)
    }

    class Beamformer {
        +estimate_doa(multichannel_signal)
        +delay_and_sum(multichannel_signal, azimuth)
    }
    
    class NoiseGenerator {
        +generate(duration)
    }

    DataMixer ..> NoiseGenerator : Uses
    DataMixer ..> BioCPPNet : Generates Training Data
    Beamformer ..> BioCPPNet : Pre-processes Input
```

## Directory Structure

```text
BioaccousticCPP/
├── data/                  # Raw and processed audio data
├── docs/                  # Project documentation & Kanban board
│   └── BioCPPNet/         # Obsidian Vault
├── results/               # General project output
├── src/                   # Source code
│   ├── models/            # Deep Learning Models
│   ├── spatial/           # Spatial Audio (Beamforming/Physics)
│   ├── metrics/           # Evaluation Metrics
│   ├── data_mixer.py      # Data Augmentation/Mixing
│   ├── noise_models.py    # Noise Generators (White, Pink, Rain)
│   ├── utils.py           # Logging & Plotting
│   └── main.py            # Main entry point
├── tests/                 # Unit tests
├── unit test results/     # Test execution reports
│   └── session_TIMESTAMP/ # Per-run output folder (plots/logs)
├── manage.py              # CLI management tool
└── project_config.yaml    # Configuration
```

## Components

### 1. Data Pipeline (`src/data_mixer.py`)
-   **Input:** Isolated vocalizations (WAV).
-   **Process:** Synthetic mixing at various SNRs + Spatialization (Virtual Array).
-   **Noise Models:** White, Pink/Brown (Wind), Rain (Impulsive).
-   **Output:** Multichannel WAV files.

### 2. Core Models
-   **[[Model_UNet|BioCPPNet U-Net]] (`src/models/unet.py`):** The main source separation model.
-   **[[Model_DAE|Denoising Autoencoder]] (`src/models/dae.py`):** Optional pre-processing stage for noise reduction.

### 3. Spatial Processing (`src/spatial/beamforming.py`)
-   **Algorithm:** [[Algorithm_GCC_PHAT|GCC-PHAT]] for TDOA/DoA estimation.
-   **Beamforming:** Delay-and-Sum.
-   **Physics:** Sub-sample delay simulation via FFT phase shifting.
-   **Hardware:** 16-channel array (simulated or miniDSP UMA-16).

### 4. Validation (`src/metrics/sisdr.py`)
-   **Metric:** Scale-Invariant Signal-to-Distortion Ratio (SI-SDR).
-   **Downstream:** Classification accuracy.

## Configuration
-   Managed via `project_config.yaml`.
-   **Logging:** All logs and plots are timestamped per session to ensure reproducibility.
