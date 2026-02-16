# Signal Specification

This document outlines the technical requirements for acoustic signal processing within the BioCPPNet project.

## 1. Sampling Rate
- **Target Rate:** 250,000 Hz (250 kHz).
- **Rationale:** Necessary to capture high-frequency bioacoustic vocalizations (e.g., bats, macaques) which can exceed 100 kHz. Adheres to the Nyquist theorem to prevent aliasing.
- **Hardware Target:** miniDSP UMA-16 or equivalent high-speed multichannel arrays.

## 2. Signal Representation

### Time Domain (Input/Output)
- **Type:** **Real-valued**.
- **Physical Meaning:** Fluctuations in air pressure (compression and rarefaction).
- **Storage Format:** PCM WAV files, 32-bit float (`float32`).

### Frequency Domain (Processing)
- **Type:** **Complex-valued**.
- **Usage:** 
    - **Precise Delays:** Beamforming and spatial simulation require sub-sample precision. This is achieved by applying phase shifts in the frequency domain ($e^{-j \omega 	au}$) to avoid the quantization errors inherent in simple time-domain sample shifting.
    - **STFT:** The U-Net model typically operates on the magnitude or complex-valued Short-Time Fourier Transform (STFT) of the signals.

## 3. Mathematical Precision
- **Internal Math:** All spatial steering and phase manipulations are performed using complex-valued arithmetic to ensure phase coherence across the multichannel array.
- **Inversion:** Final beamformed signals or separated sources are converted back to real-valued time-domain signals via Inverse FFT (IFFT) for evaluation and listening.
