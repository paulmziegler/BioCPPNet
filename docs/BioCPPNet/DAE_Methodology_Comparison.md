# Denoising Autoencoder Methodology Comparison

This document evaluates different methodologies for implementing a Denoising Autoencoder (DAE) for ultrasonic bioacoustic signals (250kHz sampling rate).

## 1. Methodologies

### A. Spectrogram-based Convolutional DAE (Recommended)
-   **Input:** Log-magnitude STFT Spectrogram.
-   **Architecture:** 2D Convolutional Encoder-Decoder (Bottleneck).
-   **Reconstruction:** Inverse STFT (ISTFT) using the *noisy* phase or Griffin-Lim.
-   **Pros:**
    -   Leverages mature 2D CNN techniques (efficiency, stability).
    -   Bioacoustic signals (chirps/pulses) have distinct T-F structures that are easy for CNNs to learn.
    -   Lower computational dimensionality compared to raw waveform (e.g., 512 frequency bins vs. 1024 raw samples).
-   **Cons:**
    -   Discarding phase information can limit performance in low SNR or overlapping speech scenarios.
    -   Requires tuning STFT parameters (Window size, Hop length).

### B. Time-Domain / Waveform DAE (e.g., Wave-U-Net, Demucs)
-   **Input:** Raw 1D waveform.
-   **Architecture:** 1D Convolutional Encoder-Decoder with large receptive fields (dilated convolutions).
-   **Pros:**
    -   End-to-end learning; preserves phase information implicitly.
    -   No STFT parameter tuning required.
-   **Cons:**
    -   **Extremely high dimensionality:** 1 second of audio = 250,000 samples.
    -   Requires very deep networks or large downsampling factors to capture long-term dependencies (e.g., reverb tails).
    -   Training is significantly slower and more memory-intensive.

### C. Complex Spectrogram DAE (Deep Complex U-Net)
-   **Input:** Complex STFT (Real + Imaginary parts).
-   **Architecture:** Complex-valued Convolutions.
-   **Pros:** Preserves phase while maintaining T-F structure.
-   **Cons:** Higher implementation complexity and computational cost than real-valued CNNs.

## 2. Recommendation

**Selected Approach: Spectrogram-based Convolutional DAE**

**Rationale:**
1.  **Efficiency:** Processing 250kHz audio is computationally demanding. STFT reduces the temporal resolution by a factor of ~256 (hop length), making training feasible on standard GPUs.
2.  **Nature of Signals:** Ultrasonic bioacoustic calls (echolocation) are highly tonal and sparse in the frequency domain, making them ideal candidates for spectrogram masking or reconstruction.
3.  **Phase Tolerance:** For *denoising* (removing wind/rain), magnitude reconstruction is usually sufficient. Phase errors typically manifest as slight roughness, which is acceptable compared to the massive noise reduction gain.

## 3. Implementation Details

-   **STFT Parameters:**
    -   `n_fft`: 1024 (256 frequency bins? No, 1024/2 + 1 = 513 bins).
    -   `hop_length`: 512 (50% overlap).
    -   `window`: Hann.
-   **Normalization:** Log-magnitude ($log(1 + |X|)$) normalized to [0, 1] or mean/std.
-   **Loss Function:** MSE on Log-Magnitude Spectrograms.
