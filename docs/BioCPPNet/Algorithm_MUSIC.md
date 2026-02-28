# MUSIC Algorithm (MUltiple SIgnal Classification)

The **MUSIC** algorithm is a high-resolution Direction of Arrival (DoA) estimation technique. Unlike basic cross-correlation (like GCC-PHAT), MUSIC can estimate the directions of multiple simultaneous sources with sub-degree precision, making it highly valuable for complex spatial tracking scenarios in the BioCPPNet system.

## Mathematical Formulation

MUSIC operates by performing an eigendecomposition on the spatial covariance matrix of the signal to separate the signal subspace from the noise subspace.

### 1. Spatial Covariance Matrix
For a multichannel signal $X(t)$, we typically compute the Short-Time Fourier Transform (STFT) to isolate a specific frequency band or the dominant frequency bin $X(f)$.
The narrowband spatial covariance matrix $R$ is:
$$ R = E[X(f) X^H(f)] \approx \frac{1}{N} \sum_{n=1}^{N} X_n(f) X_n^H(f) $$
where $X_n(f)$ is the frequency-domain snapshot at frame $n$.

### 2. Eigendecomposition
We decompose $R$ into eigenvalues and eigenvectors:
$$ R = E \Lambda E^H $$
Sorting the eigenvalues in descending order allows us to partition the eigenvectors into:
- **Signal Subspace ($E_s$)**: The eigenvectors corresponding to the largest $K$ eigenvalues (where $K$ is the number of sources).
- **Noise Subspace ($E_n$)**: The remaining eigenvectors.

A fundamental property of MUSIC is that the true steering vectors of the sources are orthogonal to the noise subspace.

### 3. The MUSIC Pseudo-Spectrum
We define a steering vector $a(	heta)$ that represents the expected phase shifts for a plane wave arriving from azimuth $	heta$.
Because $a(	heta)$ is orthogonal to $E_n$ at the true source locations, the term $a^H(	heta) E_n E_n^H a(	heta)$ will be close to zero.
We invert this to create the MUSIC pseudo-spectrum:
$$ P_{MUSIC}(	heta) = \frac{1}{a^H(	heta) E_n E_n^H a(	heta)} $$

The estimated Directions of Arrival are the peaks in this spectrum.

## Implementation Details in BioCPPNet

Given the high sampling rate (250kHz) and broadband nature of bioacoustic signals, our implementation (`src.spatial.estimators.MUSIC`):
1. Computes the STFT of the incoming multichannel audio.
2. Identifies the **dominant frequency bin** (highest average energy).
3. Constructs the spatial covariance matrix $R$ for that specific frequency band.
4. Performs the eigendecomposition to isolate the noise subspace.
5. Scans across a highly granular azimuthal grid (-180 to 180 degrees) using the array's exact geometric configuration to evaluate $P_{MUSIC}(	heta)$.

### Applications
- **High-Resolution DoA:** Provides sub-degree accuracy compared to the sample-limited resolution of GCC-PHAT.
- **Initialization for Beamforming:** Can provide highly accurate steering targets for the Delay-and-Sum beamformer to feed into the U-Net.

## Comparison with GCC-PHAT in Cocktail Party Scenarios

When dealing with a true "Cocktail Party" scenario (many overlapping bioacoustic signals in a noisy environment), **MUSIC is expected to significantly outperform GCC-PHAT**.

### 1. Handling Multiple Simultaneous Sources
*   **MUSIC:** Explicitly designed for multiple signals. By decomposing the spatial covariance matrix, it can theoretically identify up to $N-1$ distinct sources (where $N$ is the number of microphones). Each source appears as a distinct peak in the pseudo-spectrum.
*   **GCC-PHAT:** Designed primarily for single-source tracking. When multiple signals overlap, their cross-correlation peaks blur together, and weaker sources are often completely masked by the loudest dominant signal.

### 2. Resolution and Precision
*   **MUSIC:** A "high-resolution" subspace method. It yields mathematically sharp peaks because the steering vectors are orthogonal to the noise subspace. This allows MUSIC to distinguish between sources that are physically very close together (sub-degree accuracy).
*   **GCC-PHAT:** Resolution is limited by the sampling rate, leading to broader spatial "lobes" rather than sharp peaks, even when using parabolic interpolation.

### 3. Environmental Noise vs. Reverberation
*   **MUSIC:** Highly effective at isolating uncorrelated background noise (like wind or thermal microphone noise), as this noise is mathematically pushed into the "Noise Subspace," isolating the directional calls.
*   **GCC-PHAT:** The PHAT weighting makes GCC-PHAT exceptionally robust to **reverberation** (echoes) by whitening the spectrum. It is often the better choice for tracking a *single* source in a highly reflective environment (like a cave).

### Summary
*   Use **MUSIC** for the BioCPPNet multi-source separation pipeline to accurately locate individual calls in a crowded, noisy environment (the "Cocktail Party").
*   Use **GCC-PHAT** for fast, computationally cheap "Blind Beamforming" to track a single, loud target in highly reverberant conditions.
