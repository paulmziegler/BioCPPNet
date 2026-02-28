# Data Augmentation Strategy: Closing the Reality Gap

To train a robust Bioacoustic AI that functions in real-world environments (forests, caves), our synthetic data pipeline must simulate more than just anechoic mixing. This document outlines the "Dimensions of Freedom" required to bridge the gap between simulation and reality.

## 1. The Reality Gap Analysis

| Dimension | Simulation (Current) | Reality (Target) | Impact |
| :--- | :--- | :--- | :--- |
| **Reverberation** | Anechoic (Free-field) | Multipath, Echoes, RT60 > 0.5s | **High:** Echoes confuse simple TDOA estimators. |
| **Mic Matching** | Perfect Gain/Phase | Gain $\pm 1$dB, Phase Jitter | **Medium:** Degradation of beamforming null depth. |
| **Source Motion** | Stationary | Moving (Doppler, Changing TDOA) | **Low/Medium:** Critical for long calls or fly-bys. |
| **Spectral Variance** | Fixed Pitch | Pitch Shift, Time Stretch | **Medium:** Generalization across individuals/species. |
| **Multi-Source** | 1 Target + 1 Noise | $N$ Concurrent Sources | **High:** The true "Cocktail Party" problem. |

## 2. Implementation Strategy

### Hybrid Generation
We employ a hybrid approach to data generation to balance training variability with evaluation reproducibility.

*   **Training (Online):**
    *   **Method:** A PyTorch `Dataset` generates mixed/augmented audio *on-the-fly*.
    *   **Benefit:** The model never sees the exact same noise/reverb combination twice. Infinite dataset size.
    *   **Cost:** Higher CPU usage (FFT convolution/delays).

*   **Validation (Offline):**
    *   **Method:** Generate a fixed "Golden Set" of ~10 hours of audio saved to disk.
    *   **Benefit:** Consistent benchmarks (SI-SDR) to track model improvements over time.

### 3. Augmentation Techniques

#### A. Reverberation (Convolutional Reverb)
*   **Technique:** Convolve clean sources with a Room Impulse Response (RIR).
*   **Source:**
    *   *Synthetic:* Image Source Method (fast, geometric).
    *   *Real:* Recorded RIRs from forests/caves (e.g., using a starter pistol).
*   **Action:** Add `ReverbGenerator` to `src/spatial/`.

#### B. Microphone Mismatch
*   **Technique:** Apply random gain and phase offsets to each channel independently.
    *   $$ x_i(t) = (1 + \Delta g_i) \cdot x(t - \Delta 	au_i) $$
    *   Gain $\Delta g \sim U(-1dB, +1dB)$.
    *   Phase/Time Jitter $\Delta 	au \sim U(-5\mu s, +5\mu s)$.

#### C. Source Movement (Trajectory)
*   **Technique:** Time-varying delay lines.
*   **Implementation:** Update steering vectors every frame or use interpolation for continuous movement.

#### D. Spectral Augmentation
*   **Technique:**
    *   **Pitch Shift:** $\pm 10\%$ to simulate different animal sizes.
    *   **Time Stretch:** $\pm 10\%$ to simulate call duration variance.
*   **Library:** `torchaudio` or `librosa` (phase vocoder).

## 4. Phase 2: Cocktail Party Dataset Generation

For full deep learning separation (Phase 2), the `BioAcousticDataset` has been configured to simulate the complex "Cocktail Party" scenario natively:

*   **Multi-Source Overlap:** The dataset dynamically mixes a clean target source with $N$ (default: 1) competing interferer vocalizations, each assigned a random, distinct spatial azimuth.
*   **Dynamic SNR Scaling:** Interferers are scaled to have a relative SNR between -5 dB and +5 dB compared to the target, ensuring a highly challenging and realistic overlap scenario.
*   **Reverberant Interference:** Both the target and the interfering sources undergo stochastic reverb convolution (RT60 $\approx$ 0.1s - 0.5s) before reaching the virtual microphone array.
*   **Environmental Noise:** After spatial mixing, pink noise (simulating wind/thermal noise) is injected at an SNR of 10 to 20 dB to simulate realistic recording conditions.
*   **Target Output:** The dataset yields a tuple containing the `(noisy_mixture, clean_reference_target, target_azimuth)` to facilitate both spatial beamforming and direct mask estimation by the U-Net.

## 5. Pipeline Update Plan

1.  **Refactor `DataMixer`:** Support list of $N$ sources (Target + List[Interference]). *(Completed)*
2.  **Add `ReverbGenerator`:** Implement RIR convolution. *(Completed)*
3.  **Add `SensorPerturbation`:** Implement random gain/phase errors. *(Completed)*
4.  **Create `OnlineDataset`:** PyTorch wrapper for the `DataMixer`. *(Completed)*
