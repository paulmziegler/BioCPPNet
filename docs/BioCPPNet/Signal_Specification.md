# Signal Specification

This document outlines the technical requirements for acoustic signal processing within the BioCPPNet project.

## 1. Sampling Rate
- **Default Rate:** 48,000 Hz (48 kHz).
- **Configurable:** The project sample rate is now fully configurable via `project_config.yaml`. All modules, including the physics engine, data mixer, and deep learning pipeline, will dynamically adapt to the rate specified in the configuration file.
- **Rationale:** The original target of 250kHz was to capture high-frequency bioacoustics. The new default of 48kHz is a common professional audio standard that is compatible with a wider range of microphones and recording devices. The system remains capable of handling rates up to and beyond 250kHz.
- **Hardware Target:** miniDSP UMA-16 (for high-speed) or any standard multichannel audio interface.

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

## 4. Array Coordinate System

The physics engine (`src/spatial/physics.py`) uses a strict Cartesian coordinate system to calculate steering vectors and delays. All coordinates must be provided in **meters**.

### Orientation
- **+X Axis:** Forward (Azimuth 0°). This is the direction the array is "looking."
- **+Y Axis:** Left (Azimuth 90°).
- **+Z Axis:** Up (Elevation 90°).

### Mapping Real-World Measurements
If you measure your hardware by standing in front of it and looking at the face of the array, you must transform your measurements before entering them into `project_config.yaml`:

1.  **Depth (Forward/Backward):** Becomes the **X** coordinate. Flat arrays will have `X = 0.0`.
2.  **Horizontal (Left/Right):** Becomes the **Y** coordinate. **Crucially**, because you are looking at the array, your "Right" is the array's "Left". Therefore, you must **multiply your horizontal measurements by -1** (e.g., a mic 5cm to your right is at `Y = -0.05`).
3.  **Vertical (Up/Down):** Becomes the **Z** coordinate (e.g., a mic 5cm up is `Z = 0.05`).
