# GCC-PHAT (Generalized Cross-Correlation with Phase Transform)

The **GCC-PHAT** algorithm is the standard method for robust Time Difference of Arrival (TDOA) estimation in reverberant environments. It is critical for the "Blind Beamforming" capabilities of the BioCPPNet system.

## Mathematical Formulation

The core idea is to find the time lag $	au$ that maximizes the cross-correlation between two microphone signals $x_1(t)$ and $x_2(t)$.

### 1. Cross-Correlation via FFT
The cross-correlation function $R_{12}(	au)$ can be computed efficiently in the frequency domain:
$$ R_{12}(	au) = 	ext{IFFT}( X_1(\omega) \cdot X_2^*(\omega) ) $$
where $X_1(\omega)$ and $X_2(\omega)$ are the Fourier Transforms of the signals.

### 2. The Phase Transform (PHAT) Weighting
In reverberant environments, strong reflections create multiple peaks in the cross-correlation function. The PHAT weighting whitens the spectrum, making the peak sharper and more robust to reverberation:
$$ G_{PHAT}(\omega) = \frac{1}{|X_1(\omega) \cdot X_2^*(\omega)|} $$

The Generalized Cross-Correlation is then:
$$ R_{GCC-PHAT}(	au) = 	ext{IFFT}\left( \frac{X_1(\omega) \cdot X_2^*(\omega)}{|X_1(\omega) \cdot X_2^*(\omega)|} 
ight) $$

Note that the term inside the IFFT is purely the **phase difference** between the signals ($e^{j(\phi_1 - \phi_2)}$), discarding magnitude information.

### 3. TDOA Estimation
The estimated time delay is the lag that maximizes this function:
$$ \hat{	au} = \arg\max_	au R_{GCC-PHAT}(	au) $$

## Sub-sample Precision

Since the peak index $\hat{	au}$ is an integer (sample index), the resolution is limited to $\pm 1/f_s$. For a 250kHz system with 4cm spacing, delays are tiny (~116 $\mu s$ max), so sub-sample precision is critical.

### Parabolic Interpolation
We fit a parabola to the peak and its two neighbors ($R[	au-1], R[	au], R[	au+1]$) to find the fractional peak location:
$$ \delta = \frac{R[	au-1] - R[	au+1]}{2(R[	au-1] - 2R[	au] + R[	au+1])} $$
$$ \hat{	au}_{sub} = \hat{	au} + \delta $$

## Implementation Steps

1.  **Frame Segmentation:** Divide continuous audio into short frames (e.g., 20-50ms).
2.  **Windowing:** Apply Hamming/Hanning window to reduce spectral leakage.
3.  **FFT:** Compute `rfft` of both channels.
4.  **Cross-Power Spectrum:** Calculate $P = X_1 \cdot X_2^*$.
5.  **PHAT Weighting:** Normalize $P$ by $|P|$. Add epsilon $\epsilon$ to avoid division by zero.
6.  **IFFT:** Compute `irfft` to get correlation vector.
7.  **Peak Finding:** `argmax` + Parabolic Interpolation.
8.  **Output:** Estimated delay in samples.

## Applications in BioCPPNet
-   **Blind Beamforming:** Automatically steer the array towards the loudest source without knowing its location beforehand.
-   **DoA Estimation:** Convert TDOA to Azimuth angle using array geometry ($	heta = \arcsin(c \cdot 	au / d)$).
