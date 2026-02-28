from abc import ABC, abstractmethod

import numpy as np


class DoAEstimator(ABC):
    """
    Abstract Base Class for Direction of Arrival (DoA) or Time Difference of
    Arrival (TDOA) estimators.
    """

    def __init__(self, sample_rate: int, mic_positions: np.ndarray):
        self.sample_rate = sample_rate
        self.mic_positions = mic_positions
        self.n_channels = len(mic_positions)

    @abstractmethod
    def estimate(self, signal: np.ndarray) -> np.ndarray:
        """
        Estimates direction or delays.
        
        Args:
            signal: (N_channels, N_samples) input audio.
            
        Returns:
            Estimated parameters (e.g., azimuth, or delays in seconds).
        """
        pass

class GCCPHAT(DoAEstimator):
    """
    Generalized Cross-Correlation with Phase Transform (GCC-PHAT).
    Estimates TDOA between pairs of microphones.
    """
    def __init__(
        self, sample_rate: int, mic_positions: np.ndarray, ref_channel: int = 0
    ):
        super().__init__(sample_rate, mic_positions)
        self.ref_channel = ref_channel

    def estimate(
        self, signal: np.ndarray, return_diagnostics: bool = False
    ) -> np.ndarray | tuple[np.ndarray, dict]:
        """
        Estimates time delays of all channels relative to the reference channel.

        Returns:
            (N_channels,) array of delays in seconds. ref_channel delay is 0.
            If return_diagnostics is True, returns (delays, diagnostics_dict).
        """
        n_samples = signal.shape[1]

        # FFT of all channels
        # (N_channels, N_freqs)
        # Using real-fft
        X = np.fft.rfft(signal, n=n_samples, axis=1)

        # Reference channel spectrum
        X_ref = X[self.ref_channel]

        # Cross-spectrum: X_i * conj(X_ref)
        R = X * np.conj(X_ref)

        # PHAT Weighting: 1 / |R|
        # Add epsilon to avoid div by zero
        eps = 1e-12
        W = 1.0 / (np.abs(R) + eps)

        # Generalized Cross Correlation
        GCC = R * W

        # IFFT to get cross-correlation in time domain
        # (N_channels, N_samples)
        cc = np.fft.irfft(GCC, n=n_samples, axis=1)

        # Shift so that 0 lag is at the center?
        cc = np.fft.fftshift(cc, axes=1)

        # Find peaks
        peaks = np.argmax(cc, axis=1)

        # Convert peaks to delays
        # After fftshift, index 0 is -N/2. Index N/2 is 0.
        # delay_samples = peak_index - N/2
        center = n_samples // 2
        delays_samples = peaks - center

        # Parabolic Interpolation for sub-sample precision
        delays_refined = np.zeros(self.n_channels)

        for i in range(self.n_channels):
            idx = peaks[i]
            # Boundary checks
            if 0 < idx < n_samples - 1:
                y0 = cc[i, idx - 1]
                y1 = cc[i, idx]
                y2 = cc[i, idx + 1]
                denom = 2 * (y0 - 2 * y1 + y2)
                if denom != 0:
                    delta = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
                    delays_refined[i] = delays_samples[i] + delta
                else:
                    delays_refined[i] = delays_samples[i]
            else:
                delays_refined[i] = delays_samples[i]

        # Convert to seconds
        delays = delays_refined / self.sample_rate

        if return_diagnostics:
            return delays, {"cc": cc, "peaks": peaks}
        return delays

class MUSIC(DoAEstimator):
    """
    MUltiple SIgnal Classification (MUSIC) algorithm.
    Provides high-resolution Direction of Arrival (DoA) estimation.
    Uses a narrowband approximation around the dominant frequency.
    """
    def __init__(
        self, sample_rate: int, mic_positions: np.ndarray, speed_of_sound: float = 343.0, num_sources: int = 1
    ):
        super().__init__(sample_rate, mic_positions)
        self.speed_of_sound = speed_of_sound
        self.num_sources = num_sources

    def estimate(
        self, signal: np.ndarray, search_resolution: float = 1.0
    ) -> float | list[float]:
        """
        Estimates the azimuth of the dominant source(s).

        Args:
            signal: (N_channels, N_samples) input audio.
            search_resolution: Resolution of the azimuth search space in degrees.

        Returns:
            Estimated azimuth in degrees.
        """
        import scipy.linalg
        import scipy.signal

        from src.spatial.physics import (
            azimuth_elevation_to_vector,
            calculate_steering_vector,
        )

        # 1. Compute STFT
        f, t, Zxx = scipy.signal.stft(signal, fs=self.sample_rate, nperseg=256)
        
        # 2. Find the dominant frequency bin
        energy = np.mean(np.abs(Zxx)**2, axis=(0, 2))
        dom_freq_idx = np.argmax(energy)
        dom_freq = f[dom_freq_idx]
        
        # Avoid DC component
        if dom_freq == 0 and len(f) > 1:
            dom_freq_idx = 1
            dom_freq = f[1]
            
        # 3. Extract narrowband signal
        X = Zxx[:, dom_freq_idx, :]
        
        # 4. Spatial Covariance Matrix R
        n_frames = X.shape[1]
        R = (X @ X.conj().T) / n_frames
        
        # 5. Eigen decomposition
        eigenvalues, eigenvectors = scipy.linalg.eigh(R)
        
        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        # 6. Extract Noise Subspace
        En = eigenvectors[:, self.num_sources:]
        En_En_H = En @ En.conj().T
        
        # 7. Search across azimuths (-180 to 180)
        azimuths = np.arange(-180, 180, search_resolution)
        P_music = np.zeros_like(azimuths, dtype=float)
        
        for i, az in enumerate(azimuths):
            source_vec = azimuth_elevation_to_vector(az, 0.0)
            distances = calculate_steering_vector(self.mic_positions, source_vec)
            delays = -distances / self.speed_of_sound
            
            # Steering vector for this frequency
            a = np.exp(-1j * 2 * np.pi * dom_freq * delays)
            a = a.reshape(-1, 1)
            
            # P_music = 1 / (a^H * En * En^H * a)
            denom = np.abs(a.conj().T @ En_En_H @ a)[0, 0]
            if denom > 1e-12:
                P_music[i] = 1.0 / denom
                
        # 8. Find peaks
        if self.num_sources == 1:
            peak_idx = np.argmax(P_music)
            return float(azimuths[peak_idx])
        else:
            peaks, _ = scipy.signal.find_peaks(P_music)
            if len(peaks) == 0:
                # Fallback if no peaks found
                return [float(azimuths[np.argmax(P_music)])]
            
            # Sort peaks by descending value of P_music
            sorted_peaks = peaks[np.argsort(P_music[peaks])][::-1]
            top_peaks = sorted_peaks[:self.num_sources]
            return [float(azimuths[p]) for p in top_peaks]

    
