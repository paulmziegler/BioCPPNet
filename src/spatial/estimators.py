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

    
