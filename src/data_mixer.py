import numpy as np
import scipy.signal
from src.spatial.physics import (
    calculate_steering_vector,
    azimuth_elevation_to_vector,
    apply_subsample_shifts
)
from src.utils import CONFIG

class DataMixer:
    """
    Handles mixing of isolated audio sources to create synthetic training data.
    Supports high-frequency signals (up to 250kHz sampling rate).
    """
    def __init__(self, sample_rate: int = 250000):
        self.sample_rate = sample_rate
        
        # Load array configuration
        array_config = CONFIG.get("array", {})
        self.speed_of_sound = array_config.get("speed_of_sound", 343.0)
        self.mic_positions = np.array(array_config.get("geometry", []))
        
        if len(self.mic_positions) == 0:
            # Fallback to simple linear array if config missing
            self.mic_positions = np.array([
                [0.0, 0.0, 0.0],
                [0.04, 0.0, 0.0]
            ])
            
    def spatialise_signal(self, mono_signal: np.ndarray, azimuth_deg: float, elevation_deg: float = 0.0) -> np.ndarray:
        """
        Converts a mono signal into a multichannel signal by simulating
        propagation delays to the microphone array.
        
        Args:
            mono_signal: (N_samples,) 1D array.
            azimuth_deg: Source direction in XY plane.
            elevation_deg: Source elevation.
            
        Returns:
            (N_channels, N_samples) multichannel array.
        """
        # 1. Get source direction vector
        source_vec = azimuth_elevation_to_vector(azimuth_deg, elevation_deg)
        
        # 2. Calculate relative distance differences (positive = closer)
        # distance_diffs shape: (N_mics,)
        distance_diffs = calculate_steering_vector(self.mic_positions, source_vec)
        
        # 3. Convert to time delays
        # If distance is positive (closer), it arrives earlier.
        # So we want a NEGATIVE delay (shift left).
        # T = -d / c
        delays = -distance_diffs / self.speed_of_sound
        
        # 4. Apply shifts
        return apply_subsample_shifts(mono_signal, delays, self.sample_rate)

    def mix_signals(self, signal1: np.ndarray, signal2: np.ndarray, snr_db: float) -> np.ndarray:
        """
        Mixes signal2 into signal1 with a specific Signal-to-Noise Ratio (SNR).
        Handles both mono (1D) and multichannel (2D) inputs.
        
        Args:
            signal1: Primary signal (target).
            signal2: Secondary signal (interference).
            snr_db: Desired SNR in decibels.
            
        Returns:
            Mixed signal.
        """
        # Ensure dimensions match
        if signal1.ndim != signal2.ndim:
            raise ValueError("Signal dimensions must match for mixing.")
            
        # Calculate power (average across all channels if multichannel)
        p1 = self._calculate_power(signal1)
        p2 = self._calculate_power(signal2)
        
        # Avoid div by zero
        if p2 == 0:
            return signal1
            
        # Calculate scaling factor
        target_p2 = p1 / (10 ** (snr_db / 10))
        scale = np.sqrt(target_p2 / p2)
        
        mixed = signal1 + (signal2 * scale)
        
        # Normalize to prevent clipping (max amplitude 1.0)
        max_val = np.max(np.abs(mixed))
        if max_val > 0:
            mixed /= max_val
            
        return mixed

    def _calculate_power(self, signal: np.ndarray) -> float:
        return np.mean(signal ** 2)
