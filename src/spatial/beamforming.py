import numpy as np

from src.spatial.physics import (
    apply_subsample_shifts,
    azimuth_elevation_to_vector,
    calculate_steering_vector,
)
from src.utils import CONFIG


class Beamformer:
    """
    Handles spatial filtering and Direction of Arrival (DoA) estimation.
    Designed for multichannel microphone arrays.
    """
    def __init__(self, sample_rate: int = None):
        if sample_rate is None:
            self.sample_rate = CONFIG.get("audio", {}).get("sample_rate", 250000)
        else:
            self.sample_rate = sample_rate
        
        # Load array configuration
        array_config = CONFIG.get("array", {})
        self.speed_of_sound = array_config.get("speed_of_sound", 343.0)
        
        # Priority: 1. Explicit geometry (list or dict), 2. Grid parameters, 3. Fallback
        if "geometry" in array_config and array_config["geometry"]:
            geo = array_config["geometry"]
            if isinstance(geo, dict):
                # Handle dictionary mapping (e.g., channel_0: [x, y, z])
                # Sort keys numerically to ensure channel 0 is index 0, etc.
                import re
                def extract_number(s):
                    match = re.search(r'\d+', s)
                    return int(match.group()) if match else 0
                
                sorted_keys = sorted(geo.keys(), key=extract_number)
                self.mic_positions = np.array([geo[k] for k in sorted_keys])
            else:
                self.mic_positions = np.array(geo)
        elif array_config.get("type") == "grid":
            grid_config = array_config.get("grid", {})
            rows = grid_config.get("rows", 4)
            cols = grid_config.get("cols", 4)
            sx = grid_config.get("spacing_x", 0.04)
            sy = grid_config.get("spacing_y", 0.04)
            
            # Generate 4x4 grid (or rows x cols)
            # Centered around (0,0)
            x = np.arange(cols) * sx
            y = np.arange(rows) * sy
            x -= np.mean(x)
            y -= np.mean(y)
            
            xx, yy = np.meshgrid(x, y)
            self.mic_positions = np.zeros((rows * cols, 3))
            self.mic_positions[:, 0] = xx.flatten()
            self.mic_positions[:, 1] = yy.flatten()
        else:
            # Fallback
            self.mic_positions = np.array([
                [0.0, 0.0, 0.0],
                [0.04, 0.0, 0.0]
            ])
        
        self.n_channels = len(self.mic_positions)

    def estimate_doa(self, multichannel_signal: np.ndarray) -> float:
        """
        Estimates the Direction of Arrival (DoA) using MUSIC algorithm.
        
        Args:
            multichannel_signal: Input signal (Channels x Time).
            
        Returns:
            Estimated azimuth in degrees.
        """
        from src.spatial.estimators import MUSIC
        estimator = MUSIC(self.sample_rate, self.mic_positions, self.speed_of_sound)
        return estimator.estimate(multichannel_signal)

    def delay_and_sum(
        self,
        multichannel_signal: np.ndarray,
        azimuth_deg: float,
        elevation_deg: float = 0.0,
    ) -> np.ndarray:
        """
        Performs Delay-and-Sum beamforming towards a specific azimuth.
        Reverses the propagation delays to align signals from the target direction.

        Args:
            multichannel_signal: Input signal (Channels x Time).
            azimuth_deg: Target direction in degrees.
            elevation_deg: Target elevation.

        Returns:
            Beamformed single-channel signal.
        """
        # 1. Get steering vector for target direction
        source_vec = azimuth_elevation_to_vector(azimuth_deg, elevation_deg)
        
        # 2. Calculate propagation delays (relative to array center/ref)
        # These are the delays the signal experienced arriving at the mics.
        # To align them, we must apply the NEGATIVE of these relative delays.
        # Wait, check logic:
        # If Mic 2 received signal 1ms early (relative delay = -1ms),
        # we must delay it by +1ms to align with Mic 1.
        # From physics.py: calculate_steering_vector returns DISTANCE difference.
        # Positive distance = closer = earlier arrival.
        # arrival_time_diff = -distance_diff / c.
        # correction_delay = -arrival_time_diff = +distance_diff / c.
        
        distance_diffs = calculate_steering_vector(self.mic_positions, source_vec)
        correction_delays = distance_diffs / self.speed_of_sound
        
        # 3. Apply correction shifts
        aligned_signals = apply_subsample_shifts(
            multichannel_signal, correction_delays, self.sample_rate
        )
        
        # 4. Sum (and normalize by number of channels)
        beamformed_signal = np.mean(aligned_signals, axis=0)
        
        return beamformed_signal
