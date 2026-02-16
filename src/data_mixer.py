import numpy as np

from src.noise_models import ColoredNoise, RainNoise, WhiteNoise
from src.spatial.physics import (
    apply_subsample_shifts,
    azimuth_elevation_to_vector,
    calculate_steering_vector,
)
from src.spatial.reverb import ReverbGenerator
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
            
        self.reverb = ReverbGenerator(sample_rate)
            
    def spatialise_signal(
        self,
        mono_signal: np.ndarray,
        azimuth_deg: float,
        elevation_deg: float = 0.0,
        add_reverb: bool = False,
        rt60: float = 0.5,
        direct_ratio: float = 0.5,
    ) -> np.ndarray:
        """
        Converts a mono signal into a multichannel signal by simulating
        propagation delays to the microphone array. Optionally adds reverb.

        Args:
            mono_signal: (N_samples,) 1D array.
            azimuth_deg: Source direction in XY plane.
            elevation_deg: Source elevation.
            add_reverb: If True, adds stochastic reverb tail.
            rt60: Reverberation time if adding reverb.
            direct_ratio: Ratio of direct signal amplitude vs reverb (0.0 to 1.0).
                          1.0 = Only direct (no reverb tail). 0.0 = Only reverb.

        Returns:
            (N_channels, N_samples) multichannel array.
        """
        # 1. Direct Path (Precise Delay)
        source_vec = azimuth_elevation_to_vector(azimuth_deg, elevation_deg)
        distance_diffs = calculate_steering_vector(self.mic_positions, source_vec)
        delays = -distance_diffs / self.speed_of_sound
        
        direct_signal = apply_subsample_shifts(mono_signal, delays, self.sample_rate)
        
        if not add_reverb:
            return direct_signal
            
        # 2. Generate Reverb Tail
        # Use stochastic model for now (uncorrelated or partially correlated)
        n_channels = len(self.mic_positions)
        rir_tail = self.reverb.generate_stochastic_rir(
            n_channels, rt60=rt60, direct_ratio=direct_ratio
        )
        
        # 3. Convolve
        reverberant_tail = self.reverb.apply_reverb(mono_signal, rir_tail)
        
        # 4. Mix Direct + Tail
        # Ensure lengths match (convolution increases length)
        # Pad direct signal to match reverb tail length
        max_len = max(direct_signal.shape[1], reverberant_tail.shape[1])
        
        # Pad direct
        pad_direct = max_len - direct_signal.shape[1]
        if pad_direct > 0:
            direct_signal = np.pad(direct_signal, ((0,0), (0, pad_direct)))
            
        # Pad reverb (should be longer usually, but just in case)
        pad_reverb = max_len - reverberant_tail.shape[1]
        if pad_reverb > 0:
            reverberant_tail = np.pad(reverberant_tail, ((0,0), (0, pad_reverb)))
            
        return direct_signal + reverberant_tail

    def apply_sensor_perturbation(
        self,
        signal: np.ndarray,
        gain_std_db: float = 1.0,
        phase_std_deg: float = 5.0,
    ) -> np.ndarray:
        """
        Simulates hardware mismatch by applying random gain and phase offsets.

        Args:
            signal: (N_channels, N_samples)
            gain_std_db: Standard deviation of gain error in dB.
            phase_std_deg: Std of phase error (simulated via time jitter).
        """
        n_channels = signal.shape[0]
        
        # Gain mismatch
        gains_db = np.random.normal(0, gain_std_db, n_channels)
        gains_lin = 10 ** (gains_db / 20.0)
        
        # Time Jitter (Phase mismatch)
        # 5 degrees at 250kHz is tiny. 5 degrees at 1kHz is larger.
        # Let's use time jitter: +/- 1 sample (4 microseconds) covers
        # a lot of phase at high freq.
        # jitter_samples = np.random.uniform(-0.5, 0.5, n_channels)
        # We can use apply_subsample_shifts for this!

        # Convert phase_std at 50kHz (center freq) to time?
        # 360 deg = 1/50k = 20us. 5 deg = (5/360)*20us = 0.27us.
        # This is very small.
        jitter_seconds = np.random.normal(0, 1e-6, n_channels)  # 1 microsecond jitter

        # Apply jitter
        signal_jittered = apply_subsample_shifts(
            signal, jitter_seconds, self.sample_rate
        )
        # Apply gain
        return signal_jittered * gains_lin[:, None]

    def mix_signals(
        self, signal1: np.ndarray, signal2: np.ndarray, snr_db: float
    ) -> np.ndarray:
        """
        Mixes signal2 into signal1 with a specific Signal-to-Noise Ratio (SNR).
        Handles both mono (1D) and multichannel (2D) inputs.
        """
        if signal1.ndim != signal2.ndim:
            raise ValueError("Signal dimensions must match for mixing.")
            
        p1 = self._calculate_power(signal1)
        p2 = self._calculate_power(signal2)
        
        if p2 == 0:
            return signal1
            
        target_p2 = p1 / (10 ** (snr_db / 10))
        scale = np.sqrt(target_p2 / p2)
        
        mixed = signal1 + (signal2 * scale)
        
        max_val = np.max(np.abs(mixed))
        if max_val > 0:
            mixed /= max_val
            
        return mixed
        
    def mix_multiple(self, sources: list) -> np.ndarray:
        """
        Mixes an arbitrary list of sources (already scaled/spatialised).
        
        Args:
            sources: List of (N_channels, N_samples) arrays.
            
        Returns:
            Mixed signal.
        """
        if not sources:
            return np.array([])
            
        # Find max length
        max_len = max(s.shape[1] for s in sources)
        n_channels = sources[0].shape[0]
        
        mixed = np.zeros((n_channels, max_len))
        
        for s in sources:
            # Pad to match max length
            current_len = s.shape[1]
            if current_len < max_len:
                padded = np.pad(s, ((0,0), (0, max_len - current_len)))
                mixed += padded
            else:
                mixed += s
                
        # Normalize
        max_val = np.max(np.abs(mixed))
        if max_val > 0:
            mixed /= max_val
            
        return mixed

    def add_noise(
        self,
        signal: np.ndarray,
        noise_type: str = "white",
        snr_db: float = 20.0,
        **kwargs,
    ) -> np.ndarray:
        """
        Adds synthetic noise to the signal.
        """
        duration = signal.shape[-1] / self.sample_rate
        num_channels = signal.shape[0] if signal.ndim > 1 else 1
        
        # Instantiate generator
        if noise_type == 'white':
            gen = WhiteNoise(self.sample_rate)
            noise = gen.generate(duration, num_channels, std=1.0)
        elif noise_type in ['pink', 'brown']:
            gen = ColoredNoise(self.sample_rate, color=noise_type)
            noise = gen.generate(duration, num_channels, std=1.0)
        elif noise_type == 'rain':
            gen = RainNoise(self.sample_rate)
            rate = kwargs.get('rate_hz', 10.0)
            amp = kwargs.get('amplitude', 1.0)
            noise = gen.generate(duration, num_channels, rate_hz=rate, amplitude=amp)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
            
        # Ensure noise shape matches signal (handle 1D case)
        if signal.ndim == 1:
            noise = noise[0] # Take first channel
            
        # Use existing mix logic
        return self.mix_signals(signal, noise, snr_db)

    def _calculate_power(self, signal: np.ndarray) -> float:
        return np.mean(signal ** 2)
