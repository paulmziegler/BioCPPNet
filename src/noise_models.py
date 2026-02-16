import numpy as np


class NoiseGenerator:
    """
    Base class for noise generation.
    """
    def __init__(self, sample_rate: int = 250000):
        self.sample_rate = sample_rate

    def generate(self, duration: float, num_channels: int = 1) -> np.ndarray:
        raise NotImplementedError

class WhiteNoise(NoiseGenerator):
    """
    Generates Additive White Gaussian Noise (AWGN).
    Simulates thermal sensor noise.
    """

    def generate(
        self, duration: float, num_channels: int = 1, std: float = 1.0
    ) -> np.ndarray:
        n_samples = int(duration * self.sample_rate)
        return np.random.normal(0, std, size=(num_channels, n_samples))


class ColoredNoise(NoiseGenerator):
    """
    Generates colored noise with 1/f^alpha power spectral density.
    alpha=0: White noise
    alpha=1: Pink noise (Wind-like)
    alpha=2: Brown noise (Heavier wind/rumble)
    """

    def __init__(self, sample_rate: int = 250000, color: str = "pink"):
        super().__init__(sample_rate)
        self.color = color.lower()
        if self.color == "pink":
            self.alpha = 1.0
        elif self.color == "brown":
            self.alpha = 2.0
        elif self.color == "white":
            self.alpha = 0.0
        else:
            raise ValueError(f"Unknown noise color: {color}")

    def generate(
        self, duration: float, num_channels: int = 1, std: float = 1.0
    ) -> np.ndarray:
        n_samples = int(duration * self.sample_rate)
        
        # Method: Spectral shaping
        # 1. Generate White Noise
        white = np.random.normal(0, 1.0, size=(num_channels, n_samples))
        
        if self.alpha == 0.0:
            return white * std
            
        # 2. FFT
        # rfft for real input
        fft_white = np.fft.rfft(white, axis=-1)
        freqs = np.fft.rfftfreq(n_samples, d=1/self.sample_rate)
        
        # 3. Create 1/f^(alpha/2) filter (amplitude, so sqrt of power)
        # Avoid division by zero at DC (index 0)
        with np.errstate(divide='ignore'):
            scale = 1 / (freqs ** (self.alpha / 2.0))
        scale[0] = 0  # Remove DC component
        
        # 4. Apply filter
        fft_colored = fft_white * scale
        
        # 5. IFFT
        colored = np.fft.irfft(fft_colored, n=n_samples, axis=-1)
        
        # 6. Normalize to desired std
        # Calculate current std per channel
        current_std = np.std(colored, axis=-1, keepdims=True)
        # Avoid div by zero
        current_std[current_std == 0] = 1.0
        
        colored = (colored / current_std) * std
        
        return colored

class RainNoise(NoiseGenerator):
    """
    Simulates rain as sparse impulsive noise.
    """

    def generate(
        self,
        duration: float,
        num_channels: int = 1,
        rate_hz: float = 10.0,
        amplitude: float = 1.0,
    ) -> np.ndarray:
        """
        Args:
            rate_hz: Average number of drops per second.
            amplitude: Peak amplitude of drops.
        """
        n_samples = int(duration * self.sample_rate)
        noise = np.zeros((num_channels, n_samples))
        
        # Probability of a drop per sample
        p_drop = rate_hz / self.sample_rate
        
        # Generate drops
        # Using binomial or just random threshold
        # For efficiency, generate indices
        num_drops = int(n_samples * p_drop)
        
        for ch in range(num_channels):
            indices = np.random.choice(n_samples, num_drops, replace=False)
            # Random amplitudes (e.g., uniform 0.5 to 1.0 * amplitude)
            amps = np.random.uniform(0.5, 1.0, size=num_drops) * amplitude
            # Random sign? Rain drops are impacts, usually one direction? 
            # But microphone diaphragm oscillates. Let's assume bipolar spike.
            signs = np.random.choice([-1, 1], size=num_drops)
            
            noise[ch, indices] = amps * signs
            
        return noise
